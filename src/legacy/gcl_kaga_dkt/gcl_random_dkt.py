import os
import math
import json
import numpy as np
import torch
import pandas as pd
import hashlib
import pickle

# <--- 新增：修复 Tokenizer 死锁问题
# 必须在 import transformers 之前设置
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Embedding, LSTM, Linear, BCEWithLogitsLoss, ReLU
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F


hyper_params = {
    "data_params": {
        "min_len": 5,
        "max_len": 200,
        "need_text": True
    },
    "model_params": {
        "emb_dim": 128,
        "hidden_dim": 128,
        "g_layers": 2,
        "g_dropout": 0.1,
        "text_emb_model": "/home/jump/dzt/LLM/Qwen3-Embedding-0.6B",
        "text_emb_max_len": 256,
        "text_emb_batch_size": 128
    },
    "exp_params": {
        "device": "cuda:0",
        "batch_size": 1024,
        "lr": 0.01,
        "num_epochs": 50,
        "weight_decay": 1e-5,
        "num_workers": 0,
        "pin_memory": False,
        "gcl_lambda": 0.3,      # 辅助任务的权重 lambda
        "gcl_temp": 0.2,        # InfoNCE temperature
        "gcl_drop_edge": 0.2,   # 随机删边概率
        "gcl_mask_feat": 0.2    # 随机特征掩码概率
    }
}


class PathConfig:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data" / "xe"
        self.folds_dir = self.data_dir
        self.cache_dir = self.base_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.embed_model_dir = Path(hyper_params["model_params"]["text_emb_model"])
        if not self.embed_model_dir.is_absolute():
             self.embed_model_dir = self.base_dir / hyper_params["model_params"]["text_emb_model"]


class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def _compute_hash(self, texts: list, model_dir: Path, max_length: int) -> str:
        content = json.dumps(texts, sort_keys=True) + str(model_dir.name) + str(max_length)
        return hashlib.md5(content.encode()).hexdigest()

    def get_embeddings(self, texts: list, model_dir: Path, device=None, 
                       batch_size: int = 256, max_length: int = 256, 
                       normalize: bool = True, force_recompute: bool = False) -> torch.Tensor:
        cache_hash = self._compute_hash(texts, model_dir, max_length)
        cache_file = self.cache_dir / f"embeddings_{cache_hash}.pkl"
        
        if cache_file.exists() and not force_recompute:
            print(f"[Cache] 命中：{cache_file}")
            with open(cache_file, 'rb') as f:
                arr = pickle.load(f)
            t = torch.tensor(arr, dtype=torch.float32)
            return t.to(device) if device is not None else t

        print(f"[Cache] 计算并写入：{cache_file}")
        embs = self._compute_embeddings(texts, model_dir, device, batch_size, max_length, normalize)
        
        # 确保目录存在
        self.cache_dir.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(embs.cpu().numpy(), f)
        return embs.to(device) if device is not None else embs

    def _compute_embeddings(self, texts: list, model_dir: Path, device=None,
                            batch_size: int = 256, max_length: int = 256,
                            normalize: bool = True) -> torch.Tensor:
        print(f"[Embed] 使用模型：{model_dir} 计算 {len(texts)} 条文本嵌入")
        # 确保模型路径是字符串
        model_dir_str = str(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir_str, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir_str, trust_remote_code=True).to(device)
        model.eval()

        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="计算节点embedding", leave=False):
                batch = texts[i:i + batch_size]
                if not batch: continue # 处理空batch
                
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                emb = summed / lengths
                if normalize:
                    emb = F.normalize(emb, p=2, dim=1)
                all_embeds.append(emb.cpu())

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if not all_embeds:
            # 如果输入 texts 为空，返回一个正确形状的空张量
            # 假设模型输出维度，从 config 中读取（这里用 1024 占位）
            # 更好的方法是从 model.config.hidden_size 读取
            fallback_dim = 1024 
            print("警告：没有文本被编码，返回空张量")
            return torch.empty(0, fallback_dim, dtype=torch.float32)
            
        return torch.cat(all_embeds, dim=0)


class EidDataset(Dataset):
    def __init__(self, records_path: Path, data_params: Dict[str, Any]):
        self.min_len = data_params["min_len"]
        self.max_len = data_params["max_len"]

        try:
            with open(records_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Records file not found at {records_path}")
            lines = []
            
        self.records: List[Tuple[List[int], List[int]]] = []
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                eids_str = lines[i].strip()
                is_corrects_str = lines[i + 1].strip()
                # 确保在拆分后进行非空检查
                eids_int = [int(eid) for eid in eids_str.split(",") if eid.strip()]
                is_corrects_int = [int(ic) for ic in is_corrects_str.split(",") if ic.strip()]
                
                if len(eids_int) == len(is_corrects_int) and eids_int:
                    self.records.append((eids_int, is_corrects_int))

        self._filter_records()

        segmented_records = []
        for eids, is_corrects in self.records:
            for seg_e, seg_c in self._to_segments(eids, is_corrects):
                segmented_records.append((seg_e, seg_c))
        self.records = segmented_records

    def _filter_records(self):
        self.records = [rec for rec in self.records if len(rec[0]) >= self.min_len]

    def _to_segments(self, eids: List[int], is_corrects: List[int]):
        max_len = self.max_len
        # 从头开始切片
        start = 0
        while start < len(eids):
            end = start + max_len
            yield (eids[start:end], is_corrects[start:end])
            start = end # <--- 修改：确保 segments 不重叠且连续

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        return {"eids": self.records[idx][0], "is_corrects": self.records[idx][1]}


def Eid_collate_fn(batch):
    eids = [d["eids"] for d in batch]
    is_corrects = [d["is_corrects"] for d in batch]
    lengths = [len(seq) for seq in eids]
    
    padded_eids = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in eids], batch_first=True, padding_value=0)
    padded_is_corrects = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in is_corrects], batch_first=True, padding_value=0)
    
    mask = torch.zeros_like(padded_eids, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True
        
    return {"eids": padded_eids, "is_corrects": padded_is_corrects, "mask": mask}


class GraphCLUtil(Module):
    def __init__(self, temperature=0.2, drop_edge_rate=0.1, mask_feat_rate=0.1):
        super().__init__()
        self.temperature = temperature
        self.drop_edge_rate = drop_edge_rate
        self.mask_feat_rate = mask_feat_rate

    def augment(self, data: HeteroData) -> HeteroData:
        """生成图的一个视图：随机删边 + 特征掩码"""
        # 必须 Clone，防止修改原始图结构
        view = data.clone()
        
        # 1. 随机删边 (Edge Dropping)
        # 针对主要的边类型 ('exercise', 'covers', 'skill')
        edge_type = ("exercise", "covers", "skill")
        if edge_type in view.edge_index_dict:
            edge_index = view[edge_type].edge_index
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_edge_rate
            view[edge_type].edge_index = edge_index[:, mask]
            # 注意：rev_covers 会在 GraphEncoder.forward 中根据 covers 自动重建，所以这里不用手动删反向边
            
        # 2. 随机特征掩码 (Feature Masking)
        for ntype in view.node_types:
            x = view[ntype].x
            if x is not None:
                mask = torch.rand_like(x) > self.mask_feat_rate
                view[ntype].x = x * mask.float() # 将部分特征置零
        
        return view

    def compute_infonce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        计算 InfoNCE Loss
        z1, z2: [N, D] 同一节点在两个视图下的 Embedding
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 相似度矩阵 [N, N]
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # 正样本对在对角线上
        # LogSoftmax(sim_matrix) 实际上是在做分类：哪一个是对应的正样本
        # 对每一行（Query），目标是第 i 列（Key）
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class GraphEncoder(Module):
    def __init__(self, emb_dim: int, num_layers: int = 2, dropout: float = 0.1, aggr: str = "mean", residual: bool = True, layernorm: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        convs, norms = [], []
        for _ in range(num_layers):
            convs.append(HeteroConv({
                ("exercise", "covers", "skill"): SAGEConv((-1, -1), emb_dim),
                ("skill", "rev_covers", "exercise"): SAGEConv((-1, -1), emb_dim),
            }, aggr=aggr))
            norms.append(torch.nn.ModuleDict({
                "exercise": torch.nn.LayerNorm(emb_dim),
                "skill": torch.nn.LayerNorm(emb_dim),
            }) if layernorm else None)
            
        self.convs = torch.nn.ModuleList(convs)
        self.norms = torch.nn.ModuleList(norms)
        self.drop = torch.nn.Dropout(dropout)
        self.act = ReLU()

    @staticmethod
    def _ensure_reverse_edge(data: HeteroData):
        fwd_key = ("exercise", "covers", "skill")
        rev_key = ("skill", "rev_covers", "exercise")
        if fwd_key in data.edge_index_dict and rev_key not in data.edge_index_dict:
            ei = data.edge_index_dict[fwd_key]
            # 确保边索引是有效的
            if ei.numel() > 0:
                data[rev_key].edge_index = torch.stack([ei[1], ei[0]], dim=0).contiguous()
            else:
                # 如果没有边，创建一个空的
                data[rev_key].edge_index = torch.empty((2, 0), dtype=torch.long, device=ei.device)
        return data

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        data = self._ensure_reverse_edge(data)
        x_dict = {ntype: data[ntype].x for ntype in data.node_types}

        for l, conv in enumerate(self.convs):
            prev = {k: v for k, v in x_dict.items()}
            # 检查是否有边，SAGEConv 在没有边时可能会报错
            if not data.edge_index_dict:
                out = {} # 如果没有边，GNN 不做任何事
            else:
                out = conv(x_dict, data.edge_index_dict)
                
            for ntype, h in out.items():
                h = self.act(h)
                if self.norms[l] is not None:
                    h = self.norms[l][ntype](h)
                if self.residual and ntype in prev:
                    h = h + prev[ntype]
                out[ntype] = self.drop(h)
            
            # 确保没有参与 GNN 的节点特征也能传递下去
            x_dict.update(out) 

        self._last_x = x_dict
        return x_dict

    def gather_exercise_seq(self, eids: torch.Tensor, eid_lookup: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "_last_x"), "请先调用 forward(data) 完成图编码。"
        # 确保 exercise 节点存在
        if "exercise" not in self._last_x:
            print("警告: GNN 输出中没有 'exercise' 节点。")
            # 返回一个零张量作为后备
            B, S = eids.shape
            return torch.zeros(B, S, self.emb_dim, device=eids.device)

        ex_node_feats = self._last_x["exercise"]
        device = ex_node_feats.device
        eid_lookup = eid_lookup.to(device)
        
        # 确保 eids 也在正确的 device
        row_ids = eid_lookup[eids.to(device)]
        
        pad_row = torch.zeros(1, ex_node_feats.size(-1), device=device)
        table = torch.cat([pad_row, ex_node_feats], dim=0)
        
        return table[row_ids]


class KAGADKT(Module):
    def __init__(self, model_params: Dict[str, Any], gcl_params: Dict[str, Any] = None):
        super().__init__()
        self.embed_dim = model_params["emb_dim"]
        self.hidden_dim = model_params["hidden_dim"]
        
        # GNN 编码器
        self.gnn = GraphEncoder(
            emb_dim=self.embed_dim,
            num_layers=model_params.get("g_layers", 2),
            dropout=model_params.get("g_dropout", 0.1)
        )
        
        # 对比学习工具
        if gcl_params:
            self.gcl_util = GraphCLUtil(
                temperature=gcl_params.get("gcl_temp", 0.2),
                drop_edge_rate=gcl_params.get("gcl_drop_edge", 0.1),
                mask_feat_rate=gcl_params.get("gcl_mask_feat", 0.1)
            )
        else:
            self.gcl_util = None

        self.ans_embed = Embedding(3, self.embed_dim, padding_idx=0)
        self.lstm_layer = LSTM(self.embed_dim * 2, self.hidden_dim, batch_first=True)
        self.out_layer = Linear(self.hidden_dim, 1)
        
        self._graph_data: Optional[HeteroData] = None
        self._eid_lookup: Optional[torch.Tensor] = None

    def set_graph(self, data: HeteroData, eid_lookup: torch.Tensor):
        self._graph_data = data
        self._eid_lookup = eid_lookup

    def get_main_task_loss(self, eids, is_corrects, mask):
        """计算主任务 (DKT) 的 Embedding 和 Logits"""
        # 1. 使用原始图进行一次前向传播，获取当前的节点表示
        # 注意：在训练模式下，每次都需要跑一遍 GNN 以获得梯度
        self.gnn(self._graph_data)
        
        ex_seq = self.gnn.gather_exercise_seq(eids, self._eid_lookup)
        
        ans_idx = is_corrects + 1
        ans_idx = ans_idx.masked_fill(~mask, 0)
        ans_seq = self.ans_embed(ans_idx)
        
        seq_in = torch.cat([ex_seq, ans_seq], dim=-1)
        h, _ = self.lstm_layer(seq_in)
        logits = self.out_layer(h)
        return logits

    def compute_gcl_loss(self) -> torch.Tensor:
        """计算辅助任务 (GCL) 的 Loss"""
        if self.gcl_util is None or self._graph_data is None:
            return torch.tensor(0.0, device=self.out_layer.weight.device)
            
        # 1. 生成两个视图
        view1 = self.gcl_util.augment(self._graph_data)
        view2 = self.gcl_util.augment(self._graph_data)
        
        # 2. GNN 编码两个视图
        # GNN forward 会更新内部状态，但这里我们只需要返回值
        out1 = self.gnn(view1) 
        out2 = self.gnn(view2)
        
        # 3. 计算 Exercise 节点的对比损失
        # 也可以加入 Skill 节点的对比损失，这里以 Exercise 为主
        loss_ex = self.gcl_util.compute_infonce(out1['exercise'], out2['exercise'])
        loss_sk = self.gcl_util.compute_infonce(out1['skill'], out2['skill'])
        
        return loss_ex + loss_sk

    def forward(self, eids, is_corrects, mask):
        # 向后兼容
        return self.get_main_task_loss(eids, is_corrects, mask)


def build_graph_with_text(eid2sids, eid2desc, sid2desc, path_config, model_params, device):
    # <--- 修改：从多个来源构建节点ID列表，确保完整性
    exercise_ids_set = set()
    skill_ids_set = set()
    
    # 从 eid2sids 中收集 exercise IDs
    exercise_ids_set.update(eid2sids.keys())
    
    # 从 eid2desc 中收集 exercise IDs（处理字符串key）
    for k in eid2desc.keys():
        try:
            exercise_ids_set.add(int(k))
        except (ValueError, TypeError):
            continue
    
    # 从 eid2sids 中收集 skill IDs
    for sids in eid2sids.values():
        skill_ids_set.update(sids)
    
    # 从 sid2desc 中收集 skill IDs（处理字符串key）
    for k in sid2desc.keys():
        try:
            skill_ids_set.add(int(k))
        except (ValueError, TypeError):
            continue
    
    exercise_ids = sorted(list(exercise_ids_set))
    skill_ids = sorted(list(skill_ids_set))
    
    print(f"[图构建] 找到 {len(exercise_ids)} 个 exercise 节点, {len(skill_ids)} 个 skill 节点")
    
    eid2idx = {eid: i for i, eid in enumerate(exercise_ids)}
    sid2idx = {sid: i for i, sid in enumerate(skill_ids)}

    ex_texts = [eid2desc.get(str(eid), "") for eid in exercise_ids]
    sk_texts = [sid2desc.get(str(sid), "") for sid in skill_ids]

    cache = EmbeddingCache(path_config.cache_dir)
    text_emb_model_dir = path_config.embed_model_dir
    
    ex_emb = cache.get_embeddings(ex_texts, text_emb_model_dir, device=device,
                                  batch_size=model_params['text_emb_batch_size'],
                                  max_length=model_params['text_emb_max_len'])
    sk_emb = cache.get_embeddings(sk_texts, text_emb_model_dir, device=device,
                                  batch_size=model_params['text_emb_batch_size'],
                                  max_length=model_params['text_emb_max_len'])

    target_dim = model_params['emb_dim']
    
    # <--- 修改：投影层处理逻辑，确保不改变 batch 维度
    if ex_emb.numel() > 0:
        if ex_emb.shape[1] != target_dim:
            proj = Linear(ex_emb.shape[1], target_dim).to(device)
            with torch.no_grad(): 
                ex_emb = proj(ex_emb)
        # 确保 ex_emb 的第一维度等于 exercise_ids 的数量
        assert ex_emb.shape[0] == len(exercise_ids), f"Exercise embedding 数量不匹配: {ex_emb.shape[0]} vs {len(exercise_ids)}"
    else:
        ex_emb = torch.zeros(len(exercise_ids), target_dim, device=device)

    if sk_emb.numel() > 0:
        if sk_emb.shape[1] != target_dim:
            proj = Linear(sk_emb.shape[1], target_dim).to(device)
            with torch.no_grad(): 
                sk_emb = proj(sk_emb)
        # 确保 sk_emb 的第一维度等于 skill_ids 的数量
        assert sk_emb.shape[0] == len(skill_ids), f"Skill embedding 数量不匹配: {sk_emb.shape[0]} vs {len(skill_ids)}"
    else:
        sk_emb = torch.zeros(len(skill_ids), target_dim, device=device)

    # <--- 新增：打印 embedding 形状，确认数据正确
    print(f"[图构建] Exercise embedding 形状: {ex_emb.shape}")
    print(f"[图构建] Skill embedding 形状: {sk_emb.shape}")

    data = HeteroData()
    data['exercise'].x = ex_emb
    data['skill'].x = sk_emb

    # <--- 新增：立即验证数据是否被正确赋值
    print(f"[图构建] 验证 HeteroData - Exercise 节点数: {data['exercise'].x.shape[0]}")
    print(f"[图构建] 验证 HeteroData - Skill 节点数: {data['skill'].x.shape[0]}")

    # 边构建
    ex2sk = []
    for eid, sids in eid2sids.items():
        if eid in eid2idx:
            src = eid2idx[eid]
            for sid in sids:
                if sid in sid2idx:
                    dst = sid2idx[sid]
                    ex2sk.append((src, dst))
    
    print(f"[图构建] 找到 {len(ex2sk)} 条边")
    
    if ex2sk:
        data['exercise', 'covers', 'skill'].edge_index = torch.tensor(ex2sk, dtype=torch.long).t().contiguous()
    else:
        data['exercise', 'covers', 'skill'].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # <--- 新增：返回前最终检查
    print(f"[图构建] 返回前检查 - Exercise: {data['exercise'].x.shape}, Skill: {data['skill'].x.shape}")
    
    return data.to(device), eid2idx, sid2idx


def make_eid_lookup(eid2idx: Dict[int, int], max_eid: int) -> torch.Tensor:
    # 确保 max_eid 至少为 0，防止负数索引
    lookup_size = max(max_eid + 1, 1)
    lookup = torch.zeros(lookup_size, dtype=torch.long)
    
    for eid, idx in eid2idx.items():
        if 0 <= eid <= max_eid:
            lookup[eid] = int(idx) + 1 # 0 保留给 padding
    return lookup


class Experiment:
    def __init__(self, model_params, exp_params):
        self.model = None
        self.model_params = model_params
        self.exp_params = exp_params
        self.device = exp_params["device"]
        self.batch_size = exp_params["batch_size"]
        self.lr = exp_params["lr"]
        self.num_epochs = exp_params["num_epochs"]
        self.weight_decay = exp_params.get("weight_decay", 0.0)
        self.num_workers = exp_params.get("num_workers", 0)
        self.pin_memory = exp_params.get("pin_memory", False)
        self.criterion = BCEWithLogitsLoss(reduction='none')
        self.optimizer = None
        self.scheduler = None
        self._graph_data = None
        self._eid_lookup = None

    def inject_graph(self, graph_data: HeteroData, eid_lookup: torch.Tensor):
        self._graph_data = graph_data
        self._eid_lookup = eid_lookup

    def reset_config(self):
        assert self._graph_data is not None and self._eid_lookup is not None, "请先注入图"
        
        # 传递 gcl_params
        self.model = KAGADKT(
            self.model_params, 
            gcl_params=self.exp_params # 直接传 exp_params 里面包含了 gcl 配置
        ).to(self.device)
        
        self.model.set_graph(self._graph_data.to(self.device), self._eid_lookup.to(self.device))
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-6)

    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_losses, epoch_gcl_losses, epoch_bce_losses = [], [], []
        all_labels, all_preds = [], []
        
        # 获取 lambda 超参数
        lambda_gcl = self.exp_params.get("gcl_lambda", 0.0)
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {self.current_epoch:02d}", leave=False)
        
        for batch in pbar:
            eids, is_corrects, mask = batch["eids"].to(self.device), batch["is_corrects"].to(self.device), batch["mask"].to(self.device)
            
            # --- Step 1: 准备数据 ---
            input_eids, input_is_corrects = eids[:, :-1], is_corrects[:, :-1]
            input_mask = mask[:, :-1]
            labels, label_mask = is_corrects[:, 1:].float(), mask[:, 1:]
            
            if input_eids.size(1) == 0 or label_mask.sum() == 0: continue

            self.optimizer.zero_grad()

            # --- Step 2: 计算主任务 Loss (BCE) ---
            # 注意：调用 model.get_main_task_loss 会执行一次 GNN(original_graph)
            logits = self.model.get_main_task_loss(input_eids, input_is_corrects, input_mask).squeeze(-1)
            
            bce_loss = self.criterion(logits, labels)
            masked_bce_loss = (bce_loss * label_mask).sum() / label_mask.sum().clamp(min=1)

            # --- Step 3: 计算辅助任务 Loss (GCL) ---
            # 只有当 lambda > 0 时才计算
            if lambda_gcl > 0:
                gcl_loss = self.model.compute_gcl_loss()
            else:
                gcl_loss = torch.tensor(0.0, device=self.device)

            # --- Step 4: 联合优化 (Joint Optimization) ---
            total_loss = masked_bce_loss + lambda_gcl * gcl_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            # --- 记录与监控 ---
            epoch_losses.append(total_loss.item())
            epoch_bce_losses.append(masked_bce_loss.item())
            epoch_gcl_losses.append(gcl_loss.item())

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_labels.append(labels[label_mask].cpu().numpy())
                all_preds.append(probs[label_mask].cpu().numpy())
            
            pbar.set_postfix({
                "BCE": f"{masked_bce_loss.item():.4f}", 
                "GCL": f"{gcl_loss.item():.4f}",
                "Total": f"{total_loss.item():.4f}"
            })
            
        self.scheduler.step()
        
        y_true = np.concatenate(all_labels) if all_labels else np.array([])
        y_prob = np.concatenate(all_preds) if all_preds else np.array([])
        
        metrics = {
            "train_loss": np.mean(epoch_losses), 
            "train_bce": np.mean(epoch_bce_losses),
            "train_gcl": np.mean(epoch_gcl_losses)
        }

        if y_true.size > 0:
            if len(np.unique(y_true)) > 1:
                metrics["train_auc"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["train_auc"] = 0.5
            metrics["train_acc"] = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
        else:
            metrics["train_auc"] = float("nan")
            metrics["train_acc"] = float("nan")
            
        return metrics


    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        epoch_losses, all_labels, all_preds = [], [], []
        
        # <--- 修改：使用 tqdm 包装 val_loader
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        
        for batch in pbar:
            eids, is_corrects, mask = batch["eids"].to(self.device), batch["is_corrects"].to(self.device), batch["mask"].to(self.device)
            
            input_eids, input_is_corrects = eids[:, :-1], is_corrects[:, :-1]
            input_mask = mask[:, :-1] # <--- 新增
            labels, label_mask = is_corrects[:, 1:].float(), mask[:, 1:]
            
            if input_eids.size(1) == 0 or label_mask.sum() == 0: continue

            # <--- 修改：传入 input_mask
            logits = self.model(input_eids, input_is_corrects, input_mask).squeeze(-1)
            
            loss = self.criterion(logits, labels)
            masked_loss = (loss * label_mask).sum() / label_mask.sum().clamp(min=1)
            epoch_losses.append(masked_loss.item())
            
            probs = torch.sigmoid(logits)
            all_labels.append(labels[label_mask].cpu().numpy())
            all_preds.append(probs[label_mask].cpu().numpy())
            
            pbar.set_postfix(loss=masked_loss.item())

        y_true = np.concatenate(all_labels) if all_labels else np.array([])
        y_prob = np.concatenate(all_preds) if all_preds else np.array([])
        
        if y_true.size == 0:
            return {"val_loss": float("nan"), "val_auc": float("nan"), "val_acc": float("nan")}

        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5 # <--- 修改
        acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
        return {"val_loss": np.mean(epoch_losses), "val_auc": auc, "val_acc": acc}

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch # <--- 新增：用于 Tqdm
            train_metrics = self.train_one_epoch(train_loader)
            
            msg = f"[Epoch {epoch:02d}] Train: loss={train_metrics['train_loss']:.4f}, auc={train_metrics['train_auc']:.4f}, acc={train_metrics['train_acc']:.4f}"
            
            if val_loader:
                val_metrics = self.validate(val_loader)
                msg += f" | Val: loss={val_metrics['val_loss']:.4f}, auc={val_metrics['val_auc']:.4f}, acc={val_metrics['val_acc']:.4f}"
            
            print(msg) # 打印最终的 epoch 总结


def get_global_info(path_config: PathConfig, n_folds: int):
    def load_json(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f: return json.load(f)
        except FileNotFoundError:
            print(f"警告: JSON 文件未找到 {path}")
            return {}
        except json.JSONDecodeError:
            print(f"警告: JSON 文件格式错误 {path}")
            return {}
    
    eid2sids = {int(k): [int(v) for v in vs] for k, vs in load_json(path_config.folds_dir / "eid2sids.json").items()}
    eid2desc = load_json(path_config.folds_dir / "eid2desc.json")
    sid2desc = load_json(path_config.folds_dir / "sid2desc.json")

    global_eid_set = set(eid2sids.keys()) | set(int(k) for k in eid2desc.keys() if k.isdigit())
    
    for i in range(n_folds):
        for fname in ["train_records.txt", "valid_records.txt"]:
            records_path = path_config.folds_dir / f"fold{i}" / fname
            if records_path.exists():
                with open(records_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line_idx in range(0, len(lines), 2):
                    if line_idx < len(lines):
                        eids = [int(eid) for eid in lines[line_idx].strip().split(",") if eid.strip().isdigit()]
                        global_eid_set.update(eids)
    
    max_eid = max(global_eid_set) if global_eid_set else 0
    print(f"Global Max eid: {max_eid}")
    return max_eid, eid2sids, eid2desc, sid2desc


if __name__ == "__main__":
    # 设置 PyTorch 精度（如果可用）
    if hasattr(torch, "set_float32_matmul_precision"): 
        torch.set_float32_matmul_precision("medium")
        
    N_FOLDS = 5
    path_config = PathConfig()
    data_params = hyper_params["data_params"]
    model_params = hyper_params["model_params"]
    exp_params = hyper_params["exp_params"]
    device = exp_params["device"]
    
    # 确保 CUDA 可用
    if "cuda" in device and not torch.cuda.is_available():
        print(f"警告: {device} 不可用, 切换到 CPU.")
        device = "cpu"
        exp_params["device"] = "cpu"

    print(f"使用设备: {device}")
    print(f"数据目录: {path_config.data_dir}")
    print(f"模型目录: {path_config.embed_model_dir}")

    # 1. 加载全局信息和图数据
    global_max_eid, eid2sids, eid2desc, sid2desc = get_global_info(path_config, N_FOLDS)
    
    # 检查是否有数据
    if not eid2sids and not eid2desc and not sid2desc:
        print("错误：无法加载任何图数据 (eid2sids, eid2desc, sid2desc).")
        exit()

    graph_data, eid2idx, _ = build_graph_with_text(eid2sids, eid2desc, sid2desc, path_config, model_params, device)
    eid_lookup = make_eid_lookup(eid2idx, global_max_eid)

    print(f"\n图数据已构建. 节点类型: {graph_data.node_types}, 边类型: {graph_data.edge_types}")
    
    # <--- 修改：更详细的节点检查
    for node_type in graph_data.node_types:
        if hasattr(graph_data[node_type], 'x') and graph_data[node_type].x is not None:
            print(f"{node_type.capitalize()} 节点数: {graph_data[node_type].x.shape[0]}, 特征维度: {graph_data[node_type].x.shape[1]}")
        else:
            print(f"警告: {node_type} 节点没有特征!")
    
    # <--- 新增：检查边
    for edge_type in graph_data.edge_types:
        edge_index = graph_data[edge_type].edge_index
        print(f"边 {edge_type}: {edge_index.shape[1]} 条")
    
    # <--- 新增：验证图数据的完整性
    if 'exercise' not in graph_data.node_types or 'skill' not in graph_data.node_types:
        print("错误: 图数据缺少必要的节点类型!")
        exit()
    
    if graph_data['exercise'].x.shape[0] == 0 or graph_data['skill'].x.shape[0] == 0:
        print("错误: 节点特征为空!")
        exit()

    # 2. 初始化实验
    exp = Experiment(model_params, exp_params)
    # 注入图数据 (在 reset_config 之前)
    exp.inject_graph(graph_data, eid_lookup)

    fold_results = []
    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation")
    
    # 3. K-Fold 交叉验证
    for fold in range(N_FOLDS):
        print("-" * 50 + f"\nFold {fold}/{N_FOLDS-1}")
        
        # 重置模型、优化器等
        exp.reset_config()

        train_path = path_config.folds_dir / f"fold{fold}" / "train_records.txt"
        valid_path = path_config.folds_dir / f"fold{fold}" / "valid_records.txt"
        
        if not train_path.exists() or not valid_path.exists():
            print(f"警告: Fold {fold} 数据文件缺失，跳过。")
            continue

        train_dataset = EidDataset(train_path, data_params)
        valid_dataset = EidDataset(valid_path, data_params)
        
        if len(train_dataset) == 0 or len(valid_dataset) == 0:
            print(f"警告: Fold {fold} 数据集为空，跳过。")
            continue

        print(f"Train segments: {len(train_dataset)}, Validation segments: {len(valid_dataset)}")

        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=exp.batch_size, 
            shuffle=True, 
            collate_fn=Eid_collate_fn, 
            num_workers=exp.num_workers, 
            pin_memory=exp.pin_memory,
            persistent_workers=True if exp.num_workers > 0 else False # <--- 新增：提升性能
        )
        val_loader = DataLoader(
            valid_dataset, 
            batch_size=exp.batch_size, 
            shuffle=False, 
            collate_fn=Eid_collate_fn, 
            num_workers=exp.num_workers, 
            pin_memory=exp.pin_memory,
            persistent_workers=True if exp.num_workers > 0 else False # <--- 新增
        )

        # 训练和评估
        exp.fit(train_loader, val_loader)
        
        print(f"Fold {fold} 训练完成，计算最终验证集指标...")
        fold_val_metrics = exp.validate(val_loader) # 使用最佳模型（或最后epoch）
        print(f"Fold {fold} | Final Val AUC: {fold_val_metrics['val_auc']:.4f}, ACC: {fold_val_metrics['val_acc']:.4f}")
        fold_results.append(fold_val_metrics)

    # 4. 总结
    def _avg(key):
        vals = [fr[key] for fr in fold_results if fr and not np.isnan(fr[key])]
        return float(np.mean(vals)) if vals else float("nan")

    if fold_results:
        summary = {
            "mean_val_loss": _avg("val_loss"),
            "mean_val_auc": _avg("val_auc"),
            "mean_val_acc": _avg("val_acc")
        }
        print("\n" + "=" * 50 + "\nCV Summary:", f"val_auc={summary['mean_val_auc']:.4f}, val_acc={summary['mean_val_acc']:.4f}")
    else:
        print("\n" + "=" * 50 + "\nCV Summary: 没有可用的结果。")