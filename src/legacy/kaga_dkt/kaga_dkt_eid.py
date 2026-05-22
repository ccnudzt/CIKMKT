import os
import math
import json
import numpy as np
import torch
import pandas as pd

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Embedding, LSTM, Linear, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


hyper_params = {
    "data_params": {
        "min_len": 5,
        "max_len": 200,
        "need_text": False
    },
    "model_params": {
        "emb_dim": 32,        
        "hidden_dim": 32,
        "g_layers": 2,
        "g_dropout": 0.1
    },
    "exp_params": {
        "device": "cuda:1",
        "batch_size": 1024,
        "lr": 0.01,
        "num_epochs": 30,
        "weight_decay": 1e-5,
        "num_workers": 0,
        "pin_memory": False
    }
}


class PathConfig:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.base_dir / "data" / "mooc"
        self.folds_dir = self.data_dir

class EidDataset(Dataset):
    def __init__(self, eid2sids: Dict, records_path: Path, data_params: Dict[str, Any]):
        self.min_len = data_params["min_len"]
        self.max_len = data_params["max_len"]
        self.need_text = data_params["need_text"]
        self.eid2sids = eid2sids

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
                eids_int = [int(eid) for eid in eids_str.split(",") if eid]
                is_corrects_int = [int(ic) for ic in is_corrects_str.split(",") if ic]
                if len(eids_int) == len(is_corrects_int) and eids_int:
                    self.records.append((eids_int, is_corrects_int))

        self._filter_records()

        segmented_records = []
        for eids, is_corrects in self.records:
            for seg_e, seg_c in self._to_segments(eids, is_corrects):
                segmented_records.append((seg_e, seg_c))
        self.records = segmented_records

    def _filter_records(self):
        new_records = []
        for eids, is_corrects in self.records:
            if len(eids) >= self.min_len:
                new_records.append((eids, is_corrects))
        self.records = new_records

    def _to_segments(self, eids: List[int], is_corrects: List[int]):
        max_len = self.max_len
        while len(eids) > max_len:
            yield (eids[:max_len], is_corrects[:max_len])
            eids = eids[max_len:]
            is_corrects = is_corrects[max_len:]
        yield (eids, is_corrects)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        return {
            "eids": self.records[idx][0],
            "is_corrects": self.records[idx][1]
        }


def Eid_collate_fn(batch):
    eids = [d["eids"] for d in batch]
    is_corrects = [d["is_corrects"] for d in batch]
    lengths = [len(seq) for seq in eids]
    padded_eids = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in eids],
        batch_first=True,
        padding_value=0
    )
    padded_is_corrects = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in is_corrects],
        batch_first=True,
        padding_value=0
    )
    mask = torch.zeros_like(padded_eids, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True

    return {
        "eids": padded_eids,
        "is_corrects": padded_is_corrects,
        "mask": mask
    }


class GraphEncoder(Module):
    """
      nodes: 'exercise', 'skill'
      edges: ('exercise','covers','skill') + 自动补齐 ('skill','rev_covers','exercise')
    """
    def __init__(self, num_exercise: int, num_skill: int,
                 emb_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, aggr: str = "mean",
                 residual: bool = True, layernorm: bool = True):
        super().__init__()
        self.num_exercise = num_exercise
        self.num_skill = num_skill
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        self.ex_embed = Embedding(num_exercise + 1, emb_dim, padding_idx=0)
        self.sk_embed = Embedding(num_skill + 1, emb_dim, padding_idx=0)

        convs, norms = [], []
        for _ in range(num_layers):
            convs.append(HeteroConv(
                {
                    ("exercise", "covers", "skill"): SAGEConv((-1, -1), emb_dim),
                    ("skill", "rev_covers", "exercise"): SAGEConv((-1, -1), emb_dim),
                },
                aggr=aggr
            ))
            norms.append(torch.nn.ModuleDict({
                "exercise": torch.nn.LayerNorm(emb_dim),
                "skill": torch.nn.LayerNorm(emb_dim),
            }) if layernorm else None)
        self.convs = torch.nn.ModuleList(convs)
        self.norms = torch.nn.ModuleList(norms)
        self.drop = torch.nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.ex_embed.weight[1:])
        torch.nn.init.xavier_uniform_(self.sk_embed.weight[1:])
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, SAGEConv):
                    m.reset_parameters()

    @staticmethod
    def _ensure_reverse_edge(data: HeteroData):
        fwd_key = ("exercise", "covers", "skill")
        rev_key = ("skill", "rev_covers", "exercise")
        if fwd_key not in data.edge_index_dict:
            raise KeyError("缺少 ('exercise','covers','skill') 边。")
        if rev_key not in data.edge_index_dict:
            ei = data.edge_index_dict[fwd_key]
            rev_ei = torch.stack([ei[1], ei[0]], dim=0).contiguous()
            data[rev_key].edge_index = rev_ei
        return data

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        data = self._ensure_reverse_edge(data)

        x_dict = {
            "exercise": self.ex_embed.weight[1: 1 + self.num_exercise].to(device),
            "skill":    self.sk_embed.weight[1: 1 + self.num_skill].to(device),
        }

        for l, conv in enumerate(self.convs):
            prev = {k: v for k, v in x_dict.items()}
            out = conv(x_dict, data.edge_index_dict)
            for ntype, h in out.items():
                h = torch.relu(h)
                if self.norms[l] is not None:
                    h = self.norms[l][ntype](h)
                if self.residual:
                    h = h + prev[ntype]
                out[ntype] = self.drop(h)
            x_dict = out

        self._last_x = x_dict
        return x_dict

    def gather_exercise_seq(self, eids: torch.Tensor, eid_lookup: torch.Tensor) -> torch.Tensor:
        """
        eids: [B, L]，0 为 PAD；eid_lookup: [max_eid+1]，值域为图中 'exercise' 的行号+1（0为PAD）
        """
        assert hasattr(self, "_last_x"), "请先调用 forward(data) 完成图编码。"
        ex_node_feats = self._last_x["exercise"]  # [N_ex, D]
        device = ex_node_feats.device
        eid_lookup = eid_lookup.to(device)
        row_ids = eid_lookup[eids.to(device)]      # [B, L] ∈ [0..N_ex]
        pad_row = torch.zeros(1, ex_node_feats.size(-1), device=device)
        table = torch.cat([pad_row, ex_node_feats], dim=0)  # [N_ex+1, D]
        return table[row_ids]  # [B, L, D]


class KAGADKT(Module):
    def __init__(self, model_params: Dict[str, Any],
                 n_exercise: int, n_skill: int):
        super().__init__()
        self.embed_dim = model_params["emb_dim"]
        self.hidden_dim = model_params["hidden_dim"]
        g_layers = model_params.get("g_layers", 2)
        g_dropout = model_params.get("g_dropout", 0.1)

        # 图编码器
        self.gnn = GraphEncoder(
            num_exercise=n_exercise,
            num_skill=n_skill,
            emb_dim=self.embed_dim,
            num_layers=g_layers,
            dropout=g_dropout
        )

        # 作答嵌入：0=PAD, 1=错, 2=对
        self.ans_embed = Embedding(3, self.embed_dim, padding_idx=0)

        # 序列编码器（输入维度是图嵌入+作答嵌入）
        self.lstm_layer = LSTM(self.embed_dim * 2, self.hidden_dim, batch_first=True)
        self.out_layer = Linear(self.hidden_dim, 1)

        # 运行时注入
        self._graph_data: Optional[HeteroData] = None
        self._eid_lookup: Optional[torch.Tensor] = None

    def set_graph(self, data: HeteroData, eid_lookup: torch.Tensor):
        """在训练开始前注入静态图和查找表"""
        self._graph_data = data
        self._eid_lookup = eid_lookup

    def forward(self, eids: torch.Tensor, is_corrects: torch.Tensor):
        assert self._graph_data is not None and self._eid_lookup is not None, \
            "DKT 未设置图数据，请先调用 set_graph(data, eid_lookup)。"

        # 1) 图前向：得到各节点最新特征
        self.gnn(self._graph_data)

        # 2) 取序列里的题目嵌入
        ex_seq = self.gnn.gather_exercise_seq(eids, self._eid_lookup)  # [B,L,D]

        # 3) 作答（0/1） -> (1/2)，PAD(0) 保持 0
        ans_idx = (is_corrects > 0).long() + is_corrects  # 0->0, 1->2（防突变，等价于+1且把1变2）
        ans_idx = torch.where(is_corrects == 0, torch.zeros_like(ans_idx), is_corrects + 1)
        ans_seq = self.ans_embed(ans_idx)  # [B,L,D]

        # 4) 拼接 -> LSTM -> 预测
        seq_in = torch.cat([ex_seq, ans_seq], dim=-1)  # [B,L,2D]
        h, _ = self.lstm_layer(seq_in)
        logits = self.out_layer(h)                     # [B,L,1]
        return logits


def build_graph_from_eid2sids(eid2sids: Dict[int, List[int]], device: str) -> Tuple[HeteroData, Dict[int, int], Dict[int, int]]:
    """
    用 eid2sids 构建紧凑索引的二部异构图，并返回 eid->idx, sid->idx
    """
    exercise_ids = sorted(set(int(e) for e in eid2sids.keys()))
    skill_ids = sorted(set(int(s) for sids in eid2sids.values() for s in sids))

    eid2idx = {eid: i for i, eid in enumerate(exercise_ids)}
    sid2idx = {sid: i for i, sid in enumerate(skill_ids)}

    ex2sk = []
    for eid, sids in eid2sids.items():
        if eid not in eid2idx:
            continue
        src = eid2idx[eid]
        for sid in sids:
            if sid in sid2idx:
                dst = sid2idx[sid]
                ex2sk.append((src, dst))

    data = HeteroData()
    data['exercise'].num_nodes = len(exercise_ids)
    data['skill'].num_nodes = len(skill_ids)
    if len(ex2sk):
        edge_index = torch.tensor(ex2sk, dtype=torch.long).t().contiguous()
        data['exercise', 'covers', 'skill'].edge_index = edge_index
    data = data.to(device)
    return data, eid2idx, sid2idx


def make_eid_lookup(eid2idx: Dict[int, int], max_eid: int) -> torch.Tensor:
    """
    构造 raw_eid(1..max_eid; 0=PAD) -> 图中 exercise 节点行号+1 的查找表
    不在图中的 eid 映射为 0（PAD）。
    """
    lookup = torch.zeros(max_eid + 1, dtype=torch.long)
    for eid, idx in eid2idx.items():
        if 0 <= eid <= max_eid:
            lookup[eid] = int(idx) + 1
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

        # 图相关
        self._graph_data = None
        self._eid_lookup = None
        self._n_exercise = 0
        self._n_skill = 0

    def inject_graph(self, graph_data: HeteroData, eid_lookup: torch.Tensor,
                     n_exercise: int, n_skill: int):
        self._graph_data = graph_data
        self._eid_lookup = eid_lookup
        self._n_exercise = n_exercise
        self._n_skill = n_skill

    def reset_config(self):
        print("Resetting model, optimizer, and scheduler for new fold...")
        assert self._graph_data is not None and self._eid_lookup is not None, \
            "请先通过 inject_graph 注入图。"
        self.model = KAGADKT(self.model_params, self._n_exercise, self._n_skill).to(self.device)
        # 把静态图与查找表注入到模型中
        self.model.set_graph(self._graph_data, self._eid_lookup.to(self.device))
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-6)

    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_losses, all_labels, all_preds = [], [], []

        for batch in train_loader:
            eids = batch["eids"].to(self.device)
            is_corrects = batch["is_corrects"].to(self.device)
            mask = batch["mask"].to(self.device)

            input_eids = eids[:, :-1]
            input_is_corrects = is_corrects[:, :-1]
            labels = is_corrects[:, 1:].float()
            label_mask = mask[:, 1:]

            if input_eids.size(1) == 0:
                continue

            logits = self.model(input_eids, input_is_corrects).squeeze(-1)
            loss_mat = self.criterion(logits, labels)

            if label_mask.sum() == 0:
                continue
            masked_loss = loss_mat[label_mask].mean()

            self.optimizer.zero_grad()
            masked_loss.backward()
            self.optimizer.step()

            epoch_losses.append(masked_loss.item())

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_labels.append(labels[label_mask].detach().cpu().numpy())
                all_preds.append(probs[label_mask].detach().cpu().numpy())

        self.scheduler.step()

        y_true = np.concatenate(all_labels) if all_labels else np.array([])
        y_prob = np.concatenate(all_preds) if all_preds else np.array([])
        if y_true.size == 0:
            train_auc, train_acc = float("nan"), float("nan")
        else:
            train_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
            train_acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))

        return {
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "train_auc": train_auc,
            "train_acc": train_acc
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        epoch_losses, all_labels, all_preds = [], [], []

        for batch in val_loader:
            eids = batch["eids"].to(self.device)
            is_corrects = batch["is_corrects"].to(self.device)
            mask = batch["mask"].to(self.device)

            input_eids = eids[:, :-1]
            input_is_corrects = is_corrects[:, :-1]
            labels = is_corrects[:, 1:].float()
            label_mask = mask[:, 1:]

            if input_eids.size(1) == 0:
                continue

            logits = self.model(input_eids, input_is_corrects).squeeze(-1)
            loss_mat = self.criterion(logits, labels)

            if label_mask.sum() == 0:
                continue
            masked_loss = loss_mat[label_mask].mean()
            epoch_losses.append(masked_loss.item())

            probs = torch.sigmoid(logits)
            all_labels.append(labels[label_mask].detach().cpu().numpy())
            all_preds.append(probs[label_mask].detach().cpu().numpy())

        y_true = np.concatenate(all_labels) if all_labels else np.array([])
        y_prob = np.concatenate(all_preds) if all_preds else np.array([])
        if y_true.size == 0:
            val_auc, val_acc = float("nan"), float("nan")
        else:
            val_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
            val_acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))

        return {
            "val_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "val_auc": val_auc,
            "val_acc": val_acc
        }

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        hist = {"train_loss": [], "train_auc": [], "train_acc": [], "val_loss": [], "val_auc": [], "val_acc": []}
        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self.train_one_epoch(train_loader)
            hist["train_loss"].append(train_metrics["train_loss"])
            hist["train_auc"].append(train_metrics["train_auc"])
            hist["train_acc"].append(train_metrics["train_acc"])

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                hist["val_loss"].append(val_metrics["val_loss"])
                hist["val_auc"].append(val_metrics["val_auc"])
                hist["val_acc"].append(val_metrics["val_acc"])

            msg = f"[Epoch {epoch:02d}] Train: loss={train_metrics['train_loss']:.4f}"
            if not math.isnan(train_metrics["train_auc"]):
                msg += f", train_auc={train_metrics['train_auc']:.4f}, train_acc={train_metrics['train_acc']:.4f}\n"
            if val_loader is not None:
                msg += f" | Val: loss={hist['val_loss'][-1]:.4f}"
                if not math.isnan(hist['val_auc'][-1]):
                    msg += f", val_auc={hist['val_auc'][-1]:.4f}, val_acc={hist['val_acc'][-1]:.4f}\n"
            print(msg)
        return hist


def get_global_max_ids(path_config: PathConfig, n_folds: int) -> Tuple[int, int, Dict[int, List[int]]]:
    eid2sids_path = path_config.folds_dir / "eid2sids.json"
    with open(eid2sids_path, "r", encoding="utf-8") as f:
        eid2sids_raw = json.load(f)
    # 转 int
    eid2sids = {int(k): [int(v) for v in vs] for k, vs in eid2sids_raw.items()}

    global_eid_set = set(eid2sids.keys())
    global_sid_set = set(s for sids in eid2sids.values() for s in sids)

    for i in range(n_folds):
        fold_dir = path_config.folds_dir / f"fold{i}"
        for fname in ["train_records.txt", "valid_records.txt"]:
            records_path = fold_dir / fname
            if not records_path.exists():
                continue
            with open(records_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line_idx in range(0, len(lines), 2):
                eids_str = lines[line_idx].strip()
                eids = [int(eid) for eid in eids_str.split(",") if eid]
                for eid in eids:
                    global_eid_set.add(eid)
                    for sid in eid2sids.get(eid, []):
                        global_sid_set.add(sid)

    max_eid = max(global_eid_set) if global_eid_set else 0
    max_sid = max(global_sid_set) if global_sid_set else 0
    print(f"Global Max eid: {max_eid}, Global Max sid: {max_sid}")
    return max_eid, max_sid, eid2sids


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium") if hasattr(torch, "set_float32_matmul_precision") else None
    N_FOLDS = 5
    path_config = PathConfig()
    data_params = hyper_params["data_params"]
    model_params = hyper_params["model_params"]
    exp_params = hyper_params["exp_params"]

    # 1) 全局信息
    global_max_eid, global_max_sid, eid2sids = get_global_max_ids(path_config, N_FOLDS)

    # 2) 构建静态图（一次构建，跨折复用）
    device = exp_params["device"]
    graph_data, eid2idx, sid2idx = build_graph_from_eid2sids(eid2sids, device=device)
    n_exercise = graph_data["exercise"].num_nodes
    n_skill = graph_data["skill"].num_nodes
    eid_lookup = make_eid_lookup(eid2idx, global_max_eid).to(device)

    # 3) 实验对象 & 注入图
    exp = Experiment(model_params, exp_params)
    exp.inject_graph(graph_data, eid_lookup, n_exercise, n_skill)

    # 4) 交叉验证
    fold_results = []
    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation with pre-split data")
    for fold in range(N_FOLDS):
        print("-" * 50)
        print(f"Fold {fold}/{N_FOLDS-1}")

        # 该折重置模型
        exp.reset_config()

        # 数据集
        train_records_path = path_config.folds_dir / f"fold{fold}" / "train_records.txt"
        valid_records_path = path_config.folds_dir / f"fold{fold}" / "valid_records.txt"

        train_dataset = EidDataset(eid2sids, train_records_path, data_params)
        valid_dataset = EidDataset(eid2sids, valid_records_path, data_params)

        print(f"Train segments: {len(train_dataset)}, Validation segments: {len(valid_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=exp.batch_size,
            shuffle=True,
            collate_fn=Eid_collate_fn,
            num_workers=exp.num_workers,
            pin_memory=exp.pin_memory
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=exp.batch_size,
            shuffle=False,
            collate_fn=Eid_collate_fn,
            num_workers=exp.num_workers,
            pin_memory=exp.pin_memory
        )

        # 训练/验证
        exp.fit(train_loader, val_loader)
        fold_val_metrics = exp.validate(val_loader)
        print(f"Fold {fold} | Val Loss: {fold_val_metrics['val_loss']:.4f}, "
              f"AUC: {fold_val_metrics['val_auc'] if not math.isnan(fold_val_metrics['val_auc']) else 'nan'}, "
              f"ACC: {fold_val_metrics['val_acc'] if not math.isnan(fold_val_metrics['val_acc']) else 'nan'}")
        fold_results.append(fold_val_metrics)

    # 5) 汇总
    def _avg(key):
        vals = [fr[key] for fr in fold_results if not np.isnan(fr[key])]
        return float(np.mean(vals)) if len(vals) else float("nan")

    summary = {
        "fold_results": fold_results,
        "mean_val_loss": _avg("val_loss"),
        "mean_val_auc": _avg("val_auc"),
        "mean_val_acc": _avg("val_acc")
    }
    print("\n" + "=" * 50)
    print("CV Summary:",
          f"val_loss={summary['mean_val_loss']:.4f}, "
          f"val_auc={summary['mean_val_auc'] if not math.isnan(summary['mean_val_auc']) else 'nan'}, "
          f"val_acc={summary['mean_val_acc'] if not math.isnan(summary['mean_val_acc']) else 'nan'}")