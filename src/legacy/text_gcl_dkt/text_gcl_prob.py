import os
import math
import json
import numpy as np
import torch
import pandas as pd
import hashlib
import pickle

# <--- NEW: Fix tokenizer deadlock issue
# Must be set BEFORE importing transformers
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
from torch_geometric.utils import degree
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
        "batch_size": 512,
        "lr": 0.001,
        "num_epochs": 1,
        "weight_decay": 1e-5,
        "num_workers": 0,
        "pin_memory": False,

        # <--- MODIFIED: Directed perturbation for contrastive learning
        "gcl_lambda": 0.2,       # Loss weight
        "gcl_temp": 0.2,         # Temperature
        "gcl_p_min": 0.05,       # Minimum drop probability (protects long-tail/cold items)
        "gcl_p_max": 0.5,        # Maximum drop probability (perturbs popular/redundant items)
        "gcl_mask_feat": 0.1     # Feature masking remains random
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
            print(f"[Cache] Hit: {cache_file}")
            with open(cache_file, 'rb') as f:
                arr = pickle.load(f)
            t = torch.tensor(arr, dtype=torch.float32)
            return t.to(device) if device is not None else t

        print(f"[Cache] Computing and saving: {cache_file}")
        embs = self._compute_embeddings(texts, model_dir, device, batch_size, max_length, normalize)

        # Ensure directory exists
        self.cache_dir.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(embs.cpu().numpy(), f)
        return embs.to(device) if device is not None else embs

    def _compute_embeddings(self, texts: list, model_dir: Path, device=None,
                            batch_size: int = 256, max_length: int = 256,
                            normalize: bool = True) -> torch.Tensor:
        print(f"[Embed] Using model: {model_dir} to compute embeddings for {len(texts)} texts")
        model_dir_str = str(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir_str, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir_str, trust_remote_code=True).to(device)
        model.eval()

        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing node embeddings", leave=False):
                batch = texts[i:i + batch_size]
                if not batch: continue  # Handle empty batches

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
            fallback_dim = 1024
            print("Warning: No text was encoded; returning empty tensor")
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
        start = 0
        while start < len(eids):
            end = start + max_len
            yield (eids[start:end], is_corrects[start:end])
            start = end  # <--- MODIFIED: Ensure non-overlapping, continuous segments

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
    def __init__(self, temperature=0.2, mask_feat_rate=0.1):
        super().__init__()
        self.temperature = temperature
        # Note: edge drop rate is no longer fixed—it's passed externally via probabilities
        self.mask_feat_rate = mask_feat_rate

    def augment(self, data: HeteroData) -> HeteroData:
        """Generate one view of the graph: structure-aware edge dropping + random feature masking"""
        view = data.clone()

        # --- 1. Structure-aware Edge Dropping ---
        edge_type = ("exercise", "covers", "skill")

        if edge_type in view.edge_index_dict:
            edge_index = view[edge_type].edge_index

            # Check if drop probabilities are precomputed
            if hasattr(view[edge_type], 'edge_drop_prob'):
                probs = view[edge_type].edge_drop_prob
                keep_probs = 1.0 - probs
                keep_mask = torch.bernoulli(keep_probs).to(torch.bool)
                view[edge_type].edge_index = edge_index[:, keep_mask]
            else:
                # Fallback: random 10% drop if no probabilities provided
                mask = torch.rand(edge_index.size(1), device=edge_index.device) > 0.1
                view[edge_type].edge_index = edge_index[:, mask]

        # --- 2. Random Feature Masking ---
        for ntype in view.node_types:
            x = view[ntype].x
            if x is not None:
                mask = torch.rand_like(x) > self.mask_feat_rate
                view[ntype].x = x * mask.float()

        return view

    def compute_infonce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE Loss (SimCLR style).
        Positive pairs: same node across two views.
        All other pairs are negatives.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
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
            if ei.numel() > 0:
                data[rev_key].edge_index = torch.stack([ei[1], ei[0]], dim=0).contiguous()
            else:
                data[rev_key].edge_index = torch.empty((2, 0), dtype=torch.long, device=ei.device)
        return data

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        data = self._ensure_reverse_edge(data)
        x_dict = {ntype: data[ntype].x for ntype in data.node_types}

        for l, conv in enumerate(self.convs):
            prev = {k: v for k, v in x_dict.items()}
            if not data.edge_index_dict:
                out = {}
            else:
                out = conv(x_dict, data.edge_index_dict)

            for ntype, h in out.items():
                h = self.act(h)
                if self.norms[l] is not None:
                    h = self.norms[l][ntype](h)
                if self.residual and ntype in prev:
                    h = h + prev[ntype]
                out[ntype] = self.drop(h)

            x_dict.update(out)

        self._last_x = x_dict
        return x_dict

    def gather_exercise_seq(self, eids: torch.Tensor, eid_lookup: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "_last_x"), "Please call forward(data) first to encode the graph."
        if "exercise" not in self._last_x:
            print("Warning: GNN output does not contain 'exercise' nodes.")
            B, S = eids.shape
            return torch.zeros(B, S, self.emb_dim, device=eids.device)

        ex_node_feats = self._last_x["exercise"]
        device = ex_node_feats.device
        eid_lookup = eid_lookup.to(device)
        row_ids = eid_lookup[eids.to(device)]

        pad_row = torch.zeros(1, ex_node_feats.size(-1), device=device)
        table = torch.cat([pad_row, ex_node_feats], dim=0)

        return table[row_ids]


class KAGADKT(Module):
    def __init__(self, model_params: Dict[str, Any], gcl_params: Dict[str, Any] = None):
        super().__init__()
        self.embed_dim = model_params["emb_dim"]
        self.hidden_dim = model_params["hidden_dim"]

        # GNN Encoder
        self.gnn = GraphEncoder(
            emb_dim=self.embed_dim,
            num_layers=model_params.get("g_layers", 2),
            dropout=model_params.get("g_dropout", 0.1)
        )

        # Contrastive Learning Utility
        if gcl_params:
            self.gcl_util = GraphCLUtil(
                temperature=gcl_params.get("gcl_temp", 0.2),
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
        """Compute main task (DKT) logits and embeddings"""
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
        """Compute auxiliary contrastive learning loss"""
        if self.gcl_util is None or self._graph_data is None:
            return torch.tensor(0.0, device=self.out_layer.weight.device)

        view1 = self.gcl_util.augment(self._graph_data)
        view2 = self.gcl_util.augment(self._graph_data)

        out1 = self.gnn(view1)
        out2 = self.gnn(view2)

        loss_ex = self.gcl_util.compute_infonce(out1['exercise'], out2['exercise'])
        loss_sk = self.gcl_util.compute_infonce(out1['skill'], out2['skill'])

        return loss_ex + loss_sk

    def forward(self, eids, is_corrects, mask):
        """Standard forward interface pointing to main task logic"""
        return self.get_main_task_loss(eids, is_corrects, mask)


def precompute_edge_drop_probs(data: HeteroData, p_min: float, p_max: float):
    """
    Compute edge drop probabilities based on node degrees for structure-aware perturbation.
    Formula: w_ij = log(d_q) + log(d_k)
             p_ij = min(p_max, p_min + (p_max - p_min) * norm(w_ij))
    """
    print("[GCL] Precomputing structure-aware edge drop probabilities...")

    edge_type = ("exercise", "covers", "skill")
    if edge_type not in data.edge_index_dict:
        return data

    edge_index = data[edge_type].edge_index
    src, dst = edge_index[0], edge_index[1]

    num_ex_nodes = data['exercise'].x.size(0)
    num_sk_nodes = data['skill'].x.size(0)

    deg_src = degree(src, num_nodes=num_ex_nodes)
    deg_dst = degree(dst, num_nodes=num_sk_nodes)

    w = torch.log(deg_src[src] + 1) + torch.log(deg_dst[dst] + 1)

    w_min, w_max = w.min(), w.max()
    if w_max - w_min > 1e-6:
        w_norm = (w - w_min) / (w_max - w_min)
    else:
        w_norm = torch.zeros_like(w)

    drop_probs = p_min + (p_max - p_min) * w_norm
    drop_probs = torch.clamp(drop_probs, max=p_max)

    data[edge_type].edge_drop_prob = drop_probs

    print(f"[GCL] Done. Prob Range: [{drop_probs.min():.4f}, {drop_probs.max():.4f}], Mean: {drop_probs.mean():.4f}")
    return data


def build_graph_with_text(eid2sids, eid2desc, sid2desc, path_config, model_params, device):
    # <--- MODIFIED: Collect node IDs from multiple sources for completeness
    exercise_ids_set = set()
    skill_ids_set = set()

    exercise_ids_set.update(eid2sids.keys())

    for k in eid2desc.keys():
        try:
            exercise_ids_set.add(int(k))
        except (ValueError, TypeError):
            continue

    for sids in eid2sids.values():
        skill_ids_set.update(sids)

    for k in sid2desc.keys():
        try:
            skill_ids_set.add(int(k))
        except (ValueError, TypeError):
            continue

    exercise_ids = sorted(list(exercise_ids_set))
    skill_ids = sorted(list(skill_ids_set))

    print(f"[Graph Build] Found {len(exercise_ids)} exercise nodes, {len(skill_ids)} skill nodes")

    eid2idx = {eid: i for i, eid in enumerate(exercise_ids)}
    sid2idx = {sid: i for i, sid in enumerate(skill_ids)}

    # <--- MODIFIED: Extract three types of text descriptions
    ex_texts_mem = []
    ex_texts_und = []
    ex_texts_skl = []

    empty_desc = {"memory": "", "understand": "", "skill": ""}

    for eid in exercise_ids:
        raw_desc = eid2desc.get(str(eid), empty_desc)

        if isinstance(raw_desc, dict):
            ex_texts_mem.append(raw_desc.get("memory", ""))
            ex_texts_und.append(raw_desc.get("understand", ""))
            ex_texts_skl.append(raw_desc.get("skill", ""))
        elif isinstance(raw_desc, str):
            ex_texts_mem.append(raw_desc)
            ex_texts_und.append("")
            ex_texts_skl.append("")
        else:
            ex_texts_mem.append("")
            ex_texts_und.append("")
            ex_texts_skl.append("")

    sk_texts = [sid2desc.get(str(sid), "") for sid in skill_ids]

    cache = EmbeddingCache(path_config.cache_dir)
    text_emb_model_dir = path_config.embed_model_dir

    print("[Graph Build] Computing Exercise Memory Embeddings...")
    ex_emb_mem = cache.get_embeddings(ex_texts_mem, text_emb_model_dir, device=device,
                                      batch_size=model_params['text_emb_batch_size'],
                                      max_length=model_params['text_emb_max_len'])

    print("[Graph Build] Computing Exercise Understand Embeddings...")
    ex_emb_und = cache.get_embeddings(ex_texts_und, text_emb_model_dir, device=device,
                                      batch_size=model_params['text_emb_batch_size'],
                                      max_length=model_params['text_emb_max_len'])

    print("[Graph Build] Computing Exercise Skill Embeddings...")
    ex_emb_skl = cache.get_embeddings(ex_texts_skl, text_emb_model_dir, device=device,
                                      batch_size=model_params['text_emb_batch_size'],
                                      max_length=model_params['text_emb_max_len'])

    sk_emb = cache.get_embeddings(sk_texts, text_emb_model_dir, device=device,
                                  batch_size=model_params['text_emb_batch_size'],
                                  max_length=model_params['text_emb_max_len'])

    target_dim = model_params['emb_dim']

    # <--- MODIFIED: Fusion mechanism (Concatenation + Projection)
    if ex_emb_mem.numel() > 0:
        ex_emb_combined = torch.cat([ex_emb_mem, ex_emb_und, ex_emb_skl], dim=1)

        if ex_emb_combined.shape[1] != target_dim:
            print(f"[Graph Build] Fusing Exercise Embeddings: {ex_emb_combined.shape[1]} -> {target_dim}")
            proj = Linear(ex_emb_combined.shape[1], target_dim).to(device)
            with torch.no_grad():
                ex_emb = proj(ex_emb_combined)
        else:
            ex_emb = ex_emb_combined

        assert ex_emb.shape[0] == len(exercise_ids), f"Exercise embedding count mismatch: {ex_emb.shape[0]} vs {len(exercise_ids)}"
    else:
        ex_emb = torch.zeros(len(exercise_ids), target_dim, device=device)

    if sk_emb.numel() > 0:
        if sk_emb.shape[1] != target_dim:
            proj = Linear(sk_emb.shape[1], target_dim).to(device)
            with torch.no_grad():
                sk_emb = proj(sk_emb)
        assert sk_emb.shape[0] == len(skill_ids), f"Skill embedding count mismatch: {sk_emb.shape[0]} vs {len(skill_ids)}"
    else:
        sk_emb = torch.zeros(len(skill_ids), target_dim, device=device)

    print(f"[Graph Build] Exercise embedding shape: {ex_emb.shape}")
    print(f"[Graph Build] Skill embedding shape: {sk_emb.shape}")

    data = HeteroData()
    data['exercise'].x = ex_emb
    data['skill'].x = sk_emb

    print(f"[Graph Build] Verified HeteroData - Exercise nodes: {data['exercise'].x.shape[0]}")
    print(f"[Graph Build] Verified HeteroData - Skill nodes: {data['skill'].x.shape[0]}")

    # Build edges
    ex2sk = []
    for eid, sids in eid2sids.items():
        if eid in eid2idx:
            src = eid2idx[eid]
            for sid in sids:
                if sid in sid2idx:
                    dst = sid2idx[sid]
                    ex2sk.append((src, dst))

    print(f"[Graph Build] Found {len(ex2sk)} edges")

    if ex2sk:
        data['exercise', 'covers', 'skill'].edge_index = torch.tensor(ex2sk, dtype=torch.long).t().contiguous()
    else:
        data['exercise', 'covers', 'skill'].edge_index = torch.empty((2, 0), dtype=torch.long)

    print(f"[Graph Build] Final check - Exercise: {data['exercise'].x.shape}, Skill: {data['skill'].x.shape}")

    return data.to(device), eid2idx, sid2idx


def make_eid_lookup(eid2idx: Dict[int, int], max_eid: int) -> torch.Tensor:
    lookup_size = max(max_eid + 1, 1)
    lookup = torch.zeros(lookup_size, dtype=torch.long)

    for eid, idx in eid2idx.items():
        if 0 <= eid <= max_eid:
            lookup[eid] = int(idx) + 1  # 0 reserved for padding
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
        assert self._graph_data is not None and self._eid_lookup is not None, "Please inject graph first"

        self.model = KAGADKT(
            self.model_params,
            gcl_params=self.exp_params  # Pass GCL params directly
        ).to(self.device)

        self.model.set_graph(self._graph_data.to(self.device), self._eid_lookup.to(self.device))
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-6)

    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_losses, epoch_gcl_losses, epoch_bce_losses = [], [], []
        all_labels, all_preds = [], []

        lambda_gcl = self.exp_params.get("gcl_lambda", 0.0)

        pbar = tqdm(train_loader, desc=f"Train Epoch {self.current_epoch:02d}", leave=False)

        for batch in pbar:
            eids, is_corrects, mask = batch["eids"].to(self.device), batch["is_corrects"].to(self.device), batch["mask"].to(self.device)

            input_eids, input_is_corrects = eids[:, :-1], is_corrects[:, :-1]
            input_mask = mask[:, :-1]
            labels, label_mask = is_corrects[:, 1:].float(), mask[:, 1:]

            if input_eids.size(1) == 0 or label_mask.sum() == 0: continue

            self.optimizer.zero_grad()

            logits = self.model.get_main_task_loss(input_eids, input_is_corrects, input_mask).squeeze(-1)

            bce_loss = self.criterion(logits, labels)
            masked_bce_loss = (bce_loss * label_mask).sum() / label_mask.sum().clamp(min=1)

            if lambda_gcl > 0:
                gcl_loss = self.model.compute_gcl_loss()
            else:
                gcl_loss = torch.tensor(0.0, device=self.device)

            total_loss = masked_bce_loss + lambda_gcl * gcl_loss

            total_loss.backward()
            self.optimizer.step()

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
        epoch_losses, all_labels, all_preds = [], []

        pbar = tqdm(val_loader, desc="Validating", leave=False)

        for batch in pbar:
            eids, is_corrects, mask = batch["eids"].to(self.device), batch["is_corrects"].to(self.device), batch["mask"].to(self.device)

            input_eids, input_is_corrects = eids[:, :-1], is_corrects[:, :-1]
            input_mask = mask[:, :-1]
            labels, label_mask = is_corrects[:, 1:].float(), mask[:, 1:]

            if input_eids.size(1) == 0 or label_mask.sum() == 0: continue

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

        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
        return {"val_loss": np.mean(epoch_losses), "val_auc": auc, "val_acc": acc}

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_one_epoch(train_loader)

            msg = f"[Epoch {epoch:02d}] Train: loss={train_metrics['train_loss']:.4f}, auc={train_metrics['train_auc']:.4f}, acc={train_metrics['train_acc']:.4f}"

            if val_loader:
                val_metrics = self.validate(val_loader)
                msg += f" | Val: loss={val_metrics['val_loss']:.4f}, auc={val_metrics['val_auc']:.4f}, acc={val_metrics['val_acc']:.4f}"

            print(msg)


def get_global_info(path_config: PathConfig, n_folds: int):
    def load_json(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: JSON file not found {path}")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON format {path}")
            return {}

    def load_jsonl(path):
        data = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        if "eid" in obj and "desc" in obj:
                            data[str(obj["eid"])] = obj["desc"]
                    except json.JSONDecodeError:
                        continue
            return data
        except FileNotFoundError:
            print(f"Warning: JSONL file not found {path}")
            return {}

    eid2sids = {int(k): [int(v) for v in vs] for k, vs in load_json(path_config.folds_dir / "eid2sids.json").items()}
    eid2desc = load_jsonl(path_config.folds_dir / "eid2desc_rewrite2.jsonl")
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
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    N_FOLDS = 5
    path_config = PathConfig()
    data_params = hyper_params["data_params"]
    model_params = hyper_params["model_params"]
    exp_params = hyper_params["exp_params"]
    device = exp_params["device"]

    if "cuda" in device and not torch.cuda.is_available():
        print(f"Warning: {device} unavailable, switching to CPU.")
        device = "cpu"
        exp_params["device"] = "cpu"

    print(f"Using device: {device}")
    print(f"Data directory: {path_config.data_dir}")
    print(f"Embedding model directory: {path_config.embed_model_dir}")

    global_max_eid, eid2sids, eid2desc, sid2desc = get_global_info(path_config, N_FOLDS)

    if not eid2sids and not eid2desc and not sid2desc:
        print("Error: Failed to load any graph data (eid2sids, eid2desc, sid2desc).")
        exit()

    graph_data, eid2idx, _ = build_graph_with_text(eid2sids, eid2desc, sid2desc, path_config, model_params, device)
    graph_data = precompute_edge_drop_probs(
        graph_data,
        p_min=exp_params["gcl_p_min"],
        p_max=exp_params["gcl_p_max"]
    )
    eid_lookup = make_eid_lookup(eid2idx, global_max_eid)

    print(f"\nGraph built. Node types: {graph_data.node_types}, Edge types: {graph_data.edge_types}")

    for node_type in graph_data.node_types:
        if hasattr(graph_data[node_type], 'x') and graph_data[node_type].x is not None:
            print(f"{node_type.capitalize()} nodes: {graph_data[node_type].x.shape[0]}, Feature dim: {graph_data[node_type].x.shape[1]}")
        else:
            print(f"Warning: {node_type} nodes have no features!")

    for edge_type in graph_data.edge_types:
        edge_index = graph_data[edge_type].edge_index
        print(f"Edge {edge_type}: {edge_index.shape[1]} edges")

    if 'exercise' not in graph_data.node_types or 'skill' not in graph_data.node_types:
        print("Error: Graph missing required node types!")
        exit()

    if graph_data['exercise'].x.shape[0] == 0 or graph_data['skill'].x.shape[0] == 0:
        print("Error: Node features are empty!")
        exit()

    exp = Experiment(model_params, exp_params)
    exp.inject_graph(graph_data, eid_lookup)

    fold_results = []
    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation")

    for fold in range(N_FOLDS):
        print("-" * 50 + f"\nFold {fold}/{N_FOLDS-1}")

        exp.reset_config()

        train_path = path_config.folds_dir / f"fold{fold}" / "train_records.txt"
        valid_path = path_config.folds_dir / f"fold{fold}" / "valid_records.txt"

        if not train_path.exists() or not valid_path.exists():
            print(f"Warning: Fold {fold} data files missing, skipping.")
            continue

        train_dataset = EidDataset(train_path, data_params)
        valid_dataset = EidDataset(valid_path, data_params)

        if len(train_dataset) == 0 or len(valid_dataset) == 0:
            print(f"Warning: Fold {fold} dataset is empty, skipping.")
            continue

        print(f"Train segments: {len(train_dataset)}, Validation segments: {len(valid_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=exp.batch_size,
            shuffle=True,
            collate_fn=Eid_collate_fn,
            num_workers=exp.num_workers,
            pin_memory=exp.pin_memory,
            persistent_workers=True if exp.num_workers > 0 else False
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=exp.batch_size,
            shuffle=False,
            collate_fn=Eid_collate_fn,
            num_workers=exp.num_workers,
            pin_memory=exp.pin_memory,
            persistent_workers=True if exp.num_workers > 0 else False
        )

        exp.fit(train_loader, val_loader)

        print(f"Fold {fold} training complete. Computing final validation metrics...")
        fold_val_metrics = exp.validate(val_loader)
        print(f"Fold {fold} | Final Val AUC: {fold_val_metrics['val_auc']:.4f}, ACC: {fold_val_metrics['val_acc']:.4f}")
        fold_results.append(fold_val_metrics)

    def _avg(key):
        vals = [fr[key] for fr in fold_results if fr and not np.isnan(fr[key])]
        return float(np.mean(vals)) if vals else float("nan")

    if fold_results:
        summary = {
            "mean_val_loss": _avg("val_loss"),
            "mean_val_auc": _avg("val_auc"),
            "mean_val_acc": _avg("val_acc")
        }
        print("\n" + "=" * 50 + f"\nCV Summary: val_auc={summary['mean_val_auc']:.4f}, val_acc={summary['mean_val_acc']:.4f}")
    else:
        print("\n" + "=" * 50 + "\nCV Summary: No valid results available.")