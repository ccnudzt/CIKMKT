from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Embedding, LSTM, Linear, Module, ReLU

from src.models import MODEL_REGISTRY


class GraphCLUtil(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.2,
        drop_scheme: str = "random",
        drop_edge_rate: float = 0.1,
        mask_feat_rate: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.drop_scheme = drop_scheme
        self.drop_edge_rate = drop_edge_rate
        self.mask_feat_rate = mask_feat_rate

    def augment(self, data):
        view = data.clone()
        edge_type = ("exercise", "covers", "skill")
        if edge_type in view.edge_index_dict:
            edge_index = view[edge_type].edge_index
            if self.drop_scheme == "prob" and hasattr(view[edge_type], "edge_drop_prob"):
                keep_probs = 1.0 - view[edge_type].edge_drop_prob
                keep_mask = torch.bernoulli(keep_probs).to(torch.bool)
            else:
                keep_mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_edge_rate
            view[edge_type].edge_index = edge_index[:, keep_mask]

        for ntype in view.node_types:
            x = view[ntype].x
            if x is not None and self.mask_feat_rate > 0:
                mask = (torch.rand_like(x) > self.mask_feat_rate).float()
                view[ntype].x = x * mask
        return view

    def compute_infonce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim_matrix, labels)


class TextGraphEncoder(Module):
    def __init__(
        self,
        emb_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggr: str = "mean",
        residual: bool = True,
        layernorm: bool = True,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import HeteroConv, SAGEConv
        except ImportError as exc:
            raise RuntimeError("torch-geometric is required for text graph KT.") from exc

        self.emb_dim = emb_dim
        self.residual = residual
        convs, norms = [], []
        for _ in range(num_layers):
            convs.append(
                HeteroConv(
                    {
                        ("exercise", "covers", "skill"): SAGEConv((-1, -1), emb_dim),
                        ("skill", "rev_covers", "exercise"): SAGEConv((-1, -1), emb_dim),
                    },
                    aggr=aggr,
                )
            )
            norms.append(
                torch.nn.ModuleDict(
                    {
                        "exercise": torch.nn.LayerNorm(emb_dim),
                        "skill": torch.nn.LayerNorm(emb_dim),
                    }
                )
                if layernorm
                else None
            )
        self.convs = torch.nn.ModuleList(convs)
        self.norms = torch.nn.ModuleList(norms)
        self.drop = torch.nn.Dropout(dropout)
        self.act = ReLU()

    @staticmethod
    def _ensure_reverse_edge(data):
        fwd_key = ("exercise", "covers", "skill")
        rev_key = ("skill", "rev_covers", "exercise")
        if fwd_key not in data.edge_index_dict:
            raise KeyError("Graph data must contain ('exercise', 'covers', 'skill') edges.")
        if rev_key not in data.edge_index_dict:
            edge_index = data.edge_index_dict[fwd_key]
            if edge_index.numel() > 0:
                data[rev_key].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0).contiguous()
            else:
                data[rev_key].edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return data

    def forward(self, data):
        device = next(self.parameters()).device
        data = self._ensure_reverse_edge(data)
        x_dict = {ntype: data[ntype].x.to(device) for ntype in data.node_types}
        edge_index_dict = {key: value.to(device) for key, value in data.edge_index_dict.items()}

        for layer_index, conv in enumerate(self.convs):
            prev = {key: value for key, value in x_dict.items()}
            out = conv(x_dict, edge_index_dict) if edge_index_dict else {}
            for ntype, hidden in out.items():
                hidden = self.act(hidden)
                if self.norms[layer_index] is not None:
                    hidden = self.norms[layer_index][ntype](hidden)
                if self.residual and ntype in prev:
                    hidden = hidden + prev[ntype]
                out[ntype] = self.drop(hidden)
            x_dict.update(out)

        self._last_x = x_dict
        return x_dict

    def gather_exercise_seq(self, eids: torch.Tensor, eid_lookup: torch.Tensor) -> torch.Tensor:
        ex_features = self._last_x["exercise"]
        device = ex_features.device
        row_ids = eid_lookup.to(device)[eids.to(device)]
        pad_row = torch.zeros(1, ex_features.size(-1), device=device)
        table = torch.cat([pad_row, ex_features], dim=0)
        return table[row_ids]


class TextGCLKTDModel(Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        graph_data,
        eid_lookup: torch.Tensor,
        g_layers: int = 2,
        g_dropout: float = 0.1,
        g_aggr: str = "mean",
        gcl_cfg: dict | None = None,
    ):
        super().__init__()
        self.graph_data = graph_data
        self.eid_lookup = eid_lookup
        self.gnn = TextGraphEncoder(
            emb_dim=embed_dim,
            num_layers=g_layers,
            dropout=g_dropout,
            aggr=g_aggr,
        )
        self.ans_embed = Embedding(3, embed_dim, padding_idx=0)
        self.lstm_layer = LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.out_layer = Linear(hidden_dim, 1)
        self.gcl_cfg = gcl_cfg or {}
        self.gcl_util = GraphCLUtil(
            temperature=float(self.gcl_cfg.get("temperature", 0.2)),
            drop_scheme=self.gcl_cfg.get("drop_scheme", "random"),
            drop_edge_rate=float(self.gcl_cfg.get("drop_edge_rate", 0.1)),
            mask_feat_rate=float(self.gcl_cfg.get("mask_feat_rate", 0.1)),
        )

    def forward(self, qids: torch.Tensor, responses: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        self.gnn(self.graph_data)
        exercise_seq = self.gnn.gather_exercise_seq(qids, self.eid_lookup)
        ans_idx = responses + 1
        if masks is not None:
            ans_idx = ans_idx.masked_fill(~masks.bool(), 0)
        ans_seq = self.ans_embed(ans_idx)
        hidden, _ = self.lstm_layer(torch.cat([exercise_seq, ans_seq], dim=-1))
        return self.out_layer(hidden).squeeze(-1)

    def compute_aux_loss(self) -> torch.Tensor:
        if not self.gcl_cfg.get("enabled", False):
            return torch.tensor(0.0, device=self.out_layer.weight.device)

        view1 = self.gcl_util.augment(self.graph_data)
        view2 = self.gcl_util.augment(self.graph_data)
        out1 = self.gnn(view1)
        out2 = self.gnn(view2)
        loss_ex = self.gcl_util.compute_infonce(out1["exercise"], out2["exercise"])
        loss_sk = self.gcl_util.compute_infonce(out1["skill"], out2["skill"])
        return loss_ex + loss_sk


@MODEL_REGISTRY.register("text_gcl_dkt")
def build_text_gcl_dkt_model(config: dict, metadata: dict) -> TextGCLKTDModel:
    model_cfg = config["model"]
    return TextGCLKTDModel(
        embed_dim=int(model_cfg["embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        graph_data=metadata["graph_data"],
        eid_lookup=metadata["eid_lookup"],
        g_layers=int(model_cfg.get("g_layers", 2)),
        g_dropout=float(model_cfg.get("g_dropout", 0.1)),
        g_aggr=model_cfg.get("g_aggr", "mean"),
        gcl_cfg=model_cfg.get("gcl", {}),
    )
