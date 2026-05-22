from __future__ import annotations

import torch
from torch.nn import Embedding, LSTM, Linear, Module, ReLU

from src.models import MODEL_REGISTRY


class GraphEncoder(Module):
    def __init__(
        self,
        num_exercise: int,
        num_skill: int,
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
            raise RuntimeError("torch-geometric is required for KAGA-DKT.") from exc

        self.num_exercise = num_exercise
        self.num_skill = num_skill
        self.emb_dim = emb_dim
        self.residual = residual
        self.ex_embed = Embedding(num_exercise + 1, emb_dim, padding_idx=0)
        self.sk_embed = Embedding(num_skill + 1, emb_dim, padding_idx=0)

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
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.ex_embed.weight[1:])
        torch.nn.init.xavier_uniform_(self.sk_embed.weight[1:])
        for conv in self.convs:
            for module in conv.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    @staticmethod
    def _ensure_reverse_edge(data):
        fwd_key = ("exercise", "covers", "skill")
        rev_key = ("skill", "rev_covers", "exercise")
        if fwd_key not in data.edge_index_dict:
            raise KeyError("Graph data must contain ('exercise', 'covers', 'skill') edges.")
        if rev_key not in data.edge_index_dict:
            edge_index = data.edge_index_dict[fwd_key]
            data[rev_key].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0).contiguous()
        return data

    def forward(self, data):
        device = next(self.parameters()).device
        data = self._ensure_reverse_edge(data)
        x_dict = {
            "exercise": self.ex_embed.weight[1 : 1 + self.num_exercise].to(device),
            "skill": self.sk_embed.weight[1 : 1 + self.num_skill].to(device),
        }
        edge_index_dict = {key: value.to(device) for key, value in data.edge_index_dict.items()}

        for layer_index, conv in enumerate(self.convs):
            prev = {key: value for key, value in x_dict.items()}
            out = conv(x_dict, edge_index_dict)
            for ntype, hidden in out.items():
                hidden = self.act(hidden)
                if self.norms[layer_index] is not None:
                    hidden = self.norms[layer_index][ntype](hidden)
                if self.residual:
                    hidden = hidden + prev[ntype]
                out[ntype] = self.drop(hidden)
            x_dict = out

        self._last_x = x_dict
        return x_dict

    def gather_exercise_seq(self, eids: torch.Tensor, eid_lookup: torch.Tensor) -> torch.Tensor:
        ex_features = self._last_x["exercise"]
        device = ex_features.device
        row_ids = eid_lookup.to(device)[eids.to(device)]
        pad_row = torch.zeros(1, ex_features.size(-1), device=device)
        table = torch.cat([pad_row, ex_features], dim=0)
        return table[row_ids]


class KAGADKTModel(Module):
    def __init__(
        self,
        max_eid: int,
        num_exercise: int,
        num_skill: int,
        embed_dim: int,
        hidden_dim: int,
        graph_data,
        eid_lookup: torch.Tensor,
        g_layers: int = 2,
        g_dropout: float = 0.1,
        g_aggr: str = "mean",
    ):
        super().__init__()
        self.max_eid = max_eid
        self.graph_data = graph_data
        self.eid_lookup = eid_lookup
        self.gnn = GraphEncoder(
            num_exercise=num_exercise,
            num_skill=num_skill,
            emb_dim=embed_dim,
            num_layers=g_layers,
            dropout=g_dropout,
            aggr=g_aggr,
        )
        self.ans_embed = Embedding(3, embed_dim, padding_idx=0)
        self.lstm_layer = LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.out_layer = Linear(hidden_dim, 1)

    def forward(self, qids: torch.Tensor, responses: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        self.gnn(self.graph_data)
        exercise_seq = self.gnn.gather_exercise_seq(qids, self.eid_lookup)
        ans_idx = responses + 1
        if masks is not None:
            ans_idx = ans_idx.masked_fill(~masks.bool(), 0)
        ans_seq = self.ans_embed(ans_idx)
        hidden, _ = self.lstm_layer(torch.cat([exercise_seq, ans_seq], dim=-1))
        return self.out_layer(hidden).squeeze(-1)


@MODEL_REGISTRY.register("kaga_dkt")
def build_kaga_dkt_model(config: dict, metadata: dict) -> KAGADKTModel:
    model_cfg = config["model"]
    return KAGADKTModel(
        max_eid=int(metadata["max_qid"]),
        num_exercise=int(metadata["num_exercise"]),
        num_skill=int(metadata["num_skill"]),
        embed_dim=int(model_cfg["embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        graph_data=metadata["graph_data"],
        eid_lookup=metadata["eid_lookup"],
        g_layers=int(model_cfg.get("g_layers", 2)),
        g_dropout=float(model_cfg.get("g_dropout", 0.1)),
        g_aggr=model_cfg.get("g_aggr", "mean"),
    )
