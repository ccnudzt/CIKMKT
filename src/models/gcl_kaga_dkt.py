from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models import MODEL_REGISTRY
from src.models.kaga_dkt import GraphEncoder, KAGADKTModel


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
        return view

    def mask_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.mask_feat_rate <= 0:
            return embeddings
        mask = (torch.rand_like(embeddings) > self.mask_feat_rate).float()
        return embeddings * mask

    def compute_infonce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim_matrix, labels)


class GCLKAGADKTModel(KAGADKTModel):
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
        gcl_cfg: dict | None = None,
    ):
        super().__init__(
            max_eid=max_eid,
            num_exercise=num_exercise,
            num_skill=num_skill,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            graph_data=graph_data,
            eid_lookup=eid_lookup,
            g_layers=g_layers,
            g_dropout=g_dropout,
            g_aggr=g_aggr,
        )
        self.gcl_cfg = gcl_cfg or {}
        self.gcl_util = GraphCLUtil(
            temperature=float(self.gcl_cfg.get("temperature", 0.2)),
            drop_scheme=self.gcl_cfg.get("drop_scheme", "random"),
            drop_edge_rate=float(self.gcl_cfg.get("drop_edge_rate", 0.1)),
            mask_feat_rate=float(self.gcl_cfg.get("mask_feat_rate", 0.1)),
        )

    def compute_aux_loss(self) -> torch.Tensor:
        if not self.gcl_cfg.get("enabled", False):
            return torch.tensor(0.0, device=self.out_layer.weight.device)

        view1 = self.gcl_util.augment(self.graph_data)
        view2 = self.gcl_util.augment(self.graph_data)
        out1 = self.gnn(view1)
        out2 = self.gnn(view2)
        exercise_1 = self.gcl_util.mask_features(out1["exercise"])
        exercise_2 = self.gcl_util.mask_features(out2["exercise"])
        skill_1 = self.gcl_util.mask_features(out1["skill"])
        skill_2 = self.gcl_util.mask_features(out2["skill"])
        loss_ex = self.gcl_util.compute_infonce(exercise_1, exercise_2)
        loss_sk = self.gcl_util.compute_infonce(skill_1, skill_2)
        return loss_ex + loss_sk


@MODEL_REGISTRY.register("gcl_kaga_dkt")
def build_gcl_kaga_dkt_model(config: dict, metadata: dict) -> GCLKAGADKTModel:
    model_cfg = config["model"]
    return GCLKAGADKTModel(
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
        gcl_cfg=model_cfg.get("gcl", {}),
    )
