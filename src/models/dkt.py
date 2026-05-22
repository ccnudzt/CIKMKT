from __future__ import annotations

import torch
from torch.nn import Embedding, LSTM, Linear, Module

from src.models import MODEL_REGISTRY


class DKTModel(Module):
    def __init__(
        self,
        max_qid: int,
        embed_dim: int,
        hidden_dim: int,
        r_pad: int = 2,
        q_text_embeddings: torch.Tensor | None = None,
        freeze_q_embed: bool = True,
    ):
        super().__init__()
        if q_text_embeddings is not None:
            if tuple(q_text_embeddings.shape) != (max_qid + 1, embed_dim):
                raise ValueError(
                    f"q_text_embeddings must have shape {(max_qid + 1, embed_dim)}, got {tuple(q_text_embeddings.shape)}"
                )
            self.q_embed = Embedding.from_pretrained(q_text_embeddings, freeze=freeze_q_embed, padding_idx=0)
        else:
            self.q_embed = Embedding(max_qid + 1, embed_dim, padding_idx=0)

        self.r_embed = Embedding(3, embed_dim, padding_idx=r_pad)
        self.lstm_layer = LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.out_layer = Linear(hidden_dim, 1)

    def forward(self, qids: torch.Tensor, responses: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        q_embed = self.q_embed(qids)
        r_embed = self.r_embed(responses)
        hidden, _ = self.lstm_layer(torch.cat([q_embed, r_embed], dim=-1))
        return self.out_layer(hidden).squeeze(-1)


def _build_dkt_model(config: dict, metadata: dict) -> DKTModel:
    model_cfg = config["model"]
    return DKTModel(
        max_qid=int(metadata["max_qid"]),
        embed_dim=int(model_cfg["embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        r_pad=int(metadata.get("r_pad", 2)),
        q_text_embeddings=metadata.get("q_text_embeddings"),
        freeze_q_embed=bool(model_cfg.get("freeze_q_embed", True)),
    )


@MODEL_REGISTRY.register("dkt")
def build_dkt_model(config: dict, metadata: dict) -> DKTModel:
    return _build_dkt_model(config, metadata)


@MODEL_REGISTRY.register("cog_dkt")
def build_cog_dkt_model(config: dict, metadata: dict) -> DKTModel:
    return _build_dkt_model(config, metadata)
