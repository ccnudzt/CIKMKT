import json
import random
import torch
import numpy as np
from pathlib import Path


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_max_qid(records: list) -> int:
    """Infer the maximum question ID from records."""
    return max(qid for r in records for qid in r["qids"])


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, meta=None):
    """Save model checkpoint to disk."""
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "meta": meta or {},
    }
    torch.save(payload, path)


class EarlyStopping:
    """Early stopping helper."""
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stopped = False

    def step(self, metric: float) -> bool:
        """
        Update the early stopping state.
        Returns True if metric improved, False otherwise.
        """
        improved = False
        if self.best is None:
            self.best = metric
            improved = True
        elif self.mode == "max" and metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            improved = True
        elif self.mode == "min" and metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return improved


def build_qid_text_embeddings(
    qid2cog_path: str,
    model_path: str,
    max_qid: int,
    embed_dim: int,
    device: str = "cpu",
    batch_size: int = 64,
    agg: str = "mean",
) -> torch.Tensor:
    """
    Load qid2cog.jsonl and build a text embedding matrix of shape (max_qid+1, embed_dim).
    Each question has three cognitive descriptions (memory, understand, skill),
    which are embedded and aggregated into a single vector.

    The resulting tensor can be used as a fixed embedding table to replace nn.Embedding.

    Args:
        qid2cog_path: path to qid2cog.jsonl
        model_path:   path to the Qwen3-Embedding model directory
        max_qid:      maximum question ID (index 0 is reserved for padding)
        embed_dim:    output embedding dimension; if the model output dim differs,
                      a linear projection is applied
        device:       device for the final tensor ("cpu" or "cuda")
        batch_size:   number of texts per inference batch
        agg:          aggregation method for the three cog descriptions: "mean" | "sum" | "max"

    Returns:
        emb_matrix: FloatTensor of shape (max_qid+1, embed_dim); row 0 is all-zeros (padding)
    """
    from transformers import AutoTokenizer, AutoModel

    # ---------- 1. Load qid2cog.jsonl ----------
    qid2cog: dict[int, list[str]] = {}
    with open(qid2cog_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = int(obj["qid"])
            desc: dict = obj["desc"]
            # Collect available cognitive texts in a fixed order
            texts = [desc.get("memory", ""), desc.get("understand", ""), desc.get("skill", "")]
            qid2cog[qid] = texts

    # ---------- 2. Load the text embedding model ----------
    model_path = str(Path(model_path).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    text_model.eval()
    text_model = text_model.to(device)

    def _encode_texts(texts: list[str]) -> np.ndarray:
        """Encode a list of texts and return mean-pooled numpy array (N, H)."""
        all_vecs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start: start + batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                output = text_model(**encoded)
                # Mean pooling over token dimension
                hidden = output.last_hidden_state          # (N, seq, H)
                attn = encoded["attention_mask"].unsqueeze(-1).float()
                vecs = (hidden * attn).sum(1) / attn.sum(1)  # (N, H)
            all_vecs.append(vecs.cpu().float().numpy())
        return np.concatenate(all_vecs, axis=0)            # (N, H)

    # ---------- 3. Flatten all cog texts, encode in batches ----------
    # Build ordered list: for each qid in 1..max_qid, keep 3 texts
    all_texts: list[str] = []
    ordered_qids: list[int] = []

    for qid in range(1, max_qid + 1):
        texts = qid2cog.get(qid, ["", "", ""])
        all_texts.extend(texts)
        ordered_qids.append(qid)

    print(f"[build_qid_text_embeddings] Encoding {len(all_texts)} texts for {len(ordered_qids)} questions ...")
    all_vecs = _encode_texts(all_texts)  # (max_qid*3, H)
    model_hidden_dim = all_vecs.shape[1]

    # Reshape to (max_qid, 3, H)
    cog_vecs = all_vecs.reshape(max_qid, 3, model_hidden_dim)

    # ---------- 4. Aggregate the 3 cognitive texts ----------
    if agg == "mean":
        qid_vecs = cog_vecs.mean(axis=1)   # (max_qid, H)
    elif agg == "sum":
        qid_vecs = cog_vecs.sum(axis=1)
    elif agg == "max":
        qid_vecs = cog_vecs.max(axis=1)
    else:
        raise ValueError(f"Unknown agg='{agg}', choose from 'mean', 'sum', 'max'.")

    # ---------- 5. Project to embed_dim if necessary ----------
    if model_hidden_dim != embed_dim:
        print(
            f"[build_qid_text_embeddings] Projecting from {model_hidden_dim} → {embed_dim} "
            f"using a random linear layer (frozen). Consider training this projection."
        )
        proj = torch.nn.Linear(model_hidden_dim, embed_dim, bias=False)
        torch.nn.init.xavier_uniform_(proj.weight)
        proj.eval()
        with torch.no_grad():
            qid_vecs_t = torch.tensor(qid_vecs, dtype=torch.float32)
            qid_vecs = proj(qid_vecs_t).numpy()   # (max_qid, embed_dim)
    else:
        qid_vecs = torch.tensor(qid_vecs, dtype=torch.float32)
        qid_vecs = qid_vecs.numpy()

    # ---------- 6. Build final matrix: row 0 = zeros (padding) ----------
    emb_matrix = np.zeros((max_qid + 1, embed_dim), dtype=np.float32)
    emb_matrix[1:] = qid_vecs  # rows 1..max_qid

    return torch.tensor(emb_matrix, dtype=torch.float32).to(device)