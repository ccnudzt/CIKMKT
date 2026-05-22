from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute AUC and accuracy for binary predictions."""
    if labels.size == 0 or probs.size == 0:
        return {"auc": float("nan"), "acc": float("nan")}

    if len(np.unique(labels)) > 1:
        auc = float(roc_auc_score(labels, probs))
    else:
        auc = float("nan")

    acc = float(accuracy_score(labels.astype(int), (probs >= 0.5).astype(int)))
    return {"auc": auc, "acc": acc}
