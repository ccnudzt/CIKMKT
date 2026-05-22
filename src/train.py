import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# Response padding ID is pulled out as a constant for clarity
R_PAD = 2 

def train_one_epoch(model, train_loader, optimizer, criterion, device, desc="Train"):
    model.train()
    epoch_losses = []
    all_labels, all_preds = [], []

    pbar = tqdm(train_loader, desc=desc, leave=False)
    for batch in pbar:
        qids = batch["qids"].to(device)
        rs = batch["rs"].to(device)
        masks = batch["masks"].to(device)

        input_qids = qids[:, :-1]
        input_rs = rs[:, :-1]

        labels = rs[:, 1:].float()
        label_masks = masks[:, 1:].bool()
        label_masks = label_masks & (rs[:, 1:] != R_PAD)

        logits = model(input_qids, input_rs)  # (B, L-1)

        loss_mat = criterion(logits, labels)
        denom = label_masks.sum().clamp(min=1)
        loss = (loss_mat * label_masks).sum() / denom

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_labels.append(labels[label_masks].detach().cpu().numpy())
            all_preds.append(probs[label_masks].detach().cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])

    if len(all_labels) == 0:
        return avg_loss, float("nan"), float("nan")

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels.astype(int), (all_preds >= 0.5).astype(int))
    return avg_loss, auc, acc


@torch.no_grad()
def validate(model, val_loader, device, desc="Val"):
    model.eval()
    all_labels, all_preds = [], []

    pbar = tqdm(val_loader, desc=desc, leave=False)
    for batch in pbar:
        qids = batch["qids"].to(device)
        rs = batch["rs"].to(device)
        masks = batch["masks"].to(device)

        input_qids = qids[:, :-1]
        input_rs = rs[:, :-1]

        labels = rs[:, 1:].float()
        label_masks = masks[:, 1:].bool()
        label_masks = label_masks & (rs[:, 1:] != R_PAD)

        logits = model(input_qids, input_rs)
        probs = torch.sigmoid(logits)

        all_labels.append(labels[label_masks].cpu().numpy())
        all_preds.append(probs[label_masks].cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])

    if len(all_labels) == 0:
        return float("nan"), float("nan")

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels.astype(int), (all_preds >= 0.5).astype(int))
    return auc, acc