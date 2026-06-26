import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from models.multitask_gnn import MultiTaskGNN
from models.uncertainty_loss import HomoscedasticUncertaintyLoss
from config import NODE_DIM, EDGE_DIM, HIDDEN_DIM, DROPOUT, NUM_TASKS, TASK_NAMES
from training.dataset import Tox21Dataset, TASK_NAMES as DS_TASK_NAMES
from training.scaffold_split import ScaffoldSplitter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 20


def compute_metrics(logits, labels, mask):
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    mask_np = mask.cpu().numpy()

    aucs = []
    f1s = []
    accs = []
    for t in range(NUM_TASKS):
        task_mask = mask_np[:, t] == 1
        if task_mask.sum() < 5:
            continue
        y_true = labels_np[task_mask, t]
        y_prob = probs[task_mask, t]
        y_pred = (y_prob > 0.5).astype(int)

        if len(np.unique(y_true)) > 1:
            try:
                aucs.append(roc_auc_score(y_true, y_prob))
            except ValueError:
                pass
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        accs.append(accuracy_score(y_true, y_pred))

    return {
        "auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
        "acc_mean": float(np.mean(accs)) if accs else 0.0,
        "auc_list": aucs,
    }


def train_epoch(model, criterion, optimizer, loader):
    model.train()
    total_loss = 0
    all_logits, all_labels, all_masks = [], [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss, _, _ = criterion(logits, batch.y, batch.mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch.y.cpu())
        all_masks.append(batch.mask.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    mask = torch.cat(all_masks)
    metrics = compute_metrics(logits, labels, mask)

    return total_loss / len(loader), metrics


def validate_epoch(model, criterion, loader):
    model.eval()
    total_loss = 0
    all_logits, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss, _, _ = criterion(logits, batch.y, batch.mask)
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())
            all_masks.append(batch.mask.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    mask = torch.cat(all_masks)
    metrics = compute_metrics(logits, labels, mask)

    return total_loss / len(loader), metrics


def main():
    print("=" * 60)
    print("ToxiPredict - Multi-Task GNN Training")
    print(f"Device: {DEVICE}")
    print(f"Number of tasks: {NUM_TASKS}")
    print(f"Tasks: {DS_TASK_NAMES}")
    print("=" * 60)

    print("\nLoading Tox21 dataset...")
    dataset = Tox21Dataset(root="data/tox21")
    print(f"Dataset size: {len(dataset)} compounds")

    densities = dataset.analyze_label_density()
    print("\nLabel density per task:")
    for d in densities:
        print(f"  {d['task']:20s} valid={d['num_valid']:5d}  "
              f"pos_ratio={d['pos_ratio']:.3f}  missing={d['missing_ratio']:.3f}")

    print("\nComputing Bemis-Murcko scaffold splits...")
    splitter = ScaffoldSplitter()
    df = pd.read_csv(dataset.csv_path)
    fold_assignments = splitter.split(dataset.smiles_list, n_splits=5)
    print(f"Scaffold split complete. Fold distribution:")
    for f in range(5):
        print(f"  Fold {f}: {(fold_assignments == f).sum()} compounds")

    fold_results = []

    for fold in range(5):
        print(f"\n{'=' * 60}")
        print(f"Starting Fold {fold + 1} / 5")
        print(f"{'=' * 60}")

        train_idx = np.where(fold_assignments != fold)[0]
        val_idx = np.where(fold_assignments == fold)[0]

        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = MultiTaskGNN(
            in_channels=NODE_DIM,
            edge_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_tasks=NUM_TASKS,
            dropout=DROPOUT,
        ).to(DEVICE)

        criterion = HomoscedasticUncertaintyLoss(num_tasks=NUM_TASKS).to(DEVICE)
        optimizer = Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )

        best_auc = 0.0
        best_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_aucs = []

        for epoch in range(EPOCHS):
            train_loss, train_metrics = train_epoch(model, criterion, optimizer, train_loader)
            val_loss, val_metrics = validate_epoch(model, criterion, val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_aucs.append(val_metrics["auc_mean"])

            if (epoch + 1) % 5 == 0 or epoch == 0:
                s_t = criterion.s.detach().cpu().numpy()
                print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val AUC: {val_metrics['auc_mean']:.4f} | "
                      f"Val F1: {val_metrics['f1_mean']:.4f} | "
                      f"s_t range: [{s_t.min():.2f}, {s_t.max():.2f}]")

            if val_metrics["auc_mean"] > best_auc:
                best_auc = val_metrics["auc_mean"]
                best_state = {
                    "model": model.state_dict(),
                    "criterion": criterion.state_dict(),
                    "epoch": epoch + 1,
                    "val_auc": val_metrics["auc_mean"],
                    "val_f1": val_metrics["f1_mean"],
                    "val_acc": val_metrics["acc_mean"],
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

        fold_results.append({
            "fold": fold + 1,
            "best_auc": best_auc,
            "best_metrics": {
                "auc": best_state["val_auc"],
                "f1": best_state["val_f1"],
                "acc": best_state["val_acc"],
            },
            "epochs_trained": epoch + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_aucs": val_aucs,
            "state_dicts": best_state,
        })

        print(f"  Fold {fold + 1} best Val AUC: {best_auc:.4f}")

    print(f"\n{'=' * 60}")
    print("Cross-Validation Results:")
    print(f"{'=' * 60}")
    aucs = [r["best_auc"] for r in fold_results]
    f1s = [r["best_metrics"]["f1"] for r in fold_results]
    accs = [r["best_metrics"]["acc"] for r in fold_results]
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    best_fold = np.argmax(aucs)
    print(f"\nBest fold: {best_fold + 1} (AUC = {aucs[best_fold]:.4f})")

    print("\nTraining final model on full dataset...")
    final_model = MultiTaskGNN(
        in_channels=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_tasks=NUM_TASKS,
        dropout=DROPOUT,
    ).to(DEVICE)

    final_criterion = HomoscedasticUncertaintyLoss(num_tasks=NUM_TASKS).to(DEVICE)
    final_optimizer = Adam(
        list(final_model.parameters()) + list(final_criterion.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    full_loader = DataLoader(
        [dataset[i] for i in range(len(dataset))],
        batch_size=BATCH_SIZE, shuffle=True,
    )

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        final_model.train()
        total_loss = 0
        for batch in full_loader:
            batch = batch.to(DEVICE)
            final_optimizer.zero_grad()
            logits = final_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss, _, _ = final_criterion(logits, batch.y, batch.mask)
            loss.backward()
            final_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(full_loader)
        s_t = final_criterion.s.detach().cpu().numpy()

        if (epoch + 1) % 10 == 0:
            print(f"  Final Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"s_t: [{s_t.min():.2f}, {s_t.max():.2f}]")

        if avg_loss < best_val_auc or best_val_auc == 0:
            best_val_auc = avg_loss
            best_state = {
                "model": final_model.state_dict(),
                "criterion": final_criterion.state_dict(),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state["model"], "checkpoints/model_final.pt")
    print("\nFinal model saved to checkpoints/model_final.pt")

    results_summary = {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_acc_mean": float(np.mean(accs)),
        "best_fold": int(best_fold + 1),
        "num_compounds": len(dataset),
        "num_tasks": NUM_TASKS,
        "epochs_trained": epoch + 1,
    }

    with open("checkpoints/training_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\nTraining complete!")
    print(json.dumps(results_summary, indent=2))
    return results_summary


if __name__ == "__main__":
    import pandas as pd
    main()
