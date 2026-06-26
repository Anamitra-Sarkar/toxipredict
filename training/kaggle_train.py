"""
ToxiPredict - Kaggle Training Script
Run this on Kaggle GPU (anamitrasarkar007 account)

This script:
1. Downloads Tox21 dataset
2. Featurizes molecules with RDKit
3. Trains MultiTaskGNN with HomoscedasticUncertaintyLoss
4. Implements 5-fold Bemis-Murcko scaffold cross-validation
5. Early stopping with patience=20
6. Uploads best checkpoint to HuggingFace Arko007/toxipredict-gnn-models
"""

import os
import json
import sys
import subprocess

subprocess.run(["pip", "install", "-q", "torch-geometric", "torch-scatter", "torch-sparse",
                 "rdkit-pypi", "safetensors", "huggingface_hub", "pandas", "scikit-learn",
                 "matplotlib"], check=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

TASK_NAMES = [
    "NR-AR", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
    "SR-MMP", "SR-p53",
]
NUM_TASKS = len(TASK_NAMES)
NODE_DIM = 15
EDGE_DIM = 6
HIDDEN_DIM = 128
DROPOUT = 0.15
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20

ATOM_ENCODER = {
    6: [1, 0, 0, 0, 0, 0, 0],
    7: [0, 1, 0, 0, 0, 0, 0],
    8: [0, 0, 1, 0, 0, 0, 0],
    16: [0, 0, 0, 1, 0, 0, 0],
    15: [0, 0, 0, 0, 1, 0, 0],
    17: [0, 0, 0, 0, 0, 1, 0],
    35: [0, 0, 0, 0, 0, 1, 0],
    9: [0, 0, 0, 0, 0, 1, 0],
    53: [0, 0, 0, 0, 0, 1, 0],
}

BOND_ENCODER = {
    Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
    Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
    Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
    Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
}


def get_atom_features(atom):
    atomic_num = atom.GetAtomicNum()
    atomic_type = ATOM_ENCODER.get(atomic_num, [0, 0, 0, 0, 0, 0, 0])
    hyb = atom.GetHybridization()
    hyb_map = {Chem.rdchem.HybridizationType.SP: [1,0,0,0,0],
               Chem.rdchem.HybridizationType.SP2: [0,1,0,0,0],
               Chem.rdchem.HybridizationType.SP3: [0,0,1,0,0],
               Chem.rdchem.HybridizationType.SP3D: [0,0,0,1,0],
               Chem.rdchem.HybridizationType.SP3D2: [0,0,0,0,1]}
    hybridization = hyb_map.get(hyb, [0,0,0,0,0])
    degree = min(atom.GetDegree(), 6) / 6.0
    valence = min(atom.GetTotalValence(), 8) / 8.0
    fc = np.clip(atom.GetFormalCharge(), -3, 3)
    charge_feat = (fc + 3) / 6.0
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    h_count = min(atom.GetTotalNumHs(), 4) / 4.0
    return atomic_type + hybridization + [degree, valence, charge_feat, aromatic, h_count]


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
    except:
        mol = Chem.RemoveHs(mol)
    atom_features = [get_atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float32)
    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_ENCODER.get(bond.GetBondType(), [0,0,0,0])
        conj = 1.0 if bond.GetIsConjugated() else 0.0
        bf = bt + [conj]
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bf, bf])
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MultiTaskGNN(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dim, num_tasks, dropout=0.15):
        super().__init__()
        self.num_tasks = num_tasks
        self.conv1 = GATConv(in_channels, hidden_dim, heads=4, concat=True, edge_dim=edge_dim, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=False, edge_dim=edge_dim, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_tasks)])

    def forward(self, x, edge_index, edge_attr, batch):
        h = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        h = F.relu(self.bn2(self.conv2(h, edge_index, edge_attr)))
        hg = global_mean_pool(h, batch)
        shared = self.fc_shared(hg)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class HomoscedasticLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(num_tasks))
        self.num_tasks = num_tasks

    def forward(self, logits, targets, mask):
        total = 0.0
        weights = torch.exp(-self.s)
        for t in range(self.num_tasks):
            valid = torch.where(mask[:, t] == 1)[0]
            if len(valid) == 0:
                continue
            bce = F.binary_cross_entropy_with_logits(
                logits[valid, t], targets[valid, t].float(), reduction='mean'
            )
            total += weights[t] * bce + 0.5 * self.s[t]
        return total, weights.detach()


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), canonical=True)
    except:
        return ""


def scaffold_split(smiles_list, n_splits=5):
    scaffolds = [get_scaffold(s) for s in smiles_list]
    groups = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        groups[sc].append(i)
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    folds = [[] for _ in range(n_splits)]
    sizes = np.zeros(n_splits, dtype=int)
    for _, indices in sorted_groups:
        fold = np.argmin(sizes)
        folds[fold].extend(indices)
        sizes[fold] += len(indices)
    assignments = np.zeros(len(smiles_list), dtype=int)
    for f, indices in enumerate(folds):
        for idx in indices:
            assignments[idx] = f
    return assignments


def compute_metrics(logits, labels, mask):
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    mask_np = mask.cpu().numpy()
    aucs, f1s, accs = [], [], []
    for t in range(NUM_TASKS):
        m = mask_np[:, t] == 1
        if m.sum() < 5:
            continue
        y_true, y_prob = labels_np[m, t], probs[m, t]
        y_pred = (y_prob > 0.5).astype(int)
        if len(np.unique(y_true)) > 1:
            try:
                aucs.append(roc_auc_score(y_true, y_prob))
            except:
                pass
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        accs.append(accuracy_score(y_true, y_pred))
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
        "acc_mean": float(np.mean(accs)) if accs else 0.0,
    }


def main():
    print("=" * 60)
    print("ToxiPredict - Kaggle Training")
    print("=" * 60)

    print("\nDownloading Tox21 dataset...")
    url = "https://github.com/deepchem/deepchem/raw/master/datasets/tox21.csv.gz"
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} compounds, {len(df.columns)} columns")

    smiles_list = df["smiles"].tolist()

    labels_list, masks_list = [], []
    for _, row in df.iterrows():
        task_labels, task_masks = [], []
        for task in TASK_NAMES:
            val = row.get(task)
            if pd.isna(val):
                task_labels.append(0)
                task_masks.append(0)
            else:
                task_labels.append(int(val))
                task_masks.append(1)
        labels_list.append(task_labels)
        masks_list.append(task_masks)

    labels_tensor = torch.tensor(labels_list, dtype=torch.float32)
    masks_tensor = torch.tensor(masks_list, dtype=torch.float32)

    print(f"\nLabel statistics:")
    for t in range(NUM_TASKS):
        valid = masks_tensor[:, t] == 1
        n_valid = valid.sum().item()
        pos_ratio = labels_tensor[valid, t].mean().item() if n_valid > 0 else 0
        missing = 1 - n_valid / len(df)
        print(f"  {TASK_NAMES[t]:15s} valid={n_valid:5d}  pos_ratio={pos_ratio:.3f}  missing={missing:.3f}")

    print("\nComputing scaffolds...")
    fold_assignments = scaffold_split(smiles_list, n_splits=5)
    for f in range(5):
        print(f"  Fold {f}: {(fold_assignments == f).sum()} compounds")

    all_fold_metrics = []

    for fold in range(5):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold + 1} / 5")
        print(f"{'=' * 50}")

        train_idx = np.where(fold_assignments != fold)[0]
        val_idx = np.where(fold_assignments == fold)[0]

        train_graphs = []
        for i in train_idx:
            g = smiles_to_graph(smiles_list[i])
            if g is not None:
                g.y = labels_tensor[i].unsqueeze(0)
                g.mask = masks_tensor[i].unsqueeze(0)
                train_graphs.append(g)

        val_graphs = []
        for i in val_idx:
            g = smiles_to_graph(smiles_list[i])
            if g is not None:
                g.y = labels_tensor[i].unsqueeze(0)
                g.mask = masks_tensor[i].unsqueeze(0)
                val_graphs.append(g)

        print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

        model = MultiTaskGNN(NODE_DIM, EDGE_DIM, HIDDEN_DIM, NUM_TASKS, DROPOUT).to(DEVICE)
        criterion = HomoscedasticLoss(NUM_TASKS).to(DEVICE)
        optimizer = Adam(list(model.parameters()) + list(criterion.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

        best_auc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss, _ = criterion(logits, batch.y, batch.mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_logits, val_labels, val_masks = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    val_logits.append(logits.cpu())
                    val_labels.append(batch.y.cpu())
                    val_masks.append(batch.mask.cpu())

            val_logits = torch.cat(val_logits)
            val_labels = torch.cat(val_labels)
            val_masks = torch.cat(val_masks)
            metrics = compute_metrics(val_logits, val_labels, val_masks)

            s_t = criterion.s.detach().cpu().numpy()
            train_loss_avg = train_loss / len(train_loader)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss_avg:.4f} | "
                      f"Val AUC: {metrics['auc_mean']:.4f} | Val F1: {metrics['f1_mean']:.4f} | "
                      f"s_t: [{s_t.min():.2f}, {s_t.max():.2f}]")

            if metrics["auc_mean"] > best_auc:
                best_auc = metrics["auc_mean"]
                best_state = {
                    "model": model.state_dict(),
                    "criterion": criterion.state_dict(),
                    "epoch": epoch + 1,
                    "val_auc": metrics["auc_mean"],
                    "val_f1": metrics["f1_mean"],
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        print(f"  Fold {fold + 1} best Val AUC: {best_auc:.4f}")
        all_fold_metrics.append(best_auc)

    print(f"\n{'=' * 60}")
    print(f"Cross-Validation Results (5-fold):")
    print(f"  AUC: {np.mean(all_fold_metrics):.4f} ± {np.std(all_fold_metrics):.4f}")
    print(f"{'=' * 60}")

    print("\nTraining final model on full dataset...")
    all_graphs = []
    for i in range(len(smiles_list)):
        g = smiles_to_graph(smiles_list[i])
        if g is not None:
            g.y = labels_tensor[i].unsqueeze(0)
            g.mask = masks_tensor[i].unsqueeze(0)
            all_graphs.append(g)

    full_loader = DataLoader(all_graphs, batch_size=BATCH_SIZE, shuffle=True)

    final_model = MultiTaskGNN(NODE_DIM, EDGE_DIM, HIDDEN_DIM, NUM_TASKS, DROPOUT).to(DEVICE)
    final_criterion = HomoscedasticLoss(NUM_TASKS).to(DEVICE)
    final_optimizer = Adam(list(final_model.parameters()) + list(final_criterion.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        final_model.train()
        total_loss = 0.0
        for batch in full_loader:
            batch = batch.to(DEVICE)
            final_optimizer.zero_grad()
            logits = final_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss, weights = final_criterion(logits, batch.y, batch.mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
            final_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(full_loader)
        s_t = final_criterion.s.detach().cpu().numpy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Task Weights: [{s_t.min():.2f}, {s_t.max():.2f}] | "
                  f"Weights: {weights.cpu().numpy().round(3)}")

        if avg_loss < best_loss:
            best_loss = avg_loss
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

    print("\nSaving checkpoint...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state["model"], "checkpoints/model_final.pt")

    print("\nUploading to HuggingFace...")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    REPO_ID = "Arko007/toxipredict-gnn-models"

    api = HfApi(token=HF_TOKEN)
    try:
        create_repo(REPO_ID, private=False, exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Repo error: {e}")

    save_file(best_state["model"], "checkpoints/model.safetensors")

    config = {
        "node_dim": NODE_DIM, "edge_dim": EDGE_DIM,
        "hidden_dim": HIDDEN_DIM, "num_tasks": NUM_TASKS,
        "dropout": DROPOUT, "task_names": TASK_NAMES,
        "architecture": "MultiTaskGNN",
        "version": "1.0.0",
        "cv_auc_mean": float(np.mean(all_fold_metrics)),
        "cv_auc_std": float(np.std(all_fold_metrics)),
    }

    with open("checkpoints/model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    api.upload_file(path_or_fileobj="checkpoints/model.safetensors",
                    path_in_repo="model.safetensors",
                    repo_id=REPO_ID, token=HF_TOKEN)
    api.upload_file(path_or_fileobj="checkpoints/model_config.json",
                    path_in_repo="model_config.json",
                    repo_id=REPO_ID, token=HF_TOKEN)

    print(f"\nModel uploaded to: https://huggingface.co/{REPO_ID}")
    print("\nTraining complete!")
    print(f"Final CV AUC: {config['cv_auc_mean']:.4f} ± {config['cv_auc_std']:.4f}")


if __name__ == "__main__":
    main()
