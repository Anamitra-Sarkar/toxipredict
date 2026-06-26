import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from featurizer import smiles_to_graph_data

TOX21_URL = "https://github.com/deepchem/deepchem/raw/master/datasets/tox21.csv.gz"

TASK_NAMES = [
    "NR-AR", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
    "SR-MMP", "SR-p53",
]

NUM_TASKS = len(TASK_NAMES)


class Tox21Dataset(Dataset):
    def __init__(self, root="data/tox21", transform=None, pre_transform=None):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.csv_path = os.path.join(root, "tox21.csv")
        if not os.path.exists(self.csv_path):
            self._download()
        self._process_csv()
        super(Tox21Dataset, self).__init__(root, transform, pre_transform)

    def _download(self):
        import urllib.request
        print("Downloading Tox21 dataset...")
        urllib.request.urlretrieve(TOX21_URL, self.csv_path)
        print("Download complete.")

    def _process_csv(self):
        df = pd.read_csv(self.csv_path)
        self.smiles_list = df["smiles"].tolist()

        self.labels = []
        self.masks = []
        for _, row in df.iterrows():
            task_labels = []
            task_masks = []
            for task in TASK_NAMES:
                val = row.get(task)
                if pd.isna(val):
                    task_labels.append(0)
                    task_masks.append(0)
                else:
                    task_labels.append(int(val))
                    task_masks.append(1)
            self.labels.append(task_labels)
            self.masks.append(task_masks)

        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.masks = torch.tensor(self.masks, dtype=torch.float32)

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles = self.smiles_list[idx]
        data = smiles_to_graph_data(smiles)
        if data is None:
            data = Data(
                x=torch.zeros((1, 15), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 6), dtype=torch.float32),
            )
        data.y = self.labels[idx].unsqueeze(0)
        data.mask = self.masks[idx].unsqueeze(0)
        return data

    def get_labels_and_masks(self):
        return self.labels, self.masks

    def analyze_label_density(self):
        labels, masks = self.get_labels_and_masks()
        densities = []
        for t in range(NUM_TASKS):
            task_mask = masks[:, t]
            task_labels = labels[:, t]
            valid = task_mask[task_mask == 1]
            if len(valid) > 0:
                pos_ratio = task_labels[task_mask == 1].mean().item()
                densities.append({
                    "task": TASK_NAMES[t],
                    "num_valid": int(task_mask.sum().item()),
                    "pos_ratio": round(pos_ratio, 4),
                    "missing_ratio": round(1 - task_mask.mean().item(), 4),
                })
        return densities
