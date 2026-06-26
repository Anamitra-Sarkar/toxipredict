import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class ScaffoldSplitter:
    @staticmethod
    def get_scaffold(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
        except Exception:
            return ""

    def split(self, smiles_list: list, n_splits: int = 5) -> np.ndarray:
        scaffolds = [self.get_scaffold(smi) for smi in smiles_list]
        scaffold_groups = {}
        for idx, sc in enumerate(scaffolds):
            if sc not in scaffold_groups:
                scaffold_groups[sc] = []
            scaffold_groups[sc].append(idx)

        sorted_scaffolds = sorted(
            scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True
        )
        folds = [[] for _ in range(n_splits)]
        fold_sizes = np.zeros(n_splits, dtype=int)

        for _, indices in sorted_scaffolds:
            min_fold = np.argmin(fold_sizes)
            folds[min_fold].extend(indices)
            fold_sizes[min_fold] += len(indices)

        assignments = np.zeros(len(smiles_list), dtype=int)
        for fold_idx, fold_indices in enumerate(folds):
            for idx in fold_indices:
                assignments[idx] = fold_idx

        return assignments
