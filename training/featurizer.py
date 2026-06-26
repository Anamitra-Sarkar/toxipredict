import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

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

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0, 0],
    Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0],
    Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0],
    Chem.rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0],
    Chem.rdchem.HybridizationType.SP3D2: [0, 0, 0, 0, 1],
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
    hybridization = HYBRIDIZATION_MAP.get(atom.GetHybridization(), [0, 0, 0, 0, 0])
    degree = min(atom.GetDegree(), 6) / 6.0
    valence = min(atom.GetTotalValence(), 8) / 8.0
    formal_charge = np.clip(atom.GetFormalCharge(), -3, 3)
    charge_feat = (formal_charge + 3) / 6.0
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    h_count = min(atom.GetTotalNumHs(), 4) / 4.0
    return atomic_type + hybridization + [degree, valence, charge_feat, aromatic, h_count]


def get_bond_features(bond):
    bond_type = BOND_ENCODER.get(bond.GetBondType(), [0, 0, 0, 0])
    conjugated = 1.0 if bond.GetIsConjugated() else 0.0
    stereo_st = bond.GetStereo()
    stereo = [1, 0, 0, 0]
    if stereo_st == Chem.rdchem.BondStereo.STEREOANY:
        stereo = [0, 1, 0, 0]
    elif stereo_st == Chem.rdchem.BondStereo.STEREOZ:
        stereo = [0, 0, 1, 0]
    elif stereo_st == Chem.rdchem.BondStereo.STEREOE:
        stereo = [0, 0, 0, 1]
    return bond_type + [conjugated] + stereo


def smiles_to_graph_data(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
    except Exception:
        mol = Chem.RemoveHs(mol)

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float32)

    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        bf = get_bond_features(bond)
        edge_features.extend([bf, bf])

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
