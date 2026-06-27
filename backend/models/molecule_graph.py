import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


ATOM_TYPES = [6, 7, 8, 16, 15, 9, 17, 35, 53]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def one_hot(val, choices):
    encoding = [0] * (len(choices) + 1)
    for i, c in enumerate(choices):
        if val == c:
            encoding[i] = 1
            return encoding
    encoding[-1] = 1
    return encoding


def get_atom_features(atom):
    feat = []
    feat += one_hot(atom.GetAtomicNum(), ATOM_TYPES)
    feat += one_hot(atom.GetDegree(), list(range(7)))
    feat += one_hot(atom.GetTotalValence(), list(range(7)))
    feat += one_hot(atom.GetFormalCharge(), list(range(-3, 4)))
    feat += one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)
    feat.append(1.0 if atom.GetIsAromatic() else 0.0)
    feat += one_hot(atom.GetTotalNumHs(), list(range(5)))
    feat.append(float(atom.GetNumRadicalElectrons()))
    feat.append(1.0 if atom.IsInRing() else 0.0)
    return feat


def get_bond_features(bond, mol):
    feat = [1 if bond.GetBondType() == bt else 0 for bt in BOND_TYPES]
    feat.append(1.0 if bond.GetIsConjugated() else 0.0)
    feat.append(1.0 if bond.IsInRing() else 0.0)
    feat += [1 if bond.GetStereo() == s else 0 for s in STEREO_TYPES]
    conf = mol.GetConformer()
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    dist = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
    feat.append(min(dist / 2.0, 1.0))
    return feat


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
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
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bf = get_bond_features(bond, mol)
        edge_features.append(bf)
        edge_features.append(bf)

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 11), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
