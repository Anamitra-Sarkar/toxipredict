import io
import base64
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_similarity_map(smiles: str, attributions: list, target_assay: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if mol.GetNumAtoms() != len(attributions):
        raise ValueError(
            f"Atom count mismatch: mol has {mol.GetNumAtoms()} atoms, "
            f"but {len(attributions)} attributions provided"
        )

    AllChem.Compute2DCoords(mol)
    weights = np.array(attributions)
    vmax = max(abs(weights).max(), 0.01)

    highlight_colors = {}
    for i, w in enumerate(weights):
        if w > 0:
            intensity = min(abs(w) / vmax, 1.0)
            highlight_colors[i] = (1, 0, 0, intensity * 0.7)
        else:
            intensity = min(abs(w) / vmax, 1.0)
            highlight_colors[i] = (0, 0, 1, intensity * 0.7)

    img = Draw.MolToImage(
        mol,
        size=(600, 500),
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightColors=highlight_colors,
        kekulize=True,
    )

    buf = io.BytesIO()
    img.save(buf, format="png")
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
