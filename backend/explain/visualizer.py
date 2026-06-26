import io
import base64
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_similarity_map(smiles: str, attributions: list, target_assay: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fig, ax = plt.subplots(figsize=(8, 6))
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        attributions,
        colorMap='RdYlBu_r',
        scale=-1,
        alpha=0.45,
    )
    ax.set_title(f"SHAP Attributions — {target_assay}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
