from rdkit import Chem

STRUCTURAL_ALERTS_DB = {
    "aromatic_amine": {
        "smarts": "c[NH2]",
        "name": "Primary Aromatic Amine",
        "detail": "Known to undergo metabolic activation forming mutagenic nitrenium ions.",
    },
    "alkyl_halide": {
        "smarts": "[CX4][Cl,Br,I]",
        "name": "Alkyl Halide",
        "detail": "Direct-acting alkylating agent; electrophilic risk of DNA damage.",
    },
    "hydrazine": {
        "smarts": "[NX3][NX3]",
        "name": "Hydrazine",
        "detail": "Associated with genotoxic risk and reactive oxygen species generation.",
    },
    "aldehyde": {
        "smarts": "[CX3H1](=O)[#6]",
        "name": "Aldehyde",
        "detail": "Highly reactive electrophile; risk of DNA-protein crosslinks.",
    },
    "nitro": {
        "smarts": "[NX3](=O)=O",
        "name": "Nitro Aromatic",
        "detail": "Undergoes metabolic reduction to reactive nitroso and hydroxylamine intermediates.",
    },
    "epoxide": {
        "smarts": "[O]1[CH][CH]1",
        "name": "Epoxide",
        "detail": "Highly strained electrophilic ring; direct DNA alkylation risk.",
    },
    "quinone": {
        "smarts": "O=c1cccc(=O)c1",
        "name": "Quinone",
        "detail": "Redox-active species generating ROS and forming covalent adducts.",
    },
    "azide": {
        "smarts": "[NX2-]N#[N+]",
        "name": "Azide",
        "detail": "Explosive and reactive; potential for bioorthogonal toxicity.",
    },
    "aflatoxin_b1_like": {
        "smarts": "O=C1OC2C3OC3C=CC2=C1",
        "name": "Aflatoxin-like Furan",
        "detail": "Metabolic epoxidation leads to DNA adduct formation.",
    },
    "beta_lactam": {
        "smarts": "O=C1[CH][CH]N1",
        "name": "Beta-Lactam",
        "detail": "Acylating agent; risk of hypersensitivity and covalent protein binding.",
    },
}


def identify_structural_alerts(smiles: str, high_attribution_indices: list = None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    alerts_found = []

    for key, alert in STRUCTURAL_ALERTS_DB.items():
        pattern = Chem.MolFromSmarts(alert["smarts"])
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            match_attributed = False
            if high_attribution_indices and match:
                match_attributed = any(idx in high_attribution_indices for idx in match)

            alerts_found.append({
                "alert_name": alert["name"],
                "details": alert["detail"],
                "matched_atoms": list(match),
                "high_attribution": match_attributed,
            })

    return alerts_found
