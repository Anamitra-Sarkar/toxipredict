import numpy as np
import torch
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from firebase_auth import verify_token
from models.molecule_graph import smiles_to_graph
from explain.shap_wrapper import GNNFeatureWrapper
from explain.visualizer import generate_similarity_map
from config import TASK_NAMES

router = APIRouter(prefix="/api")


class ExplainRequest(BaseModel):
    smiles: str
    target_task: int = 0


class ExplainResponse(BaseModel):
    target_task: str
    shap_values: list[float]
    attribution_map_base64: str
    high_impact_atoms: list[int]


def get_app_state():
    from app import model_loader
    return model_loader


@router.post("/explain")
async def explain(request: ExplainRequest, authorization: str = Header(...)):
    await verify_token(authorization)

    if request.target_task < 0 or request.target_task >= len(TASK_NAMES):
        raise HTTPException(status_code=400, detail=f"Invalid target_task. Must be 0-{len(TASK_NAMES)-1}")

    model_loader = get_app_state()
    model = model_loader.load_model()

    try:
        data = smiles_to_graph(request.smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    num_nodes = data.x.shape[0]
    predict_fn = GNNFeatureWrapper(model, data, request.target_task, device)

    random_states = np.random.randint(0, 2, size=(20, num_nodes))
    background_reference = np.vstack([np.zeros((1, num_nodes)), random_states])
    active_state = np.ones((1, num_nodes))

    import shap
    explainer = shap.KernelExplainer(predict_fn, background_reference)
    nsamples = min(500, 2 ** num_nodes)
    shap_values = explainer.shap_values(active_state, nsamples=nsamples)

    attributions = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values)
    attributions = attributions.flatten().tolist()
    high_impact = np.where(np.abs(np.array(attributions)) > 0.05)[0].tolist()

    try:
        vis_base64 = generate_similarity_map(
            request.smiles, attributions, TASK_NAMES[request.target_task]
        )
    except Exception as e:
        vis_base64 = ""

    return ExplainResponse(
        target_task=TASK_NAMES[request.target_task],
        shap_values=attributions,
        attribution_map_base64=vis_base64,
        high_impact_atoms=high_impact,
    )
