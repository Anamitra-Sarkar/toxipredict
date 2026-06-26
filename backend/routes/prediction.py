import uuid
import torch
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from firebase_auth import verify_token
from database import save_analysis
from models.molecule_graph import smiles_to_graph
from config import TASK_NAMES, TASK_CLASSES, NUM_TASKS

router = APIRouter(prefix="/api")


class PredictRequest(BaseModel):
    smiles: str


class AssayPrediction(BaseModel):
    assay: str
    target_class: str
    probability: float
    predicted_class: str


class PredictResponse(BaseModel):
    analysis_id: str
    smiles: str
    predictions: list[AssayPrediction]
    uncertainty_weights: list[float]
    molecule_num_atoms: int


def get_app_state():
    from app import model_loader
    return model_loader


@router.post("/predict")
async def predict(request: PredictRequest, authorization: str = Header(...)):
    user = await verify_token(authorization)
    uid = user["uid"]

    model_loader = get_app_state()
    model = model_loader.load_model()

    try:
        data = smiles_to_graph(request.smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr,
                       torch.zeros(data.x.shape[0], dtype=torch.long, device=device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    predictions = []
    for i in range(NUM_TASKS):
        prob = float(probs[i])
        predictions.append(AssayPrediction(
            assay=TASK_NAMES[i],
            target_class=TASK_CLASSES.get(TASK_NAMES[i], "Unknown"),
            probability=round(prob, 4),
            predicted_class="Toxic" if prob > 0.5 else "Non-Toxic",
        ))

    analysis_id = str(uuid.uuid4())

    uncertainty_weights = []
    try:
        from app import criterion
        if criterion is not None:
            uncertainty_weights = torch.exp(-criterion.s).detach().cpu().tolist()
    except Exception:
        uncertainty_weights = [1.0] * NUM_TASKS

    save_analysis(analysis_id, uid, {
        "smiles": request.smiles,
        "predictions": [p.model_dump() for p in predictions],
        "uncertainty_weights": uncertainty_weights,
        "molecule_num_atoms": data.x.shape[0],
    })

    return PredictResponse(
        analysis_id=analysis_id,
        smiles=request.smiles,
        predictions=predictions,
        uncertainty_weights=uncertainty_weights,
        molecule_num_atoms=data.x.shape[0],
    )
