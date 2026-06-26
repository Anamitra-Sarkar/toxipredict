from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from firebase_auth import verify_token
from agent.toxicologist_agent import ToxicologistAgent
from agent.structural_alerts import identify_structural_alerts
from config import TASK_NAMES

router = APIRouter(prefix="/api")


class AgentRequest(BaseModel):
    smiles: str
    predictions: list
    shap_attributions: list[float]
    high_impact_atoms: list[int]
    target_assay: str = "NR-AR"


class AgentResponse(BaseModel):
    target_assay: str
    risk_level: str
    toxicophore_assessment: str
    biochemical_mechanism: str
    structural_alerts_found: list
    bioisosteric_replacements: list
    confidence: str


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = ToxicologistAgent()
    return _agent


@router.post("/agent")
async def agent_analysis(request: AgentRequest, authorization: str = Header(...)):
    await verify_token(authorization)

    agent = get_agent()

    alerts = identify_structural_alerts(
        request.smiles,
        high_attribution_indices=request.high_impact_atoms,
    )

    try:
        report = agent.generate_report(
            smiles=request.smiles,
            target_assay=request.target_assay,
            predictions=request.predictions,
            shap_attributions=request.shap_attributions,
            high_attr_indices=request.high_impact_atoms,
            alerts_found=alerts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")

    return AgentResponse(
        target_assay=report.target_assay,
        risk_level=report.risk_level,
        toxicophore_assessment=report.toxicophore_assessment,
        biochemical_mechanism=report.biochemical_mechanism,
        structural_alerts_found=alerts,
        bioisosteric_replacements=report.bioisosteric_replacements,
        confidence=report.confidence,
    )
