from fastapi import APIRouter, Header, HTTPException, Query
from firebase_auth import verify_token
from database import get_user_analyses, get_analysis

router = APIRouter(prefix="/api")


@router.get("/history")
async def list_history(limit: int = Query(20, ge=1, le=100), authorization: str = Header(...)):
    user = await verify_token(authorization)
    uid = user["uid"]

    analyses = get_user_analyses(uid, limit=limit)
    return {"analyses": analyses}


@router.get("/history/{analysis_id}")
async def get_history_item(analysis_id: str, authorization: str = Header(...)):
    user = await verify_token(authorization)
    uid = user["uid"]

    analysis = get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if analysis.get("uid") != uid:
        raise HTTPException(status_code=403, detail="Access denied")

    return analysis
