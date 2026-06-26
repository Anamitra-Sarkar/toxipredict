import datetime
from typing import Optional
from firebase_auth import get_db


def save_analysis(analysis_id: str, uid: str, data: dict) -> None:
    db = get_db()
    doc = {
        **data,
        "analysis_id": analysis_id,
        "uid": uid,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    db.collection("analyses").document(analysis_id).set(doc)
    db.collection("users").document(uid).collection("history").document(analysis_id).set(
        {"analysis_id": analysis_id, "created_at": doc["created_at"]}
    )


def get_analysis(analysis_id: str) -> Optional[dict]:
    db = get_db()
    doc = db.collection("analyses").document(analysis_id).get()
    return doc.to_dict() if doc.exists else None


def get_user_analyses(uid: str, limit: int = 20) -> list:
    db = get_db()
    refs = (
        db.collection("users")
        .document(uid)
        .collection("history")
        .order_by("created_at", direction="DESCENDING")
        .limit(limit)
        .stream()
    )
    analysis_ids = [ref.to_dict().get("analysis_id") for ref in refs]
    results = []
    for aid in analysis_ids:
        data = get_analysis(aid)
        if data:
            results.append(data)
    return results
