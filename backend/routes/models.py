from fastapi import APIRouter
from config import TASK_NAMES, TASK_CLASSES, NUM_TASKS

router = APIRouter(prefix="/api")


@router.get("/models")
async def list_models():
    return {
        "models": [
            {
                "name": "toxipredict-gnn-v1",
                "architecture": "MultiTaskGNN_ResGATv2_JK_VN (3×ResGATv2, 4 heads, JK concat, mean+max pool, 10 task heads)",
                "num_tasks": NUM_TASKS,
                "tasks": [
                    {"assay": name, "target_class": TASK_CLASSES.get(name, "Unknown")}
                    for name in TASK_NAMES
                ],
                "version": "1.0.0",
                "status": "loaded",
            }
        ]
    }
