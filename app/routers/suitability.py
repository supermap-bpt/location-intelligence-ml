from fastapi import APIRouter, HTTPException
from app.models.requests import BatchRequest
from app.services.suitability import batch_predict_service

router = APIRouter()

@router.post("/batch-predict")
async def batch_predict(request: BatchRequest):
    try:
        return batch_predict_service(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))