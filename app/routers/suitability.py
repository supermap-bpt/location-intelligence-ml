from fastapi import APIRouter, HTTPException
from app.models.requests import BatchRequest, SuitabilityRequest
from app.services.suitability.suitability import batch_predict_service

router = APIRouter()

@router.post("/batch-predict")
async def batch_predict(request: BatchRequest):
    try:
        return await batch_predict_service(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations")
async def calculate_suitability(request: SuitabilityRequest):
    try:
        return await batch_predict_service(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))