from fastapi import APIRouter, HTTPException
from app.models.requests import AnalysisResultRequest, GridStoreRequest
from app.services.store import store_analysis_result, store_grid_score

router = APIRouter()

@router.post("/grid-score")
async def grid_score(request: GridStoreRequest):
    try:
        return store_grid_score(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis-result")
async def analysis_result(request: AnalysisResultRequest):
    try:
        return store_analysis_result(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))