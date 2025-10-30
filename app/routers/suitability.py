from fastapi import APIRouter, HTTPException
from app.models.requests import SuitabilityRequest
from app.models.responses import SuitabilityAnalysisResponse
from app.services.suitability.suitability import _run_suitability

router = APIRouter()

@router.post("/recommendations", response_model=SuitabilityAnalysisResponse)
async def calculate_suitability(request: SuitabilityRequest) -> SuitabilityAnalysisResponse:
    """
    Calculate suitability analysis for provided grid geometries based on mandatory and optional parameters.

    Returns a comprehensive analysis including:
    - Suitability classification for each grid
    - Feature scores and composite values
    - Administrative boundaries (province, district, subdistrict)
    - Classification thresholds
    """
    try:
        return await _run_suitability(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))