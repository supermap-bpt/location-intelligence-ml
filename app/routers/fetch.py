from fastapi import APIRouter, HTTPException, Query
from enum import Enum
from typing import Optional
from app.services.fetch import (
    get_grid_score,
    get_analysis_result,
    get_all_grid_scores,
    get_all_analysis_results,
)

router = APIRouter()

class Facility(str, Enum):
    sekolah = "sekolah"
    hotel = "hotel"
    pusatperbelanjaan = "pusatperbelanjaan"
    rumahsakit = "rumahsakit"

# ---------- GRID SCORES ----------
@router.get("/grid-score/{grid_id}")
async def fetch_grid_score(
    grid_id: int,
    facility: Optional[Facility] = Query(
        default=None,
        description="Optional facility layer to fetch within the selected kecamatan. "
                    "One of: sekolah, hotel, pusatperbelanjaan, rumahsakit."
    ),
):
    try:
        result = get_grid_score(grid_id, facility.value if facility else None)
        if "message" in result and result["message"] == "Grid score not found":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/grid-score")
async def fetch_all_grid_scores():
    try:
        return get_all_grid_scores()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- ANALYSIS RESULTS ----------
@router.get("/analysis-result/{result_id}")
async def fetch_analysis_result(result_id: int):
    try:
        result = get_analysis_result(result_id)
        if "message" in result and result["message"] == "Analysis result not found":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis-result")
async def fetch_all_analysis_results():
    try:
        return get_all_analysis_results()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))