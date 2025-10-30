# app/routes/facilities.py
from fastapi import APIRouter, Query
from typing import Optional, List
from app.services.facilities.facility_services import (
    list_facility_categories_service,
    get_facilities_by_categories_service,
)

router = APIRouter()


@router.get("/categories", summary="List facility categories")
def list_facility_categories():
    """
    Returns distinct categories and row counts from public.poi
    """
    return list_facility_categories_service()


@router.get("/by-categories", summary="Get facilities by selected categories")
def get_facilities_by_categories(
    categories: str = Query(..., description="Comma-separated `public.poi.kategori` values"),
    limit: int = Query(1000, ge=1, le=100000),
    bbox: Optional[str] = Query(
        None, description="Optional bbox as minx,miny,maxx,maxy (lon/lat, EPSG:4326)"
    ),
    code: Optional[str] = Query(
        None, description="Optional comma-separated district codes (kdkec) to filter facilities"
    ),
):
    # Parse inputs
    category_list: List[str] = [c.strip() for c in categories.split(",") if c.strip()]
    bbox_vals: Optional[List[float]] = None
    if bbox:
        try:
            bbox_vals = [float(x) for x in bbox.split(",")]
        except Exception:
            # Service will also validate length; raise here for clearer message
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Invalid bbox. Use minx,miny,maxx,maxy in lon/lat.")

    kdkec_list: Optional[List[str]] = None
    if code:
        kdkec_list = [k.strip() for k in code.split(",") if k.strip()]

    return get_facilities_by_categories_service(
        categories=category_list, limit=limit, bbox=bbox_vals, kdkec=kdkec_list
    )
