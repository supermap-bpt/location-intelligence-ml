# app/services/facilities_service.py
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from sqlalchemy import text, bindparam
from app.database import engine

def list_facility_categories_service() -> List[Dict[str, Any]]:
    """
    Return distinct categories from public.poi
    """
    sql = text("""
        SELECT DISTINCT kategori AS category
        FROM public.poi
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [{"category": r._mapping["category"]} for r in rows]



def get_facilities_by_categories_service(
    categories: List[str],
    limit: int,
    bbox: Optional[List[float]] = None,
    kdkec: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch facilities filtered by kategori IN (...), optional bbox, optional kdkec list, and limit.
    bbox format: [minx, miny, maxx, maxy] (lon/lat, EPSG:4326)
    kdkec: list of district codes to filter facilities
    """
    if not categories:
        raise HTTPException(status_code=400, detail="No categories provided.")

    where = ["kategori IN :cats"]
    params: Dict[str, Any] = {}
    bindparams_list = [bindparam("cats", expanding=True)]

    if bbox:
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must be [minx,miny,maxx,maxy].")
        minx, miny, maxx, maxy = bbox
        where.append("ST_Intersects(smgeometry, ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326))")
        params.update(dict(minx=minx, miny=miny, maxx=maxx, maxy=maxy))

    if kdkec:
        where.append("kdkec IN :kdkec_codes")
        bindparams_list.append(bindparam("kdkec_codes", expanding=True))

    sql = text(f"""
        SELECT
            smid AS id,
            COALESCE(nama, '') AS nama,
            kategori AS category,
            ST_Y(ST_Transform(smgeometry, 4326)) AS latitude,
            ST_X(ST_Transform(smgeometry, 4326)) AS longitude
        FROM public.poi
        WHERE {' AND '.join(where)}
        ORDER BY smid
        LIMIT :limit
    """).bindparams(*bindparams_list)

    # Build execution parameters
    exec_params = {"cats": categories, "limit": limit, **params}
    if kdkec:
        exec_params["kdkec_codes"] = kdkec

    with engine.connect() as conn:
        rows = conn.execute(sql, exec_params).fetchall()

    return [{
        "id": r._mapping["id"],
        "nama": r._mapping["nama"],
        "category": r._mapping["category"],
        "latitude": r._mapping["latitude"],
        "longitude": r._mapping["longitude"],
    } for r in rows]