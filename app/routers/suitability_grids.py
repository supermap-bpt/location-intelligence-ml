from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

import asyncpg
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely import wkt
import numpy as np

# ---------------------------
# Router
# ---------------------------
router = APIRouter(prefix="/suitability", tags=["suitability"])

# ---------------------------
# Request / Response Models
# ---------------------------

class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Any

    @validator("type")
    def _type_ok(cls, v: str) -> str:
        if v not in {"Polygon", "MultiPolygon"}:
            raise ValueError("geometry_grid.type must be Polygon or MultiPolygon")
        return v


class GridItem(BaseModel):
    geometry_grid: GeoJSONGeometry


class ParamRange(BaseModel):
    min_value: float
    max_value: float

    @validator("max_value")
    def _valid_range(cls, v: float, values: Dict[str, Any]) -> float:
        if "min_value" in values and v < values["min_value"]:
            raise ValueError("max_value must be >= min_value")
        return v


class OptionalParamSpec(BaseModel):
    weight: int = Field(default=1)
    invert: bool = Field(default=False)


class SuitabilityRequest(BaseModel):
    data: List[GridItem]
    mandatory_parameters: Dict[str, ParamRange]
    optional_parameters: Dict[str, OptionalParamSpec] = Field(default_factory=dict)


class GridScore(BaseModel):
    grid_index: int
    centroid: Tuple[float, float]
    region: Dict[str, Any]
    param_scores: Dict[str, float]
    total_score: float
    label: str


class SuitabilityResponse(BaseModel):
    bins: Dict[str, float]  # {"A": min, "B": ..., "C": ..., "D": max}
    results: List[GridScore]


# ---------------------------
# DB Utilities
# ---------------------------

# Configure your Postgres DSN (env, settings, etc.)
# Example: "postgresql://user:pass@localhost:5432/sdx"
PG_DSN = "postgresql://postgres:postgres@localhost:5432/sdx"

async def get_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(PG_DSN, min_size=1, max_size=10)

# ---------------------------
# Core Helpers
# ---------------------------

def geojson_to_polygon(geom: Dict[str, Any]) -> Polygon | MultiPolygon:
    try:
        return shape(geom)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid geometry: {e}")

def within_range(val: Optional[float], rng: ParamRange) -> bool:
    if val is None:
        return False
    return (rng.min_value is None or val >= rng.min_value) and (rng.max_value is None or val <= rng.max_value)

@dataclass
class SelectedRegionRow:
    # raw mandatory values for filtering
    raw_values: Dict[str, Optional[float]]
    # skor values for scoring
    skor_values: Dict[str, Optional[float]]
    # region identity
    region: Dict[str, Any]

def build_select_columns(params_all: List[str]) -> Tuple[str, List[str]]:
    """
    Build SELECT list for dynamic params and skor_ columns, plus region identity columns.
    Returns (select_list_sql, ordered_columns)
    """
    cols: List[str] = []
    ordered: List[str] = []

    # raw param columns
    for p in params_all:
        cols.append(f'pl."{p}" AS "{p}"')
        ordered.append(p)

    # skor_ param columns
    for p in params_all:
        sp = f"skor_{p}"
        cols.append(f'pl."{sp}" AS "{sp}"')
        ordered.append(sp)

    # region / reference columns
    region_cols = [
        'kc.kdkec::text AS "kdkec"',
        'kc.kdkab::text AS "kdkab"',
        'kc.kdprov::text AS "kdprov"',
        'kc.nmprov AS "nmprov"',
        'kc.nmkab AS "nmkab"',
        'kc.nmkec AS "nmkec"',
        'kc.latitude AS "latitude"',
        'kc.longitude AS "longitude"',
        'kc.sumber AS "sumber"',
        'kc.region_code AS "region_code"',
        'kc.periode AS "periode"'
    ]
    cols.extend(region_cols)
    ordered.extend([c.split(' AS "')[-1][:-1] for c in region_cols])  # column aliases

    return ", ".join(cols), ordered

async def fetch_best_row_for_polygon(conn: asyncpg.Connection, wkt_polygon: str, select_list: str) -> Optional[asyncpg.Record]:
    """
    Uses PostGIS to select the kecamatan row that maximally overlaps the polygon.
    """
    sql = f"""
    WITH g AS (
      SELECT ST_SetSRID(ST_GeomFromText($1), 4326) AS geom
    )
    SELECT {select_list}
    FROM public.kecamatan AS kc
    JOIN public.parameter_li AS pl
      ON pl.kdkec::text = kc.kdkec::text
    JOIN g
      ON ST_Intersects(kc.geom, g.geom)
    ORDER BY ST_Area(ST_Intersection(kc.geom, g.geom)) DESC
    LIMIT 1;
    """
    return await conn.fetchrow(sql, wkt_polygon)

def compute_total_score(
    row: SelectedRegionRow,
    mandatory: Dict[str, ParamRange],
    optional: Dict[str, OptionalParamSpec],
) -> Tuple[bool, Dict[str, float], float]:
    """
    Returns:
      - passes_mandatory (bool)
      - per_param_scores (dict)
      - total_score (float)
    """
    # 1) Filter by mandatory raw ranges
    for pname, rng in mandatory.items():
        raw_val = row.raw_values.get(pname)
        if not within_range(raw_val, rng):
            # fails mandatory
            return False, {}, 0.0

    # 2) Scoring
    per_param_scores: Dict[str, float] = {}
    total = 0.0

    # include all mandatory in scoring (weight=1, no invert by default)
    # for pname in mandatory.keys():
    #     sk = row.skor_values.get(f"skor_{pname}")
    #     if sk is None:
    #         continue
    #     v = float(sk)
    #     # mandatory: no invert config supplied, treat as non-inverted
    #     per_param_scores[pname] = v
    #     total += v

    # include optional with weight & invert
    for pname, spec in optional.items():
        sk = row.skor_values.get(f"skor_{pname}")
        if sk is None:
            continue
        v = float(sk)
        if spec.invert:
            v = 4.0 - v  # invert: 1->3, 2->2, 3->1 (assuming 1..3)
        weighted = v * float(spec.weight)
        per_param_scores[pname] = weighted
        total += weighted

    return True, per_param_scores, total

def split_into_three_bins(scores: List[float]) -> Dict[str, float]:
    """
    Return boundaries A, B, C, D such that:
      A-B = low
      B+ε - C = medium
      C+ε - D = high
    If all scores equal, make degenerate bins.
    """
    if not scores:
        return {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    a = float(min(scores))
    d = float(max(scores))
    if np.isclose(a, d):
        return {"A": a, "B": a, "C": a, "D": d}

    width = (d - a) / 3.0
    b = a + width
    c = a + 2.0 * width
    return {"A": a, "B": b, "C": c, "D": d}

def label_from_bins(val: float, bins: Dict[str, float]) -> str:
    A, B, C, D = bins["A"], bins["B"], bins["C"], bins["D"]
    if np.isclose(A, D):
        return "medium"  # degenerate case
    if val <= B:
        return "low"
    elif val <= C:
        return "medium"
    else:
        return "high"

# ---------------------------
# Endpoint
# ---------------------------

@router.post("/score-grids", response_model=SuitabilityResponse)
async def score_grids(req: SuitabilityRequest) -> SuitabilityResponse:
    # Union of all parameter names we want to SELECT from parameter_li
    mandatory_names = list(req.mandatory_parameters.keys())
    optional_names = list(req.optional_parameters.keys())
    all_param_names: List[str] = list(dict.fromkeys(mandatory_names + optional_names))

    if not all_param_names:
        raise HTTPException(status_code=400, detail="No parameters specified.")

    select_list, ordered_cols = build_select_columns(all_param_names)

    # Prepare geometries
    polygons: List[Polygon | MultiPolygon] = []
    wkts: List[str] = []
    for g in req.data:
        poly = geojson_to_polygon(g.geometry_grid.dict())
        polygons.append(poly)
        wkts.append(shape(g.geometry_grid.dict()).wkt)

    pool = await get_pool()
    results: List[GridScore] = []
    totals_for_bins: List[float] = []

    async with pool.acquire() as conn:
        for idx, wkt_poly in enumerate(wkts):
            rec = await fetch_best_row_for_polygon(conn, wkt_poly, select_list)
            # If no intersecting region, mark grid as filtered_out
            if rec is None:
                centroid = polygons[idx].centroid
                results.append(
                    GridScore(
                        grid_index=idx,
                        centroid=(float(centroid.x), float(centroid.y)),
                        region={},
                        param_scores={},
                        total_score=0.0,
                        label="filtered_out",
                    )
                )
                continue

            # Build row object
            raw_values: Dict[str, Optional[float]] = {}
            skor_values: Dict[str, Optional[float]] = {}
            region: Dict[str, Any] = {}

            for p in all_param_names:
                raw_values[p] = rec.get(p)

            for p in all_param_names:
                sp = f"skor_{p}"
                skor_values[sp] = rec.get(sp)

            region_keys = [
                "kdkec", "kdkab", "kdprov", "nmprov", "nmkab",
                "nmkec", "latitude", "longitude", "sumber",
                "region_code", "periode"
            ]
            for k in region_keys:
                region[k] = rec.get(k)

            sel = SelectedRegionRow(raw_values=raw_values, skor_values=skor_values, region=region)

            passes, per_param_scores, total = compute_total_score(
                sel,
                req.mandatory_parameters,
                req.optional_parameters,
            )

            # label later once we know bins; store now
            centroid = polygons[idx].centroid
            grid_result = GridScore(
                grid_index=idx,
                centroid=(float(centroid.x), float(centroid.y)),
                region=region,
                param_scores=per_param_scores if passes else {},
                total_score=float(total if passes else 0.0),
                label="filtered_out" if not passes else "UNASSIGNED",
            )
            results.append(grid_result)
            if passes:
                totals_for_bins.append(grid_result.total_score)

    # Compute bins on passing totals
    bins = split_into_three_bins(totals_for_bins)

    # Assign labels
    for r in results:
        if r.label == "filtered_out":
            continue
        r.label = label_from_bins(r.total_score, bins)

    return SuitabilityResponse(bins=bins, results=results)