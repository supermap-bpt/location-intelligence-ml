from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg

# Import your existing request models
from app.models.requests import (
    BatchRequest,
    SuitabilityRequest,
    GridData,
    RangeSpec,
    ConsideredParamSpec,
)

# Import response models
from app.models.responses import (
    SuitabilityAnalysisResponse,
    GridGeometryResponse,
    ThresholdsResponse,
    MandatoryParameterResponse,
    OptionalParameterResponse,
)

# -------- connection pool (self-managed so the service works without Depends) --------

_pool: Optional[asyncpg.Pool] = None

async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL env var is required for suitability service.")
        _pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
    return _pool

# --------------------------------- helpers ---------------------------------

@dataclass(frozen=True)
class SelectedRow:
    prov: str
    kab: str
    kec: str
    raw_vals: Dict[str, Optional[float]]
    skor_vals: Dict[str, Optional[float]]

def _equal_width_thresholds(values: List[float]) -> Tuple[float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return (vmin, vmin, vmin, vmax)
    w = (vmax - vmin) / 3.0
    return (vmin, vmin + w, vmin + 2 * w, vmax)

def _within_range(val: Optional[float], spec: RangeSpec) -> bool:
    return (val is not None) and (spec.min_value <= val <= spec.max_value)

def _adjust_optional_skor(skor: Optional[float], invert: bool) -> Optional[float]:
    if skor is None:
        return None
    return (4 - skor) if invert else skor  # 1<->3, 2 stays 2

def _class_from_thresholds(value: float, A: float, B: float, C: float, D: float) -> str:
    if math.isclose(A, B) and math.isclose(B, C) and math.isclose(C, D):
        return "low"
    if value <= B:
        return "low"
    elif value <= C:
        return "medium"
    return "high"

def _geojson_str(gj: Dict[str, Any]) -> str:
    return json.dumps(gj, separators=(",", ":"))

async def _fetch_best_match_row(
    pool: asyncpg.Pool,
    grid_geojson: Dict[str, Any],
    param_names_all: List[str],
) -> Optional[SelectedRow]:
    raw_cols = ", ".join([f"pl.{p}" for p in param_names_all])
    skor_cols = ", ".join([f"pl.skor_{p}" for p in param_names_all])
    geojson = _geojson_str(grid_geojson)

    sql = f"""
        WITH g AS (
          SELECT ST_SetSRID(ST_GeomFromGeoJSON($1), 4326) AS geom
        )
        SELECT
          kc.nmprov, kc.nmkab, kc.nmkec,
          {raw_cols},
          {skor_cols}
        FROM public.kecamatan AS kc
        JOIN public.parameter_li AS pl
          ON pl.kdkec::text = kc.kdkec::text
        JOIN g ON TRUE
        WHERE ST_Intersects(kc.geom, g.geom)
        ORDER BY ST_Area(ST_Intersection(kc.geom, g.geom)) DESC
        LIMIT 1;
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, geojson)
        if not row:
            return None

        raw_vals: Dict[str, Optional[float]] = {}
        skor_vals: Dict[str, Optional[float]] = {}
        for p in param_names_all:
            raw_vals[p] = row.get(p)
            s_key = f"skor_{p}"
            skor_vals[p] = row.get(s_key) if s_key in row else None

        return SelectedRow(
            prov=row["nmprov"],
            kab=row["nmkab"],
            kec=row["nmkec"],
            raw_vals=raw_vals,
            skor_vals=skor_vals,
        )

# ---------------------------- core suitability flow ----------------------------

async def _run_suitability(request: SuitabilityRequest) -> SuitabilityAnalysisResponse:
    mandatory_names = list(request.mandatory_parameters.keys())
    optional_names = list(request.optional_parameters.keys())
    param_names_all = mandatory_names + optional_names

    pool = await _get_pool()

    results: List[GridGeometryResponse] = []
    composites_valid: List[float] = []

    for idx, grid in enumerate(request.data, start=1):
        row = await _fetch_best_match_row(pool, grid.geometry_grid, param_names_all)

        if row is None:
            results.append(GridGeometryResponse(
                id=str(idx),
                predicted_class="excluded",
                grid_value=0.0,
                geometry=grid.geometry_grid,
                feature_scores={},
                provinsi=None,
                kabupaten=None,
                kecamatan=[],
            ))
            continue

        # 2) mandatory gate
        mandatory_pass = True
        feature_scores: Dict[str, float] = {}
        for m in mandatory_names:
            spec = request.mandatory_parameters[m]
            val = row.raw_vals.get(m)
            feature_scores[m] = float(val) if val is not None else 0.0
            if not _within_range(val, spec):
                mandatory_pass = False

        if not mandatory_pass:
            results.append(GridGeometryResponse(
                id=str(idx),
                predicted_class="low",
                grid_value=0.0,
                geometry=grid.geometry_grid,
                feature_scores=feature_scores,
                provinsi=row.prov,
                kabupaten=row.kab,
                kecamatan=[row.kec],
            ))
            continue

        # 3) compute scores
        mand_skors: List[float] = []
        for m in mandatory_names:
            s = row.skor_vals.get(m)
            s = 0 if s is None else float(s)
            mand_skors.append(s)
            feature_scores[f"skor_{m}"] = s
        mandatory_avg = (sum(mand_skors) / len(mand_skors)) if mand_skors else 0.0

        # Count how many optional parameters have values
        num_applied_optional = 0
        for o in optional_names:
            s = row.skor_vals.get(o)
            adj = _adjust_optional_skor(s, request.optional_parameters[o].invert)
            if adj is not None:
                num_applied_optional += 1

        # Calculate weighted sum with normalized weights
        weighted_sum = 0.0
        weight_total = 0.0
        for o in optional_names:
            spec = request.optional_parameters[o]
            s = row.skor_vals.get(o)
            adj = _adjust_optional_skor(s, spec.invert)
            if adj is not None:
                # Divide weight by number of applied parameters
                w = float(spec.weight) / num_applied_optional if num_applied_optional > 0 else 0.0
                weighted_sum += float(adj) * w
                weight_total += w
                feature_scores[f"skor_{o}"] = float(s) if s is not None else 0.0
                feature_scores[f"adj_skor_{o}"] = float(adj)

        optional_weighted_avg = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        composite = optional_weighted_avg

        composites_valid.append(composite)

        results.append(GridGeometryResponse(
            id=str(idx),
            predicted_class="",  # fill later
            grid_value=float(composite),
            geometry=grid.geometry_grid,
            feature_scores=feature_scores,
            provinsi=row.prov,
            kabupaten=row.kab,
            kecamatan=[row.kec],
        ))

    # 4) thresholds from valid composites
    valid_values = [r.grid_value for r in results if r.predicted_class == ""]
    A, B, C, D = _equal_width_thresholds(valid_values)

    # 5) classes
    for r in results:
        if r.predicted_class == "excluded":
            continue
        if r.predicted_class == "":
            r.predicted_class = _class_from_thresholds(r.grid_value, A, B, C, D)

    # Convert request parameters to response models
    mandatory_params_dict = {
        k: MandatoryParameterResponse(min_value=v.min_value, max_value=v.max_value)
        for k, v in request.mandatory_parameters.items()
    }
    optional_params_dict = {
        k: OptionalParameterResponse(weight=v.weight, invert=v.invert)
        for k, v in request.optional_parameters.items()
    }

    return SuitabilityAnalysisResponse(
        thresholds=ThresholdsResponse(
            composite=[A, B, C, D],
            explanation="A=min, D=max; B and C split into equal-width bins (low/medium/high).",
        ),
        grid_geometries=results,
        mandatory_parameters=mandatory_params_dict,
        optional_parameters=optional_params_dict,
    )

# ------------------------------ public service ------------------------------

async def batch_predict_service(
    request: Union[SuitabilityRequest, BatchRequest]
) -> Union[SuitabilityAnalysisResponse, Dict[str, Any]]:
    """
    Keeps your router import working.
    - If request is SuitabilityRequest: run the new suitability flow.
    - If request is BatchRequest: return a deprecation notice (so the endpoint doesn't crash).
    """
    # Detect by attribute presence to be robust even if models are proxied
    if hasattr(request, "mandatory_parameters") and hasattr(request, "optional_parameters"):
        # New flow
        return await _run_suitability(request)  # type: ignore[arg-type]

    # Legacy BatchRequest path
    # You can implement your old GDP-based logic here if needed.
    return {
        "status": "deprecated",
        "detail": "The /batch-predict (BatchRequest) endpoint is deprecated. "
                  "Please call /recommendations with SuitabilityRequest.",
        "grid_geometries": [],
        "thresholds": {"composite": [0, 0, 0, 0]},
    }