import numpy as np
from fastapi import HTTPException
from shapely.geometry import shape
from app.models.responses import SuitabilityCategory
from app.models.requests import BatchRequest
from app.services.suitability.intersect import get_intersect_value
from app.services.suitability.loaders import load_layer, all_layer_specs

LEGACY_ALIASES = {
    "jumlahsiswaputussekolah": "siswa_putus_sekolah",
    "peopleden":               "kepadatan_penduduk",
    "poiarea":                 "poi",
    "nearestfaskes":           "kedekatan_faskes",
    "nearest_sungai":          "kedekatan_sungai",
    "road":                    "jalan",
    # add more if you had other legacy names
}

def _normalize_weights(weights: dict, specs: dict) -> dict:
    norm = {}
    for k, v in (weights or {}).items():
        key = LEGACY_ALIASES.get(k, k)
        if key not in specs:
            raise HTTPException(status_code=400, detail=f"Unknown feature code in weights: '{k}'")
        norm[key] = v
    return norm

def _pick_gate_code(specs: dict) -> str:
    """
    Prefer an explicit flag (is_required=True) for the gate (e.g., pendapatan_per_kapita_R).
    If multiple are marked required, pick the first numeric one.
    Fallback: name heuristic for 'pendapatan_per_kapita'.
    """
    required_numeric = [c for c, s in specs.items() if s["is_required"] and s["data_type"] in {"numeric", "double precision", "real", "integer", "bigint", "smallint"}]
    if required_numeric:
        return required_numeric[0]
    for code in specs.keys():
        if "pendapatan_per_kapita" in code.lower():
            return code
    raise HTTPException(status_code=500, detail="Cannot locate GDP gate parameter (mark it is_required=true or name it pendapatan_per_kapita*).")

async def batch_predict_service(request: BatchRequest):
    try:
        specs = all_layer_specs()   # { code: {table, value_col, data_type, unit, is_required, min_value, max_value}, ... }
        gate_code = _pick_gate_code(specs)

        # Validate that all weighted features exist & are numeric
        weights = _normalize_weights(request.weights, specs)
        for code in weights.keys():
            if code not in specs:
                raise HTTPException(status_code=400, detail=f"Unknown feature code in weights: '{code}'")
            if specs[code]["data_type"] not in {"numeric", "double precision", "real", "integer", "bigint", "smallint"}:
                raise HTTPException(status_code=400, detail=f"Feature '{code}' is not numeric (data_type={specs[code]['data_type']}).")

        # Load needed layers once (features with nonzero weights + gate)
        needed = {c for c, w in weights.items() if w > 0}
        needed.add(gate_code)
        gdfs = {code: (load_layer(code) if code in needed else None) for code in specs.keys()}

        # Gate range: prefer request.mandatory_parameters, else parameter.min_value/max_value
        mandatory = request.mandatory_parameters.get("pendapatan_per_kapita", {}) or {}
        low_range = mandatory.get("min_value", specs[gate_code]["min_value"] or 0)
        high_range = mandatory.get("max_value", specs[gate_code]["max_value"] or 0)

        # Feature codes exclude the gate
        feature_codes = [c for c in specs.keys() if c != gate_code]

        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 100.0, atol=1e-6):
            raise HTTPException(status_code=400, detail=f"Weights must sum to 100. Got {total_weight}.")

        grid_scores = []
        for grid_item in request.data:
            polygon = shape(grid_item.geometry_grid)

            # compute per-feature dynamically
            feature_scores = {}
            for code in feature_codes:
                if code not in needed:
                    # not weighted → just report 0 (or skip; keeping 0 is clearer)
                    feature_scores[code] = 0.0
                    continue
                gdf = gdfs.get(code)
                val_col = specs[code]["value_col"]
                feature_scores[code] = get_intersect_value(gdf, polygon, val_col) if gdf is not None else 0.0

            grid_value_raw = sum(feature_scores.get(k, 0.0) * (weights.get(k, 0.0) / 100.0) for k in feature_scores)

            # GDP gate
            gdp_val = get_intersect_value(gdfs[gate_code], polygon, specs[gate_code]["value_col"]) if gdfs[gate_code] is not None else None
            gdp_ok = (gdp_val is not None) and (low_range <= gdp_val <= high_range)

            grid_scores.append({
                "geometry": grid_item.geometry_grid,
                "grid_value": grid_value_raw if gdp_ok else 0.0,
                "grid_value_raw": grid_value_raw,
                "feature_scores": feature_scores,
                "weights_applied": weights,
                "gdp": gdp_val,
                "predicted_class": None if gdp_ok else "low",
                "forced_by_gdp": not gdp_ok,
            })

        # thresholds (same logic as before)
        valid = [gs["grid_value"] for gs in grid_scores if not gs["forced_by_gdp"]]
        if valid:
            vmin, vmax = float(min(valid)), float(max(valid))
            spread = vmax - vmin
            if spread <= 1e-12:
                thresholds = {"high": [round(vmin, 2), round(vmax, 2)]}
            else:
                c1 = round(vmin + spread/3, 2)
                c2 = round(vmin + 2*spread/3, 2)
                thresholds = {"low": [round(vmin, 2), c1], "medium": [c1 + 0.01, c2], "high": [c2 + 0.01, round(vmax, 2)]}
        else:
            thresholds = {"low": [0.0, 0.0]}

        # categorize
        results = []
        for gs in grid_scores:
            if gs["forced_by_gdp"]:
                category = SuitabilityCategory.NOT_RECOMMENDED
            else:
                val = gs["grid_value"]
                if thresholds["low"][0] <= val <= thresholds["low"][1]:
                    category = SuitabilityCategory.NOT_RECOMMENDED
                elif "medium" in thresholds and thresholds["medium"][0] <= val <= thresholds["medium"][1]:
                    category = SuitabilityCategory.NEUTRAL
                else:
                    category = SuitabilityCategory.RECOMMENDED

            results.append({
                "predicted_class": category,
                "geometry_grid": gs["geometry"],
                "grid_value": gs["grid_value"],
                "feature_scores": gs["feature_scores"],
                "weights_applied": gs["weights_applied"],
                "gdp": gs["gdp"],
            })

        return {"data": results, "thresholds": thresholds, "low_range_gdp": low_range, "high_range_gdp": high_range}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")