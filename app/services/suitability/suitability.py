import numpy as np
from fastapi import HTTPException
from shapely.geometry import shape
from app.models.responses import SuitabilityCategory
from app.models.requests import BatchRequest
from app.services.suitability.intersect import get_intersect_value
from app.services.suitability import loaders


async def batch_predict_service(request: BatchRequest):
    try:
        # figure out which layers we actually need
        all_weights = {k for grid in request.data for k, w in grid.weights.items() if w > 0}

        gdfs = {
            "siswa": loaders.get_siswa_putus_sekolah_geodataframe() if "jumlahsiswaputussekolah" in all_weights else None,
            "kemiskinan": loaders.get_kemiskinan_geodataframe() if "kemiskinan" in all_weights else None,
            "penduduk": loaders.get_kepadatan_penduduk_geodataframe() if "peopleden" in all_weights else None,
            "poi": loaders.get_poi_geodataframe() if "poiarea" in all_weights else None,
            "sungai": loaders.get_kedekatan_sungai_geodataframe() if "nearest_sungai" in all_weights else None,
            "faskes": loaders.get_kedekatan_faskes_geodataframe() if "nearestfaskes" in all_weights else None,
            "road": loaders.get_kedekatan_jalan_geodataframe() if "road" in all_weights else None,
            "slope": loaders.get_slope_geodataframe() if "slope" in all_weights else None,
            "gdp": loaders.get_gdp_geodataframe(),  # global
        }

        grid_scores = []

        for grid_item in request.data:
            polygon = shape(grid_item.geometry_grid)

            # --- Per-polygon feature scoring (ALWAYS) ---
            weights = dict(grid_item.weights or {})
            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 100.0, atol=1e-6):
                raise HTTPException(status_code=400, detail=f"Weights must sum to 100. Got {total_weight}.")

            feature_scores = {
                "jumlahsiswaputussekolah": get_intersect_value(gdfs["siswa"], polygon, 's_siswaputussekolah') if gdfs["siswa"] is not None else 0.0,
                "kemiskinan":              get_intersect_value(gdfs["kemiskinan"], polygon, 's_kemiskinan')     if gdfs["kemiskinan"] is not None else 0.0,
                "peopleden":               get_intersect_value(gdfs["penduduk"], polygon, 's_pddk')             if gdfs["penduduk"] is not None else 0.0,
                "poiarea":                 get_intersect_value(gdfs["poi"], polygon, 's_poi')                   if gdfs["poi"] is not None else 0.0,
                "nearest_sungai":          get_intersect_value(gdfs["sungai"], polygon, 's_sungai')             if gdfs["sungai"] is not None else 0.0,
                "nearestfaskes":           get_intersect_value(gdfs["faskes"], polygon, 's_faskes')             if gdfs["faskes"] is not None else 0.0,
                "road":                    get_intersect_value(gdfs["road"], polygon, 's_road')                 if gdfs["road"] is not None else 0.0,
                "slope":                   get_intersect_value(gdfs["slope"], polygon, 's_slope')               if gdfs["slope"] is not None else 0.0,
            }

            grid_value_raw = sum(feature_scores.get(k, 0.0) * (weights.get(k, 0.0) / 100.0) for k in weights)

            # --- Global GDP gate ---
            gdp_value = get_intersect_value(gdfs["gdp"], polygon, 'pendapatan') if gdfs["gdp"] is not None else None
            gdp_in_range = (gdp_value is not None) and (request.low_range <= gdp_value <= request.high_range)

            if not gdp_in_range:
                # Forced low by GDP (but keep feature_scores & weights for every polygon)
                grid_scores.append({
                    "geometry": grid_item.geometry_grid,
                    "grid_value": 0.0,
                    "grid_value_raw": grid_value_raw,
                    "feature_scores": feature_scores,
                    "weights_applied": weights,
                    "gdp": gdp_value,
                    "predicted_class": "low",         # mark as low immediately
                    "forced_by_gdp": True
                })
            else:
                grid_scores.append({
                    "geometry": grid_item.geometry_grid,
                    "grid_value": grid_value_raw,
                    "grid_value_raw": grid_value_raw,
                    "feature_scores": feature_scores,
                    "weights_applied": weights,
                    "gdp": gdp_value,
                    "predicted_class": None,          # to be assigned by thresholds
                    "forced_by_gdp": False
                })

        # --- Thresholds (quantile-based) only on GDP-valid items ---
        valid_values = [gs["grid_value"] for gs in grid_scores if not gs["forced_by_gdp"]]
        thresholds = {}

        if len(valid_values) >= 3:
            q33, q66 = np.percentile(valid_values, [33.33, 66.67])
            # inclusive on lower bounds to avoid gaps
            thresholds = {
                "low":    [float(min(valid_values)), float(q33)],
                "medium": [float(q33), float(q66)],
                "high":   [float(q66), float(max(valid_values))]
            }
        elif len(valid_values) == 2:
            lo, hi = sorted(valid_values)
            thresholds = {
                "medium": [float(lo), float(lo)],
                "high":   [float(lo), float(hi)]
            }
        elif len(valid_values) == 1:
            v = float(valid_values[0])
            thresholds = {"high": [v, v]}
        else:
            # no valid (GDP-passing) values; keep everything as low
            thresholds = {"low": [0.0, 0.0]}

        # --- Assign categories (keep GDP-forced low as low) ---
        results = []
        for gs in grid_scores:
            if gs["predicted_class"] == "low" and gs["forced_by_gdp"]:
                category = SuitabilityCategory.NOT_RECOMMENDED
            else:
                val = gs["grid_value"]
                if "low" in thresholds and thresholds["low"][0] <= val <= thresholds["low"][1]:
                    category = SuitabilityCategory.NOT_RECOMMENDED
                elif "medium" in thresholds and thresholds["medium"][0] <= val <= thresholds["medium"][1]:
                    category = SuitabilityCategory.NEUTRAL
                elif "high" in thresholds and val >= thresholds["high"][0]:
                    category = SuitabilityCategory.RECOMMENDED
                else:
                    category = SuitabilityCategory.NEUTRAL

            results.append({
                "predicted_class": category,
                "geometry_grid": gs["geometry"],
                "grid_value": gs["grid_value"],
                "feature_scores": gs["feature_scores"],     # always present now
                "weights_applied": gs["weights_applied"],   # always present now
                "gdp": gs["gdp"]
            })

        return {
            "data": results,
            "thresholds": thresholds,
            "low_range_gdp": request.low_range,
            "high_range_gdp": request.high_range
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")