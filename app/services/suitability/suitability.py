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
                grid_scores.append({
                    "geometry": grid_item.geometry_grid,
                    "grid_value": 0.0,
                    "grid_value_raw": grid_value_raw,
                    "feature_scores": feature_scores,
                    "weights_applied": weights,
                    "gdp": gdp_value,
                    "predicted_class": "low",
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
                    "predicted_class": None,
                    "forced_by_gdp": False
                })

        # --- Thresholds: fair thirds with rounding and +0.01 gap ---
        valid_values = [gs["grid_value"] for gs in grid_scores if not gs["forced_by_gdp"]]

        if len(valid_values) >= 1:
            vmin = float(min(valid_values))
            vmax = float(max(valid_values))
            spread = vmax - vmin

            if spread <= 1e-12:
                thresholds = {"high": [round(vmin, 2), round(vmax, 2)]}
            else:
                c1 = vmin + spread / 3.0
                c2 = vmin + 2.0 * spread / 3.0

                # Round values
                vmin = round(vmin, 2)
                c1   = round(c1, 2)
                c2   = round(c2, 2)
                vmax = round(vmax, 2)

                # Apply +0.01 step between categories
                thresholds = {
                    "low":    [vmin, c1],
                    "medium": [c1 + 0.01, c2],
                    "high":   [c2 + 0.01, vmax],
                }
        else:
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
                "feature_scores": gs["feature_scores"],
                "weights_applied": gs["weights_applied"],
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