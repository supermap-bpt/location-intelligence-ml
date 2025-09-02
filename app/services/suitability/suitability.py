import numpy as np
from fastapi import HTTPException
from shapely.geometry import shape
from app.models.requests import BatchRequest
from app.services.suitability.intersect import get_intersect_value
from app.services.suitability import loaders


async def batch_predict_service(request: BatchRequest):
    try:
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
            "gdp": loaders.get_gdp_geodataframe(),
        }

        grid_scores = []
        # --- Step 1: GDP check ---
        for grid_item in request.data:
            polygon = shape(grid_item.geometry_grid)
            gdp_value = get_intersect_value(gdfs["gdp"], polygon, 'pendapatan') if gdfs["gdp"] is not None else None

            if gdp_value is None or not (request.low_range <= gdp_value <= request.high_range):
                grid_scores.append({
                    "geometry": grid_item.geometry_grid,
                    "grid_value": 0.0,
                    "gdp": gdp_value,
                    "category": "low"
                })
                continue

            # --- Step 2: feature scoring ---
            weights = grid_item.weights
            feature_scores = {
                "jumlahsiswaputussekolah": get_intersect_value(gdfs["siswa"], polygon, 's_siswaputussekolah') if gdfs["siswa"] is not None else 0.0,
                "kemiskinan": get_intersect_value(gdfs["kemiskinan"], polygon, 's_kemiskinan') if gdfs["kemiskinan"] is not None else 0.0,
                "peopleden": get_intersect_value(gdfs["penduduk"], polygon, 's_pddk') if gdfs["penduduk"] is not None else 0.0,
                "poiarea": get_intersect_value(gdfs["poi"], polygon, 's_poi') if gdfs["poi"] is not None else 0.0,
                "nearest_sungai": get_intersect_value(gdfs["sungai"], polygon, 's_sungai') if gdfs["sungai"] is not None else 0.0,
                "nearestfaskes": get_intersect_value(gdfs["faskes"], polygon, 's_faskes') if gdfs["faskes"] is not None else 0.0,
                "road": get_intersect_value(gdfs["road"], polygon, 's_road') if gdfs["road"] is not None else 0.0,
                "slope": get_intersect_value(gdfs["slope"], polygon, 's_slope') if gdfs["slope"] is not None else 0.0,
            }

            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 100.0, atol=1e-6):
                raise HTTPException(status_code=400, detail=f"Weights must sum to 100. Got {total_weight}.")

            grid_value = sum(feature_scores.get(k, 0.0) * (weights.get(k, 0.0) / 100.0) for k in weights)

            grid_scores.append({
                "geometry": grid_item.geometry_grid,
                "grid_value": grid_value,
                "gdp": gdp_value,
                "category": None
            })

        # --- Step 3: thresholds ---
        unique_scores = sorted(set(gs["grid_value"] for gs in grid_scores if gs["category"] is None))
        thresholds = {}
        if len(unique_scores) == 1:
            thresholds = {"high": [unique_scores[0], 1e10]}
        elif len(unique_scores) == 2:
            thresholds = {"medium": [unique_scores[0], unique_scores[0]], "high": [unique_scores[0] + 1, unique_scores[1]]}
        elif len(unique_scores) == 3:
            thresholds = {"low": [unique_scores[0], unique_scores[0]], "medium": [unique_scores[1], unique_scores[1]], "high": [unique_scores[1] + 1, unique_scores[2]]}
        elif len(unique_scores) >= 4:
            thresholds = {"low": [unique_scores[0], unique_scores[1]], "medium": [unique_scores[1], unique_scores[2]], "high": [unique_scores[2] + 1, unique_scores[3]]}

        # --- Step 4: assign categories ---
        results = []
        for gs in grid_scores:
            if gs["category"] == "low":
                results.append({"geometry_grid": gs["geometry"], "grid_value": gs["grid_value"], "category": "low"})
                continue

            val = gs["grid_value"]
            if "low" in thresholds and thresholds["low"][0] <= val <= thresholds["low"][1]:
                category = "low"
            elif "medium" in thresholds and thresholds["medium"][0] <= val <= thresholds["medium"][1]:
                category = "medium"
            elif "high" in thresholds and val >= thresholds["high"][0]:
                category = "high"
            else:
                category = "medium"

            results.append({"geometry_grid": gs["geometry"], "grid_value": gs["grid_value"], "category": category})

        return {"data": results, "thresholds": thresholds, "low_range_gdp": request.low_range, "high_range_gdp": request.high_range}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
