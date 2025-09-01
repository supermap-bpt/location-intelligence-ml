from datetime import datetime
from app.models.responses import SuitabilityResponse, SuitabilityCategory
from app.models.requests import BatchRequest, GridData
from app.config import METRICS
import numpy as np
from shapely.geometry import shape
from app.services.data_loader import (
    get_siswa_putus_sekolah_geodataframe,
    get_kemiskinan_geodataframe,
    get_kepadatan_penduduk_geodataframe,
    get_poi_geodataframe,
    get_gdp_geodataframe,
    get_kedekatan_sungai_geodataframe,
    get_kedekatan_faskes_geodataframe,
    get_kedekatan_jalan_geodataframe,
    get_slope_geodataframe
)

def get_intersect_value(gdf, polygon, score_col):
    """
    Helper function to calculate intersection value between GeoDataFrame and polygon
    """
    if gdf is None or gdf.empty:
        return 0
    gdf = gdf.to_crs("EPSG:4326")
    gdf['intersection'] = gdf.geometry.intersection(polygon)
    gdf = gdf[gdf['intersection'].area > 0]
    if gdf.empty:
        return 0
    gdf['intersection_area'] = gdf['intersection'].area
    max_idx = gdf['intersection_area'].idxmax()
    return float(gdf.loc[max_idx, score_col])

def batch_predict_service(request: BatchRequest):
    """
    Batch suitability prediction service based on GDP filtering and weighted feature scoring
    """
    try:
        # --- Step 0: preload GeoDataFrames once (only features with weight > 0) ---
        all_weights = {k for grid in request.data for k, w in grid.weights.items() if w > 0}

        # Load required GeoDataFrames
        gdfs = {
            "siswa": get_siswa_putus_sekolah_geodataframe() if "jumlahsiswaputussekolah" in all_weights else None,
            "kemiskinan": get_kemiskinan_geodataframe() if "kemiskinan" in all_weights else None,
            "penduduk": get_kepadatan_penduduk_geodataframe() if "peopleden" in all_weights else None,
            "poi": get_poi_geodataframe() if "poiarea" in all_weights else None,
            "sungai": get_kedekatan_sungai_geodataframe() if "nearest_sungai" in all_weights else None,
            "faskes": get_kedekatan_faskes_geodataframe() if "nearestfaskes" in all_weights else None,
            "road": get_kedekatan_jalan_geodataframe() if "road" in all_weights else None,
            "slope": get_slope_geodataframe() if "slope" in all_weights else None,
            "gdp": get_gdp_geodataframe(),
        }

        grid_scores = []

        # --- Step 1: calculate GDP first ---
        for grid_item in request.data:
            polygon = shape(grid_item.geometry_grid)

            gdp_value = get_intersect_value(gdfs["gdp"], polygon, 'pendapatan') if gdfs["gdp"] is not None else None

            # If GDP is not valid / outside range -> directly categorize as low
            if gdp_value is None or not (request.low_range <= gdp_value <= request.high_range):
                grid_scores.append({
                    "geometry": grid_item.geometry_grid,
                    "grid_value": 0.0,   # don't need to calculate
                    "gdp": gdp_value,
                    "category": "low"
                })
                continue

            # --- Step 2: calculate other features only if GDP passes filter ---
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

            # Validate weight sum
            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 100.0, atol=1e-6):
                raise ValueError(f"Weights must sum to 100. Got {total_weight}.")

            # Weighted sum = grid_value
            grid_value = sum(feature_scores.get(k, 0.0) * (weights.get(k, 0.0) / 100.0) for k in weights)

            grid_scores.append({
                "geometry": grid_item.geometry_grid,
                "grid_value": grid_value,
                "gdp": gdp_value,
                "category": None  # classify later
            })

        # --- Step 3: determine thresholds (only for grids with values) ---
        unique_scores = sorted(set(gs["grid_value"] for gs in grid_scores if gs["category"] is None))
        n_unique = len(unique_scores)

        thresholds = {}
        if n_unique == 1:
            thresholds = {"high": [unique_scores[0], 1e10]}  # Use large finite number instead of inf
        elif n_unique == 2:
            thresholds = {
                "medium": [unique_scores[0], unique_scores[0]],
                "high": [unique_scores[0] + 1, unique_scores[1]]
            }
        elif n_unique == 3:
            thresholds = {
                "low": [unique_scores[0], unique_scores[0]],
                "medium": [unique_scores[1], unique_scores[1]],
                "high": [unique_scores[1] + 1, unique_scores[2]]
            }
        elif n_unique >= 4:
            thresholds = {
                "low": [unique_scores[0], unique_scores[1]],
                "medium": [unique_scores[1], unique_scores[2]],
                "high": [unique_scores[2] + 1, unique_scores[3]]
            }

        # --- Step 4: assign category for grids that passed GDP filter ---
        results = []
        for gs in grid_scores:
            if gs["category"] == "low":  # already marked as GDP failed
                results.append(SuitabilityResponse(
                    predicted_class=SuitabilityCategory.NOT_RECOMMENDED,
                    confidence=0.0,
                    mean_absolute_error=METRICS["MAE"],
                    mean_squared_error=METRICS["MSE"],
                    root_mean_squared_error=METRICS["RMSE"],
                    r2_score=METRICS["R2"],
                    feature_scores={},
                    weights_applied={},
                    input_polygon=gs["geometry"].get("coordinates", []),
                    timestamp=datetime.now().isoformat(),
                    grid_id=None
                ))
                continue

            val = gs["grid_value"]
            category = None

            if "low" in thresholds and thresholds["low"][0] <= val <= thresholds["low"][1]:
                category = SuitabilityCategory.NOT_RECOMMENDED
            elif "medium" in thresholds and thresholds["medium"][0] <= val <= thresholds["medium"][1]:
                category = SuitabilityCategory.NEUTRAL
            elif "high" in thresholds and val >= thresholds["high"][0]:
                category = SuitabilityCategory.RECOMMENDED

            # Fallback to ensure no null category
            if category is None:
                category = SuitabilityCategory.NEUTRAL

            # Create feature scores dictionary for this grid
            grid_feature_scores = {
                "jumlahsiswaputussekolah": get_intersect_value(gdfs["siswa"], shape(gs["geometry"]), 's_siswaputussekolah') if gdfs["siswa"] is not None else 0.0,
                "kemiskinan": get_intersect_value(gdfs["kemiskinan"], shape(gs["geometry"]), 's_kemiskinan') if gdfs["kemiskinan"] is not None else 0.0,
                "peopleden": get_intersect_value(gdfs["penduduk"], shape(gs["geometry"]), 's_pddk') if gdfs["penduduk"] is not None else 0.0,
                "poiarea": get_intersect_value(gdfs["poi"], shape(gs["geometry"]), 's_poi') if gdfs["poi"] is not None else 0.0,
                "nearest_sungai": get_intersect_value(gdfs["sungai"], shape(gs["geometry"]), 's_sungai') if gdfs["sungai"] is not None else 0.0,
                "nearestfaskes": get_intersect_value(gdfs["faskes"], shape(gs["geometry"]), 's_faskes') if gdfs["faskes"] is not None else 0.0,
                "road": get_intersect_value(gdfs["road"], shape(gs["geometry"]), 's_road') if gdfs["road"] is not None else 0.0,
                "slope": get_intersect_value(gdfs["slope"], shape(gs["geometry"]), 's_slope') if gdfs["slope"] is not None else 0.0,
            }

            # Find the original grid item to get weights
            original_item = next((item for item in request.data 
                                if item.geometry_grid == gs["geometry"]), None)
            
            results.append(SuitabilityResponse(
                predicted_class=category,
                confidence=min(1.0, val / 10.0),  # Simple confidence calculation
                mean_absolute_error=METRICS["MAE"],
                mean_squared_error=METRICS["MSE"],
                root_mean_squared_error=METRICS["RMSE"],
                r2_score=METRICS["R2"],
                feature_scores=grid_feature_scores,
                weights_applied=original_item.weights if original_item else {},
                input_polygon=gs["geometry"].get("coordinates", []),
                timestamp=datetime.now().isoformat(),
                grid_id=None
            ))

        return {
            "data": [result.dict() for result in results],
            "thresholds": thresholds,
            "low_range": request.low_range,
            "high_range": request.high_range
        }

    except Exception as e:
        raise ValueError(f"Unexpected error in batch prediction: {str(e)}")