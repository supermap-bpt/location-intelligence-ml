from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple
from shapely.geometry import shape
import geopandas as gpd
from datetime import datetime

app = FastAPI()

# Load the trained classifier model
model = joblib.load("model/suitability_classifier.pkl")

# Hardcoded model accuracy (computed offline on test/validation set)
MODEL_ACCURACY = 0.92

# Category enum matching model output: 0, 1, 2
class SuitabilityCategory(str, Enum):
    NOT_RECOMMENDED = "not recommended"
    NEUTRAL = "neutral"
    RECOMMENDED = "recommended"

# Request schema
class SuitabilityRequest(BaseModel):
    geometry_grid: Dict
    feature_scores: Dict[str, float]
    weights: Dict[str, float]

# Response schema
class SuitabilityResponse(BaseModel):
    predicted_class: SuitabilityCategory
    confidence: float
    model_accuracy: float
    feature_scores: Dict[str, float]
    weights_applied: Dict[str, float]
    input_polygon: List[List[Tuple[float, float]]]
    timestamp: str

# Health check schema
class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    dependencies: dict

# Class mapping
label_mapping = {
    0: SuitabilityCategory.NOT_RECOMMENDED,
    1: SuitabilityCategory.NEUTRAL,
    2: SuitabilityCategory.RECOMMENDED
}

# Extract polygon coordinates
def extract_coordinates(geojson: Dict) -> List[List[Tuple[float, float]]]:
    if geojson["type"] == "Polygon":
        return geojson["coordinates"]
    raise ValueError("Only Polygon geometry type is supported")

@app.post("/predict", response_model=SuitabilityResponse)
async def predict_suitability(request: SuitabilityRequest):
    try:
        coords = extract_coordinates(request.geometry_grid)
        polygon = shape(request.geometry_grid)
        _ = gpd.GeoDataFrame([{'geometry': polygon}], crs="EPSG:4326").to_crs(epsg=3857)

        feature_map = {
            "population_density": "pop",
            "poi": "poi",
            "roads": "road",
            "healtcare_nearby": "healthcare",
            "far_from_river": "river",
            "slope": "slope"
        }

        model_input = {
            model_key: request.feature_scores[api_key]
            for api_key, model_key in feature_map.items()
        }
        input_df = pd.DataFrame([model_input])

        predicted_class_idx = int(model.predict(input_df)[0])
        predicted_probs = model.predict_proba(input_df)[0]
        category = label_mapping[predicted_class_idx]
        confidence = round(max(predicted_probs), 4)

        total_weight = sum(request.weights.values())
        norm_weights = {
            k: round(v / total_weight, 3) for k, v in request.weights.items()
        }

        return {
            "predicted_class": category,
            "confidence": confidence,
            "model_accuracy": MODEL_ACCURACY,
            "feature_scores": request.feature_scores,
            "weights_applied": norm_weights,
            "input_polygon": coords,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        dependencies = {
            "geopandas": gpd.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__[:5]
        }
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "dependencies": dependencies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")