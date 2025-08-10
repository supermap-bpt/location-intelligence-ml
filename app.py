from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Any
from shapely.geometry import shape
import geopandas as gpd
from datetime import datetime

app = FastAPI()

# Load the trained classifier model
model = joblib.load("model/random_forest_model.pkl")

# Hardcoded model accuracy (computed offline on test/validation set)
MODEL_ACCURACY = 0.92

# Category enum matching model output: 'low', 'medium', 'high'
class SuitabilityCategory(str, Enum):
    NOT_RECOMMENDED = "low"
    NEUTRAL = "medium"
    RECOMMENDED = "high"

# Request schema
class SuitabilityRequest(BaseModel):
    geometry_grid: Dict
    feature_scores: Dict[str, Any]
    weights: Dict[str, Any]

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

# Class mapping from model output to our categories
label_mapping = {
    'low': SuitabilityCategory.NOT_RECOMMENDED,
    'medium': SuitabilityCategory.NEUTRAL,
    'high': SuitabilityCategory.RECOMMENDED
}

# Feature mapping - matches your training data columns
feature_map = {
    "jumlahsiswaputussekolah": "jumlahsiswaputussekolah",
    "kemiskinan": "kemiskinan",
    "nearest_sungai": "nearest_sungai",
    "nearestfaskes": "nearestfaskes",
    "peopleden": "peopleden",
    "poiarea": "poiarea",
    "road": "road",
    "slope": "slope"
}

def extract_coordinates(geojson: Dict) -> List[List[Tuple[float, float]]]:
    if not isinstance(geojson, dict):
        raise ValueError("geometry_grid must be a GeoJSON object (dict).")
    if geojson.get("type") == "Polygon":
        coords = geojson.get("coordinates")
        if coords is None:
            raise ValueError("Polygon geometry must include 'coordinates'.")
        return coords
    raise ValueError("Only Polygon geometry type is supported")

@app.post("/predict", response_model=SuitabilityResponse)
async def predict_suitability(request: SuitabilityRequest):
    try:
        # Validate and extract geometry
        coords = extract_coordinates(request.geometry_grid)
        polygon = shape(request.geometry_grid)
        _ = gpd.GeoDataFrame([{'geometry': polygon}], crs="EPSG:4326").to_crs(epsg=3857)

        # Clean & validate features
        cleaned_scores = {}
        for api_key, model_key in feature_map.items():
            if api_key not in request.feature_scores:
                raise HTTPException(status_code=400, detail=f"Missing feature_scores key: {api_key}")
            
            try:
                # Convert all values to float as your model expects numerical inputs
                cleaned_scores[api_key] = float(request.feature_scores[api_key])
            except (ValueError, TypeError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {api_key}. Must be numeric. Error: {str(e)}"
                )

        # Clean & validate weights
        total_weight = 0.0
        converted_weights = {}
        for api_key in feature_map.keys():
            if api_key not in request.weights:
                raise HTTPException(status_code=400, detail=f"Missing weight key: {api_key}")
            
            try:
                weight = float(request.weights[api_key])
                if weight < 0:
                    raise ValueError(f"Weight for {api_key} must be non-negative")
                converted_weights[api_key] = weight
                total_weight += weight
            except (ValueError, TypeError) as e:
                raise HTTPException(status_code=400, detail=str(e))

        if not np.isclose(total_weight, 100.0, atol=1e-6):
            raise HTTPException(
                status_code=400,
                detail=f"Weights must sum to 100. Got {total_weight}."
            )

        # Prepare model input - using the same column names as in training
        model_input = {
            model_key: cleaned_scores[api_key]
            for api_key, model_key in feature_map.items()
        }
        input_df = pd.DataFrame([model_input])

        # Predict
        try:
            predicted_class = model.predict(input_df)[0]  # This will return 'low', 'medium', or 'high'
            predicted_probs = model.predict_proba(input_df)[0]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction failed: {str(e)}"
            )

        # Map the predicted class to our categories
        category = label_mapping.get(predicted_class, SuitabilityCategory.NEUTRAL)
        
        # Get confidence as the probability of the predicted class
        class_index = list(model.classes_).index(predicted_class)
        confidence = float(predicted_probs[class_index])
        
        # Normalize weights for response
        norm_weights = {k: round(v / 100.0, 4) for k, v in converted_weights.items()}

        return {
            "predicted_class": category,
            "confidence": confidence,
            "model_accuracy": MODEL_ACCURACY,
            "feature_scores": cleaned_scores,
            "weights_applied": norm_weights,
            "input_polygon": coords,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        dependencies = {
            "geopandas": getattr(gpd, "__version__", "unknown"),
            "pandas": getattr(pd, "__version__", "unknown"),
            "numpy": getattr(np, "__version__", "unknown"),
            "joblib": getattr(joblib, "__version__", "unknown")
        }
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "dependencies": dependencies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")