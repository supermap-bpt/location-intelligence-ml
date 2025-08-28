from fastapi import APIRouter, HTTPException
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib

from app.models.responses import HealthCheckResponse
from app.config import MODEL_PATH

router = APIRouter()

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

@router.get("/", response_model=HealthCheckResponse)
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