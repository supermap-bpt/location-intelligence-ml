from fastapi import APIRouter, HTTPException
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib

from app.models.responses import HealthCheckResponse

router = APIRouter()

@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")