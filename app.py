from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Tuple, Any
from shapely.geometry import shape, mapping
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib

# SQLAlchemy & GeoAlchemy2
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import Geometry
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App ---
app = FastAPI()

# Load the trained classifier model
model = joblib.load("model/random_forest_model.pkl")

# Hardcoded model accuracy (computed offline on test/validation set)
MODEL_ACCURACY = 0.92

# Category enum matching model output: 'low', 'medium', 'high'
# --- DB Config ---
DATABASE_URL = "postgresql://postgres:HansAngela09@localhost:5432/batas_wilayah_indonesia"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# --- DB Model ---
class Provinsi(Base):
    __tablename__ = "provinsi"
    __table_args__ = {'schema': 'dataset_wilayah_indonesia'}
    
    id = Column(Integer, primary_key=True, index=True)
    kode_provinsi = Column(String)
    nama_provinsi = Column(String)
    geom = Column(Geometry("MULTIPOLYGON"))

# --- Enum ---
class SuitabilityCategory(str, Enum):
    NOT_RECOMMENDED = "low"
    NEUTRAL = "medium"
    RECOMMENDED = "high"

# --- Request Schema ---
class SuitabilityRequest(BaseModel):
    geometry_grid: Dict
    feature_scores: Dict[str, Any]
    weights: Dict[str, Any]

# --- Response Schema ---
class SuitabilityResponse(BaseModel):
    predicted_class: SuitabilityCategory
    confidence: float
    model_accuracy: float
    feature_scores: Dict[str, float]
    weights_applied: Dict[str, float]
    input_polygon: List[List[Tuple[float, float]]]
    timestamp: str

# --- Health Check Schema ---
class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    dependencies: dict

# --- Label Mapping ---
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

# --- Predict Endpoint ---
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

# --- Health Check ---
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

@app.get("/provinsi")
def get_provinsi():
    try:
        df = pd.read_sql_query(
            "SELECT kode_provinsi, nama_provinsi, latitude, longitude, rings FROM dataset_wilayah_indonesia.provinsi",
            con=engine
        )

        result = []
        for _, row in df.iterrows():
            # Asumsikan rings disimpan sebagai stringified list
            try:
                rings = eval(row["rings"])  # pastikan isinya list of coordinates
            except Exception:
                rings = []

            result.append({
                "kode_provinsi": row["kode_provinsi"],
                "nama_provinsi": row["nama_provinsi"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "rings": rings
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kota-kabupaten")
def get_kota_kabupaten(kode_provinsi: str):
    try:
        query = """
            SELECT kode_kota_kabupaten, nama_kota_kabupaten, latitude, longitude, rings
            FROM dataset_wilayah_indonesia.kota_kabupaten
            WHERE kode_provinsi = %(kode_provinsi)s
        """
        df = pd.read_sql_query(query, con=engine, params={"kode_provinsi": kode_provinsi})

        result = []
        for _, row in df.iterrows():
            try:
                rings = eval(row["rings"])
            except Exception:
                rings = []

            result.append({
                "kode_kota_kabupaten": row["kode_kota_kabupaten"],
                "nama_kota_kabupaten": row["nama_kota_kabupaten"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "rings": rings
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kecamatan")
def get_kecamatan(kode_kota_kabupaten: str):
    try:
        query = """
            SELECT kode_kecamatan, nama_kecamatan, latitude, longitude, rings
            FROM dataset_wilayah_indonesia.kecamatan
            WHERE kode_kota_kabupaten = %(kode_kota_kabupaten)s
        """
        df = pd.read_sql_query(query, con=engine, params={"kode_kota_kabupaten": kode_kota_kabupaten})

        result = []
        for _, row in df.iterrows():
            try:
                rings = eval(row["rings"])
            except Exception:
                rings = []

            result.append({
                "kode_kecamatan": row["kode_kecamatan"],
                "nama_kecamatan": row["nama_kecamatan"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "rings": rings
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kelurahan")
def get_kelurahan(kode_kecamatan: str):
    try:
        query = """
            SELECT kode_kelurahan, nama_kelurahan, latitude, longitude, rings
            FROM dataset_wilayah_indonesia.kelurahan
            WHERE kode_kecamatan = %(kode_kecamatan)s
        """
        df = pd.read_sql_query(query, con=engine, params={"kode_kecamatan": kode_kecamatan})

        result = []
        for _, row in df.iterrows():
            try:
                rings = eval(row["rings"])
            except Exception:
                rings = []

            result.append({
                "kode_kelurahan": row["kode_kelurahan"],
                "nama_kelurahan": row["nama_kelurahan"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "rings": rings
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Tambahkan ini setelah app dibuat
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)