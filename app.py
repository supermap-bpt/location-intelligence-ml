from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Tuple
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

# --- FastAPI App ---
app = FastAPI()

# --- Load model ---
model = joblib.load("model/suitability_classifier.pkl")
MODEL_ACCURACY = 0.92

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
    NOT_RECOMMENDED = "not recommended"
    NEUTRAL = "neutral"
    RECOMMENDED = "recommended"

# --- Request Schema ---
class SuitabilityRequest(BaseModel):
    geometry_grid: Dict
    feature_scores: Dict[str, float]
    weights: Dict[str, float]

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
    0: SuitabilityCategory.NOT_RECOMMENDED,
    1: SuitabilityCategory.NEUTRAL,
    2: SuitabilityCategory.RECOMMENDED
}

# --- Helper ---
def extract_coordinates(geojson: Dict) -> List[List[Tuple[float, float]]]:
    if geojson["type"] == "Polygon":
        return geojson["coordinates"]
    raise ValueError("Only Polygon geometry type is supported")

# --- Predict Endpoint ---
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

# --- Health Check ---
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