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

# Engine untuk dummy_bps
DATABASE_URL_DUMMY_BPS = "postgresql://postgres:HansAngela09@localhost:5432/dummy_bps"
engine_dummy_bps = create_engine(DATABASE_URL_DUMMY_BPS)

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

def get_siswa_putus_sekolah_geodataframe():
    sql = """
        SELECT wadmkc, s_siswaputussekolah, ST_AsGeoJSON(geometry) as geojson
        FROM siswa_putus_sekolah
        WHERE geometry IS NOT NULL
    """
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

def get_kemiskinan_geodataframe():
    sql = """
        SELECT wadmkc, s_kemiskinan, ST_AsGeoJSON(geometry) as geojson
        FROM kemiskinan
        WHERE geometry IS NOT NULL
    """
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

def get_kepadatan_penduduk_geodataframe():
    sql = """
        SELECT wadmkc, s_pddk, ST_AsGeoJSON(geometry) as geojson
        FROM kepadatan_penduduk
        WHERE geometry IS NOT NULL
    """
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

def get_poi_geodataframe():
    sql = """
        SELECT wadmkc, s_poi, ST_AsGeoJSON(geometry) as geojson
        FROM poi
        WHERE geometry IS NOT NULL
    """
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

def get_kedekatan_sungai_geodataframe():
    sql = """
        SELECT s_sungai, ST_AsGeoJSON(geometry) as geojson
        FROM kedekatan_sungai
        WHERE geometry IS NOT NULL
    """
    df = pd.read_sql_query(sql, con=engine_dummy_bps)  # atau engine yang sesuai database
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

def get_kedekatan_faskes_geodataframe():
    sql = """
        SELECT s_faskes, ST_AsGeoJSON(geometry) as geojson
        FROM kedekatan_faskes
        WHERE geometry IS NOT NULL
    """
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

def get_kedekatan_jalan_geodataframe():
    sql = """
        SELECT s_road, ST_AsGeoJSON(geometry) as geojson
        FROM jalan
        WHERE geometry IS NOT NULL
    """
    df = pd.read_sql_query(sql, con=engine_dummy_bps)  # sesuaikan engine kalau beda
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

def get_slope_geodataframe():
    sql = """
        SELECT s_slope, ST_AsGeoJSON(geometry) as geojson
        FROM slope
        WHERE geometry IS NOT NULL
    """
    df = pd.read_sql_query(sql, con=engine_dummy_bps)  # sesuaikan engine kalau beda
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

# --- Predict Endpoint ---
@app.post("/predict", response_model=SuitabilityResponse)
async def predict_suitability(request: SuitabilityRequest):
    try:
        polygon = shape(request.geometry_grid)
        input_gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs="EPSG:4326")

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

        # Set fitur yang beratnya > 0
        non_zero_features = {k for k, w in converted_weights.items() if w > 0}

        # Fungsi bantu untuk intersect dan ambil nilai max intersection area
        def get_intersect_value(gdf_func, polygon, score_col, feature_key):
            if feature_key not in non_zero_features:
                return 0
            gdf = gdf_func()
            gdf = gdf.to_crs("EPSG:4326")
            gdf['intersection'] = gdf.geometry.intersection(polygon)
            gdf = gdf[gdf['intersection'].area > 0]
            if gdf.empty:
                return 0
            gdf['intersection_area'] = gdf['intersection'].area
            max_idx = gdf['intersection_area'].idxmax()
            return float(gdf.loc[max_idx, score_col])

        # Ambil nilai tiap fitur dengan intersect hanya jika weight > 0
        nilai_siswa = get_intersect_value(get_siswa_putus_sekolah_geodataframe, polygon, 's_siswaputussekolah', "jumlahsiswaputussekolah")
        nilai_kemiskinan = get_intersect_value(get_kemiskinan_geodataframe, polygon, 's_kemiskinan', "kemiskinan")
        nilai_penduduk = get_intersect_value(get_kepadatan_penduduk_geodataframe, polygon, 's_pddk', "peopleden")
        nilai_poi = get_intersect_value(get_poi_geodataframe, polygon, 's_poi', "poiarea")
        nilai_sungai = get_intersect_value(get_kedekatan_sungai_geodataframe, polygon, 's_sungai', "nearest_sungai")
        nilai_faskes = get_intersect_value(get_kedekatan_faskes_geodataframe, polygon, 's_faskes', "nearestfaskes")
        nilai_road = get_intersect_value(get_kedekatan_jalan_geodataframe, polygon, 's_road', "road")
        nilai_slope = get_intersect_value(get_slope_geodataframe, polygon, 's_slope', "slope")

        # Update feature_scores
        cleaned_scores = {
            "jumlahsiswaputussekolah": nilai_siswa,
            "kemiskinan": nilai_kemiskinan,
            "peopleden": nilai_penduduk,
            "poiarea": nilai_poi,
            "nearest_sungai": nilai_sungai,
            "nearestfaskes": nilai_faskes,
            "road": nilai_road,
            "slope": nilai_slope
        }

        # Jika ada fitur lain di request.feature_scores (misal input manual), cek dan masukkan
        for api_key in feature_map.keys():
            if api_key not in cleaned_scores:
                if api_key not in request.feature_scores:
                    raise HTTPException(status_code=400, detail=f"Missing feature_scores key: {api_key}")
                try:
                    cleaned_scores[api_key] = float(request.feature_scores[api_key])
                except (ValueError, TypeError) as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value for {api_key}. Must be numeric. Error: {str(e)}"
                    )

        # Prepare input untuk model
        model_input = {model_key: cleaned_scores[api_key] for api_key, model_key in feature_map.items()}
        input_df = pd.DataFrame([model_input])

        predicted_class = model.predict(input_df)[0]
        predicted_probs = model.predict_proba(input_df)[0]
        category = label_mapping.get(predicted_class, SuitabilityCategory.NEUTRAL)
        class_index = list(model.classes_).index(predicted_class)
        confidence = float(predicted_probs[class_index])
        norm_weights = {k: round(v / 100.0, 4) for k, v in converted_weights.items()}

        return {
            "predicted_class": category,
            "confidence": confidence,
            "model_accuracy": MODEL_ACCURACY,
            "feature_scores": cleaned_scores,
            "weights_applied": norm_weights,
            "input_polygon": extract_coordinates(request.geometry_grid),
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