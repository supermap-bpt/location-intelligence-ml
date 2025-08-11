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

        # Ambil data dari DB
        wilayah_siswa_gdf = get_siswa_putus_sekolah_geodataframe()
        wilayah_kemiskinan_gdf = get_kemiskinan_geodataframe()
        wilayah_penduduk_gdf = get_kepadatan_penduduk_geodataframe()
        wilayah_poi_gdf = get_poi_geodataframe()

        # Samakan CRS
        wilayah_siswa_gdf = wilayah_siswa_gdf.to_crs("EPSG:4326")
        wilayah_kemiskinan_gdf = wilayah_kemiskinan_gdf.to_crs("EPSG:4326")
        wilayah_penduduk_gdf = wilayah_penduduk_gdf.to_crs("EPSG:4326")
        wilayah_poi_gdf = wilayah_poi_gdf.to_crs("EPSG:4326")
        input_gdf = input_gdf.to_crs("EPSG:4326")

        # Intersect siswa_putus_sekolah
        wilayah_siswa_gdf['intersection'] = wilayah_siswa_gdf.geometry.intersection(polygon)
        wilayah_siswa_gdf = wilayah_siswa_gdf[wilayah_siswa_gdf['intersection'].area > 0]
        nilai_siswa = 0.0
        if not wilayah_siswa_gdf.empty:
            wilayah_siswa_gdf['intersection_area'] = wilayah_siswa_gdf['intersection'].area
            max_idx_siswa = wilayah_siswa_gdf['intersection_area'].idxmax()
            nilai_siswa = float(wilayah_siswa_gdf.loc[max_idx_siswa, 's_siswaputussekolah'])

        # Intersect kemiskinan
        wilayah_kemiskinan_gdf['intersection'] = wilayah_kemiskinan_gdf.geometry.intersection(polygon)
        wilayah_kemiskinan_gdf = wilayah_kemiskinan_gdf[wilayah_kemiskinan_gdf['intersection'].area > 0]
        nilai_kemiskinan = 0.0
        if not wilayah_kemiskinan_gdf.empty:
            wilayah_kemiskinan_gdf['intersection_area'] = wilayah_kemiskinan_gdf['intersection'].area
            max_idx_kemiskinan = wilayah_kemiskinan_gdf['intersection_area'].idxmax()
            nilai_kemiskinan = float(wilayah_kemiskinan_gdf.loc[max_idx_kemiskinan, 's_kemiskinan'])

        # Intersect kepadatan_penduduk
        wilayah_penduduk_gdf['intersection'] = wilayah_penduduk_gdf.geometry.intersection(polygon)
        wilayah_penduduk_gdf = wilayah_penduduk_gdf[wilayah_penduduk_gdf['intersection'].area > 0]
        nilai_penduduk = 0.0
        if not wilayah_penduduk_gdf.empty:
            wilayah_penduduk_gdf['intersection_area'] = wilayah_penduduk_gdf['intersection'].area
            max_idx_penduduk = wilayah_penduduk_gdf['intersection_area'].idxmax()
            nilai_penduduk = float(wilayah_penduduk_gdf.loc[max_idx_penduduk, 's_pddk'])

        # Intersect poi
        wilayah_poi_gdf['intersection'] = wilayah_poi_gdf.geometry.intersection(polygon)
        wilayah_poi_gdf = wilayah_poi_gdf[wilayah_poi_gdf['intersection'].area > 0]
        nilai_poi = 0.0
        if not wilayah_poi_gdf.empty:
            wilayah_poi_gdf['intersection_area'] = wilayah_poi_gdf['intersection'].area
            max_idx_poi = wilayah_poi_gdf['intersection_area'].idxmax()
            nilai_poi = float(wilayah_poi_gdf.loc[max_idx_poi, 's_poi'])
        
        # --- Kedekatan Sungai ---
        wilayah_sungai_gdf = get_kedekatan_sungai_geodataframe()
        wilayah_sungai_gdf = wilayah_sungai_gdf.to_crs("EPSG:4326")
        wilayah_sungai_gdf['intersection'] = wilayah_sungai_gdf.geometry.intersection(polygon)
        wilayah_sungai_gdf = wilayah_sungai_gdf[wilayah_sungai_gdf['intersection'].area > 0]
        nilai_sungai = 0  # default jika tidak ada intersection
        if not wilayah_sungai_gdf.empty:
            wilayah_sungai_gdf['intersection_area'] = wilayah_sungai_gdf['intersection'].area
            max_idx_sungai = wilayah_sungai_gdf['intersection_area'].idxmax()
            nilai_sungai = int(wilayah_sungai_gdf.loc[max_idx_sungai, 's_sungai'])
        
        # Ambil data kedekatan faskes
        wilayah_faskes_gdf = get_kedekatan_faskes_geodataframe()
        wilayah_faskes_gdf = wilayah_faskes_gdf.to_crs("EPSG:4326")
        wilayah_faskes_gdf['intersection'] = wilayah_faskes_gdf.geometry.intersection(polygon)
        wilayah_faskes_gdf = wilayah_faskes_gdf[wilayah_faskes_gdf['intersection'].area > 0]

        nilai_faskes = 0  # default kalau tidak ada intersection
        if not wilayah_faskes_gdf.empty:
            wilayah_faskes_gdf['intersection_area'] = wilayah_faskes_gdf['intersection'].area
            max_idx_faskes = wilayah_faskes_gdf['intersection_area'].idxmax()
            nilai_faskes = int(wilayah_faskes_gdf.loc[max_idx_faskes, 's_faskes'])

        # Ambil data kedekatan jalan
        wilayah_jalan_gdf = get_kedekatan_jalan_geodataframe()
        wilayah_jalan_gdf = wilayah_jalan_gdf.to_crs("EPSG:4326")
        wilayah_jalan_gdf['intersection'] = wilayah_jalan_gdf.geometry.intersection(polygon)
        wilayah_jalan_gdf = wilayah_jalan_gdf[wilayah_jalan_gdf['intersection'].area > 0]

        nilai_road = 0  # default kalau tidak ada intersection
        if not wilayah_jalan_gdf.empty:
            wilayah_jalan_gdf['intersection_area'] = wilayah_jalan_gdf['intersection'].area
            max_idx_jalan = wilayah_jalan_gdf['intersection_area'].idxmax()
            nilai_road = int(wilayah_jalan_gdf.loc[max_idx_jalan, 's_road'])

        # Ambil data slope
        wilayah_slope_gdf = get_slope_geodataframe()
        wilayah_slope_gdf = wilayah_slope_gdf.to_crs("EPSG:4326")
        wilayah_slope_gdf['intersection'] = wilayah_slope_gdf.geometry.intersection(polygon)
        wilayah_slope_gdf = wilayah_slope_gdf[wilayah_slope_gdf['intersection'].area > 0]

        nilai_slope = 0  # default jika tidak ada intersection
        if not wilayah_slope_gdf.empty:
            wilayah_slope_gdf['intersection_area'] = wilayah_slope_gdf['intersection'].area
            max_idx_slope = wilayah_slope_gdf['intersection_area'].idxmax()
            nilai_slope = int(wilayah_slope_gdf.loc[max_idx_slope, 's_slope'])
            
        # Update feature_scores
        cleaned_scores = {}
        for api_key, model_key in feature_map.items():
            if api_key == "jumlahsiswaputussekolah":
                cleaned_scores[api_key] = nilai_siswa
            elif api_key == "kemiskinan":
                cleaned_scores[api_key] = nilai_kemiskinan
            elif api_key == "peopleden":
                cleaned_scores[api_key] = nilai_penduduk
            elif api_key == "poiarea":  # Asumsi poiarea adalah nilai dari s_poi
                cleaned_scores[api_key] = nilai_poi
            elif api_key == "nearest_sungai":
                cleaned_scores[api_key] = nilai_sungai
            elif api_key == "nearestfaskes":
                cleaned_scores[api_key] = nilai_faskes
            elif api_key == "road":
                cleaned_scores[api_key] = nilai_road
            elif api_key == "slope":
                cleaned_scores[api_key] = nilai_slope
            else:
                if api_key not in request.feature_scores:
                    raise HTTPException(status_code=400, detail=f"Missing feature_scores key: {api_key}")
                try:
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