from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from shapely import wkb
from shapely.geometry import shape, mapping
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
import json

app = FastAPI()

# Load model
model = joblib.load("model/random_forest_model_regressor.pkl")
MEAN_ABSOLUTE_ERROR = 0.0412
MEAN_SQUARED_ERROR = 0.0031
ROOT_MEAN_SQUARED_ERROR = 0.0560
R2_SCORE = 0.8643

# Database configuration
DATABASE_URL = "postgresql://postgres:12345678@localhost:5432/batas_wilayah"
DATABASE_URL_DUMMY_BPS = "postgresql://postgres:12345678@localhost:5432/dummy_parameter_bps"
engine = create_engine(DATABASE_URL)
engine_dummy_bps = create_engine(DATABASE_URL_DUMMY_BPS)

# Enums and Models
class SuitabilityCategory(str, Enum):
    NOT_RECOMMENDED = "low"
    NEUTRAL = "medium"
    RECOMMENDED = "high"

class SuitabilityRequest(BaseModel):
    geometry_grid: Dict
    feature_scores: Dict[str, Any]
    weights: Dict[str, Any]

class BatchSuitabilityRequest(BaseModel):
    geometry_grid: List[Dict]
    feature_scores: Dict[str, Any]
    weights: Dict[str, Any]
    grid_ids: Optional[List[str]] = None

class SuitabilityResponse(BaseModel):
    predicted_class: SuitabilityCategory
    confidence: float
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r2_score: float
    feature_scores: Dict[str, float]
    weights_applied: Dict[str, float]
    input_polygon: List[List[Tuple[float, float]]]
    timestamp: str
    grid_id: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    dependencies: dict

class HotelItem(BaseModel):
    nama: str
    geometry: Optional[dict]

class PendidikanItem(BaseModel):
    namobj: str
    geometry: Optional[dict]

class PusatPerbelanjaanItem(BaseModel):
    namobj: str
    geometry: Optional[dict]

class RumahSakitItem(BaseModel):
    namobj: str
    geometry: Optional[dict]

# Label and feature mapping
label_mapping = {
    'low': SuitabilityCategory.NOT_RECOMMENDED,
    'medium': SuitabilityCategory.NEUTRAL,
    'high': SuitabilityCategory.RECOMMENDED
}

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

# Helper functions
def extract_coordinates(geojson: Dict) -> List[List[Tuple[float, float]]]:
    if not isinstance(geojson, dict):
        raise ValueError("geometry_grid must be a GeoJSON object (dict).")
    if geojson.get("type") == "Polygon":
        coords = geojson.get("coordinates")
        if coords is None:
            raise ValueError("Polygon geometry must include 'coordinates'.")
        return coords
    raise ValueError("Only Polygon geometry type is supported")

def get_intersect_value(gdf, polygon, score_col):
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

# Data loading functions
def get_siswa_putus_sekolah_geodataframe():
    sql = "SELECT wadmkc, s_siswaputussekolah, ST_AsGeoJSON(smgeometry) as geojson FROM siswa_putus_sekolah WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kemiskinan_geodataframe():
    sql = "SELECT wadmkc, s_kemiskinan, ST_AsGeoJSON(smgeometry) as geojson FROM kemiskinan WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kepadatan_penduduk_geodataframe():
    sql = "SELECT wadmkc, s_pddk, ST_AsGeoJSON(smgeometry) as geojson FROM kepadatan_penduduk WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_poi_geodataframe():
    sql = "SELECT wadmkc, s_poi, ST_AsGeoJSON(smgeometry) as geojson FROM poi WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kedekatan_sungai_geodataframe():
    sql = "SELECT s_sungai, ST_AsGeoJSON(smgeometry) as geojson FROM kedekatan_sungai WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kedekatan_faskes_geodataframe():
    sql = "SELECT s_faskes, ST_AsGeoJSON(smgeometry) as geojson FROM kedekatan_faskes WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kedekatan_jalan_geodataframe():
    sql = "SELECT s_road, ST_AsGeoJSON(smgeometry) as geojson FROM jalan WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_slope_geodataframe():
    sql = "SELECT s_slope, ST_AsGeoJSON(smgeometry) as geojson FROM slope WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# Endpoints
@app.post("/batch-predict", response_model=List[SuitabilityResponse])
async def batch_predict_suitability(request: BatchSuitabilityRequest):
    try:
        # Validate weights
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
            raise HTTPException(status_code=400, detail=f"Weights must sum to 100. Got {total_weight}.")

        non_zero_features = {k for k, w in converted_weights.items() if w > 0}
        
        # Pre-load all needed GeoDataFrames
        gdfs = {
            "siswa": get_siswa_putus_sekolah_geodataframe() if "jumlahsiswaputussekolah" in non_zero_features else None,
            "kemiskinan": get_kemiskinan_geodataframe() if "kemiskinan" in non_zero_features else None,
            "penduduk": get_kepadatan_penduduk_geodataframe() if "peopleden" in non_zero_features else None,
            "poi": get_poi_geodataframe() if "poiarea" in non_zero_features else None,
            "sungai": get_kedekatan_sungai_geodataframe() if "nearest_sungai" in non_zero_features else None,
            "faskes": get_kedekatan_faskes_geodataframe() if "nearestfaskes" in non_zero_features else None,
            "road": get_kedekatan_jalan_geodataframe() if "road" in non_zero_features else None,
            "slope": get_slope_geodataframe() if "slope" in non_zero_features else None
        }

        results = []
        for i, grid in enumerate(request.geometry_grid):
            try:
                polygon = shape(grid)
                input_gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs="EPSG:4326")

                # Get values for each feature
                cleaned_scores = {
                    "jumlahsiswaputussekolah": get_intersect_value(gdfs["siswa"], polygon, 's_siswaputussekolah'),
                    "kemiskinan": get_intersect_value(gdfs["kemiskinan"], polygon, 's_kemiskinan'),
                    "peopleden": get_intersect_value(gdfs["penduduk"], polygon, 's_pddk'),
                    "poiarea": get_intersect_value(gdfs["poi"], polygon, 's_poi'),
                    "nearest_sungai": get_intersect_value(gdfs["sungai"], polygon, 's_sungai'),
                    "nearestfaskes": get_intersect_value(gdfs["faskes"], polygon, 's_faskes'),
                    "road": get_intersect_value(gdfs["road"], polygon, 's_road'),
                    "slope": get_intersect_value(gdfs["slope"], polygon, 's_slope')
                }

                # Prepare model input
                model_input = {model_key: cleaned_scores[api_key] for api_key, model_key in feature_map.items()}
                input_df = pd.DataFrame([model_input])

                # Predict with regressor
                reg_value = float(model.predict(input_df)[0])
                
                # Calculate weighted sum
                norm_weights = {k: round(v / 100.0, 4) for k, v in converted_weights.items()}
                weighted_sum = sum(cleaned_scores[k] * norm_weights[k] for k in feature_map.keys())
                
                # Combine regressor value and weighted sum (adjust as needed)
                final_value = max(0.0, min(1.0, (0.7 * reg_value + 0.3 * weighted_sum)))  # Weighted combination

                # Classify into 3 categories
                if final_value < 0.33:
                    predicted_class = SuitabilityCategory.NOT_RECOMMENDED
                elif final_value < 0.66:
                    predicted_class = SuitabilityCategory.NEUTRAL
                else:
                    predicted_class = SuitabilityCategory.RECOMMENDED

                # Calculate confidence based on distance from thresholds
                if predicted_class == SuitabilityCategory.NOT_RECOMMENDED:
                    confidence = (0.33 - final_value) / 0.33
                elif predicted_class == SuitabilityCategory.NEUTRAL:
                    if final_value < 0.5:
                        confidence = (final_value - 0.33) / 0.17
                    else:
                        confidence = (0.66 - final_value) / 0.16
                else:
                    confidence = (final_value - 0.66) / 0.34
                
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]

                # Prepare response
                result = {
                    "predicted_class": predicted_class,
                    "confidence": round(confidence, 4),
                    "mean_absolute_error": MEAN_ABSOLUTE_ERROR,
                    "mean_squared_error": MEAN_SQUARED_ERROR,
                    "root_mean_squared_error": ROOT_MEAN_SQUARED_ERROR,
                    "r2_score": R2_SCORE,
                    "feature_scores": cleaned_scores,
                    "weights_applied": norm_weights,
                    "input_polygon": extract_coordinates(grid),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add grid_id if provided
                if request.grid_ids and i < len(request.grid_ids):
                    result["grid_id"] = request.grid_ids[i]

                results.append(result)
                
            except Exception as e:
                print(f"Error processing grid {i}: {str(e)}")
                # Optionally include error information in response
                error_result = {
                    "error": str(e),
                    "grid_id": request.grid_ids[i] if (request.grid_ids and i < len(request.grid_ids)) else None,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
                continue

        return results

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
        # Ambil geom + latitude & longitude dari PostGIS
        df = pd.read_sql_query(
            """
            SELECT
                kode_provinsi,
                nama_provinsi,
                latitude,
                longitude,
                ST_AsGeoJSON(geom) AS geom_json
            FROM provinsi
            """,
            con=engine
        )

        result = []
        for _, row in df.iterrows():
            try:
                geom_obj = json.loads(row["geom_json"])
                rings = geom_obj.get("coordinates", [])
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
            SELECT
                kode_kota_kabupaten,
                nama_kota_kabupaten,
                latitude,
                longitude,
                ST_AsGeoJSON(geom) AS geom_json
            FROM kota_kabupaten
            WHERE kode_provinsi = %(kode_provinsi)s
        """
        df = pd.read_sql_query(query, con=engine, params={"kode_provinsi": kode_provinsi})

        result = []
        for _, row in df.iterrows():
            try:
                geom_obj = json.loads(row["geom_json"])
                rings = geom_obj.get("coordinates", [])
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
            SELECT
                kode_kecamatan,
                nama_kecamatan,
                latitude,
                longitude,
                ST_AsGeoJSON(geom) AS geom_json
            FROM kecamatan
            WHERE kode_kota_kabupaten = %(kode_kota_kabupaten)s
        """
        df = pd.read_sql_query(query, con=engine, params={"kode_kota_kabupaten": kode_kota_kabupaten})

        result = []
        for _, row in df.iterrows():
            try:
                geom_obj = json.loads(row["geom_json"])
                rings = geom_obj.get("coordinates", [])
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

@app.get("/kelurahan-desa")
def get_kelurahan(kode_kecamatan: str):
    try:
        query = """
            SELECT
                kode_kelurahan_desa AS kode_kelurahan,
                nama_kelurahan_desa AS nama_kelurahan,
                latitude,
                longitude,
                ST_AsGeoJSON(geom) AS geom_json
            FROM kelurahan_desa
            WHERE kode_kecamatan = %(kode_kecamatan)s
        """
        df = pd.read_sql_query(query, con=engine, params={"kode_kecamatan": kode_kecamatan})

        result = []
        for _, row in df.iterrows():
            try:
                geom_obj = json.loads(row["geom_json"])
                rings = geom_obj.get("coordinates", [])
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

@app.get("/get-hotel", response_model=List[HotelItem])
def get_hotel(nmkec: str = Query(..., description="Nama kecamatan (bisa multiple, dipisah koma)")):
    try:
        # Pisahkan kecamatan berdasarkan koma, hilangkan spasi
        kecamatan_list = [k.strip() for k in nmkec.split(",") if k.strip()]

        if not kecamatan_list:
            raise HTTPException(
                status_code=400,
                detail="Minimal satu nama kecamatan harus disediakan"
            )

        query = text("""
            SELECT nama, smgeometry
            FROM "Hotel_P"
            WHERE LOWER(nmkec) IN (SELECT LOWER(UNNEST(:nmkec_list)))
        """)

        with engine_dummy_bps.connect() as conn:
            results = conn.execute(query, {"nmkec_list": kecamatan_list}).fetchall()

        hotels_list = []
        for nama, geom_wkb in results:
            geom_geojson = None
            if geom_wkb:
                if isinstance(geom_wkb, memoryview):
                    geom_bytes = geom_wkb.tobytes()
                elif isinstance(geom_wkb, str):
                    geom_bytes = bytes.fromhex(geom_wkb)
                elif isinstance(geom_wkb, bytes):
                    geom_bytes = geom_wkb
                else:
                    raise TypeError(f"Tipe geometry tidak dikenal: {type(geom_wkb)}")
                shapely_geom = wkb.loads(geom_bytes)
                geom_geojson = mapping(shapely_geom)

            hotels_list.append({
                "nama": nama,
                "geometry": geom_geojson
            })

        return hotels_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-pendidikan", response_model=List[PendidikanItem])
def get_pendidikan(nmkec: str = Query(..., description="Nama kecamatan (bisa multiple, dipisah koma)")):
    try:
        kecamatan_list = [k.strip() for k in nmkec.split(",") if k.strip()]

        if not kecamatan_list:
            raise HTTPException(status_code=400, detail="Minimal satu nama kecamatan harus disediakan")

        query = text("""
            SELECT namobj, smgeometry
            FROM "Pendidikan_P"
            WHERE LOWER(nmkec) IN (SELECT LOWER(UNNEST(:nmkec_list)))
        """)

        with engine_dummy_bps.connect() as conn:
            results = conn.execute(query, {"nmkec_list": kecamatan_list}).fetchall()

        pendidikan_list = []
        for namobj, geom_wkb in results:
            geom_geojson = None
            if geom_wkb:
                if isinstance(geom_wkb, memoryview):
                    geom_bytes = geom_wkb.tobytes()
                elif isinstance(geom_wkb, str):
                    geom_bytes = bytes.fromhex(geom_wkb)
                elif isinstance(geom_wkb, bytes):
                    geom_bytes = geom_wkb
                else:
                    raise TypeError(f"Tipe geometry tidak dikenal: {type(geom_wkb)}")
                shapely_geom = wkb.loads(geom_bytes)
                geom_geojson = mapping(shapely_geom)

            pendidikan_list.append({
                "namobj": namobj,
                "geometry": geom_geojson
            })

        return pendidikan_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-pusatperbelanjaan", response_model=List[PusatPerbelanjaanItem])
def get_pusatperbelanjaan(nmkec: str = Query(..., description="Nama kecamatan (bisa multiple, dipisah koma)")):
    try:
        kecamatan_list = [k.strip() for k in nmkec.split(",") if k.strip()]

        if not kecamatan_list:
            raise HTTPException(status_code=400, detail="Minimal satu nama kecamatan harus disediakan")

        query = text("""
            SELECT namobj, smgeometry
            FROM "PusatPerbelanjaan_P"
            WHERE LOWER(nmkec) IN (SELECT LOWER(UNNEST(:nmkec_list)))
        """)

        with engine_dummy_bps.connect() as conn:
            results = conn.execute(query, {"nmkec_list": kecamatan_list}).fetchall()

        pusatperbelanjaan_list = []
        for namobj, geom_wkb in results:
            if namobj is None:
                continue  # skip kalau nama objek kosong

            geom_geojson = None
            if geom_wkb:
                if isinstance(geom_wkb, memoryview):
                    geom_bytes = geom_wkb.tobytes()
                elif isinstance(geom_wkb, str):
                    geom_bytes = bytes.fromhex(geom_wkb)
                elif isinstance(geom_wkb, bytes):
                    geom_bytes = geom_wkb
                else:
                    raise TypeError(f"Tipe geometry tidak dikenal: {type(geom_wkb)}")
                shapely_geom = wkb.loads(geom_bytes)
                geom_geojson = mapping(shapely_geom)

            pusatperbelanjaan_list.append({
                "namobj": namobj,
                "geometry": geom_geojson
            })

        return pusatperbelanjaan_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-rumahsakit", response_model=List[RumahSakitItem])
def get_rumahsakit(nmkec: str = Query(..., description="Nama kecamatan (bisa multiple dipisah koma)")):
    """
    Ambil daftar rumah sakit berdasarkan nama kecamatan (bisa multiple)
    """
    try:
        # Split multiple kecamatan names
        kecamatan_list = [k.strip() for k in nmkec.split(",") if k.strip()]
        
        if not kecamatan_list:
            raise HTTPException(
                status_code=400,
                detail="Minimal satu nama kecamatan harus disediakan"
            )

        # Build query with IN clause
        query = text("""
            SELECT namobj, smgeometry
            FROM "RumahSakit_P"
            WHERE LOWER(nmkec) IN (SELECT LOWER(UNNEST(:nmkec_list)))
        """)

        with engine_dummy_bps.connect() as conn:
            results = conn.execute(query, {"nmkec_list": kecamatan_list}).fetchall()

        rumah_sakit_list = []
        for namobj, geom_wkb in results:
            geom_geojson = None
            if geom_wkb:
                if isinstance(geom_wkb, memoryview):
                    geom_bytes = geom_wkb.tobytes()
                elif isinstance(geom_wkb, str):
                    geom_bytes = bytes.fromhex(geom_wkb)
                elif isinstance(geom_wkb, bytes):
                    geom_bytes = geom_wkb
                else:
                    raise TypeError(f"Tipe geometry tidak dikenal: {type(geom_wkb)}")

                shapely_geom = wkb.loads(geom_bytes)
                geom_geojson = mapping(shapely_geom)

            rumah_sakit_list.append({
                "namobj": namobj,
                "geometry": geom_geojson
            })

        return rumah_sakit_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)