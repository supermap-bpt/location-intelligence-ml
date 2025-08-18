import json
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from shapely import wkb
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
import json

class GridData(BaseModel):
    geometry_grid: Dict[str, Any]
    feature_scores: Dict[str, float]
    weights: Dict[str, float]

class BatchRequest(BaseModel):
    data: List[GridData]
    low_range: float
    high_range: float

app = FastAPI()

# Load model
model = joblib.load("model/random_forest_model_regressor.pkl")
MEAN_ABSOLUTE_ERROR = 0.0412
MEAN_SQUARED_ERROR = 0.0031
ROOT_MEAN_SQUARED_ERROR = 0.0560
R2_SCORE = 0.8643

# Database configuration
DATABASE_URL = "postgresql://postgres@localhost:5432/region_indonesia"
DATABASE_URL_DUMMY_BPS = "postgresql://postgres@localhost:5432/BPS_LI"
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

class BufferRequest(BaseModel):
    buffer_polygons: List[Any]
    recommended_area: List[Any]
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

FACILITY_CONFIG = {
    "hotel": {"table": "Hotel_P", "name_col": "nama", "geom_col": "smgeometry"},
    "rumah_sakit": {"table": "RumahSakit_P", "name_col": "nama", "geom_col": "smgeometry"},
    "sekolah": {"table": "Sekolah_P", "name_col": "namobj", "geom_col": "smgeometry"},
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
@app.post("/batch-predict")
async def batch_predict_suitability(request: BatchRequest):
    try:
        grid_scores = []

        # Step 1: calculate weighted scores (grid_value)
        for grid_item in request.data:
            weights = grid_item.weights
            feature_scores = grid_item.feature_scores

            # validate weight sum
            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 100.0, atol=1e-6):
                raise HTTPException(status_code=400, detail=f"Weights must sum to 100. Got {total_weight}.")

            # weighted sum = grid_value
            grid_value = sum(feature_scores[k] * weights[k] for k in weights if k in feature_scores)

            # get GDP separately (no weight)
            gdp_value = feature_scores.get("gdp", None)

            grid_scores.append({
                "geometry": grid_item.geometry_grid,
                "grid_value": grid_value,
                "gdp": gdp_value
            })

        # Step 2: thresholds
        unique_scores = sorted(set(gs["grid_value"] for gs in grid_scores))
        n_unique = len(unique_scores)

        thresholds = {}
        if n_unique == 1:
            thresholds = {"high": [unique_scores[0], None]}
        elif n_unique == 2:
            thresholds = {
                "medium": [unique_scores[0], unique_scores[0]],
                "high": [unique_scores[1], None]
            }
        elif n_unique == 3:
            thresholds = {
                "low": [unique_scores[0], unique_scores[0]],
                "medium": [unique_scores[1], unique_scores[1]],
                "high": [unique_scores[2], None]
            }
        else:  # >= 4
            thresholds = {
                "low": [unique_scores[0], unique_scores[1]],
                "medium": [unique_scores[1] + 1, unique_scores[-2]],
                "high": [unique_scores[-2] + 1, None]
            }

        results = []

        # Step 3: classify + GDP downgrade
        for gs in grid_scores:
            val = gs["grid_value"]
            category = None

            if "low" in thresholds and thresholds["low"][0] <= val <= thresholds["low"][1]:
                category = "low"
            elif "medium" in thresholds and thresholds["medium"][0] <= val <= thresholds["medium"][1]:
                category = "medium"
            elif "high" in thresholds and (val >= thresholds["high"][0]):
                category = "high"

            # downgrade to low if GDP out of range
            if category in ["medium", "high"]:
                if gs["gdp"] is None or not (request.low_range <= gs["gdp"] <= request.high_range):
                    category = "low"

            if category:
                results.append({
                    "geometry_grid": gs["geometry"],
                    "grid_value": gs["grid_value"],
                    "category": category
                })

        return {
            "data": results,
            "thresholds": thresholds,
            "low_range": request.low_range,
            "high_range": request.high_range
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Get Facilities Data ---
@app.get("/facilities")
def get_facilities(types: str = Query(..., description="Comma-separated facility types (e.g. hotel,sekolah)")):
    try:
        dfs = []
        requested_types = [t.strip().lower() for t in types.split(",")]

        for ftype in requested_types:
            if ftype not in FACILITY_CONFIG:
                raise HTTPException(status_code=400, detail=f"Unknown facility type: {ftype}")

            config = FACILITY_CONFIG[ftype]
            q = f"""
                SELECT smid as id, {config['name_col']} as nama, '{ftype}' as type,
                       ST_Y({config['geom_col']}) as latitude,
                       ST_X({config['geom_col']}) as longitude
                FROM "{config['table']}"
            """
            dfs.append(pd.read_sql_query(q, con=engine_dummy_bps))

        if not dfs:
            return []

        df = pd.concat(dfs, ignore_index=True)
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Buffer Result ---
@app.post("/buffer-result")
def buffer_result(req: BufferRequest):
    # --- Step 1: Convert inputs to Shapely ---
    buffer_shapes = [shape(p) for p in req.buffer_polygons]
    recommended_shapes = [shape(p) for p in req.recommended_area]

    # Merge buffer polygons into one geometry
    buffer_union = unary_union(buffer_shapes)

    # Crop recommended areas (remove overlaps with buffer)
    cropped = [r.difference(buffer_union) for r in recommended_shapes if not r.is_empty]

    # --- Step 2: Fetch LahanKosong_P filtered by cropped polygons ---
    lahan_kosong = []
    with engine_dummy_bps.connect() as conn:
        for c in cropped:
            if c.is_empty:
                continue

            # Convert cropped polygon to WKT
            cropped_wkt = c.wkt  

            query = text("""
                SELECT smid, smuserid, ST_AsGeoJSON(smgeometry) as geometry,
                       userid, namobj, remark, kdprov, kdkab, kdkec,
                       nmprov, nmkab, nmkec, region_code
                FROM "LahanKosong_P"
                WHERE ST_Intersects(
                    smgeometry,
                    ST_GeomFromText(:cropped_wkt, 4326)
                );
            """)

            result = conn.execute(query, {"cropped_wkt": cropped_wkt}).fetchall()

            for row in result:
                lahan_kosong.append({
                    "smid": row.smid,
                    "smuserid": row.smuserid,
                    "geometry": json.loads(row.geometry),
                    "userid": row.userid,
                    "namobj": row.namobj,
                    "remark": row.remark,
                    "kdprov": row.kdprov,
                    "kdkab": row.kdkab,
                    "kdkec": row.kdkec,
                    "nmprov": row.nmprov,
                    "nmkab": row.nmkab,
                    "nmkec": row.nmkec,
                    "region_code": row.region_code
                })

    # --- Step 3: Return cropped polygons & lahan kosong ---
    return {
        "cropped_polygons": [mapping(c) for c in cropped if not c.is_empty],
        "lahan_kosong": lahan_kosong
    }

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