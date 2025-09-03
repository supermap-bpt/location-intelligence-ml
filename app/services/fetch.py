import json
from sqlalchemy import text
from app.database import engine_dummy_bps, engine
from typing import Dict, List, Any


def safe_json_load(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    return json.loads(value)


def _feature(geometry_geojson: str, properties: Dict) -> Dict:
    return {
        "type": "Feature",
        "geometry": safe_json_load(geometry_geojson) if isinstance(geometry_geojson, str) else geometry_geojson,
        "properties": properties,
    }


def _feature_collection(features: List[Dict]) -> Dict:
    return {"type": "FeatureCollection", "features": features}

def _fetch_kecamatan_features_by_codes(codes: List[str]) -> Dict[str, Dict]:
    """
    Fetch kecamatan polygons for a list of kode_kecamatan from the *engine_query* DB.
    Returns a mapping: kode_kecamatan -> GeoJSON Feature
    """
    if not codes:
        return {}

    # Deduplicate for query
    unique_codes = sorted(set([c for c in codes if c]))

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT
                k.kode_kecamatan,
                k.nama_kecamatan,
                ST_AsGeoJSON(k.geom) AS geom
            FROM kecamatan k
            WHERE k.kode_kecamatan = ANY(:codes)
        """), {"codes": unique_codes}).mappings().all()

    return {
        r["kode_kecamatan"]: _feature(
            r["geom"],
            {
                "kode_kecamatan": r["kode_kecamatan"],
                "nama_kecamatan": r["nama_kecamatan"],
            },
        )
        for r in rows
    }


# -------- GRID SCORES --------
def get_grid_score(grid_id: int):
    # single record variant (also includes kecamatan polygons)
    with engine_dummy_bps.begin() as conn:
        r = conn.execute(text("""
            SELECT *
            FROM grid_scores
            WHERE id = :id
        """), {"id": grid_id}).mappings().first()

    if not r:
        return {"message": "Grid score not found"}

    codes = [str(c) for c in (safe_json_load(r["kode_kecamatan"]) or [])]
    code_to_feature = _fetch_kecamatan_features_by_codes(codes)

    return {
        "id": r["id"],
        "nama_layer": r["nama_layer"],
        "kode_provinsi": r["kode_provinsi"],
        "kode_kota_kabupaten": r["kode_kota_kabupaten"],
        "kode_kecamatan": codes,
        "thresholds": safe_json_load(r["thresholds"]),
        "low_range_gdp": r["low_range_gdp"],
        "high_range_gdp": r["high_range_gdp"],
        "grid_geometries": safe_json_load(r["grid_geometries"]),
        "feature_scores": safe_json_load(r["feature_scores"]),
        "weights_applied": safe_json_load(r["weights_applied"]),
        "kecamatan_regions": _feature_collection([code_to_feature[c] for c in codes if c in code_to_feature]),
        "created_at": r["created_at"],
    }

def get_all_grid_scores():
    with engine_dummy_bps.begin() as conn:
        query = text("""
            SELECT *
            FROM grid_scores
            ORDER BY created_at DESC
        """)
        results = conn.execute(query).mappings().all()

        return [
            {
                "id": r["id"],
                "nama_layer": r["nama_layer"],
                "kode_provinsi": r["kode_provinsi"],
                "kode_kota_kabupaten": r["kode_kota_kabupaten"],
                "kode_kecamatan": safe_json_load(r["kode_kecamatan"]),
                "thresholds": safe_json_load(r["thresholds"]),
                "low_range_gdp": r["low_range_gdp"],
                "high_range_gdp": r["high_range_gdp"],
                "grid_geometries": safe_json_load(r["grid_geometries"]),
                "feature_scores": safe_json_load(r["feature_scores"]),
                "weights_applied": safe_json_load(r["weights_applied"]),
                "created_at": r["created_at"],
            }
            for r in results
        ]


# -------- ANALYSIS RESULTS --------
def get_analysis_result(result_id: int):
    with engine_dummy_bps.begin() as conn:
        query = text("""
            SELECT *
            FROM analysis_results
            WHERE id = :id
        """)
        result = conn.execute(query, {"id": result_id}).mappings().first()
        if not result:
            return {"message": "Analysis result not found"}

        return {
            "id": result["id"],
            "nama_layer": result["nama_layer"],
            "lahan_kosong": safe_json_load(result["lahan_kosong"]),
            "selected_facilites": safe_json_load(result["selected_facilites"]),
            "grid_geometries": safe_json_load(result["grid_geometries"]),
            "feature_scores": safe_json_load(result["feature_scores"]),
            "weights_applied": safe_json_load(result["weights_applied"]),
            "created_at": result["created_at"],
        }


def get_all_analysis_results():
    with engine_dummy_bps.begin() as conn:
        query = text("""
            SELECT *
            FROM analysis_results
            ORDER BY created_at DESC
        """)
        results = conn.execute(query).mappings().all()

        return [
            {
                "id": r["id"],
                "nama_layer": r["nama_layer"],
                "lahan_kosong": safe_json_load(r["lahan_kosong"]),
                "selected_facilites": safe_json_load(r["selected_facilites"]),
                "grid_geometries": safe_json_load(r["grid_geometries"]),
                "feature_scores": safe_json_load(r["feature_scores"]),
                "weights_applied": safe_json_load(r["weights_applied"]),
                "created_at": r["created_at"],
            }
            for r in results
        ]