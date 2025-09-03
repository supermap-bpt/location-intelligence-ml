import json
from sqlalchemy import text
from app.database import engine_dummy_bps


def safe_json_load(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    return json.loads(value)


# -------- GRID SCORES --------
def get_grid_score(grid_id: int):
    with engine_dummy_bps.begin() as conn:
        query = text("""
            SELECT *
            FROM grid_scores
            WHERE id = :id
        """)
        result = conn.execute(query, {"id": grid_id}).mappings().first()
        if not result:
            return {"message": "Grid score not found"}

        return {
            "id": result["id"],
            "nama_layer": result["nama_layer"],
            "kode_provinsi": result["kode_provinsi"],
            "kode_kota_kabupaten": result["kode_kota_kabupaten"],
            "kode_kecamatan": safe_json_load(result["kode_kecamatan"]),
            "thresholds": safe_json_load(result["thresholds"]),
            "low_range_gdp": result["low_range_gdp"],
            "high_range_gdp": result["high_range_gdp"],
            "grid_geometries": safe_json_load(result["grid_geometries"]),
            "feature_scores": safe_json_load(result["feature_scores"]),
            "weights_applied": safe_json_load(result["weights_applied"]),
            "created_at": result["created_at"],
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