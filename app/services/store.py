import json
from sqlalchemy import text
from app.database import engine_dummy_bps
from app.models.requests import AnalysisResultRequest, GridStoreRequest

def store_grid_score(request: GridStoreRequest):
    with engine_dummy_bps.begin() as conn:
        query = text("""
            INSERT INTO grid_scores (
                nama_layer,
                kode_provinsi,
                kode_kota_kabupaten,
                kode_kecamatan,
                high_range,
                medium_range,
                low_range,
                selected_facilites,
                grid_geometries
            )
            VALUES (
                :nama_layer,
                :kode_provinsi,
                :kode_kota_kabupaten,
                :kode_kecamatan,
                :high_range,
                :medium_range,
                :low_range,
                CAST(:selected_facilites AS JSONB),
                CAST(:grid_geometries AS JSONB)
            )
            RETURNING id
        """)

        result = conn.execute(query, {
            "nama_layer": request.nama_layer,
            "kode_provinsi": request.kode_provinsi,
            "kode_kota_kabupaten": request.kode_kota_kabupaten,
            "kode_kecamatan": request.kode_kecamatan,
            "high_range": request.high_range,
            "medium_range": request.medium_range,
            "low_range": request.low_range,
            "selected_facilites": json.dumps([f.dict() for f in request.selected_facilites]),
            "grid_geometries": json.dumps([g.dict() for g in request.grid_geometries]),
        })
        inserted_id = result.scalar_one()
        return {"message": "Grid score stored successfully", "id": inserted_id}
    
def store_analysis_result(request: AnalysisResultRequest):
    with engine_dummy_bps.begin() as conn:
        query = text("""
            INSERT INTO analysis_results (
                nama_layer,
                lahan_kosong,
                selected_facilites,
                grid_geometries
            )
            VALUES (
                :nama_layer,
                CAST(:lahan_kosong AS JSONB),
                CAST(:selected_facilites AS JSONB),
                CAST(:grid_geometries AS JSONB)
            )
            RETURNING id
        """)

        result = conn.execute(query, {
            "nama_layer": request.nama_layer,
            "lahan_kosong": json.dumps(request.lahan_kosong),
            "selected_facilites": json.dumps(request.selected_facilites),
            "grid_geometries": json.dumps(request.grid_geometries),
        })
        inserted_id = result.scalar_one()
        return {"message": "Analysis result stored successfully", "id": inserted_id}