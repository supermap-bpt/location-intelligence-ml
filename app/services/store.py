import json
from sqlalchemy import text
from app.database import engine
from app.models.requests import AnalysisResultRequest, GridStoreRequest


def store_grid_score(request: GridStoreRequest):
    try:
        with engine.begin() as conn:
            # Extract weights_applied from optional_parameters as a convenience
            weights_applied = None
            if request.optional_parameters:
                weights_applied = {
                    param_name: param_spec.weight
                    for param_name, param_spec in request.optional_parameters.items()
                }

            query = text("""
                INSERT INTO grid_scores (
                    nama_layer,
                    deskripsi_layer,
                    kode_provinsi,
                    kode_kota_kabupaten,
                    kode_kecamatan,
                    thresholds,
                    mandatory_parameters,
                    optional_parameters,
                    weights_applied,
                    grid_geometries
                )
                VALUES (
                    :nama_layer,
                    :deskripsi_layer,
                    :kode_provinsi,
                    :kode_kota_kabupaten,
                    CAST(:kode_kecamatan AS JSONB),
                    CAST(:thresholds AS JSONB),
                    CAST(:mandatory_parameters AS JSONB),
                    CAST(:optional_parameters AS JSONB),
                    CAST(:weights_applied AS JSONB),
                    CAST(:grid_geometries AS JSONB)
                )
                RETURNING id
            """)

            result = conn.execute(query, {
                "nama_layer": request.nama_layer,
                "deskripsi_layer": request.deskripsi_layer,
                "kode_provinsi": request.kode_provinsi,
                "kode_kota_kabupaten": request.kode_kota_kabupaten,
                "kode_kecamatan": json.dumps(request.kode_kecamatan),
                "thresholds": json.dumps(request.thresholds.model_dump()),
                "mandatory_parameters": json.dumps({k: v.model_dump() for k, v in request.mandatory_parameters.items()}),
                "optional_parameters": json.dumps({k: v.model_dump() for k, v in request.optional_parameters.items()}) if request.optional_parameters else None,
                "weights_applied": json.dumps(weights_applied) if weights_applied else None,
                "grid_geometries": json.dumps([g.model_dump() for g in request.grid_geometries]),
            })
            inserted_id = result.scalar_one()
            return {"message": "Grid score stored successfully", "id": inserted_id}
    except Exception as e:
        import traceback
        print("❌ ERROR store_grid_score:", e)
        traceback.print_exc()
        raise

def store_analysis_result(request: AnalysisResultRequest):
    with engine.begin() as conn:
        query = text("""
            INSERT INTO analysis_results (
                nama_layer,
                deskripsi_layer,
                grid_layer_name,
                kode_provinsi,
                kode_kota_kabupaten,
                kode_kecamatan,
                lahan_kosong,
                selected_facilites,
                clip_geometries,
                ukuran_buffer,
                selected_fasilitas
            )
            VALUES (
                :nama_layer,
                :deskripsi_layer,
                :grid_layer_name,
                :kode_provinsi,
                :kode_kota_kabupaten,
                CAST(:kode_kecamatan AS JSONB),
                CAST(:lahan_kosong AS JSONB),
                CAST(:selected_facilites AS JSONB),
                CAST(:clip_geometries AS JSONB),
                :ukuran_buffer,
                :selected_fasilitas
            )
            RETURNING id
        """)

        result = conn.execute(query, {
            "nama_layer": request.nama_layer,
            "deskripsi_layer": request.deskripsi_layer,
            "grid_layer_name": request.grid_layer_name,
            "kode_provinsi": request.kode_provinsi,
            "kode_kota_kabupaten": request.kode_kota_kabupaten,
            "kode_kecamatan": json.dumps(request.kode_kecamatan),
            "lahan_kosong": json.dumps(request.lahan_kosong),
            "selected_facilites": json.dumps(request.selected_facilites),
            "clip_geometries": json.dumps(request.grid_geometries),
            "ukuran_buffer": request.ukuran_buffer,
            "selected_fasilitas": request.selected_fasilitas
        })
        inserted_id = result.scalar_one()
        return {"message": "Analysis result stored successfully", "id": inserted_id}