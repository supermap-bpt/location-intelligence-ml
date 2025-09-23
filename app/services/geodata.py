from sqlalchemy import text
from app.database import engine, engine_dummy_bps
from app.config import FACILITY_CONFIG
from typing import List

# ==================
# REGION ENDPOINTS
# ==================

from fastapi import HTTPException
from sqlalchemy import text
import json

def get_provinsi_service():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        kode_provinsi,
                        nama_provinsi,
                        latitude,
                        longitude,
                        ST_AsGeoJSON(geom) AS geom_json
                    FROM provinsi
                    ORDER BY nama_provinsi ASC
                """)
            ).fetchall()

            provinsi_list = []
            for row in result:
                geom_json = row._mapping.get("geom_json")
                try:
                    geom_obj = json.loads(geom_json) if geom_json else {}
                    rings = geom_obj.get("coordinates", [])
                except Exception:
                    rings = []

                provinsi_list.append({
                    "kode_provinsi": row._mapping["kode_provinsi"],
                    "nama_provinsi": row._mapping["nama_provinsi"],
                    "latitude": row._mapping["latitude"],
                    "longitude": row._mapping["longitude"],
                    "rings": rings
                })

            return provinsi_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_provinsi_no_rings_service():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        kode_provinsi,
                        nama_provinsi,
                        latitude,
                        longitude,
                        ST_X(ST_Centroid(geom)) AS longitude,  -- Ambil koordinat X dari centroid
                        ST_Y(ST_Centroid(geom)) AS latitude     -- Ambil koordinat Y dari centroid
                    FROM provinsi
                    ORDER BY nama_provinsi ASC
                """)
            ).fetchall()

            provinsi_list = []
            for row in result:
                provinsi_list.append({
                    "kode_provinsi": row._mapping["kode_provinsi"],
                    "nama_provinsi": row._mapping["nama_provinsi"],
                    "latitude": row._mapping["latitude"],  # Koordinat Y (latitude)
                    "longitude": row._mapping["longitude"],  # Koordinat X (longitude)
                })

            return provinsi_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_kota_kabupaten_service(kode_provinsi: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        kode_kota_kabupaten,
                        nama_kota_kabupaten,
                        latitude,
                        longitude,
                        ST_AsGeoJSON(geom) AS geom_json
                    FROM kota_kabupaten
                    WHERE kode_provinsi = :kode_prov
                    ORDER BY nama_kota_kabupaten ASC
                """),
                {"kode_prov": kode_provinsi}
            ).fetchall()

            kota_list = []
            for row in result:
                geom_json = row._mapping.get("geom_json")
                try:
                    geom_obj = json.loads(geom_json) if geom_json else {}
                    rings = geom_obj.get("coordinates", [])
                except Exception:
                    rings = []

                kota_list.append({
                    "kode_kota_kabupaten": row._mapping["kode_kota_kabupaten"],
                    "nama_kota_kabupaten": row._mapping["nama_kota_kabupaten"],
                    "latitude": row._mapping["latitude"],
                    "longitude": row._mapping["longitude"],
                    "rings": rings
                })

            return kota_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_kota_kabupaten_no_rings_service(kode_provinsi: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        kode_kota_kabupaten,
                        nama_kota_kabupaten,
                        latitude,
                        longitude
                    FROM kota_kabupaten
                    WHERE kode_provinsi = :kode_prov
                    ORDER BY nama_kota_kabupaten ASC
                """),
                {"kode_prov": kode_provinsi}
            ).fetchall()

            kota_list = []
            for row in result:
                kota_list.append({
                    "kode_kota_kabupaten": row._mapping["kode_kota_kabupaten"],
                    "nama_kota_kabupaten": row._mapping["nama_kota_kabupaten"],
                    "latitude": row._mapping["latitude"],
                    "longitude": row._mapping["longitude"]
                })

            return kota_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def get_kecamatan_service(kode_kota_kabupaten: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        kode_kecamatan,
                        nama_kecamatan,
                        latitude,
                        longitude,
                        ST_AsGeoJSON(geom) AS geom_json
                    FROM kecamatan
                    WHERE kode_kota_kabupaten = :kode_kota
                    ORDER BY nama_kecamatan ASC
                """),
                {"kode_kota": kode_kota_kabupaten}
            ).fetchall()

            kecamatan_list = []
            for row in result:
                geom_json = row._mapping.get("geom_json")
                try:
                    geom_obj = json.loads(geom_json) if geom_json else {}
                    rings = geom_obj.get("coordinates", [])
                except Exception:
                    rings = []

                kecamatan_list.append({
                    "kode_kecamatan": row._mapping["kode_kecamatan"],
                    "nama_kecamatan": row._mapping["nama_kecamatan"],
                    "latitude": row._mapping["latitude"],
                    "longitude": row._mapping["longitude"],
                    "rings": rings
                })

            return kecamatan_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_kecamatan_no_rings_service(kode_kota_kabupaten: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        kode_kecamatan,
                        nama_kecamatan,
                        latitude,
                        longitude
                    FROM kecamatan
                    WHERE kode_kota_kabupaten = :kode_kota
                    ORDER BY nama_kecamatan ASC
                """),
                {"kode_kota": kode_kota_kabupaten}
            ).fetchall()

            kecamatan_list = []
            for row in result:
                kecamatan_list.append({
                    "kode_kecamatan": row._mapping["kode_kecamatan"],
                    "nama_kecamatan": row._mapping["nama_kecamatan"],
                    "latitude": row._mapping["latitude"],
                    "longitude": row._mapping["longitude"]
                })

            return kecamatan_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_kelurahan_service(kode_kecamatan: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        kode_kelurahan_desa AS kode_kelurahan,
                        nama_kelurahan_desa AS nama_kelurahan,
                        latitude,
                        longitude,
                        ST_AsGeoJSON(geom) AS geom_json
                    FROM kelurahan_desa
                    WHERE kode_kecamatan = :kode_kec
                    ORDER BY nama_kelurahan ASC
                """),
                {"kode_kec": kode_kecamatan}
            ).fetchall()

            kelurahan_list = []
            for row in result:
                geom_json = row._mapping.get("geom_json")
                try:
                    geom_obj = json.loads(geom_json) if geom_json else {}
                    rings = geom_obj.get("coordinates", [])
                except Exception:
                    rings = []

                kelurahan_list.append({
                    "kode_kelurahan": row._mapping["kode_kelurahan"],
                    "nama_kelurahan": row._mapping["nama_kelurahan"],
                    "latitude": row._mapping["latitude"],
                    "longitude": row._mapping["longitude"],
                    "rings": rings
                })

            return kelurahan_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_kelurahan_no_rings_service(kode_kecamatan: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        kode_kelurahan_desa AS kode_kelurahan,
                        nama_kelurahan_desa AS nama_kelurahan,
                        latitude,
                        longitude
                    FROM kelurahan_desa
                    WHERE kode_kecamatan = :kode_kec
                    ORDER BY nama_kelurahan ASC
                """),
                {"kode_kec": kode_kecamatan}
            ).fetchall()

            kelurahan_list = []
            for row in result:
                kelurahan_list.append({
                    "kode_kelurahan": row._mapping["kode_kelurahan"],
                    "nama_kelurahan": row._mapping["nama_kelurahan"],
                    "latitude": row._mapping["latitude"],
                    "longitude": row._mapping["longitude"]
                })

            return kelurahan_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================
# FACILITIES ENDPOINTS
# ==================

def get_facilities_service(types: str):
    try:
        facilities = []
        requested_types = [t.strip().lower() for t in types.split(",")]

        with engine_dummy_bps.connect() as conn:
            for ftype in requested_types:
                if ftype not in FACILITY_CONFIG:
                    raise HTTPException(status_code=400, detail=f"Unknown facility type: {ftype}")

                cfg = FACILITY_CONFIG[ftype]
                query = text(f"""
                    SELECT 
                        smid AS id,
                        {cfg['name_col']} AS nama,
                        '{ftype}' AS type,
                        ST_Y({cfg['geom_col']}) AS latitude,
                        ST_X({cfg['geom_col']}) AS longitude
                    FROM "{cfg['table']}"
                """)
                result = conn.execute(query).fetchall()

                for row in result:
                    facilities.append({
                        "id": row._mapping["id"],
                        "nama": row._mapping["nama"],
                        "type": row._mapping["type"],
                        "latitude": row._mapping["latitude"],
                        "longitude": row._mapping["longitude"],
                    })

        return facilities

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def get_hotels_service(nmkec: List[str]):
    try:
        if not nmkec:
            raise HTTPException(status_code=400, detail="Minimal satu nama kecamatan harus disediakan")

        # Convert to lowercase for case-insensitive comparison
        kecamatan_list = [k.strip().lower() for k in nmkec if k.strip()]
        
        query = text("""
            SELECT namobj, ST_AsGeoJSON(smgeometry)::json AS geometry
            FROM "Hotel_P"
            WHERE LOWER(nmkec) = ANY(:nmkec_list)
        """)

        with engine_dummy_bps.connect() as conn:
            result = conn.execute(query, {"nmkec_list": kecamatan_list}).fetchall()

        return [dict(r._mapping) for r in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lakukan modifikasi yang sama untuk fungsi lainnya
def get_pendidikan_service(nmkec: List[str]):
    try:
        if not nmkec:
            raise HTTPException(status_code=400, detail="Minimal satu nama kecamatan harus disediakan")

        kecamatan_list = [k.strip().lower() for k in nmkec if k.strip()]
        
        query = text("""
            SELECT namobj, ST_AsGeoJSON(smgeometry)::json AS geometry
            FROM "Sekolah_P"
            WHERE LOWER(nmkec) = ANY(:nmkec_list)
        """)

        with engine_dummy_bps.connect() as conn:
            result = conn.execute(query, {"nmkec_list": kecamatan_list}).fetchall()

        return [dict(r._mapping) for r in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_pusatperbelanjaan_service(nmkec: List[str]):
    try:
        if not nmkec:
            raise HTTPException(status_code=400, detail="Minimal satu nama kecamatan harus disediakan")

        kecamatan_list = [k.strip().lower() for k in nmkec if k.strip()]
        
        query = text("""
            SELECT namobj, ST_AsGeoJSON(smgeometry)::json AS geometry
            FROM "PusatPerbelanjaan_P"
            WHERE LOWER(nmkec) = ANY(:nmkec_list)
        """)

        with engine_dummy_bps.connect() as conn:
            result = conn.execute(query, {"nmkec_list": kecamatan_list}).fetchall()

        return [dict(r._mapping) for r in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_rumahsakit_service(nmkec: List[str]):
    try:
        if not nmkec:
            raise HTTPException(status_code=400, detail="Minimal satu nama kecamatan harus disediakan")

        kecamatan_list = [k.strip().lower() for k in nmkec if k.strip()]
        
        query = text("""
            SELECT namobj, ST_AsGeoJSON(smgeometry)::json AS geometry
            FROM "RumahSakit_P"
            WHERE LOWER(nmkec) = ANY(:nmkec_list)
        """)

        with engine_dummy_bps.connect() as conn:
            result = conn.execute(query, {"nmkec_list": kecamatan_list}).fetchall()

        return [dict(r._mapping) for r in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))