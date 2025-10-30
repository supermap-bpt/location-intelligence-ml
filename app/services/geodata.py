from sqlalchemy import text
from app.database import engine, engine_dummy_bps
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
                        kdprov as kode_provinsi,
                        nmprov as nama_provinsi,
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
                        kdprov as kode_provinsi,
                        nmprov as nama_provinsi,
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
                        kdkab as kode_kota_kabupaten,
                        nmkab as nama_kota_kabupaten,
                        latitude,
                        longitude,
                        ST_AsGeoJSON(geom) AS geom_json
                    FROM kota_kabupaten
                    WHERE kdprov = :kode_prov
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
                        kdkab AS kode_kota_kabupaten,
                        nmkab AS nama_kota_kabupaten,
                        latitude,
                        longitude
                    FROM kota_kabupaten
                    WHERE kdprov = :kode_prov
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
                        kdkec AS kode_kecamatan,
                        nmkec AS nama_kecamatan,
                        latitude,
                        longitude,
                        ST_AsGeoJSON(geom) AS geom_json
                    FROM kecamatan
                    WHERE kdkab = :kode_kota
                    ORDER BY nmkec ASC
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
                        kdkec AS kode_kecamatan,
                        nmkec AS nama_kecamatan,
                        latitude,
                        longitude
                    FROM kecamatan
                    WHERE kdkab = :kode_kota
                    ORDER BY nmkec ASC
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
                        kddesa AS kode_kelurahan,
                        nmdesa AS nama_kelurahan,
                        latitude,
                        longitude,
                        ST_AsGeoJSON(geom) AS geom_json
                    FROM kelurahan_desa
                    WHERE kdkec = :kode_kec
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
                        kddes AS kode_kelurahan,
                        kdkec AS nama_kelurahan,
                        latitude,
                        longitude
                    FROM kelurahan_desa
                    WHERE kdkec = :kode_kec
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