from fastapi import APIRouter, HTTPException
from app.services.geodata import get_provinsi_service, get_kota_kabupaten_service, get_kecamatan_service, get_kelurahan_service, get_provinsi_no_rings_service, get_kota_kabupaten_no_rings_service, get_kecamatan_no_rings_service, get_kelurahan_no_rings_service

router = APIRouter()

@router.get("/provinsi")
def get_provinsi():
    return get_provinsi_service()

@router.get("/provinsi-no-rings")
def get_provinsi():
    return get_provinsi_no_rings_service()

@router.get("/kota-kabupaten")
def get_kota_kabupaten(kode_provinsi: str):
    return get_kota_kabupaten_service(kode_provinsi)

@router.get("/kota-kabupaten-no-rings")
def get_kota_kabupaten(kode_provinsi: str):
    return get_kota_kabupaten_no_rings_service(kode_provinsi)

@router.get("/kecamatan")
def get_kecamatan(kode_kota_kabupaten: str):
    return get_kecamatan_service(kode_kota_kabupaten)

@router.get("/kecamatan-no-rings")
def get_kecamatan(kode_kota_kabupaten: str):
    return get_kecamatan_no_rings_service(kode_kota_kabupaten)

@router.get("/kelurahan-desa")
def get_kelurahan(kode_kecamatan: str):
    return get_kelurahan_service(kode_kecamatan)

@router.get("/kelurahan-desa-no-rings")
def get_kelurahan(kode_kecamatan: str):
    return get_kelurahan_no_rings_service(kode_kecamatan)
