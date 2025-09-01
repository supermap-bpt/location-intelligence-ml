from fastapi import APIRouter, Query, HTTPException
from typing import List
from app.models.facilities import HotelItem, PendidikanItem, PusatPerbelanjaanItem, RumahSakitItem
from app.services.geodata import (
    get_facilities_service, 
    get_hotels_service, 
    get_pendidikan_service, 
    get_pusatperbelanjaan_service, 
    get_rumahsakit_service
)

router = APIRouter()

@router.get("", summary="Get facilities by type")
def get_facilities(types: str = Query(..., description="Comma-separated facility types (e.g. hotel,sekolah)")):
    return get_facilities_service(types)

# Ubah parameter menjadi List[str] untuk multiple values
@router.get("/hotel", response_model=list[HotelItem])
def get_hotel(nmkec: List[str] = Query(..., description="List of kecamatan names")):
    return get_hotels_service(nmkec)

@router.get("/pendidikan", response_model=list[PendidikanItem])
def get_pendidikan(nmkec: List[str] = Query(..., description="List of kecamatan names")):
    return get_pendidikan_service(nmkec)

@router.get("/pusatperbelanjaan", response_model=list[PusatPerbelanjaanItem])
def get_pusatperbelanjaan(nmkec: List[str] = Query(..., description="List of kecamatan names")):
    return get_pusatperbelanjaan_service(nmkec)

@router.get("/rumahsakit", response_model=list[RumahSakitItem])
def get_rumahsakit(nmkec: List[str] = Query(..., description="List of kecamatan names")):
    return get_rumahsakit_service(nmkec)