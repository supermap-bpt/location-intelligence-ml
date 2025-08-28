from pydantic import BaseModel
from typing import Optional, Dict

class HotelItem(BaseModel):
    nama: str
    geometry: Optional[Dict]

class PendidikanItem(BaseModel):
    namobj: str
    geometry: Optional[Dict]

class PusatPerbelanjaanItem(BaseModel):
    namobj: str
    geometry: Optional[Dict]

class RumahSakitItem(BaseModel):
    namobj: str
    geometry: Optional[Dict]
