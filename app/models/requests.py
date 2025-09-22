from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class GridData(BaseModel):
    geometry_grid: Dict[str, Any]
    weights: Dict[str, float]

class BatchRequest(BaseModel):
    data: List[GridData]
    low_range: float
    high_range: float

class BufferGrid(BaseModel):
    geometry: Dict[str, Any]
    properties: Dict[str, Any]

class BufferRequest(BaseModel):
    buffer_polygons: List[Any]
    recommended_area: List[BufferGrid]
    luas_area: Optional[float] = 0
    
class Facility(BaseModel):
    id: int
    type: str
    name: str
    geometry: Dict[str, Any]

class GridGeometry(BaseModel):
    id: str
    predicted_class: Optional[str]
    grid_value: float
    geometry: Dict[str, Any]
    feature_scores: Dict[str, float]
    weights_applied: Dict[str, float]
    provinsi: Optional[str] = None
    kabupaten: Optional[str] = None
    kecamatan: List[str] = []

class GridStoreRequest(BaseModel):
    nama_layer: str
    kode_provinsi: str             
    kode_kota_kabupaten: str          
    kode_kecamatan: List[str]
    low_range_gdp: float
    high_range_gdp: float
    thresholds: Dict[str, List[float]]
    grid_geometries: List[GridGeometry]

class AnalysisResultRequest(BaseModel):
    nama_layer: str
    kode_provinsi: str
    kode_kota_kabupaten: str
    kode_kecamatan: List[str]
    lahan_kosong: List[Dict[str, Any]]
    selected_facilites: List[Dict[str, Any]]
    grid_geometries: List[Dict[str, Any]]
    grid_layer_name: Optional[str] = None
    ukuran_buffer: Optional[float] = None
    selected_fasilitas: Optional[str] = None