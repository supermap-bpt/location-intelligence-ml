from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class GridData(BaseModel):
    geometry_grid: Dict[str, Any]
    weights: Dict[str, float]

class BatchRequest(BaseModel):
    data: List[GridData]
    low_range: float
    high_range: float

class BufferRequest(BaseModel):
    buffer_polygons: List[Any]
    recommended_area: List[Any]

class Facility(BaseModel):
    id: int
    type: str
    name: str
    geometry: Dict[str, Any]

class GridGeometry(BaseModel):
    id: int
    predicted_class: Optional[str]
    grid_value: float
    geometry: Dict[str, Any]
    feature_scores: Dict[str, float]
    weights_applied: Dict[str, float]

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
    lahan_kosong: List[Dict[str, Any]]
    selected_facilites: List[Dict[str, Any]]
    grid_geometries: List[Dict[str, Any]]