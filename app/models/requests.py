from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class FeatureScores(BaseModel):
    gdp: float

class GridData(BaseModel):
    geometry_grid: Dict[str, Any]

class RangeSpec(BaseModel):
    min_value: float
    max_value: float
    
class ConsideredParamSpec(BaseModel):
    invert: bool = False
    weight: float

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
    provinsi: Optional[str] = None
    kabupaten: Optional[str] = None
    kecamatan: List[str] = []

class Thresholds(BaseModel):
    composite: List[float]
    explanation: str

class GridStoreRequest(BaseModel):
    nama_layer: str
    deskripsi_layer: Optional[str] = None
    kode_provinsi: str
    kode_kota_kabupaten: str
    kode_kecamatan: List[str]
    thresholds: Thresholds
    grid_geometries: List[GridGeometry]
    mandatory_parameters: Dict[str, RangeSpec]
    optional_parameters: Optional[Dict[str, ConsideredParamSpec]] = None

class AnalysisResultRequest(BaseModel):
    nama_layer: str
    deskripsi_layer: Optional[str] = None
    kode_provinsi: str
    kode_kota_kabupaten: str
    kode_kecamatan: List[str]
    lahan_kosong: List[Dict[str, Any]]
    selected_facilites: List[Dict[str, Any]]
    grid_geometries: List[Dict[str, Any]]
    grid_layer_name: Optional[str] = None
    ukuran_buffer: Optional[float] = None
    selected_fasilitas: Optional[str] = None

class SuitabilityRequest(BaseModel):
    data: List[GridData]
    mandatory_parameters: Dict[str, RangeSpec]
    optional_parameters: Dict[str, ConsideredParamSpec]