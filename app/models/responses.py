from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

class SuitabilityCategory(str, Enum):
    NOT_RECOMMENDED = "low"
    NEUTRAL = "medium"
    RECOMMENDED = "high"

class SuitabilityResponse(BaseModel):
    """Legacy response model - deprecated"""
    predicted_class: str
    grid_value: float
    feature_scores: Dict[str, float]
    weights_applied: Dict[str, float]
    input_polygon: List[List[Tuple[float, float]]]
    timestamp: str
    grid_id: Optional[str] = None

class GridGeometryResponse(BaseModel):
    """Response model for individual grid geometry in suitability analysis"""
    id: str
    predicted_class: str
    grid_value: float
    geometry: Dict[str, Any]
    feature_scores: Dict[str, float]
    provinsi: Optional[str] = None
    kabupaten: Optional[str] = None
    kecamatan: List[str] = Field(default_factory=list)

class ThresholdsResponse(BaseModel):
    """Thresholds for suitability classification"""
    composite: List[float]
    explanation: str

class MandatoryParameterResponse(BaseModel):
    """Response model for mandatory parameter"""
    min_value: float
    max_value: float

class OptionalParameterResponse(BaseModel):
    """Response model for optional parameter"""
    weight: float
    invert: bool

class SuitabilityAnalysisResponse(BaseModel):
    """Complete response model for suitability analysis"""
    thresholds: ThresholdsResponse
    grid_geometries: List[GridGeometryResponse]
    mandatory_parameters: Dict[str, MandatoryParameterResponse]
    optional_parameters: Dict[str, OptionalParameterResponse]

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    dependencies: dict

class Parameter(BaseModel):
    id: int
    category_id: int
    code: str
    name: str
    description: Optional[str]
    isRequired: Optional[bool]
    data_type: Optional[str]
    unit: Optional[str]
    min_value: Optional[float]
    max_value: Optional[float]
    isActive: Optional[bool]
    default_value: Optional[str]

class ParameterCategory(BaseModel):
    id: int
    classification_id: int
    code: str
    name: str
    description: Optional[str]
    parameters: List[Parameter] = Field(default_factory=list)

class ParameterClassification(BaseModel):
    id: int
    code: str
    name: str
    description: Optional[str]
    ParameterCategories: List[ParameterCategory] = Field(default_factory=list)
