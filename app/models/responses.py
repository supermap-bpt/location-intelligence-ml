from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

class SuitabilityCategory(str, Enum):
    NOT_RECOMMENDED = "low"
    NEUTRAL = "medium"
    RECOMMENDED = "high"

class SuitabilityResponse(BaseModel):
    predicted_class: SuitabilityCategory
    confidence: float
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r2_score: float
    feature_scores: Dict[str, float]
    weights_applied: Dict[str, float]
    input_polygon: List[List[Tuple[float, float]]]
    timestamp: str
    grid_id: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    dependencies: dict
