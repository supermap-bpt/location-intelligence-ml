from datetime import datetime
from app.models.responses import SuitabilityResponse, SuitabilityCategory
from app.models.requests import BatchRequest
from app.config import METRICS

def batch_predict_service(request: BatchRequest):
    """
    Mocked suitability prediction service.
    In real implementation, load model & compute predictions here.
    """
    results = []
    for i, grid in enumerate(request.data):
        # Mock classification rule based on feature sum
        score_sum = sum(grid.feature_scores.values())
        if score_sum < request.low_range:
            category = SuitabilityCategory.NOT_RECOMMENDED
        elif score_sum > request.high_range:
            category = SuitabilityCategory.RECOMMENDED
        else:
            category = SuitabilityCategory.NEUTRAL

        resp = SuitabilityResponse(
            predicted_class=category,
            confidence=round(min(1.0, score_sum / 10.0), 4),
            mean_absolute_error=METRICS["MAE"],
            mean_squared_error=METRICS["MSE"],
            root_mean_squared_error=METRICS["RMSE"],
            r2_score=METRICS["R2"],
            feature_scores=grid.feature_scores,
            weights_applied=grid.weights,
            input_polygon=grid.geometry_grid.get("coordinates", []),
            timestamp=datetime.now().isoformat(),
            grid_id=f"grid_{i+1}"
        )
        results.append(resp.dict())
    return results