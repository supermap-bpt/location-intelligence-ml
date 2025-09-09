from fastapi import APIRouter, HTTPException, status
from app.services.delete import delete_analysis_result, delete_grid_score

router = APIRouter(tags=["Delete"])

@router.delete("/analysis-result/{result_id}", 
               status_code=status.HTTP_200_OK,
               summary="Delete an analysis result by ID",
               description="Permanently delete an analysis result from the database.")
async def delete_analysis_result_endpoint(result_id: int):
    """
    Delete a specific analysis result by its ID.
    
    - **result_id**: The ID of the analysis result to delete
    """
    try:
        # FIX: Add 'await' here too
        result = delete_analysis_result(result_id)
        
        if not result["success"]:
            if "not found" in result["message"].lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result["message"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result["message"]
                )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
        
@router.delete("/grid-score/{grid_id}",
               status_code=status.HTTP_200_OK,
               summary="Delete a grid score by ID",
               description="Permanently delete a grid score from the database.")
async def delete_grid_score_endpoint(grid_id: int):
    """
    Delete a specific grid score by its ID.
    
    - **grid_id**: The ID of the grid score to delete
    """
    try:
        # FIX: Add 'await' before the async service function call
        result = delete_grid_score(grid_id)
        
        if not result["success"]:
            if "not found" in result["message"].lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result["message"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result["message"]
                )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )