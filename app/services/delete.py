import logging
from sqlalchemy import text
from app.database import engine_dummy_bps

logger = logging.getLogger(__name__)

def delete_analysis_result(analysis_id: int) -> dict:
    """
    Delete an analysis result by ID.
    Returns success message or error details.
    """
    try:
        with engine_dummy_bps.connect() as conn:
            # First check if the record exists
            result = conn.execute(text("""
                SELECT id FROM analysis_results WHERE id = :id
            """), {"id": analysis_id})
            result = result.mappings().first()
            
            if not result:
                return {
                    "success": False,
                    "message": f"Analysis result with ID {analysis_id} not found",
                    "deleted_id": analysis_id
                }
            
            # Delete the record
            conn.execute(text("""
                DELETE FROM analysis_results WHERE id = :id
            """), {"id": analysis_id})
            
            # Commit the transaction
            conn.commit()
            
            logger.info(f"Successfully deleted analysis result with ID: {analysis_id}")
            
            return {
                "success": True,
                "message": f"Analysis result with ID {analysis_id} deleted successfully",
                "deleted_id": analysis_id
            }
            
    except Exception as e:
        logger.error(f"Error deleting analysis result {analysis_id}: {str(e)}")
        return {
            "success": False,
            "message": f"Error deleting analysis result: {str(e)}",
            "deleted_id": analysis_id
        }

def delete_grid_score(grid_id: int) -> dict:
    """
    Delete a grid score by ID.
    Returns success message or error details.
    """
    try:
        with engine_dummy_bps.connect() as conn:
            # First check if the record exists
            result = conn.execute(text("""
                SELECT id FROM grid_scores WHERE id = :id
            """), {"id": grid_id})
            result = result.mappings().first()
            
            if not result:
                return {
                    "success": False,
                    "message": f"Grid score with ID {grid_id} not found",
                    "deleted_id": grid_id
                }
            
            # Delete the record
            conn.execute(text("""
                DELETE FROM grid_scores WHERE id = :id
            """), {"id": grid_id})
            
            # Commit the transaction
            conn.commit()
            
            logger.info(f"Successfully deleted grid score with ID: {grid_id}")
            
            return {
                "success": True,
                "message": f"Grid score with ID {grid_id} deleted successfully",
                "deleted_id": grid_id
            }
            
    except Exception as e:
        logger.error(f"Error deleting grid score {grid_id}: {str(e)}")
        return {
            "success": False,
            "message": f"Error deleting grid score: {str(e)}",
            "deleted_id": grid_id
        }