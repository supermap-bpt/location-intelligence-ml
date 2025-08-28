from fastapi import APIRouter, HTTPException
from app.models.requests import BufferRequest
from app.services.buffer import buffer_result_service

router = APIRouter()

@router.post("/result")
def buffer_result(req: BufferRequest):
    try:
        return buffer_result_service(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
