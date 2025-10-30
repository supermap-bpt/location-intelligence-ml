# app/routers/parameters.py
from fastapi import APIRouter, HTTPException, Query
from app.services import parameters

router = APIRouter()

@router.get("/groups", summary="List parameter groups with code & label")
def get_parameter_groups():
    return parameters.list_category_groups_with_items()