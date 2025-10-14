# app/routers/parameters.py
from fastapi import APIRouter, HTTPException, Query
from app.services import parameters

router = APIRouter()

@router.get("/tree")
def get_parameter_tree():
    return parameters.list_parameter_tree()

@router.get("/classifications")
def get_classifications():
    return parameters.list_classifications()

@router.get("/categories")
def get_categories(classification_id: int = Query(None)):
    return parameters.list_categories(classification_id)

@router.get("/list")
def get_parameters(q: str = Query(None), category_id: int = Query(None)):
    return parameters.list_parameters(q, category_id)

# --- your new static routes go BEFORE the {param_id} route ---
@router.get("/wajib-groups")
def http_list_wajib_groups():
    return parameters.list_wajib_groups()

@router.get("/wajib-tree")
def http_list_wajib_tree():
    return parameters.list_wajib_tree()

# Put this LAST (or change to /id/{param_id})
@router.get("/id/{param_id}")   # <- alternative to avoid any future collisions
def get_parameter(param_id: int):
    result = parameters.get_parameter(param_id)
    if not result:
        raise HTTPException(status_code=404, detail="Parameter not found")
    return result

@router.get("/dipertimbangkan-groups")
def http_list_dipertimbangkan_groups():
    return parameters.list_dipertimbangkan_groups()

@router.get("/dipertimbangkan-tree")
def http_list_dipertimbangkan_tree():
    return parameters.list_dipertimbangkan_tree()

# keep the id route last or namespaced:
@router.get("/id/{param_id}")
def get_parameter(param_id: int):
    row = parameters.get_parameter(param_id)
    if not row:
        raise HTTPException(status_code=404, detail="Parameter not found")
    return row
