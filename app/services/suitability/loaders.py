# app/services/suitability/loaders.py
from app.services.suitability.layer_specs import geodf_for_code, build_layer_specs

def load_layer(code: str):
    return geodf_for_code(code)

def all_layer_specs():
    return build_layer_specs()
