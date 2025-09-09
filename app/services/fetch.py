import json
from typing import Dict, List, Any, Optional

from sqlalchemy import text

from app.database import engine_dummy_bps, engine


# =========================
# Helpers & Config
# =========================

def safe_json_load(value: Any) -> Optional[Any]:
    """
    Safely parse JSON from string, returning original value if parsing fails.
    Handles empty strings, None values, and invalid JSON.
    """
    if value is None:
        return None
    
    if isinstance(value, str):
        # Handle empty strings
        if not value.strip():
            return None
        
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Return original string if it's not valid JSON
            return value
    
    # Return non-string values as-is
    return value


def _feature(geometry_geojson: str, properties: Dict) -> Dict:
    return {
        "type": "Feature",
        "geometry": safe_json_load(geometry_geojson) if isinstance(geometry_geojson, str) else geometry_geojson,
        "properties": properties,
    }


def _feature_collection(features: List[Dict]) -> Dict:
    return {"type": "FeatureCollection", "features": features}


def _qident(schema: str, table: str) -> str:
    """Quote identifiers to preserve case and ensure correct schema resolution."""
    return f'"{schema}"."{table}"'


# ---- Admin (kecamatan) lives on `engine`
_ADMIN_BOUNDS = ("public", "kecamatan")     # change schema/table if needed
_ADMIN_CODE_COL = "kode_kecamatan"
_ADMIN_GEOM_COL = "geom"                    # change if your geom column differs


# =========================
# Fetch admin polygons (engine)
# =========================

def _fetch_kecamatan_features_by_codes(codes: List[str]) -> Dict[str, Dict]:
    """
    Fetch kecamatan polygons for a list of kode_kecamatan from the spatial DB (engine).
    Returns a mapping: kode_kecamatan -> GeoJSON Feature
    """
    if not codes:
        return {}

    unique_codes = sorted({c for c in codes if c})

    adm_schema, adm_table = _ADMIN_BOUNDS
    adm_fqtn = _qident(adm_schema, adm_table)

    with engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT
                k.{_ADMIN_CODE_COL} AS kode_kecamatan,
                k.nama_kecamatan,
                ST_AsGeoJSON(k.{_ADMIN_GEOM_COL}) AS geom
            FROM {adm_fqtn} k
            WHERE k.{_ADMIN_CODE_COL} = ANY(:codes)
        """), {"codes": unique_codes}).mappings().all()

    return {
        r["kode_kecamatan"]: _feature(
            r["geom"],
            {
                "kode_kecamatan": r["kode_kecamatan"],
                "nama_kecamatan": r.get("nama_kecamatan"),
            },
        )
        for r in rows
    }


# =========================
# Facilities config (engine_dummy_bps)
# =========================
# Map each facility to (schema, table, geom_col, kec_name_col_in_poi)
_FACILITY_TABLES = {
    "sekolah": ("public", "Sekolah_P", "smgeometry", "nmkec"),
    "hotel": ("public", "Hotel_P", "smgeometry", "nmkec"),
    "pusatperbelanjaan": ("public", "PusatPerbelanjaan_P", "smgeometry", "nmkec"),
    "rumahsakit": ("public", "RumahSakit_P", "smgeometry", "nmkec"),
}


# =========================
# Fetch facilities (attribute filter by kdkec)
# =========================

def _fetch_facilities_in_kecamatan(codes: List[str], facility_key: Optional[str]) -> Optional[List[Dict]]:
    """
    Filter POI by *kecamatan name*:
      public.kecamatan.nama_kecamatan  (engine)
        ↔ POI.{nmkec}                  (engine_dummy_bps)

    Returns a list of GeoJSON Feature dicts (NOT a FeatureCollection).
    """
    if not facility_key or not codes:
        return None

    fac = _FACILITY_TABLES.get(facility_key)
    if not fac:
        return None

    fac_schema, fac_table, geom_col, poi_kec_name_col = fac
    fac_fqtn = _qident(fac_schema, fac_table)

    # 1) Get the kecamatan *names* for the provided codes from the admin DB (engine)
    adm_schema, adm_table = _ADMIN_BOUNDS
    adm_fqtn = _qident(adm_schema, adm_table)

    unique_codes = sorted({c for c in codes if c})

    with engine.begin() as conn:
        name_rows = conn.execute(text(f"""
            SELECT
                { _ADMIN_CODE_COL } AS kode_kecamatan,
                nama_kecamatan
            FROM {adm_fqtn}
            WHERE { _ADMIN_CODE_COL } = ANY(:codes)
        """), {"codes": unique_codes}).mappings().all()

    if not name_rows:
        return []

    # Normalize names to UPPER for robust comparison
    nmkec_names = sorted({
        (r["nama_kecamatan"] or "").strip().upper()
        for r in name_rows
        if r.get("nama_kecamatan")
    })

    if not nmkec_names:
        return []

    # 2) Query POI by kecamatan name (on engine_dummy_bps)
    # Use UPPER() on the POI side and compare with the pre-uppercased list
    sql = f"""
        SELECT
            -- Echo the matched POI name column as the filter key
            f.{poi_kec_name_col}        AS _nama_kecamatan_filter,
            ST_AsGeoJSON(f.{geom_col})  AS _geom_geojson,
            f.*
        FROM {fac_fqtn} AS f
        WHERE UPPER(f.{poi_kec_name_col}) = ANY(:nmkec_names)
    """

    with engine_dummy_bps.begin() as conn:
        rows = conn.execute(
            text(sql),
            {"nmkec_names": nmkec_names},
        ).mappings().all()

    features: List[Dict[str, Any]] = []
    for r in rows:
        gj = r.get("_geom_geojson")
        if not gj:
            continue
        props = {k: v for k, v in r.items() if k != "_geom_geojson"}
        features.append(_feature(gj, props))

    return features

# =========================
# Public API functions
# =========================

def get_grid_score(grid_id: int, facility_key: Optional[str] = None):
    """
    Fetch a single grid score record (from engine_dummy_bps) and enrich response with:
      - kecamatan polygon FeatureCollection (from engine)
      - optional facilities list (array of GeoJSON Features) filtered by kdkec (from engine_dummy_bps)
    """
    with engine_dummy_bps.begin() as conn:
        r = conn.execute(text("""
            SELECT *
            FROM grid_scores
            WHERE id = :id
        """), {"id": grid_id}).mappings().first()

    if not r:
        return {"message": "Grid score not found"}

    codes = [str(c) for c in (safe_json_load(r["kode_kecamatan"]) or [])]
    code_to_feature = _fetch_kecamatan_features_by_codes(codes)

    response: Dict[str, Any] = {
        "id": r["id"],
        "nama_layer": r["nama_layer"],
        "kode_provinsi": r["kode_provinsi"],
        "kode_kota_kabupaten": r["kode_kota_kabupaten"],
        "kode_kecamatan": codes,
        "thresholds": safe_json_load(r["thresholds"]),
        "low_range_gdp": r["low_range_gdp"],
        "high_range_gdp": r["high_range_gdp"],
        "grid_geometries": safe_json_load(r["grid_geometries"]),
        "feature_scores": safe_json_load(r["feature_scores"]),
        "weights_applied": safe_json_load(r["weights_applied"]),
        "kecamatan_regions": _feature_collection([code_to_feature[c] for c in codes if c in code_to_feature]),
        "created_at": r["created_at"],
    }

    facilities_list = _fetch_facilities_in_kecamatan(codes, facility_key)
    if facilities_list is not None:
        response["facility_requested"] = facility_key
        response["facilities"] = facilities_list  # plain array (not FeatureCollection)

    return response


def get_all_grid_scores():
    with engine_dummy_bps.begin() as conn:
        rows = conn.execute(text("""
            SELECT
                id,
                nama_layer,
                kode_provinsi,
                kode_kota_kabupaten,
                thresholds,
                created_at
            FROM grid_scores
            ORDER BY created_at DESC, id DESC
        """)).mappings().all()

    return [
        {
            **dict(r),
            "thresholds": safe_json_load(r["thresholds"]) if r.get("thresholds") else None
        }
        for r in rows
    ]

def get_analysis_result(analysis_id: int):
    with engine_dummy_bps.begin() as conn:
        r = conn.execute(text("""
            SELECT *
            FROM analysis_results
            WHERE id = :id
        """), {"id": analysis_id}).mappings().first()

    if not r:
        return {"message": "Analysis result not found"}

    return {k: (safe_json_load(v) if isinstance(v, str) else v) for k, v in dict(r).items()}


def get_all_analysis_results():
    with engine_dummy_bps.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, nama_layer, created_at
            FROM analysis_results
            ORDER BY created_at DESC, id DESC
        """)).mappings().all()
    return [dict(r) for r in rows]