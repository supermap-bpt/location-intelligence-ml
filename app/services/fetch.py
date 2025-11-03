import json
from typing import Dict, List, Any, Optional

from sqlalchemy import text

from app.database import engine


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
_ADMIN_CODE_COL = "kdkec"
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
                k.nmkec as nama_kecamatan,
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
# Helpers: lookup names by kode
# =========================

def _get_provinsi_name(kode_provinsi: str) -> Optional[str]:
    if not kode_provinsi:
        return None
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT nmprov as nama_provinsi
            FROM public.provinsi
            WHERE kdprov = :kode
        """), {"kode": kode_provinsi}).mappings().first()
    return row["nama_provinsi"] if row else None


def _get_kota_kabupaten_name(kode_kota_kabupaten: str) -> Optional[str]:
    if not kode_kota_kabupaten:
        return None
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT nmkab as nama_kota_kabupaten
            FROM public.kota_kabupaten
            WHERE kdkab = :kode
        """), {"kode": kode_kota_kabupaten}).mappings().first()
    return row["nama_kota_kabupaten"] if row else None


def _get_kecamatan_names(codes: List[str]) -> Dict[str, str]:
    """Return dict {kode_kecamatan: nama_kecamatan}"""
    if not codes:
        return {}
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT kdkec as kode_kecamatan, nmkec as nama_kecamatan
            FROM public.kecamatan
            WHERE kdkec = ANY(:codes)
        """), {"codes": codes}).mappings().all()
    return {r["kode_kecamatan"]: r["nama_kecamatan"] for r in rows}


# =========================
# Facilities config (engine)
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
        ↔ POI.{nmkec}                  (engine)

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
                nmkec as nama_kecamatan
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

    # 2) Query POI by kecamatan name (on engine)
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

    with engine.begin() as conn:
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


def _fetch_poi_by_categories(codes: List[str], categories: Optional[List[str]]) -> Optional[List[Dict]]:
    """
    Fetch POI from public.poi table filtered by:
      - kdkec IN (codes)
      - kategori IN (categories)

    Returns a list of GeoJSON Feature dicts (NOT a FeatureCollection).
    """
    if not categories or not codes:
        return None

    unique_codes = sorted({c for c in codes if c})
    unique_categories = sorted({cat.strip().upper() for cat in categories if cat.strip()})

    if not unique_codes or not unique_categories:
        return None

    sql = text("""
        SELECT
            smid AS id,
            COALESCE(nama, '') AS nama,
            kategori AS category,
            kdkec AS kode_kecamatan,
            ST_AsGeoJSON(ST_Transform(smgeometry, 4326)) AS geom_json
        FROM public.poi
        WHERE kdkec = ANY(:codes)
          AND UPPER(kategori) = ANY(:categories)
        ORDER BY smid
        LIMIT 1000
    """)

    with engine.begin() as conn:
        rows = conn.execute(sql, {
            "codes": unique_codes,
            "categories": unique_categories
        }).mappings().all()

    features: List[Dict[str, Any]] = []
    for r in rows:
        gj = r.get("geom_json")
        if not gj:
            continue
        props = {
            "id": r["id"],
            "nama": r["nama"],
            "category": r["category"],
            "kode_kecamatan": r["kode_kecamatan"]
        }
        features.append(_feature(gj, props))

    return features

# =========================
# Public API functions
# =========================

def get_grid_score(grid_id: int, facility_key: Optional[str] = None, categories: Optional[List[str]] = None):
    """
    Fetch a single grid score record (from engine) and enrich response with:
      - provinsi, kota/kabupaten, kecamatan (kode + nama)
      - kecamatan polygon FeatureCollection (from engine)
      - optional facilities list (array of GeoJSON Features) filtered by kdkec (from engine)
      - optional POI/categories list (array of GeoJSON Features) filtered by kdkec and kategori
    """
    with engine.begin() as conn:
        r = conn.execute(text("""
            SELECT *
            FROM grid_scores
            WHERE id = :id
        """), {"id": grid_id}).mappings().first()

    if not r:
        return {"message": "Grid score not found"}

    # Parse kode
    kode_provinsi = r.get("kode_provinsi")
    kode_kota = r.get("kode_kota_kabupaten")
    codes = [str(c) for c in (safe_json_load(r["kode_kecamatan"]) or [])]

    # Ambil nama dari engine
    provinsi_name = _get_provinsi_name(kode_provinsi)
    kota_name = _get_kota_kabupaten_name(kode_kota)
    kecamatan_names = _get_kecamatan_names(codes)

    # Fetch polygons
    code_to_feature = _fetch_kecamatan_features_by_codes(codes)

    response: Dict[str, Any] = {
        "id": r["id"],
        "nama_layer": r["nama_layer"],
        "deskripsi_layer": r.get("deskripsi_layer"),
        "provinsi": {
            "kode_provinsi": kode_provinsi,
            "nama_provinsi": provinsi_name,
        },
        "kota_kabupaten": {
            "kode_kota_kabupaten": kode_kota,
            "nama_kota_kabupaten": kota_name,
        },
        "kecamatan": [
            {
                "kode_kecamatan": c,
                "nama_kecamatan": kecamatan_names.get(c, c)
            }
            for c in codes
        ],
        "thresholds": safe_json_load(r["thresholds"]),
        "mandatory_parameters": safe_json_load(r.get("mandatory_parameters")),
        "optional_parameters": safe_json_load(r.get("optional_parameters")),
        "weights_applied": safe_json_load(r.get("weights_applied")),
        "grid_geometries": safe_json_load(r["grid_geometries"]),
        "kecamatan_regions": _feature_collection(
            [code_to_feature[c] for c in codes if c in code_to_feature]
        ),
        "created_at": r["created_at"],
    }

    facilities_list = _fetch_facilities_in_kecamatan(codes, facility_key)
    if facilities_list is not None:
        response["facility_requested"] = facility_key
        response["facilities"] = facilities_list

    # Fetch POI by categories if provided
    poi_list = _fetch_poi_by_categories(codes, categories)
    if poi_list is not None:
        response["categories_requested"] = categories
        response["poi"] = poi_list

    return response

def get_all_grid_scores():
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT
                id,
                nama_layer,
                deskripsi_layer,
                kode_provinsi,
                kode_kota_kabupaten,
                kode_kecamatan,
                thresholds,
                created_at
            FROM grid_scores
            ORDER BY created_at DESC, id DESC
        """)).mappings().all()

    results = []
    for r in rows:
        kode_provinsi = r.get("kode_provinsi")
        kode_kota = r.get("kode_kota_kabupaten")
        codes = [str(c) for c in (safe_json_load(r["kode_kecamatan"]) or [])]

        provinsi_name = _get_provinsi_name(kode_provinsi)
        kota_name = _get_kota_kabupaten_name(kode_kota)
        kecamatan_names = _get_kecamatan_names(codes)

        results.append({
            "id": r["id"],
            "nama_layer": r["nama_layer"],
            "deskripsi_layer": r.get("deskripsi_layer"),
            "provinsi": {
                "kode_provinsi": kode_provinsi,
                "nama_provinsi": provinsi_name,
            },
            "kota_kabupaten": {
                "kode_kota_kabupaten": kode_kota,
                "nama_kota_kabupaten": kota_name,
            },
            "kecamatan": [
                {
                    "kode_kecamatan": c,
                    "nama_kecamatan": kecamatan_names.get(c, c)
                }
                for c in codes
            ],
            "thresholds": safe_json_load(r["thresholds"]) if r.get("thresholds") else None,
            "created_at": r["created_at"],
        })

    return results


def get_analysis_result(analysis_id: int) -> Dict[str, Any]:
    with engine.begin() as conn:
        r = conn.execute(text("""
            SELECT *
            FROM analysis_results
            WHERE id = :id
        """), {"id": analysis_id}).mappings().first()

    if not r:
        return {"message": "Analysis result not found"}

    # --- Parse kode
    kode_provinsi = r.get("kode_provinsi")
    kode_kota = r.get("kode_kota_kabupaten")
    codes = [str(c) for c in (safe_json_load(r["kode_kecamatan"]) or [])]

    # --- Ambil nama dari engine
    provinsi_name = _get_provinsi_name(kode_provinsi)
    kota_name = _get_kota_kabupaten_name(kode_kota)
    kecamatan_names = _get_kecamatan_names(codes)

    # --- Fetch polygons
    code_to_feature = _fetch_kecamatan_features_by_codes(codes)

    # --- Build response eksplisit (kayak get_grid_score)
    response: Dict[str, Any] = {
        "id": r["id"],
        "nama_layer": r["nama_layer"],
        "deskripsi_layer": r.get("deskripsi_layer"),
        "grid_layer_name": r.get("grid_layer_name"),
        "provinsi": {
            "kode_provinsi": kode_provinsi,
            "nama_provinsi": provinsi_name,
        },
        "kota_kabupaten": {
            "kode_kota_kabupaten": kode_kota,
            "nama_kota_kabupaten": kota_name,
        },
        "kecamatan": [
            {
                "kode_kecamatan": c,
                "nama_kecamatan": kecamatan_names.get(c, c)
            }
            for c in codes
        ],
        "lahan_kosong": safe_json_load(r.get("lahan_kosong")),
        "selected_facilites": safe_json_load(r.get("selected_facilites")),
        "selected_fasilitas": r.get("selected_fasilitas"),
        "clip_geometries": safe_json_load(r.get("clip_geometries")),
        "ukuran_buffer": r.get("ukuran_buffer"),
        "created_at": r.get("created_at"),
        "kecamatan_regions": _feature_collection(
            [code_to_feature[c] for c in codes if c in code_to_feature]
        ),
    }

    return response

def get_all_analysis_results():
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, nama_layer, deskripsi_layer, grid_layer_name, kode_provinsi, kode_kota_kabupaten, kode_kecamatan, selected_fasilitas, created_at
            FROM analysis_results
            ORDER BY created_at DESC, id DESC
        """)).mappings().all()

    results = []
    for r in rows:
        kode_provinsi = r.get("kode_provinsi")
        kode_kota = r.get("kode_kota_kabupaten")
        codes_kecamatan = [str(c) for c in (safe_json_load(r["kode_kecamatan"]) or [])]

        provinsi_name = _get_provinsi_name(kode_provinsi)
        kota_name = _get_kota_kabupaten_name(kode_kota)
        kecamatan_names = _get_kecamatan_names(codes_kecamatan)

        results.append({
            "id": r["id"],
            "nama_layer": r["nama_layer"],
            "deskripsi_layer": r.get("deskripsi_layer"),
            "grid_layer_name": r["grid_layer_name"],
            "provinsi": {
                "kode_provinsi": kode_provinsi,
                "nama_provinsi": provinsi_name,
            },
            "kota_kabupaten": {
                "kode_kota_kabupaten": kode_kota,
                "nama_kota_kabupaten": kota_name,
            },
            "kecamatan": [
                {
                    "kode_kecamatan": c,
                    "nama_kecamatan": kecamatan_names.get(c, c)
                }
                for c in codes_kecamatan
            ],
            "selected_fasilitas": r["selected_fasilitas"],
            "created_at": r["created_at"],
        })

    return results
