# app/services/suitability/layer_specs.py
from __future__ import annotations
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from sqlalchemy import text
from functools import lru_cache
from typing import Dict, Any
from app.database import engine_dummy_bps

NUMERIC_TYPES = {"numeric", "double precision", "real", "integer", "bigint", "smallint"}

@lru_cache(maxsize=1)
def build_layer_specs(schema: str = "public") -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict keyed by parameter.code (== table name), for active rows only:
    {
      "<code>": {
        "table": "<code>",
        "value_col": "<score_col>",
        "data_type": "<param_data_type>",
        "unit": "<unit or None>",
        "is_required": <bool>,
        "min_value": <num or None>,
        "max_value": <num or None>,
      },
      ...
    }
    """
    q = f'''
      SELECT code, score_col, data_type::text AS data_type, unit, is_required,
             min_value, max_value
      FROM "{schema}".parameter
      WHERE is_active = true
      ORDER BY id ASC
    '''
    df = pd.read_sql_query(q, con=engine_dummy_bps)

    specs: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        code = r["code"]
        value_col = r["score_col"]
        if not code or not value_col:
            raise RuntimeError(f"parameter row missing code/score_col: {r.to_dict()}")

        # Validate column existence
        cols = pd.read_sql_query(
            text("""
              SELECT column_name, data_type
              FROM information_schema.columns
              WHERE table_schema = :schema AND table_name = :table
            """),
            con=engine_dummy_bps,
            params={"schema": schema, "table": code},
        )

        names = set(cols["column_name"].tolist())
        if value_col not in names:
            raise RuntimeError(f"[parameter.code={code}] value column '{value_col}' not found in table.")

        specs[code] = {
            "table": code,
            "value_col": value_col,
            "data_type": (r["data_type"] or "").lower(),
            "unit": r.get("unit"),
            "is_required": bool(r.get("is_required", False)),
            "min_value": r.get("min_value"),
            "max_value": r.get("max_value"),
        }
    return specs

def geodf_for_code(code: str, schema: str = "public") -> gpd.GeoDataFrame:
    """Generic loader using the dynamic specs; tries with wadmkc then without."""
    specs = build_layer_specs(schema)
    if code not in specs:
        raise KeyError(f"Unknown or inactive parameter code: {code}")

    table = specs[code]["table"]
    value_col = specs[code]["value_col"]

    for cols in (["wadmkc", value_col], [value_col]):
        try:
            sql = f'''
              SELECT {", ".join(cols)}, ST_AsGeoJSON(smgeometry) AS geojson
              FROM "{schema}"."{table}"
              WHERE smgeometry IS NOT NULL
            '''
            df = pd.read_sql_query(sql, con=engine_dummy_bps)
            df["geometry"] = df["geojson"].apply(lambda x: shape(json.loads(x) if isinstance(x, str) else x))
            df = df.drop(columns=["geojson"])
            return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        except Exception:
            continue
    raise RuntimeError(f"Failed to load GeoDataFrame for code={code} (table={table}).")