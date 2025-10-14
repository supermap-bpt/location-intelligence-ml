# app/services/parameters.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from app.database import engine_dummy_bps

# ---------------------------------------------
# Helpers: DB runners
# ---------------------------------------------
def _fetch_all(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    with engine_dummy_bps.begin() as conn:
        rows = conn.execute(text(sql), params or {}).mappings().all()
    return [dict(r) for r in rows]

def _fetch_one(sql: str, params: Dict[str, Any] | None = None) -> Optional[Dict[str, Any]]:
    with engine_dummy_bps.begin() as conn:
        row = conn.execute(text(sql), params or {}).mappings().first()
    return dict(row) if row else None

def _execute(sql: str, params: Dict[str, Any] | None = None) -> int:
    with engine_dummy_bps.begin() as conn:
        res = conn.execute(text(sql), params or {})
        # For DML, rowcount gives affected rows
        return res.rowcount or 0


# ---------------------------------------------
# READ: Nested classification → categories → parameters
# ---------------------------------------------
def list_parameter_tree() -> List[Dict[str, Any]]:
    """
    Returns:
    [
      {
        "id": ...,
        "code": "...",
        "name": "...",
        "description": "...",
        "ParameterCategories": [
          {
            "id": ...,
            "classification_id": ...,
            "code": "...",
            "name": "...",
            "description": "...",
            "parameters": [
              {
                "id": ...,
                "category_id": ...,
                "code": "...",
                "name": "...",
                "description": "...",
                "isRequired": ...,
                "data_type": "...",
                "unit": "...",
                "min_value": ...,
                "max_value": ...,
                "isActive": ...,
                "default_value": null
              }
            ]
          }
        ]
      }
    ]
    """
    # 1) Classifications
    classifications = _fetch_all(
        """
        SELECT id, code, name, description
        FROM parameter_classification
        ORDER BY name;
        """
    )

    if not classifications:
        return []

    # 2) Categories for all classifications
    cat_rows = _fetch_all(
        """
        SELECT id, classification_id, code, name, description
        FROM parameter_category
        WHERE classification_id = ANY(:cls_ids)
        ORDER BY name;
        """,
        {"cls_ids": [c["id"] for c in classifications]},
    )
    cats_by_cls: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in cat_rows:
        c["parameters"] = []  # slot for children
        cats_by_cls[c["classification_id"]].append(c)

    # 3) Parameters for all categories (with API-friendly aliases)
    if cat_rows:
        prm_rows = _fetch_all(
            """
            SELECT
              id,
              category_id,
              code,
              name,
              description,
              is_required            AS "isRequired",
              data_type::text        AS data_type,
              unit,
              min_value,
              max_value,
              is_active              AS "isActive",
              NULL::text             AS default_value
            FROM parameter
            WHERE category_id = ANY(:cat_ids)
            ORDER BY name;
            """,
            {"cat_ids": [c["id"] for c in cat_rows]},
        )
    else:
        prm_rows = []

    prms_by_cat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in prm_rows:
        prms_by_cat[p["category_id"]].append(p)

    # 4) Attach parameters to categories
    for c in cat_rows:
        c["parameters"] = prms_by_cat.get(c["id"], [])

    # 5) Attach categories to classifications
    for cls in classifications:
        cls["ParameterCategories"] = cats_by_cls.get(cls["id"], [])

    return classifications


# ---------------------------------------------
# READ: Flat lists (optional utility endpoints)
# ---------------------------------------------
def list_classifications() -> List[Dict[str, Any]]:
    return _fetch_all(
        """
        SELECT id, code, name, description
        FROM parameter_classification
        ORDER BY name;
        """
    )

def list_categories(classification_id: Optional[int] = None) -> List[Dict[str, Any]]:
    sql = """
        SELECT id, classification_id, code, name, description
        FROM parameter_category
    """
    params: Dict[str, Any] = {}
    if classification_id:
        sql += " WHERE classification_id = :cid"
        params["cid"] = classification_id
    sql += " ORDER BY name;"
    return _fetch_all(sql, params)

def list_parameters(q: Optional[str] = None, category_id: Optional[int] = None) -> List[Dict[str, Any]]:
    sql = """
        SELECT
          id,
          category_id,
          code,
          name,
          description,
          is_required  AS "isRequired",
          data_type::text AS data_type,
          unit,
          min_value,
          max_value,
          is_active    AS "isActive",
          NULL::text   AS default_value
        FROM parameter
    """
    clauses = []
    params: List[Any] = []
    if q:
        clauses.append("lower(name) LIKE %s")
        params.append(f"%{q.lower()}%")
    if category_id:
        clauses.append("category_id = %s")
        params.append(category_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY name;"

    # NOTE: mixing %s vs named params is okay with text() but we need tuple/list
    with engine_dummy_bps.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------
# CRUD: parameter
# ---------------------------------------------
def get_parameter(param_id: int) -> Optional[Dict[str, Any]]:
    return _fetch_one(
        """
        SELECT
          id,
          category_id,
          code,
          name,
          description,
          is_required  AS "isRequired",
          data_type::text AS data_type,
          unit,
          min_value,
          max_value,
          is_active    AS "isActive",
          NULL::text   AS default_value
        FROM parameter
        WHERE id = :id;
        """,
        {"id": param_id},
    )

def create_parameter(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects keys:
      category_id, code, name, description, is_required, data_type, unit, min_value, max_value, is_active
    """
    row = _fetch_one(
        """
        INSERT INTO parameter
          (category_id, code, name, description, is_required, data_type, unit, min_value, max_value, is_active)
        VALUES
          (:category_id, :code, :name, :description, :is_required, :data_type, :unit, :min_value, :max_value, :is_active)
        RETURNING
          id, category_id, code, name, description,
          is_required  AS "isRequired",
          data_type::text AS data_type,
          unit, min_value, max_value,
          is_active    AS "isActive",
          NULL::text   AS default_value;
        """,
        payload,
    )
    assert row is not None
    return row

def update_parameter(param_id: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Only updates provided fields.
    Allowed: name, description, is_required, data_type, unit, min_value, max_value, is_active
    """
    allowed = ["name","description","is_required","data_type","unit","min_value","max_value","is_active"]
    sets: List[str] = []
    params: Dict[str, Any] = {"id": param_id}
    for k in allowed:
        if k in payload and payload[k] is not None:
            sets.append(f"{k} = :{k}")
            params[k] = payload[k]
    if not sets:
        return get_parameter(param_id)

    sql = f"""
      UPDATE parameter
      SET {', '.join(sets)}
      WHERE id = :id
      RETURNING
        id, category_id, code, name, description,
        is_required  AS "isRequired",
        data_type::text AS data_type,
        unit, min_value, max_value,
        is_active    AS "isActive",
        NULL::text   AS default_value;
    """
    return _fetch_one(sql, params)

def delete_parameter(param_id: int) -> int:
    return _execute("DELETE FROM parameter WHERE id = :id;", {"id": param_id})


# ---------------------------------------------
# UPSERT by PATH (classification → category → parameter)
# ---------------------------------------------
def upsert_parameter_by_path(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects keys:
      classification_code, classification_name,
      category_code, category_name,
      parameter_code, parameter_name,
      is_required, data_type, unit, description
    """
    sql = """
    WITH cls AS (
      INSERT INTO parameter_classification(code, name)
      VALUES (:classification_code, :classification_name)
      ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
      RETURNING id
    ), cat AS (
      INSERT INTO parameter_category(classification_id, code, name)
      SELECT id, :category_code, :category_name FROM cls
      ON CONFLICT (classification_id, code) DO UPDATE SET name = EXCLUDED.name
      RETURNING id
    ), prm AS (
      INSERT INTO parameter(category_id, code, name, is_required, data_type, unit, description)
      SELECT id, :parameter_code, :parameter_name, :is_required, :data_type, :unit, :description
      FROM cat
      ON CONFLICT (category_id, code) DO UPDATE SET
        name = EXCLUDED.name,
        is_required = EXCLUDED.is_required,
        data_type   = EXCLUDED.data_type,
        unit        = EXCLUDED.unit,
        description = EXCLUDED.description
      RETURNING
        id, category_id, code, name, description,
        is_required  AS "isRequired",
        data_type::text AS data_type,
        unit, min_value, max_value,
        is_active    AS "isActive",
        NULL::text   AS default_value
    )
    SELECT * FROM prm;
    """
    row = _fetch_one(sql, payload)
    assert row is not None
    return row

# --- NEW: only "parameter wajib" as groups for your UI ---
def list_wajib_groups():
    """
    Returns:
    [
      {"label": <category.name>, "items": [<parameter.name>, ...]},
      ...
    ]
    """
    # 1) get classification id for 'wajib'
    cls = _fetch_one(
        "SELECT id FROM parameter_classification WHERE code = 'wajib';"
    )
    if not cls:
        return []  # nothing seeded yet

    # 2) categories under 'wajib'
    cats = _fetch_all(
        """
        SELECT id, name
        FROM parameter_category
        WHERE classification_id = :cid
        ORDER BY name;
        """,
        {"cid": cls["id"]},
    )
    if not cats:
        return []

    cat_ids = [c["id"] for c in cats]

    # 3) parameters under those categories
    params = _fetch_all(
        """
        SELECT category_id, name
        FROM parameter
        WHERE category_id = ANY(:cat_ids)
        ORDER BY name;
        """,
        {"cat_ids": cat_ids},
    )

    # 4) group → { label, items[] }
    items_by_cat = {c["id"]: [] for c in cats}
    for p in params:
        items_by_cat[p["category_id"]].append(p["name"])

    groups = [{"label": c["name"], "items": items_by_cat.get(c["id"], [])} for c in cats]
    return groups


# --- (optional) if you also want a tree for wajib only ---
def list_wajib_tree():
    """
    Returns classification 'wajib' with its categories + parameters (your Pydantic models can use this).
    """
    cls = _fetch_one(
        "SELECT id, code, name, description FROM parameter_classification WHERE code = 'wajib';"
    )
    if not cls:
        return None

    cats = _fetch_all(
        """
        SELECT id, classification_id, code, name, description
        FROM parameter_category
        WHERE classification_id = :cid
        ORDER BY name;
        """,
        {"cid": cls["id"]},
    )
    if not cats:
        cls["ParameterCategories"] = []
        return cls

    params = _fetch_all(
        """
        SELECT
          id, category_id, code, name, description,
          is_required AS "isRequired",
          data_type::text AS data_type,
          unit, min_value, max_value,
          is_active AS "isActive",
          NULL::text AS default_value
        FROM parameter
        WHERE category_id = ANY(:cat_ids)
        ORDER BY name;
        """,
        {"cat_ids": [c["id"] for c in cats]},
    )
    by_cat = {c["id"]: [] for c in cats}
    for p in params:
        by_cat[p["category_id"]].append(p)

    for c in cats:
        c["parameters"] = by_cat.get(c["id"], [])

    cls["ParameterCategories"] = cats
    return cls

# --- NEW: only "parameter yang dipertimbangkan" as groups for your UI ---
def list_dipertimbangkan_groups():
    """
    Returns:
    [
      {"label": <category.name>, "items": [<parameter.name>, ...]},
      ...
    ]
    """
    # 1) classification id
    cls = _fetch_one(
        "SELECT id FROM parameter_classification WHERE code = 'dipertimbangkan';"
    )
    if not cls:
        return []

    # 2) categories under it
    cats = _fetch_all(
        """
        SELECT id, name
        FROM parameter_category
        WHERE classification_id = :cid
        ORDER BY name;
        """,
        {"cid": cls["id"]},
    )
    if not cats:
        return []

    cat_ids = [c["id"] for c in cats]

    # 3) parameters in those categories
    params = _fetch_all(
        """
        SELECT category_id, name
        FROM parameter
        WHERE category_id = ANY(:cat_ids)
        ORDER BY name;
        """,
        {"cat_ids": cat_ids},
    )

    # 4) group to UI shape
    items_by_cat = {c["id"]: [] for c in cats}
    for p in params:
        items_by_cat[p["category_id"]].append(p["name"])

    return [{"label": c["name"], "items": items_by_cat.get(c["id"], [])} for c in cats]


# --- OPTIONAL: full tree for "dipertimbangkan" ---
def list_dipertimbangkan_tree():
    cls = _fetch_one(
        "SELECT id, code, name, description FROM parameter_classification WHERE code = 'dipertimbangkan';"
    )
    if not cls:
        return None

    cats = _fetch_all(
        """
        SELECT id, classification_id, code, name, description
        FROM parameter_category
        WHERE classification_id = :cid
        ORDER BY name;
        """,
        {"cid": cls["id"]},
    )
    if not cats:
        cls["ParameterCategories"] = []
        return cls

    params = _fetch_all(
        """
        SELECT
          id, category_id, code, name, description,
          is_required AS "isRequired",
          data_type::text AS data_type,
          unit, min_value, max_value,
          is_active AS "isActive",
          NULL::text AS default_value
        FROM parameter
        WHERE category_id = ANY(:cat_ids)
        ORDER BY name;
        """,
        {"cat_ids": [c["id"] for c in cats]},
    )

    by_cat = {c["id"]: [] for c in cats}
    for p in params:
        by_cat[p["category_id"]].append(p)
    for c in cats:
        c["parameters"] = by_cat.get(c["id"], [])

    cls["ParameterCategories"] = cats
    return cls
