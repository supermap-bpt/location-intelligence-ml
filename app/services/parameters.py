# services/parameters.py
from typing import List, Dict, Any, DefaultDict
from collections import defaultdict
from sqlalchemy import text
from app.database import engine


def list_category_groups_with_items() -> List[Dict[str, Any]]:
    """
    Mixed shape per category:
      - with sub-categories: { label, subgroups: [{ sub_label, items: [{...}] }, ...] }
      - without sub-categories: { label, items: [{...}] }

    Each item now includes: id, code, label, min_value, max_value
    """

    # 1) Categories
    sql_cats = text("""
        SELECT id, code, name
        FROM public.parameter_category
        ORDER BY name ASC;
    """)

    # 2) Sub-categories (may be empty for some categories)
    sql_subs = text("""
        SELECT id, category_id, code, name, display_order
        FROM public.parameter_sub_category
        ORDER BY category_id, display_order, name;
    """)

    # 3) Parameters (active only), include sub_category_id and range fields
    sql_params = text("""
        SELECT
          p.id,
          p.category_id,
          p.sub_category_id,
          p.code,
          COALESCE(NULLIF(btrim(p.label), ''), p.code) AS label,
          p.display_order,
          p.min_value,
          p.max_value
        FROM public.parameter p
        WHERE p.is_active
        ORDER BY p.category_id, p.sub_category_id NULLS FIRST,
                 p.display_order NULLS LAST, p.code;
    """)

    with engine.connect() as conn:
        cats   = conn.execute(sql_cats).mappings().all()
        subs   = conn.execute(sql_subs).mappings().all()
        params = conn.execute(sql_params).mappings().all()

    if not cats:
        return []

    # Build lookup maps for sub-categories
    subs_by_cat: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    sub_id_to_obj: Dict[int, Dict[str, Any]] = {}
    for s in subs:
        obj = {
            "id": s["id"],
            "category_id": s["category_id"],
            "code": s["code"],
            "name": s["name"],
            "display_order": s["display_order"],
        }
        subs_by_cat[s["category_id"]].append(obj)
        sub_id_to_obj[s["id"]] = obj

    # Bucket items by (category_id -> sub_category_id -> list[items])
    items_bucket: DefaultDict[int, DefaultDict[Any, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for p in params:
        cat_id = p["category_id"]
        sub_id = p["sub_category_id"]  # can be None
        items_bucket[cat_id][sub_id].append({
            "id": p["id"],
            "code": p["code"],
            "label": p["label"],
            "min_value": p["min_value"],   # <-- added
            "max_value": p["max_value"],   # <-- added
        })

    # Assemble final structure (mixed shape)
    groups: List[Dict[str, Any]] = []
    for c in cats:
        cat_id = c["id"]
        cat_name = c["name"]

        has_real_subs = len(subs_by_cat.get(cat_id, [])) > 0

        if has_real_subs:
            # Build subgroups, including a virtual bucket for NULL sub_category_id (if any)
            subgroups: List[Dict[str, Any]] = []

            # Real sub-categories from the table (ordered already by display_order, name)
            for s in subs_by_cat.get(cat_id, []):
                subgroups.append({
                    "sub_label": s["name"],
                    "items": items_bucket[cat_id].get(s["id"], []),
                })

            # Items without a sub_category_id → virtual subgroup
            null_items = items_bucket[cat_id].get(None, [])
            if null_items:
                subgroups.insert(0, {
                    "sub_label": None,  # or "Tanpa Sub-kategori"
                    "items": null_items,
                })

            groups.append({
                "label": cat_name,
                "subgroups": subgroups,
            })

        else:
            # No sub-categories → return items directly
            flat_items: List[Dict[str, Any]] = []
            for _, items in items_bucket[cat_id].items():
                flat_items.extend(items)

            groups.append({
                "label": cat_name,
                "items": flat_items,
            })

    return groups