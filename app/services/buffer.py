import json
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from app.database import engine_dummy_bps
from sqlalchemy import text
from app.models.requests import BufferRequest


def buffer_result_service(req: BufferRequest):
    # --- Step 1: Convert inputs to Shapely ---
    buffer_shapes = [shape(p) for p in req.buffer_polygons]
    recommended_shapes = [shape(p) for p in req.recommended_area]

    # Merge buffer polygons into one geometry
    buffer_union = unary_union(buffer_shapes)

    # Crop recommended areas (remove overlaps with buffer)
    cropped = [r.difference(buffer_union) for r in recommended_shapes if not r.is_empty]

    # --- Step 2: Gabungkan semua cropped jadi satu union besar ---
    cropped_union = unary_union([c for c in cropped if not c.is_empty])
    if cropped_union.is_empty:
        return {
            "cropped_polygons": [],
            "lahan_kosong": []
        }

    cropped_wkt = cropped_union.wkt

    lahan_kosong = []
    with engine_dummy_bps.connect() as conn:
        query = text("""
            SELECT smid, MAX(smuserid) as smuserid, ST_AsGeoJSON(MAX(smgeometry)) as geometry,
                   MAX(userid) as userid, MAX(namobj) as namobj, MAX(remark) as remark,
                   MAX(kdprov) as kdprov, MAX(kdkab) as kdkab, MAX(kdkec) as kdkec,
                   MAX(nmprov) as nmprov, MAX(nmkab) as nmkab, MAX(nmkec) as nmkec,
                   MAX(region_code) as region_code, MAX(luas) as luas,
                   MAX(x_centroid) as x_centroid, MAX(y_centroid) as y_centroid
            FROM "LahanKosong_Area"
            WHERE ST_Intersects(
                smgeometry,
                ST_GeomFromText(:cropped_wkt, 4326)
            )
            AND luas >= :luas_area
            GROUP BY smid;
        """)

        result = conn.execute(query, {
            "cropped_wkt": cropped_wkt,
            "luas_area": req.luas_area or 0
        }).fetchall()

        for row in result:
            lahan_kosong.append({
                "smid": row.smid,
                "smuserid": row.smuserid,
                "geometry": json.loads(row.geometry),
                "userid": row.userid,
                "namobj": row.namobj,
                "remark": row.remark,
                "kdprov": row.kdprov,
                "kdkab": row.kdkab,
                "kdkec": row.kdkec,
                "nmprov": row.nmprov,
                "nmkab": row.nmkab,
                "nmkec": row.nmkec,
                "region_code": row.region_code,
                "luas": row.luas,
                "x_centroid": row.x_centroid,
                "y_centroid": row.y_centroid
            })

    # --- Step 3: Return hasil ---
    return {
        "cropped_polygons": [mapping(c) for c in cropped if not c.is_empty],
        "lahan_kosong": lahan_kosong
    }
