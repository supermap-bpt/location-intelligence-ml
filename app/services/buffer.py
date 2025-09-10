import json
from shapely.geometry import shape
from shapely.ops import unary_union
from app.database import engine_dummy_bps
from sqlalchemy import text
from shapely.geometry import shape, mapping
from app.models.requests import BufferRequest

def buffer_result_service(req: BufferRequest):
    # --- Step 1: Convert inputs to Shapely ---
    buffer_shapes = [shape(p) for p in req.buffer_polygons]
    recommended_shapes = [shape(p) for p in req.recommended_area]

    # Merge buffer polygons into one geometry
    buffer_union = unary_union(buffer_shapes)

    # Crop recommended areas (remove overlaps with buffer)
    cropped = [r.difference(buffer_union) for r in recommended_shapes if not r.is_empty]

    # --- Step 2: Fetch LahanKosong_P filtered by cropped polygons ---
    lahan_kosong = []
    with engine_dummy_bps.connect() as conn:
        for c in cropped:
            if c.is_empty:
                continue

            # Convert cropped polygon to WKT
            cropped_wkt = c.wkt  

            query = text("""
                SELECT smid, smuserid, ST_AsGeoJSON(smgeometry) as geometry,
                       userid, namobj, remark, kdprov, kdkab, kdkec,
                       nmprov, nmkab, nmkec, region_code, luas, x_centroid, y_centroid
                FROM "LahanKosong_Area"
                WHERE ST_Intersects(
                    smgeometry,
                    ST_GeomFromText(:cropped_wkt, 4326)
                );
            """)

            result = conn.execute(query, {"cropped_wkt": cropped_wkt}).fetchall()

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

    # --- Step 3: Return cropped polygons & lahan kosong ---
    return {
        "cropped_polygons": [mapping(c) for c in cropped if not c.is_empty],
        "lahan_kosong": lahan_kosong
    }