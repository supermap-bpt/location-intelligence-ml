from shapely.geometry import shape
from shapely.ops import unary_union

def buffer_result_service(req):
    """
    Calculates buffer intersection with recommended areas.
    """
    try:
        # Convert GeoJSON to shapely polygons
        buffer_polys = [shape(poly) for poly in req.buffer_polygons]
        recommended_polys = [shape(poly) for poly in req.recommended_area]

        buffer_union = unary_union(buffer_polys)
        rec_union = unary_union(recommended_polys)

        intersection = buffer_union.intersection(rec_union)

        return {
            "buffer_area": buffer_union.area,
            "recommended_area": rec_union.area,
            "intersection_area": intersection.area,
            "intersection_geometry": intersection.__geo_interface__
        }
    except Exception as e:
        raise Exception(f"Buffer processing failed: {str(e)}")