import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon


def get_intersect_value(gdf: gpd.GeoDataFrame, polygon: Polygon | MultiPolygon, score_col: str) -> float:
    """
    Helper function to calculate intersection value between GeoDataFrame and polygon.
    Returns the score of the feature with the maximum intersection area.
    """
    if gdf is None or gdf.empty:
        return 0.0

    # Ensure consistent CRS
    gdf = gdf.to_crs("EPSG:4326")

    # Calculate intersection
    gdf['intersection'] = gdf.geometry.intersection(polygon)
    gdf = gdf[gdf['intersection'].area > 0]

    if gdf.empty:
        return 0.0

    # Compute intersection areas
    gdf['intersection_area'] = gdf['intersection'].area

    # Pick the row with the max intersection area
    max_idx = gdf['intersection_area'].idxmax()
    return float(gdf.loc[max_idx, score_col])
