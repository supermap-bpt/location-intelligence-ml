import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL_DUMMY_BPS = os.getenv("DATABASE_URL_DUMMY_BPS")
engine_dummy_bps = create_engine(DATABASE_URL_DUMMY_BPS)

def get_siswa_putus_sekolah_geodataframe():
    sql = "SELECT wadmkc, s_siswaputussekolah, ST_AsGeoJSON(smgeometry) as geojson FROM siswa_putus_sekolah WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kemiskinan_geodataframe():
    sql = "SELECT wadmkc, s_kemiskinan, ST_AsGeoJSON(smgeometry) as geojson FROM kemiskinan WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kepadatan_penduduk_geodataframe():
    sql = "SELECT wadmkc, s_pddk, ST_AsGeoJSON(smgeometry) as geojson FROM kepadatan_penduduk WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_poi_geodataframe():
    sql = "SELECT wadmkc, s_poi, ST_AsGeoJSON(smgeometry) as geojson FROM poi WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_gdp_geodataframe():
    sql = """SELECT wadmkc, pendapatan, ST_AsGeoJSON(smgeometry) as geojson FROM "pendapatan_per_kapita_R" WHERE smgeometry IS NOT NULL"""
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kedekatan_sungai_geodataframe():
    sql = "SELECT s_sungai, ST_AsGeoJSON(smgeometry) as geojson FROM kedekatan_sungai WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kedekatan_faskes_geodataframe():
    sql = "SELECT s_faskes, ST_AsGeoJSON(smgeometry) as geojson FROM kedekatan_faskes WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_kedekatan_jalan_geodataframe():
    sql = "SELECT s_road, ST_AsGeoJSON(smgeometry) as geojson FROM jalan WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

def get_slope_geodataframe():
    sql = "SELECT s_slope, ST_AsGeoJSON(smgeometry) as geojson FROM slope WHERE smgeometry IS NOT NULL"
    df = pd.read_sql_query(sql, con=engine_dummy_bps)
    df['geometry'] = df['geojson'].apply(lambda x: shape(eval(x) if isinstance(x, str) else x))
    return gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")