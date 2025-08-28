DATABASE_URL = "postgresql://postgres@localhost:5432/region_indonesia"
DATABASE_URL_DUMMY_BPS = "postgresql://postgres@localhost:5432/BPS_LI"

MODEL_PATH = "model/random_forest_model_regressor.pkl"

METRICS = {
    "MAE": 0.0412,
    "MSE": 0.0031,
    "RMSE": 0.0560,
    "R2": 0.8643
}

FACILITY_CONFIG = {
    "hotel": {"table": "Hotel_P", "name_col": "nama", "geom_col": "smgeometry"},
    "rumah_sakit": {"table": "RumahSakit_P", "name_col": "namobj", "geom_col": "smgeometry"},
    "sekolah": {"table": "Sekolah_P", "name_col": "namobj", "geom_col": "smgeometry"},
    "pusat_perbelanjaan": {"table": "PusatPerbelanjaan_P", "name_col": "namobj", "geom_col": "smgeometry"}
}