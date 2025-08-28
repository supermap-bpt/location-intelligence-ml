import os
from dotenv import load_dotenv

# load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_DUMMY_BPS = os.getenv("DATABASE_URL_DUMMY_BPS")

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
