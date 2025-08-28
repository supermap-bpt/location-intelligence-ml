from sqlalchemy import create_engine
from app.config import DATABASE_URL, DATABASE_URL_DUMMY_BPS

engine = create_engine(DATABASE_URL)
engine_dummy_bps = create_engine(DATABASE_URL_DUMMY_BPS)