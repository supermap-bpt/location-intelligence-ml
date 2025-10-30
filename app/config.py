import os
from dotenv import load_dotenv

# load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_DUMMY_BPS = os.getenv("DATABASE_URL_DUMMY_BPS")