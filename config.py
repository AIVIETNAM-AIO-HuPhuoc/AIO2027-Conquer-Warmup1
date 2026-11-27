import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent


def load_env_files() -> None:
    """Automatically load environment variables (including `export KEY="VAL"`) from `.env` and `.env.local`."""
    for env_name in (".env", ".env.local"):
        env_path = BASE_DIR / env_name
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and val not in ("your_kaggle_username", "your_kaggle_api_key"):
                    os.environ[key] = val


load_env_files()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
Purchases_CHUNKS_DIR = PROCESSED_DATA_DIR / "purchases_by_month"
OUTPUT_DIR = BASE_DIR / "outputs"

# Kaggle dataset & kernel identifiers
# Reference notebook: https://www.kaggle.com/code/juncuong/warnup
# Input dataset (2.43 GB, 5 monthly CSVs): https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop
KAGGLE_DATASET = "mkechinov/ecommerce-events-history-in-cosmetics-shop"
KAGGLE_KERNEL = "juncuong/warnup"
EXPECTED_RAW_CSVS = [
    "2019-Oct.csv",
    "2019-Nov.csv",
    "2019-Dec.csv",
    "2020-Jan.csv",
    "2020-Feb.csv",
]

# Month mapping from raw CSV filenames
MONTH_MAP = {
    "2019-Oct": "2019-10",
    "2019-Nov": "2019-11",
    "2019-Dec": "2019-12",
    "2020-Jan": "2020-01",
    "2020-Feb": "2020-02",
}

ALL_COLS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

USE_COLS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "brand",
    "price",
    "user_id",
    "user_session",
]

DTYPES = {
    "event_type": "category",
    "product_id": "int32",
    "category_id": "int64",
    "brand": "object",
    "price": "float64",  # float64 to prevent rounding error across >20M rows
    "user_id": "int64",
    "user_session": "object",
}

# Pre-processed artifact paths (CSV & small chunks)
PREPROCESSED_SENTINEL_CSV = PROCESSED_DATA_DIR / "pre-processed.csv"
USER_BEHAVIOR_CSV = PROCESSED_DATA_DIR / "all_user_behavior_5m.csv"
ANOMALIES_CSV = PROCESSED_DATA_DIR / "anomalies_price_le0.csv"
AUDIT_MISSING_CSV = PROCESSED_DATA_DIR / "audit_missing_summary.csv"
AUDIT_DUPLICATES_CSV = PROCESSED_DATA_DIR / "audit_duplicates_nov.csv"
MONTHLY_SUMMARY_CSV = PROCESSED_DATA_DIR / "monthly_cleaning_summary.csv"

# RFM output artifact paths
RFM_SCORED_BASE_CSV = OUTPUT_DIR / "rfm_scored_base.csv"
RFM_11_PROFILES_CSV = OUTPUT_DIR / "rfm_11_segments_profiles.csv"
RFM_11_SUMMARY_CSV = OUTPUT_DIR / "rfm_11_segments_summary.csv"
RFM_5_PROFILES_CSV = OUTPUT_DIR / "rfm_5_tiers_profiles.csv"
RFM_5_SUMMARY_CSV = OUTPUT_DIR / "rfm_5_tiers_summary.csv"


def ensure_dirs() -> None:
    """Ensure all data and output directories exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, Purchases_CHUNKS_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
