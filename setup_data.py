"""
Part 1: Setup & Data Pre-processing Script (Kernel-Only Fetch)
- Checks if pre-processed CSV files (`data/processed/pre-processed.csv` and broken-down CSVs) exist.
- If missing, fetches the dataset/outputs EXCLUSIVELY from the Kaggle notebook kernel `juncuong/warnup`
  (`https://www.kaggle.com/code/juncuong/warnup`).
- Processes and breaks down the data into small, focused CSV files in `data/processed/` for Git tracking,
  downstream RFM scoring, and Streamlit deployment.
"""

import argparse
import gc
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import (
    ALL_COLS,
    ANOMALIES_CSV,
    AUDIT_DUPLICATES_CSV,
    AUDIT_MISSING_CSV,
    DTYPES,
    EXPECTED_RAW_CSVS,
    KAGGLE_KERNEL,
    MONTH_MAP,
    MONTHLY_SUMMARY_CSV,
    PREPROCESSED_SENTINEL_CSV,
    PROCESSED_DATA_DIR,
    Purchases_CHUNKS_DIR,
    RAW_DATA_DIR,
    USE_COLS,
    USER_BEHAVIOR_CSV,
    ensure_dirs,
)


def is_preprocessed_ready() -> bool:
    """Check whether pre-processed CSV artifacts already exist."""
    month_chunks = list(Purchases_CHUNKS_DIR.glob("purchases_*.csv"))
    return (
        PREPROCESSED_SENTINEL_CSV.exists()
        and PREPROCESSED_SENTINEL_CSV.stat().st_size > 0
        and USER_BEHAVIOR_CSV.exists()
        and len(month_chunks) >= 5
    )


def validate_raw_files(raw_dir: Path = RAW_DATA_DIR) -> List[str]:
    """
    Verify that the 5 expected raw monthly CSV files (or kernel checkpoints) exist in `data/raw/`.
    If missing, abort immediately with a clear error message.
    """
    existing_csvs = [
        str(raw_dir / fname)
        for fname in EXPECTED_RAW_CSVS
        if (raw_dir / fname).exists() and (raw_dir / fname).stat().st_size > 0
    ]
    has_checkpoints = (
        (raw_dir / "clean_purchases_5m.parquet").exists()
        and (raw_dir / "all_user_behavior_5m.parquet").exists()
    )

    if len(existing_csvs) < len(EXPECTED_RAW_CSVS) and not has_checkpoints:
        missing = [fname for fname in EXPECTED_RAW_CSVS if not (raw_dir / fname).exists()]
        print(
            f"[Setup Error] Cannot run `--from-raw`: Missing raw CSV files in '{raw_dir}':\n"
            f"  Missing: {', '.join(missing)}\n"
            f"Please place all 5 raw CSV files ({', '.join(EXPECTED_RAW_CSVS)}) into '{raw_dir}' "
            f"before running `python3 setup_data.py --from-raw` (or `make setup-from-raw`).",
            file=sys.stderr,
        )
        sys.exit(1)

    return sorted(existing_csvs)


def validate_kaggle_env() -> None:
    """
    Verify that KAGGLE_USERNAME and KAGGLE_KEY environment variables are configured
    (not empty and not placeholder values) before running `--from-kernel`.
    """
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    placeholders = {"", "your_kaggle_username", "your_kaggle_api_key"}
    if username in placeholders or key in placeholders:
        print(
            "[Setup Error] Cannot run `--from-kernel`: Missing Kaggle credentials (`KAGGLE_USERNAME` and `KAGGLE_KEY`).\n"
            "Please configure them in `.env` or `.env.local`:\n"
            '  export KAGGLE_USERNAME="your_actual_username"\n'
            '  export KAGGLE_KEY="your_actual_api_key"\n'
            "before running `python3 setup_data.py --from-kernel` (or `make setup-from-kernel`).",
            file=sys.stderr,
        )
        sys.exit(1)


def fetch_from_kernel(kernel_slug: str = KAGGLE_KERNEL, target_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Fetch data exclusively from the Kaggle notebook kernel `juncuong/warnup`
    (`https://www.kaggle.com/code/juncuong/warnup`) via the Kaggle API (`kernels_output`).
    Requires valid KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """
    validate_kaggle_env()
    os.makedirs(target_dir, exist_ok=True)
    print(f"[Setup] Fetching outputs from Kaggle kernel '{kernel_slug}' (https://www.kaggle.com/code/{kernel_slug})...")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.kernels_output(kernel_slug, path=str(target_dir))
        print(f"[Setup] Successfully downloaded kernel outputs from '{kernel_slug}' into: {target_dir}")
        return target_dir
    except Exception as e:
        print(
            f"[Setup Error] Failed to fetch data from Kaggle kernel '{kernel_slug}' "
            f"(https://www.kaggle.com/code/{kernel_slug}).\n"
            f"Please verify your `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` / `.env.local`.\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def run_data_audit(csv_files: List[str]) -> None:
    """
    Cell 1 logic: Audit missing values across raw CSV files and anomalies/duplicates on sample month (2019-Nov).
    Saves compact summary CSVs for dashboard inspection.
    """
    print("\n=== 1. AUDITING MISSING VALUES ACROSS ALL FILES ===")
    total_raw_rows = 0
    missing_counts = pd.Series(0, index=ALL_COLS, dtype="int64")
    pur_missing_core = 0

    for f in csv_files:
        df_raw = pd.read_csv(f, usecols=ALL_COLS, dtype={"event_type": "category", "price": "float64"})
        total_raw_rows += len(df_raw)
        missing_counts += df_raw.isna().sum()
        pur_rows = df_raw[df_raw["event_type"] == "purchase"]
        pur_missing_core += pur_rows[["user_id", "user_session", "price", "event_time"]].isna().any(axis=1).sum()
        del df_raw, pur_rows
        gc.collect()

    missing_summary = pd.DataFrame(
        {
            "column": ALL_COLS,
            "missing_rows": missing_counts.values,
            "missing_pct": (missing_counts.values / max(total_raw_rows, 1) * 100).round(2),
            "total_raw_rows": total_raw_rows,
            "purchase_missing_core": pur_missing_core,
        }
    )
    missing_summary.to_csv(AUDIT_MISSING_CSV, index=False)
    print(f"Total raw rows: {total_raw_rows:,} | Purchase rows missing core IDs: {pur_missing_core}")
    print(missing_summary[["column", "missing_rows", "missing_pct"]].to_string(index=False))

    nov_files = [f for f in csv_files if "2019-Nov" in os.path.basename(f)]
    sample_file = nov_files[0] if nov_files else csv_files[0]
    sample_df = pd.read_csv(sample_file, usecols=USE_COLS, dtype=DTYPES)

    print(f"\n=== 2. AUDITING ANOMALIES & DUPLICATES ({os.path.basename(sample_file)}: {len(sample_df):,} ROWS) ===")
    dup_mask = sample_df.duplicated(keep=False)
    dup_by_event = sample_df[dup_mask].groupby("event_type", observed=True).size()
    total_by_event = sample_df.groupby("event_type", observed=True).size()
    audit_dup = (
        pd.DataFrame(
            {
                "total_events": total_by_event,
                "exact_duplicate_rows": dup_by_event,
                "duplicate_pct": (dup_by_event / total_by_event * 100).round(2),
            }
        )
        .reset_index()
        .rename(columns={"index": "event_type"})
    )
    audit_dup.to_csv(AUDIT_DUPLICATES_CSV, index=False)
    print(audit_dup.to_string(index=False))

    del sample_df, dup_mask
    gc.collect()


def _build_and_save_small_csvs(
    purchases: pd.DataFrame,
    user_behavior: pd.DataFrame,
    anomalies: Optional[pd.DataFrame] = None,
    monthly_metrics: Optional[List[dict]] = None,
) -> None:
    """
    Break down the processed data into small CSV files in `data/processed/`:
    - `purchases_by_month/purchases_<YYYY-MM>.csv`
    - `clean_purchases_5m.csv`
    - `all_user_behavior_5m.csv`
    - `anomalies_price_le0.csv`
    - `monthly_cleaning_summary.csv`
    - `pre-processed.csv` (buyer-level pre-aggregated table combining purchase metrics + session behavior)
    """
    ensure_dirs()

    # Ensure event_time is datetime and month column exists
    purchases["event_time"] = pd.to_datetime(
        purchases["event_time"].astype(str).str.replace(" UTC", "", regex=False),
        errors="coerce",
    )
    purchases = purchases.dropna(subset=["event_time"])
    if "month" not in purchases.columns:
        purchases["month"] = purchases["event_time"].dt.strftime("%Y-%m")

    # 1. Break down purchases into monthly small CSV chunks
    computed_monthly = []
    for month_label, df_m in purchases.groupby("month"):
        chunk_path = Purchases_CHUNKS_DIR / f"purchases_{month_label}.csv"
        df_m.to_csv(chunk_path, index=False)
        computed_monthly.append(
            {
                "month": month_label,
                "purchase_rows": len(df_m),
                "unique_buyers": int(df_m["user_id"].nunique()),
                "revenue": round(float(df_m["price"].sum()), 2),
            }
        )

    # 2. Save core CSV artifacts (purchases are saved as 5 small monthly files in purchases_by_month/)
    user_behavior.to_csv(USER_BEHAVIOR_CSV, index=False)

    if anomalies is not None and len(anomalies) > 0:
        anomalies.to_csv(ANOMALIES_CSV, index=False)

    summary_rows = monthly_metrics if monthly_metrics else computed_monthly
    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("month").to_csv(MONTHLY_SUMMARY_CSV, index=False)

    # 3. Build and save pre-processed.csv (buyer-level pre-aggregated dataset)
    snapshot_date = purchases["event_time"].max() + pd.Timedelta(days=1)
    buyer_pre = purchases.groupby("user_id", as_index=False).agg(
        last_purchase=("event_time", "max"),
        first_purchase=("event_time", "min"),
        recency=("event_time", lambda x: (snapshot_date - x.max()).days),
        frequency=("user_session", "nunique"),
        monetary=("price", "sum"),
        total_items=("product_id", "count"),
        unique_products=("product_id", "nunique"),
    )
    buyer_pre = buyer_pre.merge(user_behavior, on="user_id", how="left")
    buyer_pre.to_csv(PREPROCESSED_SENTINEL_CSV, index=False)

    print("\n=== SUMMARY OF PRE-PROCESSED CSV BREAKDOWN ===")
    print(f"- Valid purchases: {len(purchases):,} rows | Unique buyers: {purchases['user_id'].nunique():,} users")
    print(f"- Clean total revenue: ${purchases['price'].sum():,.2f}")
    if "is_suspected_bot" in user_behavior.columns:
        print(f"- Suspected Bot/Scraper accounts flagged: {int(user_behavior['is_suspected_bot'].sum()):,} users")
    print(f"- Saved broken-down CSV files to: {PROCESSED_DATA_DIR}")


def run_preprocessing_from_kernel_dir(kernel_dir: Path, run_audit: bool = True) -> None:
    """
    Process files fetched from the `juncuong/warnup` kernel:
    - If raw monthly CSVs are present in `kernel_dir`, runs the full Cell 1 & Cell 2 cleaning logic.
    - If the kernel's parquet checkpoints (`clean_purchases_5m.parquet`, `all_user_behavior_5m.parquet`,
      `anomalies_price_le0.parquet`) are present, normalizes and splits them directly into the small CSV files.
    """
    ensure_dirs()

    pur_pq = kernel_dir / "clean_purchases_5m.parquet"
    beh_pq = kernel_dir / "all_user_behavior_5m.parquet"
    anom_pq = kernel_dir / "anomalies_price_le0.parquet"

    if pur_pq.exists() and beh_pq.exists():
        print(f"[Setup] Processing kernel checkpoints from '{KAGGLE_KERNEL}' into broken-down CSV files...")
        purchases = pd.read_parquet(pur_pq)
        user_behavior = pd.read_parquet(beh_pq)
        anomalies = pd.read_parquet(anom_pq) if anom_pq.exists() else None

        # Ensure bot flag is present per Cell 2 logic
        if "is_suspected_bot" not in user_behavior.columns:
            user_behavior["is_suspected_bot"] = (
                (user_behavior["purchase_sessions"] == 0)
                & ((user_behavior["total_carts"] > 50) | (user_behavior["total_views"] > 200))
            ).astype("int8")

        _build_and_save_small_csvs(purchases, user_behavior, anomalies)
        return

    # Otherwise process raw monthly CSV files from kernel_dir using Cell 1 & Cell 2 logic
    raw_csvs = sorted(
        [
            f
            for f in glob.glob(str(kernel_dir / "*.csv"))
            if any(m in os.path.basename(f) for m in MONTH_MAP.keys())
        ]
    )
    if not raw_csvs:
        raise FileNotFoundError(
            f"No kernel checkpoints (`clean_purchases_5m.parquet`) or monthly CSV files found in {kernel_dir} "
            f"from kernel '{KAGGLE_KERNEL}'."
        )

    if run_audit:
        run_data_audit(raw_csvs)

    print("\n=== RUNNING 5-MONTH DATA CLEANING & SESSION-LEVEL AGGREGATION ===")
    purchase_chunks = []
    anomaly_chunks = []
    user_session_stats_list = []
    monthly_metrics = []

    for file_path in raw_csvs:
        fname = os.path.basename(file_path).replace(".csv", "")
        month_label = MONTH_MAP.get(fname, fname)

        df = pd.read_csv(file_path, usecols=USE_COLS, dtype=DTYPES)
        raw_len = len(df)

        # Step 1: Drop missing core identifiers & standardize timestamps
        df = df.dropna(subset=["user_id", "user_session", "price"])
        df["event_time"] = pd.to_datetime(
            df["event_time"].astype(str).str.replace(" UTC", "", regex=False),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        df = df.dropna(subset=["event_time"])
        df["month"] = month_label

        # Step 2: Separate price <= 0 anomalies
        df_anom = df[df["price"] <= 0].copy()
        anomaly_chunks.append(df_anom)
        df = df[df["price"] > 0]

        # Step 3: Selective Deduplication
        df_non_pur = df[df["event_type"] != "purchase"].drop_duplicates()
        df_pur = df[df["event_type"] == "purchase"].copy()
        df_pur["dup_seq"] = df_pur.groupby(["user_session", "event_time", "product_id"]).cumcount() + 1
        df_pur = df_pur[df_pur["dup_seq"] <= 2].drop(columns=["dup_seq"])

        df_clean = pd.concat([df_non_pur, df_pur], ignore_index=True)

        # Step 4: Session-level behavior aggregation per user
        df_clean["is_view"] = (df_clean["event_type"] == "view").astype("int16")
        df_clean["is_cart"] = (df_clean["event_type"] == "cart").astype("int16")
        df_clean["is_remove"] = (df_clean["event_type"] == "remove_from_cart").astype("int16")
        df_clean["is_pur"] = (df_clean["event_type"] == "purchase").astype("int16")

        sess_agg = df_clean.groupby(["user_id", "user_session"], as_index=False).agg(
            views=("is_view", "sum"),
            carts=("is_cart", "sum"),
            removes=("is_remove", "sum"),
            purchases=("is_pur", "sum"),
        )
        sess_agg["has_cart"] = (sess_agg["carts"] > 0).astype("int8")
        sess_agg["has_pur"] = (sess_agg["purchases"] > 0).astype("int8")
        sess_agg["abandoned_cart"] = ((sess_agg["carts"] > 0) & (sess_agg["purchases"] == 0)).astype("int8")

        user_agg = sess_agg.groupby("user_id", as_index=False).agg(
            total_sessions=("user_session", "count"),
            cart_sessions=("has_cart", "sum"),
            purchase_sessions=("has_pur", "sum"),
            abandoned_cart_sessions=("abandoned_cart", "sum"),
            total_views=("views", "sum"),
            total_carts=("carts", "sum"),
            total_removes=("removes", "sum"),
        )
        user_session_stats_list.append(user_agg)
        purchase_chunks.append(df_pur)

        month_rev = float(df_pur["price"].sum())
        monthly_metrics.append(
            {
                "month": month_label,
                "raw_rows": raw_len,
                "clean_rows": len(df_clean),
                "anomaly_rows": len(df_anom),
                "purchase_rows": len(df_pur),
                "unique_buyers": int(df_pur["user_id"].nunique()),
                "revenue": round(month_rev, 2),
            }
        )
        print(
            f"[{month_label}] Raw: {raw_len:,} -> Clean: {len(df_clean):,} | "
            f"Purchases: {len(df_pur):,} | Rev: ${month_rev:,.2f}"
        )

        del df, df_anom, df_non_pur, df_pur, df_clean, sess_agg, user_agg
        gc.collect()

    purchases = pd.concat(purchase_chunks, ignore_index=True)
    anomalies = pd.concat(anomaly_chunks, ignore_index=True)
    user_behavior = (
        pd.concat(user_session_stats_list, ignore_index=True)
        .groupby("user_id", as_index=False)
        .sum()
    )
    user_behavior["is_suspected_bot"] = (
        (user_behavior["purchase_sessions"] == 0)
        & ((user_behavior["total_carts"] > 50) | (user_behavior["total_views"] > 200))
    ).astype("int8")

    _build_and_save_small_csvs(purchases, user_behavior, anomalies, monthly_metrics)


def ensure_preprocessed_data(
    force: bool = False,
    run_audit: bool = True,
    from_kernel: bool = False,
    from_raw: bool = False,
) -> Path:
    """
    Entry point for Part 1 setup:
    1. Checks if `pre-processed.csv` and broken-down CSVs exist.
    2. If `--from-raw` is specified: strictly validates that the 5 raw CSV files exist in `data/raw/`;
       aborts if they are missing.
    3. If `--from-kernel` is specified: strictly validates that `KAGGLE_USERNAME` and `KAGGLE_KEY`
       are set in the environment (`.env` / `.env.local`); aborts if missing.
    4. If neither flag is passed (e.g., called automatically by `main.py` when `pre-processed.csv` is missing):
       checks if `data/raw/` already has the raw files, or if Kaggle env vars are set; otherwise aborts
       and instructs the user to run `setup_data.py` first.
    """
    ensure_dirs()
    if not force and is_preprocessed_ready():
        print(f"[Setup] Pre-processed CSV files already exist at {PROCESSED_DATA_DIR}. Skipping setup.")
        return PREPROCESSED_SENTINEL_CSV

    if from_raw:
        validate_raw_files(RAW_DATA_DIR)
        print(f"[Setup] Processing local raw files from: {RAW_DATA_DIR}")
        run_preprocessing_from_kernel_dir(RAW_DATA_DIR, run_audit=run_audit)
        return PREPROCESSED_SENTINEL_CSV

    if from_kernel:
        validate_kaggle_env()
        kernel_dir = fetch_from_kernel(KAGGLE_KERNEL, RAW_DATA_DIR)
        run_preprocessing_from_kernel_dir(kernel_dir, run_audit=run_audit)
        return PREPROCESSED_SENTINEL_CSV

    # When invoked without explicit flags (e.g. from main.py when pre-processed.csv is missing):
    has_all_raw = all((RAW_DATA_DIR / f).exists() for f in EXPECTED_RAW_CSVS) or (
        (RAW_DATA_DIR / "clean_purchases_5m.parquet").exists()
        and (RAW_DATA_DIR / "all_user_behavior_5m.parquet").exists()
    )
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    has_kaggle_env = username not in ("", "your_kaggle_username") and key not in ("", "your_kaggle_api_key")

    if has_all_raw:
        print(f"[Setup] Found raw files in {RAW_DATA_DIR}. Running pre-processing...")
        run_preprocessing_from_kernel_dir(RAW_DATA_DIR, run_audit=run_audit)
        return PREPROCESSED_SENTINEL_CSV
    elif has_kaggle_env:
        kernel_dir = fetch_from_kernel(KAGGLE_KERNEL, RAW_DATA_DIR)
        run_preprocessing_from_kernel_dir(kernel_dir, run_audit=run_audit)
        return PREPROCESSED_SENTINEL_CSV
    else:
        print(
            "[Setup Error] Pre-processed data (`data/processed/pre-processed.csv`) was not found, and setup cannot run automatically:\n"
            "  - Neither the 5 raw CSV files were found in `data/raw/` (for `--from-raw`),\n"
            "  - Nor were `KAGGLE_USERNAME` and `KAGGLE_KEY` configured in `.env` / `.env.local` (for `--from-kernel`).\n\n"
            "Please either:\n"
            "  1. Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` and run: `make setup-from-kernel`\n"
            "  2. Or place the 5 raw CSV files into `data/raw/` and run: `make setup-from-raw`",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup & Pre-process RFM Data from Kaggle Kernel 'juncuong/warnup' or local data/raw folder into small CSVs."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--from-kernel",
        action="store_true",
        help="Fetch outputs from Kaggle notebook kernel 'juncuong/warnup'.",
    )
    source_group.add_argument(
        "--from-raw",
        action="store_true",
        help="Process local raw CSV files placed in the data/raw/ folder.",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run pre-processing even if pre-processed.csv exists.")
    parser.add_argument("--skip-audit", action="store_true", help="Skip Cell 1 missing/duplicate audit step.")
    args = parser.parse_args()

    ensure_preprocessed_data(
        force=args.force,
        run_audit=not args.skip_audit,
        from_kernel=args.from_kernel,
        from_raw=args.from_raw,
    )
