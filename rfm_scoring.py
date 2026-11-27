"""
RFM Base Calculation & Scoring Module (Cell 3 Logic)
- Computes Recency, Frequency, Monetary, AOV, and Basket Size per customer.
- Applies Iterative Quantile Peeling for F_score (to handle the ~78.9% F=1 skew).
- Merges session-level behavioral metrics (conversion rate & cart abandonment rate).
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    PREPROCESSED_SENTINEL_CSV,
    Purchases_CHUNKS_DIR,
    RFM_SCORED_BASE_CSV,
    USER_BEHAVIOR_CSV,
    ensure_dirs,
)


def dynamic_peeling_score(series: pd.Series, num_bins: int = 5, ascending: bool = True) -> pd.Series:
    """
    Iterative Quantile Peeling algorithm for skewed discrete distributions (e.g., Frequency).
    Prevents splitting identical mass points (like F=1, which accounts for ~78.91% of buyers)
    across multiple score tiers.
    """
    scores = pd.Series(index=series.index, dtype="int64")
    remaining_mask = pd.Series(True, index=series.index)
    current_score = 1
    rem_bins = num_bins
    unique_vals = sorted(series.unique(), reverse=(not ascending))

    for val in unique_vals:
        if rem_bins <= 1:
            break
        rem_count = remaining_mask.sum()
        if rem_count == 0:
            break
        val_mask = remaining_mask & (series == val)
        if (val_mask.sum() / rem_count) > (1.0 / rem_bins):
            scores[val_mask] = current_score
            remaining_mask = remaining_mask & (~val_mask)
            current_score += 1
            rem_bins -= 1
        else:
            break

    if remaining_mask.sum() > 0:
        if rem_bins == 1:
            scores[remaining_mask] = current_score
        else:
            tail_labels = list(range(current_score, num_bins + 1))
            scores[remaining_mask] = pd.qcut(
                series[remaining_mask].rank(method="first", ascending=ascending),
                q=rem_bins,
                labels=tail_labels,
            ).astype("int64")
    return scores


def compute_rfm_scores(
    preprocessed_df: Optional[pd.DataFrame] = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Compute R, F, M metrics, AOV, avg_basket_size, R/F/M scores, and session conversion rates.
    Can load directly from `pre-processed.csv` (fast path) or compute from `clean_purchases_5m.csv` + `all_user_behavior_5m.csv`.
    """
    ensure_dirs()

    if preprocessed_df is not None:
        rfm_base = preprocessed_df.copy()
    elif PREPROCESSED_SENTINEL_CSV.exists():
        rfm_base = pd.read_csv(PREPROCESSED_SENTINEL_CSV)
    else:
        chunk_files = sorted(Purchases_CHUNKS_DIR.glob("purchases_*.csv"))
        purchases = pd.concat((pd.read_csv(f) for f in chunk_files), ignore_index=True)
        user_behavior = pd.read_csv(USER_BEHAVIOR_CSV)
        purchases["event_time"] = pd.to_datetime(purchases["event_time"])
        snapshot_date = purchases["event_time"].max() + pd.Timedelta(days=1)

        rfm_base = purchases.groupby("user_id", as_index=False).agg(
            last_purchase=("event_time", "max"),
            first_purchase=("event_time", "min"),
            recency=("event_time", lambda x: (snapshot_date - x.max()).days),
            frequency=("user_session", "nunique"),
            monetary=("price", "sum"),
            total_items=("product_id", "count"),
            unique_products=("product_id", "nunique"),
        )
        rfm_base = rfm_base.merge(user_behavior, on="user_id", how="left")

    rfm_base["aov"] = rfm_base["monetary"] / rfm_base["frequency"]
    rfm_base["avg_basket_size"] = rfm_base["total_items"] / rfm_base["frequency"]

    # Score R, F, M
    rfm_base["R_score"] = pd.qcut(
        rfm_base["recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    rfm_base["F_score"] = dynamic_peeling_score(
        rfm_base["frequency"], num_bins=5, ascending=True
    ).astype(int)
    rfm_base["M_score"] = pd.qcut(
        rfm_base["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Session conversion & cart abandonment metrics
    rfm_base["session_conversion_rate"] = np.where(
        rfm_base["total_sessions"] > 0,
        rfm_base["purchase_sessions"] / rfm_base["total_sessions"],
        0.0,
    )
    rfm_base["true_cart_abandonment_rate"] = np.where(
        rfm_base["cart_sessions"] > 0,
        rfm_base["abandoned_cart_sessions"] / rfm_base["cart_sessions"],
        0.0,
    )

    if save_output:
        rfm_base.to_csv(RFM_SCORED_BASE_CSV, index=False)

    print("=== RFM SCORE DISTRIBUTION SUMMARY ===")
    print(rfm_base[["recency", "frequency", "monetary", "aov"]].describe().round(2))
    return rfm_base
