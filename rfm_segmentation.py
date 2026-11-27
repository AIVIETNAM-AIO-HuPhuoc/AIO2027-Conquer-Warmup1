"""
RFM Segmentation Module (Cell 4 & Cell 5 Logic)
- Direction A (11 Detailed RFM Segments)
- Direction B (5 Customer Value Tiers)
Both operate independently on `rfm_scored_base`.
"""

from typing import Optional, Tuple

import pandas as pd

from config import (
    RFM_11_PROFILES_CSV,
    RFM_11_SUMMARY_CSV,
    RFM_5_PROFILES_CSV,
    RFM_5_SUMMARY_CSV,
    RFM_SCORED_BASE_CSV,
    ensure_dirs,
)


def assign_11_segments_row(row: pd.Series) -> str:
    """Assign one of 11 detailed RFM segments based on R_score, F_score, M_score."""
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    fm = (f + m) / 2.0

    if r == 1 and f >= 4 and m >= 4:
        return "Cannot Lose Them"
    elif r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers"
    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    elif r >= 4 and 2 <= fm < 4:
        return "Potential Loyalist"
    elif r >= 4 and f == 1:
        return "New Customers"
    elif r == 3 and fm >= 2:
        return "Need Attention"
    elif r == 3 and fm < 2:
        return "Promising"
    elif r == 2 and fm <= 2.5:
        return "About To Sleep"
    elif r == 1 and f == 1 and m <= 2:
        return "Lost"
    else:
        return "Hibernating"


def assign_5_tiers_row(row: pd.Series) -> str:
    """
    Assign one of 5 Customer Tiers directly from R_score, F_score, M_score:
    - 1. Top Tier     : R in [4..5], F in [4..5], M in [4..5]
    - 2. High Tier    : (R in [3..5], F in [3..5], M in [3..5]) or (R == 1, F in [4..5], M in [4..5])
    - 3. Mid-High Tier: (R in [4..5], 2 <= (F+M)/2 < 4) or (R in [1..2], F in [3..5], M in [3..5])
    - 4. Mid Tier     : (R == 3) or (R in [4..5], (F+M)/2 < 2)
    - 5. Low Tier     : Remaining cases
    """
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    fm = (f + m) / 2.0

    if r >= 4 and f >= 4 and m >= 4:
        return "1. Top Tier"
    elif (r >= 3 and f >= 3 and m >= 3) or (r == 1 and f >= 4 and m >= 4):
        return "2. High Tier"
    elif (r >= 4 and 2.0 <= fm < 4.0) or (r <= 2 and f >= 3 and m >= 3):
        return "3. Mid-High Tier"
    elif (r == 3) or (r >= 4 and fm < 2.0):
        return "4. Mid Tier"
    else:
        return "5. Low Tier"


def _build_segment_summary(df: pd.DataFrame, group_col: str, sort_by_revenue: bool = True) -> pd.DataFrame:
    """Aggregate segment-level KPI metrics."""
    total_cust = max(len(df), 1)
    total_rev = max(df["monetary"].sum(), 1e-9)

    summary = df.groupby(group_col, as_index=False).agg(
        customer_count=("user_id", "count"),
        total_revenue=("monetary", "sum"),
        avg_recency_days=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_aov=("aov", "mean"),
        weighted_conv_rate=(
            "purchase_sessions",
            lambda x: x.sum() / max(df.loc[x.index, "total_sessions"].sum(), 1),
        ),
        avg_cart_abandon=("true_cart_abandonment_rate", "mean"),
    )
    summary["customer_share_pct"] = (summary["customer_count"] / total_cust * 100).round(2)
    summary["revenue_share_pct"] = (summary["total_revenue"] / total_rev * 100).round(2)

    if sort_by_revenue:
        summary = summary.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)
    else:
        summary = summary.sort_values(by=group_col, ascending=True).reset_index(drop=True)

    return summary


def run_11_segments(
    rfm_base: Optional[pd.DataFrame] = None,
    save_output: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Direction A: 11 Detailed RFM Segments."""
    ensure_dirs()
    rfm_11 = rfm_base.copy() if rfm_base is not None else pd.read_csv(RFM_SCORED_BASE_CSV)
    rfm_11["segment_11"] = rfm_11.apply(assign_11_segments_row, axis=1)

    summary_11 = _build_segment_summary(rfm_11, group_col="segment_11", sort_by_revenue=True)

    if save_output:
        rfm_11.to_csv(RFM_11_PROFILES_CSV, index=False)
        summary_11.to_csv(RFM_11_SUMMARY_CSV, index=False)

    print("\n=== [DIRECTION A] 11 RFM SEGMENTS SUMMARY ===")
    cols = [
        "segment_11",
        "customer_count",
        "customer_share_pct",
        "total_revenue",
        "revenue_share_pct",
        "avg_recency_days",
        "avg_frequency",
        "avg_monetary",
    ]
    print(summary_11[cols].to_string(index=False))
    return rfm_11, summary_11


def run_5_tiers(
    rfm_base: Optional[pd.DataFrame] = None,
    save_output: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Direction B: 5 Customer Tiers."""
    ensure_dirs()
    rfm_5 = rfm_base.copy() if rfm_base is not None else pd.read_csv(RFM_SCORED_BASE_CSV)
    rfm_5["customer_tier"] = rfm_5.apply(assign_5_tiers_row, axis=1)

    summary_5 = _build_segment_summary(rfm_5, group_col="customer_tier", sort_by_revenue=False)

    if save_output:
        rfm_5.to_csv(RFM_5_PROFILES_CSV, index=False)
        summary_5.to_csv(RFM_5_SUMMARY_CSV, index=False)

    print("\n=== [DIRECTION B] 5 CUSTOMER TIERS SUMMARY ===")
    cols = [
        "customer_tier",
        "customer_count",
        "customer_share_pct",
        "total_revenue",
        "revenue_share_pct",
        "avg_recency_days",
        "avg_frequency",
        "avg_monetary",
    ]
    print(summary_5[cols].to_string(index=False))
    return rfm_5, summary_5
