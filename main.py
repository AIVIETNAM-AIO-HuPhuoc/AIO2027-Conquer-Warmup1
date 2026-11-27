"""
Main Aggregator & Streamlit Dashboard Entry Point (`main.py`)
- Aggregates Part 1 (`setup_data.py`) and Part 2 (`rfm_scoring.py`, `rfm_segmentation.py`, `visualization.py`).
- Can run as a CLI pipeline (`python main.py`) OR as an interactive Streamlit web application (`streamlit run main.py`).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from matplotlib.figure import Figure

from config import (
    AUDIT_DUPLICATES_CSV,
    AUDIT_MISSING_CSV,
    MONTHLY_SUMMARY_CSV,
    OUTPUT_DIR,
    PREPROCESSED_SENTINEL_CSV,
    RFM_11_PROFILES_CSV,
    RFM_11_SUMMARY_CSV,
    RFM_5_SUMMARY_CSV,
    RFM_SCORED_BASE_CSV,
    Purchases_CHUNKS_DIR,
)
from rfm_scoring import compute_rfm_scores
from rfm_segmentation import run_11_segments, run_5_tiers
from setup_data import ensure_preprocessed_data, is_preprocessed_ready
from visualization import (
    plot_brand_mix_by_segment,
    plot_conversion_vs_abandonment,
    plot_monthly_revenue_orders,
    plot_rfm_distributions,
    plot_rfm_distributions_report,
    plot_rfm_heatmap,
    plot_segment_behavior_11,
    plot_segment_revenue_vs_customers,
    plot_segment_share_11,
    plot_segment_spend_11,
)


def load_purchases() -> pd.DataFrame:
    """Load the clean purchase rows (only the columns the report charts need) from the monthly chunks."""
    chunk_files = sorted(Purchases_CHUNKS_DIR.glob("purchases_*.csv"))
    cols = ["user_id", "user_session", "brand", "price", "month"]
    return pd.concat((pd.read_csv(f, usecols=cols) for f in chunk_files), ignore_index=True)


def build_report_figures(rfm_11: pd.DataFrame, summary_11: pd.DataFrame, purchases: pd.DataFrame) -> Dict[str, Figure]:
    """Build the Direction A report charts (Hình 1–6), keyed by their output PNG filename."""
    return {
        "chart1_monthly_revenue_orders.png": plot_monthly_revenue_orders(purchases),
        "chart2_rfm_distribution.png": plot_rfm_distributions_report(rfm_11),
        "chart3_segment_customer_vs_revenue_share.png": plot_segment_share_11(summary_11),
        "chart4_segment_avg_monetary_aov.png": plot_segment_spend_11(summary_11),
        "chart5_segment_conversion_abandon.png": plot_segment_behavior_11(summary_11),
        "chart6_brand_mix_by_segment.png": plot_brand_mix_by_segment(purchases, rfm_11),
    }


def run_full_pipeline(
    force_preprocess: bool = False,
    save_plots: bool = True,
    show_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the complete end-to-end pipeline:
    1. Setup & pre-process raw Kaggle data into small CSVs (if `pre-processed.csv` is not present).
    2. Compute RFM metrics & Iterative Quantile Peeling F_score (`rfm_scoring.py`).
    3. Run Direction A (11 Detailed RFM Segments) and Direction B (5 Customer Tiers) (`rfm_segmentation.py`).
    4. Optionally export static visualization charts to `outputs/` and/or display them (`visualization.py`).
    """
    import matplotlib.pyplot as plt

    ensure_preprocessed_data(force=force_preprocess)

    rfm_base = compute_rfm_scores(save_output=True)
    rfm_11, summary_11 = run_11_segments(rfm_base, save_output=True)
    rfm_5, summary_5 = run_5_tiers(rfm_base, save_output=True)

    if save_plots or show_plots:
        fig_11 = plot_segment_revenue_vs_customers(
            summary_11, "segment_11", "11 RFM Segments: Customer Share vs. Revenue Share"
        )
        fig_11_conv = plot_conversion_vs_abandonment(summary_11, "segment_11")
        fig_5 = plot_segment_revenue_vs_customers(
            summary_5, "customer_tier", "5 Customer Tiers: Customer Share vs. Revenue Share"
        )
        fig_5_conv = plot_conversion_vs_abandonment(summary_5, "customer_tier")
        fig_heat = plot_rfm_heatmap(rfm_base)
        fig_dist = plot_rfm_distributions(rfm_base)
        report_figs = build_report_figures(rfm_11, summary_11, load_purchases())

        if save_plots:
            fig_11.savefig(OUTPUT_DIR / "chart_11_segments_share.png", bbox_inches="tight")
            fig_11_conv.savefig(OUTPUT_DIR / "chart_11_segments_conversion.png", bbox_inches="tight")
            fig_5.savefig(OUTPUT_DIR / "chart_5_tiers_share.png", bbox_inches="tight")
            fig_5_conv.savefig(OUTPUT_DIR / "chart_5_tiers_conversion.png", bbox_inches="tight")
            fig_heat.savefig(OUTPUT_DIR / "chart_rfm_heatmap.png", bbox_inches="tight")
            fig_dist.savefig(OUTPUT_DIR / "chart_rfm_distributions.png", bbox_inches="tight")
            for filename, fig in report_figs.items():
                fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")  # dpi=300 for Docs/Slides
            print(f"\n[Pipeline] Saved visualization charts and CSV summaries to: {OUTPUT_DIR}")

        if show_plots:
            print("[Pipeline] Displaying charts from core RFM logic...")
            plt.show()
        else:
            plt.close("all")

    return {
        "rfm_base": rfm_base,
        "rfm_11": rfm_11,
        "summary_11": summary_11,
        "rfm_5": rfm_5,
        "summary_5": summary_5,
    }


def _is_running_in_streamlit() -> bool:
    """Detect whether main.py is being executed by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def render_streamlit_app() -> None:
    """Render the interactive Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="AIO2027 — RFM Customer Segmentation",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 eCommerce Cosmetics Shop — RFM Customer Segmentation")
    st.caption(
        "**AIO2027 Warm-up Conquer Project 01** | Group: Road-to-Papers | "
        "Dataset: eCommerce Events History in Cosmetics Shop (10/2019 – 02/2020)"
    )

    with st.sidebar:
        st.header("⚙️ Data & Pipeline Controls")
        data_ready = is_preprocessed_ready()
        if data_ready:
            st.success("Pre-processed CSVs detected (`pre-processed.csv`).")
        else:
            st.warning("Pre-processed CSVs not found.")

        if st.button("🔄 Run / Refresh Full RFM Pipeline", use_container_width=True):
            with st.spinner("Running Kaggle setup & RFM segmentation pipeline..."):
                run_full_pipeline(force_preprocess=False, save_plots=True)
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.subheader("📂 Upload Pre-processed CSV (Optional)")
        st.caption("For lightweight cloud deployment without downloading 20M raw rows:")
        uploaded_file = st.file_uploader("Upload `pre-processed.csv` or `rfm_scored_base.csv`", type=["csv"])

    @st.cache_data(show_spinner="Loading & scoring RFM data...")
    def load_or_compute_data(
        uploaded_bytes: Optional[bytes] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if uploaded_bytes is not None:
            import io

            df_in = pd.read_csv(io.BytesIO(uploaded_bytes))
            rfm_base_df = compute_rfm_scores(preprocessed_df=df_in, save_output=False)
            rfm_11_df, sum_11 = run_11_segments(rfm_base_df, save_output=False)
            _, sum_5 = run_5_tiers(rfm_base_df, save_output=False)
            return rfm_base_df, rfm_11_df, sum_11, sum_5

        if (
            RFM_SCORED_BASE_CSV.exists()
            and RFM_11_PROFILES_CSV.exists()
            and RFM_11_SUMMARY_CSV.exists()
            and RFM_5_SUMMARY_CSV.exists()
        ):
            return (
                pd.read_csv(RFM_SCORED_BASE_CSV),
                pd.read_csv(RFM_11_PROFILES_CSV),
                pd.read_csv(RFM_11_SUMMARY_CSV),
                pd.read_csv(RFM_5_SUMMARY_CSV),
            )

        if PREPROCESSED_SENTINEL_CSV.exists():
            rfm_base_df = compute_rfm_scores(save_output=True)
            rfm_11_df, sum_11 = run_11_segments(rfm_base_df, save_output=True)
            _, sum_5 = run_5_tiers(rfm_base_df, save_output=True)
            return rfm_base_df, rfm_11_df, sum_11, sum_5

        results = run_full_pipeline(force_preprocess=False, save_plots=False)
        return results["rfm_base"], results["rfm_11"], results["summary_11"], results["summary_5"]

    @st.cache_data(show_spinner="Loading purchase rows for report charts...")
    def load_purchases_cached() -> Optional[pd.DataFrame]:
        if not any(Purchases_CHUNKS_DIR.glob("purchases_*.csv")):
            return None
        return load_purchases()

    try:
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        rfm_base, rfm_11, summary_11, summary_5 = load_or_compute_data(uploaded_bytes)
    except Exception as e:
        st.error(f"Data not yet initialized: {e}")
        st.info(
            "Run `python setup_data.py` locally to fetch & pre-process the Kaggle dataset into `data/processed/pre-processed.csv`, "
            "or upload a pre-processed CSV file in the sidebar."
        )
        return

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unique Buyers", f"{len(rfm_base):,}")
    col2.metric("Total Clean Revenue", f"${rfm_base['monetary'].sum():,.2f}")
    col3.metric("Avg Order Value (AOV)", f"${rfm_base['aov'].mean():,.2f}")
    col4.metric("Median Recency", f"{rfm_base['recency'].median():.0f} days")

    tab_overview, tab_11, tab_5, tab_explorer = st.tabs(
        [
            "🧹 Data Audit & RFM Overview",
            "🎯 Direction A: 11 RFM Segments",
            "🏆 Direction B: 5 Customer Tiers",
            "🔍 Customer Profile Explorer",
        ]
    )

    with tab_overview:
        st.subheader("RFM Distributions & Score Heatmap")
        st.pyplot(plot_rfm_distributions(rfm_base))

        c_left, c_right = st.columns([1.1, 1])
        with c_left:
            st.pyplot(plot_rfm_heatmap(rfm_base))
        with c_right:
            st.markdown("#### Descriptive Statistics (`R, F, M, AOV`)")
            st.dataframe(
                rfm_base[["recency", "frequency", "monetary", "aov", "avg_basket_size"]].describe().round(2),
                use_container_width=True,
            )
            if MONTHLY_SUMMARY_CSV.exists():
                st.markdown("#### 5-Month Cleaning Summary")
                st.dataframe(pd.read_csv(MONTHLY_SUMMARY_CSV), use_container_width=True)

        if AUDIT_MISSING_CSV.exists() or AUDIT_DUPLICATES_CSV.exists():
            with st.expander("📋 Raw Data Quality Audit Tables (Cell 1)"):
                ac1, ac2 = st.columns(2)
                if AUDIT_MISSING_CSV.exists():
                    ac1.markdown("**Missing Values (5 Months)**")
                    ac1.dataframe(pd.read_csv(AUDIT_MISSING_CSV), use_container_width=True)
                if AUDIT_DUPLICATES_CSV.exists():
                    ac2.markdown("**Exact Duplicates (2019-Nov Sample)**")
                    ac2.dataframe(pd.read_csv(AUDIT_DUPLICATES_CSV), use_container_width=True)

    with tab_11:
        st.subheader("Direction A: 11 Detailed RFM Segments")
        purchases = load_purchases_cached()
        if purchases is None:
            st.info("Purchase chunks not found in `data/processed/purchases_by_month/` — skipping Hình 1 & Hình 6.")
            st.pyplot(plot_rfm_distributions_report(rfm_11))
            st.pyplot(plot_segment_share_11(summary_11))
            st.pyplot(plot_segment_spend_11(summary_11))
            st.pyplot(plot_segment_behavior_11(summary_11))
        else:
            for fig in build_report_figures(rfm_11, summary_11, purchases).values():
                st.pyplot(fig)

        st.dataframe(summary_11, use_container_width=True)

    with tab_5:
        st.subheader("Direction B: 5 Customer Value Tiers")
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.pyplot(
                plot_segment_revenue_vs_customers(
                    summary_5, "customer_tier", "5 Tiers: Customer Share vs. Revenue Share"
                )
            )
        with c2:
            st.pyplot(plot_conversion_vs_abandonment(summary_5, "customer_tier"))

        st.dataframe(summary_5, use_container_width=True)

    with tab_explorer:
        st.subheader("Lookup & Filter Individual Customer RFM Profiles")
        min_m, max_m = float(rfm_base["monetary"].min()), float(rfm_base["monetary"].max())
        spend_Filter = st.slider("Minimum Total Spend ($)", min_value=0.0, max_value=min(1000.0, max_m), value=0.0)
        filtered = rfm_base[rfm_base["monetary"] >= spend_Filter].sort_values("monetary", ascending=False)
        st.write(f"Showing top {min(500, len(filtered)):,} of {len(filtered):,} matching customers:")
        st.dataframe(filtered.head(500), use_container_width=True)


if __name__ == "__main__":
    if _is_running_in_streamlit():
        render_streamlit_app()
    else:
        parser = argparse.ArgumentParser(description="Run the aggregated RFM Segmentation Pipeline.")
        parser.add_argument("--force-setup", action="store_true", help="Force re-run Kaggle fetch and pre-processing.")
        parser.add_argument("--no-plots", action="store_true", help="Skip saving static PNG charts.")
        parser.add_argument("--show-plots", action="store_true", help="Display interactive Matplotlib chart windows.")
        args = parser.parse_args()

        run_full_pipeline(
            force_preprocess=args.force_setup,
            save_plots=not args.no_plots,
            show_plots=args.show_plots,
        )
