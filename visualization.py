"""
Visualization Module (Cell 6 Logic)
Provides Matplotlib/Seaborn figure builders for both CLI report export and Streamlit rendering:
- Customer Share vs. Revenue Share across Segments / Tiers
- RFM Heatmap (R_score vs. F_score with average Monetary value)
- Recency, Frequency, Monetary distributions
- Session Conversion Rate vs. Cart Abandonment Rate by Segment
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


def plot_segment_revenue_vs_customers(summary_df: pd.DataFrame, label_col: str, title: str) -> plt.Figure:
    """Horizontal bar comparison of Customer Share (%) vs Revenue Share (%)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df_plot = summary_df.copy()

    y = np.arange(len(df_plot))
    height = 0.38

    bars_cust = ax.barh(
        y - height / 2,
        df_plot["customer_share_pct"],
        height,
        label="Customer Share (%)",
        color="#4C78A8",
    )
    bars_rev = ax.barh(
        y + height / 2,
        df_plot["revenue_share_pct"],
        height,
        label="Revenue Share (%)",
        color="#F58518",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot[label_col])
    ax.invert_yaxis()
    ax.set_xlabel("Percentage (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")

    for bar in bars_cust:
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5)
    for bar in bars_rev:
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5)

    max_val = max(df_plot["customer_share_pct"].max(), df_plot["revenue_share_pct"].max())
    ax.set_xlim(0, max_val * 1.18)
    fig.tight_layout()
    return fig


def plot_rfm_heatmap(rfm_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of Average Monetary Value ($) across R_score and F_score grid."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pivot_m = rfm_df.pivot_table(
        index="F_score",
        columns="R_score",
        values="monetary",
        aggfunc="mean",
    ).sort_index(ascending=False)

    sns.heatmap(
        pivot_m,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Avg Monetary ($)"},
        ax=ax,
    )
    ax.set_title("RFM Heatmap: Avg Monetary ($) by R_score & F_score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recency Score (R_score)")
    ax.set_ylabel("Frequency Score (F_score)")
    fig.tight_layout()
    return fig


def plot_rfm_distributions(rfm_df: pd.DataFrame) -> plt.Figure:
    """Histograms of Recency, Frequency (clipped at 10+), and Log10(Monetary)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    sns.histplot(rfm_df["recency"], bins=30, kde=True, color="#2b5c8f", ax=axes[0])
    axes[0].set_title("Recency Distribution (Days)")
    axes[0].set_xlabel("Days Since Last Purchase")

    freq_clipped = rfm_df["frequency"].clip(upper=10)
    sns.countplot(x=freq_clipped, color="#388e3c", ax=axes[1])
    axes[1].set_title("Frequency Distribution (Clipped at 10+)")
    axes[1].set_xlabel("Unique Purchase Sessions")

    sns.histplot(np.log1p(rfm_df["monetary"]), bins=30, kde=True, color="#d95f02", ax=axes[2])
    axes[2].set_title("Log(1 + Monetary) Distribution")
    axes[2].set_xlabel("log(1 + Total Spend $)")

    fig.tight_layout()
    return fig


def plot_conversion_vs_abandonment(summary_df: pd.DataFrame, label_col: str) -> plt.Figure:
    """Scatter / Bubble chart of Conversion Rate vs Cart Abandonment Rate per segment."""
    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot = summary_df.copy()

    x = df_plot["avg_cart_abandon"] * 100
    y = df_plot["weighted_conv_rate"] * 100
    sizes = (df_plot["total_revenue"] / df_plot["total_revenue"].max()) * 900 + 120

    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=np.arange(len(df_plot)),
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidth=1,
    )

    for _, row in df_plot.iterrows():
        ax.annotate(
            row[label_col],
            (row["avg_cart_abandon"] * 100, row["weighted_conv_rate"] * 100),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_title("Session Conversion Rate vs. Cart Abandonment Rate (Bubble = Revenue)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Avg True Cart Abandonment Rate (%)")
    ax.set_ylabel("Weighted Session Conversion Rate (%)")
    fig.tight_layout()
    return fig
