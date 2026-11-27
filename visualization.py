"""
Visualization Module (Cell 6 Logic)
Provides Matplotlib/Seaborn figure builders for both CLI report export and Streamlit rendering:
- Customer Share vs. Revenue Share across Segments / Tiers
- RFM Heatmap (R_score vs. F_score with average Monetary value)
- Recency, Frequency, Monetary distributions
- Session Conversion Rate vs. Cart Abandonment Rate by Segment
- Report charts for Direction A (Hình 1–6, Vietnamese labels): monthly revenue & orders, RFM distributions,
  11-segment share / spend / behavior, and brand revenue mix per segment
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


# ==============================================================================
# Report charts — Direction A (11 RFM Segments), Hình 1–6 for the Vietnamese report
# ==============================================================================

# Fixed segment order for every report chart: highest -> lowest customer value
SEGMENT_11_ORDER = [
    "Champions", "Loyal Customers", "Potential Loyalist", "New Customers", "Promising",
    "Need Attention", "About To Sleep", "At Risk", "Cannot Lose Them", "Hibernating", "Lost",
]
# Dark -> light palette following SEGMENT_11_ORDER
SEGMENT_11_COLORS = dict(zip(SEGMENT_11_ORDER, sns.light_palette("#2E5E1E", n_colors=14, reverse=True)[:11]))
C_MAIN, C_ACCENT = "#2E5E1E", "#E0A526"  # Colors for two-metric comparison charts


def _segment_view(summary_11: pd.DataFrame) -> pd.DataFrame:
    """Index the 11-segment summary by segment name in SEGMENT_11_ORDER."""
    return summary_11.set_index("segment_11").reindex(SEGMENT_11_ORDER)


def _grouped_barh(ax, df: pd.DataFrame, cols, labels, colors, fmt) -> None:
    """Draw two horizontal bars side by side per segment, with data labels."""
    y = np.arange(len(df))
    h = 0.4
    for i, (col, label, color) in enumerate(zip(cols, labels, colors)):
        bars = ax.barh(y + (i - 0.5) * h, df[col], height=h, color=color, label=label)
        ax.bar_label(bars, labels=[fmt(v) for v in df[col]], padding=3, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.set_xlim(0, df[cols].max().max() * 1.15)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)


def plot_monthly_revenue_orders(purchases: pd.DataFrame) -> plt.Figure:
    """Hình 1: monthly revenue (bars) and number of orders = unique purchase sessions (line)."""
    monthly = purchases.groupby("month", observed=True).agg(
        revenue=("price", "sum"),
        orders=("user_session", "nunique"),
    ).sort_index()
    months = monthly.index.astype(str)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    bars = ax1.bar(months, monthly["revenue"] / 1e6, color=C_MAIN, width=0.6, label="Doanh thu")
    ax1.bar_label(bars, labels=[f"${v / 1e6:.2f}M" for v in monthly["revenue"]], padding=3, fontsize=9)
    ax1.set_ylabel("Doanh thu (triệu $)")
    # Bars use the lower ~60% of the height so the order line above never overlaps their labels
    ax1.set_ylim(0, monthly["revenue"].max() / 1e6 * 1.65)

    ax2 = ax1.twinx()
    ax2.plot(months, monthly["orders"] / 1e3, color=C_ACCENT, marker="o", lw=2.5, label="Số đơn hàng")
    for x, v in zip(months, monthly["orders"]):
        ax2.annotate(f"{v / 1e3:,.1f}K", (x, v / 1e3), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9, color=C_ACCENT, fontweight="bold")
    ax2.set_ylim(-monthly["orders"].max() / 1e3 * 1.6, monthly["orders"].max() / 1e3 * 1.25)
    ax2.set_yticks([])
    ax2.grid(False)

    h1, _ = ax1.get_legend_handles_labels()
    h2, _ = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, ["Doanh thu (triệu $)", "Số đơn hàng (nghìn phiên mua)"],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax1.set_title("Hình 1. Doanh thu & số đơn hàng theo tháng (10/2019 – 02/2020)", fontweight="bold", pad=30)
    return fig


def plot_rfm_distributions_report(rfm_df: pd.DataFrame) -> plt.Figure:
    """Hình 2: Recency histogram, Frequency bars (10+ grouped), Monetary histogram clipped at P99."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) Recency
    r = rfm_df["recency"]
    axes[0].hist(r, bins=range(0, int(r.max()) + 6, 5), color=C_MAIN, edgecolor="white")
    axes[0].axvline(r.median(), color=C_ACCENT, ls="--", lw=2, label=f"Trung vị = {r.median():.0f} ngày")
    axes[0].set_title("(a) Recency (ngày)")
    axes[0].set_xlabel("Số ngày từ lần mua cuối đến mốc chốt")
    axes[0].set_ylabel("Số khách hàng")
    axes[0].legend()

    # (b) Frequency: group >= 10 orders into one bar because the tail is very long
    f_counts = rfm_df["frequency"].clip(upper=10).value_counts().sort_index()
    f_pct = f_counts / f_counts.sum() * 100
    f_labels = ["10+" if i == 10 else str(i) for i in f_counts.index]
    bars = axes[1].bar(f_labels, f_counts.values, color=C_MAIN)
    axes[1].bar_label(bars, labels=[f"{p:.1f}%" for p in f_pct], padding=2, fontsize=8)
    axes[1].set_title(f"(b) Frequency — {f_pct.iloc[0]:.2f}% khách chỉ mua 1 đơn")
    axes[1].set_xlabel("Số đơn hàng (phiên mua)")
    axes[1].set_ylim(0, f_counts.max() * 1.12)

    # (c) Monetary: clipped at P99 for readability (hidden tail is stated on the axis)
    m = rfm_df["monetary"]
    p99 = m.quantile(0.99)
    axes[2].hist(m[m <= p99], bins=50, color=C_MAIN, edgecolor="white")
    axes[2].axvline(m.median(), color=C_ACCENT, ls="--", lw=2, label=f"Trung vị = ${m.median():.2f}")
    axes[2].set_title(f"(c) Monetary (\\$) — hiển thị ≤ P99 (\\${p99:,.0f})")  # escaped $ avoids mathtext
    axes[2].set_xlabel(f"Tổng chi tiêu 5 tháng ({(m > p99).sum():,} khách > P99 bị ẩn)")
    axes[2].legend()

    fig.suptitle(f"Hình 2. Phân phối của Recency, Frequency, Monetary ({len(rfm_df):,} khách hàng)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_segment_share_11(summary_11: pd.DataFrame) -> plt.Figure:
    """Hình 3: customer share (%) vs revenue share (%) for the 11 segments."""
    seg = _segment_view(summary_11)
    fig, ax = plt.subplots(figsize=(11, 7))
    _grouped_barh(ax, seg, ["customer_share_pct", "revenue_share_pct"],
                  ["Tỷ trọng khách hàng (%)", "Tỷ trọng doanh thu (%)"],
                  [C_ACCENT, C_MAIN], lambda v: f"{v:.2f}%")
    ax.set_xlabel(f"% trên tổng {int(summary_11['customer_count'].sum()):,} khách hàng / tổng doanh thu sạch")
    ax.set_title("Hình 3. Quy mô khách hàng và đóng góp doanh thu của 11 phân khúc RFM", fontweight="bold", pad=30)
    return fig


def plot_segment_spend_11(summary_11: pd.DataFrame) -> plt.Figure:
    """Hình 4: average monetary per customer and average order value (AOV) for the 11 segments."""
    seg = _segment_view(summary_11)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    colors = [SEGMENT_11_COLORS[s] for s in seg.index]

    for ax, col, title in zip(axes, ["avg_monetary", "avg_aov"],
                              ["(a) Chi tiêu trung bình / khách ($)", "(b) Giá trị đơn hàng trung bình — AOV ($)"]):
        bars = ax.barh(seg.index, seg[col], color=colors)
        ax.bar_label(bars, labels=[f"${v:,.2f}" for v in seg[col]], padding=3, fontsize=9)
        ax.set_xlim(0, seg[col].max() * 1.2)
        ax.set_title(title)
    axes[0].invert_yaxis()

    fig.suptitle("Hình 4. Chênh lệch sức mua giữa 11 phân khúc RFM", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_segment_behavior_11(summary_11: pd.DataFrame) -> plt.Figure:
    """Hình 5: weighted session conversion rate vs true cart abandonment rate for the 11 segments."""
    behavior = _segment_view(summary_11)[["weighted_conv_rate", "avg_cart_abandon"]] * 100
    fig, ax = plt.subplots(figsize=(11, 7))
    _grouped_barh(ax, behavior, ["weighted_conv_rate", "avg_cart_abandon"],
                  ["Tỷ lệ chuyển đổi phiên (%)", "Tỷ lệ bỏ giỏ hàng thực (%)"],
                  [C_MAIN, C_ACCENT], lambda v: f"{v:.1f}%")
    ax.set_xlabel("%")
    ax.set_title("Hình 5. Tỷ lệ chuyển đổi phiên và tỷ lệ bỏ giỏ hàng theo 11 phân khúc RFM", fontweight="bold", pad=30)
    return fig


def plot_brand_mix_by_segment(purchases: pd.DataFrame, rfm_11: pd.DataFrame) -> plt.Figure:
    """
    Hình 6: share (%) of each segment's revenue coming from the top-10 brands (each column sums to 100%).
    The "Toàn shop" column is the shop-wide baseline to compare every segment against.
    """
    no_brand, other = "(Không rõ brand)", "(Brand khác)"

    pur_seg = purchases[["user_id", "brand", "price"]].merge(
        rfm_11[["user_id", "segment_11"]], on="user_id", how="inner"
    )
    pur_seg["brand"] = pur_seg["brand"].fillna(no_brand)

    # Top 10 brands by shop-wide revenue; all remaining named brands grouped as "Brand khác"
    top10 = pur_seg.loc[pur_seg["brand"] != no_brand].groupby("brand")["price"].sum().nlargest(10).index.tolist()
    pur_seg["brand_group"] = np.where(pur_seg["brand"].isin(top10 + [no_brand]), pur_seg["brand"], other)

    rev = pur_seg.pivot_table(index="brand_group", columns="segment_11", values="price", aggfunc="sum", fill_value=0)
    rev.insert(0, "Toàn shop", rev.sum(axis=1))
    brand_share = (rev / rev.sum() * 100).reindex(index=top10 + [other, no_brand], columns=["Toàn shop"] + SEGMENT_11_ORDER)

    n_cust = rfm_11["segment_11"].value_counts()
    col_labels = [f"Toàn shop\n(n={len(rfm_11):,})"] + [f"{s}\n(n={n_cust.get(s, 0):,})" for s in SEGMENT_11_ORDER]

    # Two blocks: top-10 brands (green scale) and the two grouped rows (grey scale) so large rows don't wash out colors
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [10, 2], "hspace": 0.04})
    sns.heatmap(brand_share.loc[top10], ax=ax_top, cmap="Greens", annot=True, fmt=".1f", cbar=False,
                linewidths=0.5, linecolor="white", xticklabels=False)
    sns.heatmap(brand_share.loc[[other, no_brand]], ax=ax_bot, cmap="Greys", vmin=0, vmax=100, annot=True, fmt=".1f",
                cbar=False, linewidths=0.5, linecolor="white", xticklabels=col_labels)
    for ax in (ax_top, ax_bot):
        ax.axvline(1, color="black", lw=2)  # separates the "Toàn shop" baseline from the 11 segments
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
    ax_bot.tick_params(axis="x", rotation=35)
    plt.setp(ax_bot.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax_top.set_title("Hình 6. Cơ cấu doanh thu theo thương hiệu trong từng phân khúc RFM\n"
                     "(Ô = % doanh thu của cột đó; mỗi cột cộng lại = 100%)", fontweight="bold")
    return fig
