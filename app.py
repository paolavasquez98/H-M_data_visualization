"""
H&M Fashion Intelligence Dashboard
────────────────────────────────────────────────────────────────────────────────
A single, narrative-driven interactive dashboard for fashion analysts and
retail strategy teams.

Story thread: Who buys what → in what colours → when → through which channel
              → and what drives revenue?

Design rationale (addressing lecturer feedback):
  ① Normalized heatmaps (row / col / raw toggle) replace raw-count charts that
    made low-volume categories invisible.
  ② 100 % stacked bar replaces the diverging bar for channel comparison —
    the question "which channel dominates?" is answered by proportion, not
    absolute volume.
  ③ 5 index groups (Baby/Children, Divided, Ladieswear, Menswear, Sport) replace
    19 product groups — categories are grouped, scale is standardised.
  ④ Indexed trend lines (100 = first month) put every segment on the same scale
    so seasonal shapes can be compared rather than volumes.
  ⑤ Colour heatmap uses perceived_colour_master_name (20 master colours) with
    the same row / col / raw toggle so the user decides the reference.

Run: python app.py  →  http://127.0.0.1:8050
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(__file__)
ART_PATH = os.path.join(BASE, "articles.csv")
CUS_PATH = os.path.join(BASE, "customers.csv")
TX_PATH  = os.path.join(BASE, "transactions_train.csv")

# ── Brand palette (one colour per product line) ───────────────────────────────
INDEX_COLORS = {
    "Baby/Children": "#FF9F9F",
    "Divided":       "#E8AC00",
    "Ladieswear":    "#E10028",   # H&M red
    "Menswear":      "#1F4E79",
    "Sport":         "#00875A",
}
INDEX_GROUPS = list(INDEX_COLORS.keys())

# ── Age bins ──────────────────────────────────────────────────────────────────
AGE_BINS   = [15, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# ── Load & merge ──────────────────────────────────────────────────────────────
print("Loading data ...")
art = pd.read_csv(ART_PATH, usecols=[
    "article_id", "index_group_name",
    "colour_group_name", "perceived_colour_master_name",
])
cust = pd.read_csv(CUS_PATH, usecols=["customer_id", "age", "club_member_status"])
tx   = pd.read_csv(TX_PATH, parse_dates=["t_dat"],
                   usecols=["t_dat", "customer_id", "article_id",
                            "price", "sales_channel_id"])

tx = tx.merge(art,  on="article_id",  how="left")
tx = tx.merge(cust, on="customer_id", how="left")
print(f"Rows after merge: {len(tx):,}")

# Derived columns
tx["age_bin"] = pd.cut(tx["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
tx["month"]   = tx["t_dat"].dt.to_period("M").dt.to_timestamp()

# Keep only rows with a known, valid index group and an age
tx_clean = tx.dropna(subset=["age_bin", "index_group_name"]).copy()
tx_clean  = tx_clean[tx_clean["index_group_name"].isin(INDEX_GROUPS)]

# ── Pre-aggregations (computed once at startup) ───────────────────────────────

# 1. Age x IndexGroup  (heatmap 1)
g_age_idx = (tx_clean
             .groupby(["age_bin", "index_group_name"], observed=True)
             .size().reset_index(name="count"))

# 2. Age x Master Colour x IndexGroup  (heatmap 2)
#    Use perceived_colour_master_name (~20 master tones), keep top 12 by volume
top_colors = (tx_clean
              .dropna(subset=["perceived_colour_master_name"])
              .groupby("perceived_colour_master_name")
              .size().nlargest(12).index.tolist())
tx_col = tx_clean.dropna(subset=["perceived_colour_master_name"])
tx_col  = tx_col[tx_col["perceived_colour_master_name"].isin(top_colors)]
g_age_color = (tx_col
               .groupby(["age_bin", "perceived_colour_master_name",
                          "index_group_name"], observed=True)
               .size().reset_index(name="count"))

# 3. Monthly volume x IndexGroup, indexed to first available month = 100 (trend)
g_monthly = (tx_clean
             .groupby(["month", "index_group_name"])
             .size().reset_index(name="count"))
first_counts = (g_monthly.sort_values("month")
                          .groupby("index_group_name")
                          .first()[["count"]]
                          .rename(columns={"count": "base"}))
g_monthly = g_monthly.merge(first_counts, on="index_group_name")
g_monthly["indexed"] = g_monthly["count"] / g_monthly["base"] * 100

# 4. Sales channel split x IndexGroup  (100% stacked bar)
g_channel = (tx_clean
             .groupby(["index_group_name", "sales_channel_id"])
             .size().reset_index(name="count"))
ch_piv = (g_channel
          .pivot(index="index_group_name", columns="sales_channel_id", values="count")
          .fillna(0))
ch_piv.columns = [f"ch{int(c)}" for c in ch_piv.columns]
if "ch1" not in ch_piv.columns:
    ch_piv["ch1"] = 0
if "ch2" not in ch_piv.columns:
    ch_piv["ch2"] = 0
ch_piv["total"]   = ch_piv["ch1"] + ch_piv["ch2"]
ch_piv["Store%"]  = ch_piv["ch1"] / ch_piv["total"] * 100
ch_piv["Online%"] = ch_piv["ch2"] / ch_piv["total"] * 100

# 5. Revenue bubble x IndexGroup  (scatter)
g_bubble = (tx_clean
            .groupby("index_group_name")
            .agg(volume=("article_id", "count"),
                 avg_price=("price",   "mean"),
                 revenue=("price",    "sum"))
            .reset_index())

print("Pre-aggregation done.\n")

# ── Helper functions ──────────────────────────────────────────────────────────

def pivot_normalize(df, idx_col, col_col, val_col, mode, idx_order=None, col_order=None):
    """Pivot and optionally normalize by row, column, or leave raw."""
    piv = df.pivot(index=idx_col, columns=col_col, values=val_col).fillna(0)
    if idx_order:
        piv = piv.reindex([r for r in idx_order if r in piv.index])
    if col_order:
        piv = piv.reindex(columns=[c for c in col_order if c in piv.columns])
    if mode == "row":
        row_sums = piv.sum(axis=1).replace(0, np.nan)
        piv = piv.div(row_sums, axis=0) * 100
    elif mode == "col":
        col_sums = piv.sum(axis=0).replace(0, np.nan)
        piv = piv.div(col_sums, axis=1) * 100
    return piv


def make_heatmap(piv, colorscale, norm_mode, xangle=-25):
    """Return a styled Plotly Heatmap figure from a pivot DataFrame."""
    suffix = "%" if norm_mode != "raw" else ""
    fmt    = ".1f" if norm_mode != "raw" else ",.0f"
    cell_text = [[f"{v:{fmt}}{suffix}" for v in row] for row in piv.values]
    fig = go.Figure(go.Heatmap(
        z=piv.values,
        x=piv.columns.tolist(),
        y=[str(y) for y in piv.index],
        colorscale=colorscale,
        text=cell_text,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black"},
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> x <b>%{x}</b><br>%{text}<extra></extra>",
        colorbar=dict(thickness=10, title=dict(text=suffix or "n", side="right")),
    ))
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(tickangle=xangle, tickfont=dict(size=11), title=None),
        yaxis=dict(tickfont=dict(size=11), title=None),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ── Dash layout helpers ───────────────────────────────────────────────────────

def card(*children, **extra_style):
    base = {
        "background": "white",
        "borderRadius": "10px",
        "padding": "16px 20px",
        "boxShadow": "0 1px 6px rgba(0,0,0,0.08)",
    }
    base.update(extra_style)
    return html.Div(list(children), style=base)


def panel_header(num, title, subtitle):
    return html.Div([
        html.Div([
            html.Span(num,
                      style={"color": "#E10028", "fontWeight": 800,
                             "fontSize": "15px", "marginRight": "6px"}),
            html.Span(title,
                      style={"fontWeight": 700, "fontSize": "14px", "color": "#222"}),
        ]),
        html.P(subtitle,
               style={"margin": "2px 0 8px", "fontSize": "11px", "color": "#999",
                      "lineHeight": "1.4"}),
    ])


NORM_OPTS = [
    {"label": "% of age group",  "value": "row"},
    {"label": "% of category",   "value": "col"},
    {"label": "Raw count",       "value": "raw"},
]

radio_style = {"fontSize": "11px", "marginBottom": "8px"}

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="H&M Fashion Intelligence")

app.layout = html.Div([

    # ── Sticky header ─────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("H&M",
                      style={"color": "#E10028", "fontWeight": 900,
                             "fontSize": "24px", "marginRight": "8px",
                             "letterSpacing": "-0.5px"}),
            html.Span("Fashion Intelligence",
                      style={"fontWeight": 300, "fontSize": "20px", "color": "#333"}),
            html.P(
                "Discover who buys what, in which colours, when, and through which channel "
                "- all panels update together when you change the filter below.",
                style={"margin": "2px 0 0", "fontSize": "11.5px", "color": "#888"}),
        ], style={"flex": 1}),

        html.Div([
            html.Label("Filter product lines:",
                       style={"fontSize": "11px", "fontWeight": 700,
                              "color": "#555", "display": "block",
                              "marginBottom": "5px"}),
            dcc.Checklist(
                id="group-filter",
                options=[{"label": f"  {g}", "value": g} for g in INDEX_GROUPS],
                value=INDEX_GROUPS,
                inline=True,
                style={"fontSize": "12.5px"},
                inputStyle={"marginRight": "4px", "accentColor": "#E10028"},
                labelStyle={"marginRight": "18px", "cursor": "pointer"},
            ),
        ]),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "32px",
        "padding": "14px 28px",
        "background": "white",
        "borderBottom": "3px solid #E10028",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
        "position": "sticky", "top": 0, "zIndex": 999,
    }),

    # ── Main content ──────────────────────────────────────────────────────────
    html.Div([

        # ── Row 1: two heatmaps ────────────────────────────────────────────────
        html.Div([

            card(
                panel_header("(1)", "Who buys what?",
                             "Share of purchases by age group across product lines. "
                             "Switch normalisation to compare within-age or within-category."),
                dcc.RadioItems(id="norm1", options=NORM_OPTS, value="row",
                               inline=True, style=radio_style),
                dcc.Graph(id="heatmap-cat",
                          config={"displayModeBar": False},
                          style={"height": "285px"}),
                flex="1",
            ),

            card(
                panel_header("(2)", "What colours do they prefer?",
                             "Top 12 master colour tones by age group - "
                             "reveals aesthetic differences across generations. "
                             "Normalise by row to see each age group's colour palette."),
                dcc.RadioItems(id="norm2", options=NORM_OPTS, value="row",
                               inline=True, style=radio_style),
                dcc.Graph(id="heatmap-color",
                          config={"displayModeBar": False},
                          style={"height": "285px"}),
                flex="1",
            ),

        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # ── Row 2: trend (full width) ──────────────────────────────────────────
        card(
            panel_header("(3)", "How do sales evolve over time?",
                         "Monthly transaction volume indexed to first period (100 = baseline). "
                         "All lines share the same scale so seasonal shapes - not volumes - can be compared."),
            dcc.Graph(id="trend-chart",
                      config={"displayModeBar": False},
                      style={"height": "265px"}),
            marginBottom="12px",
        ),

        # ── Row 3: channel + bubble ────────────────────────────────────────────
        html.Div([

            card(
                panel_header("(4)", "Where do they shop?",
                             "Store vs Online proportion per product line (100% = total). "
                             "A proportional scale makes channel strategy comparable across segments."),
                dcc.Graph(id="channel-chart",
                          config={"displayModeBar": False},
                          style={"height": "265px"}),
                flex="1",
            ),

            card(
                panel_header("(5)", "What drives revenue?",
                             "Average price vs transaction volume. "
                             "Bubble area = total revenue - reveals which segments punch above their weight."),
                dcc.Graph(id="bubble-chart",
                          config={"displayModeBar": False},
                          style={"height": "265px"}),
                flex="1",
            ),

        ], style={"display": "flex", "gap": "12px"}),

    ], style={
        "padding": "16px 20px",
        "background": "#F3F3F3",
        "minHeight": "calc(100vh - 78px)",
    }),

], style={
    "fontFamily": "'Segoe UI', system-ui, -apple-system, sans-serif",
    "margin": 0, "padding": 0,
})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("heatmap-cat", "figure"),
    [Input("group-filter", "value"), Input("norm1", "value")],
)
def update_cat_heatmap(groups, norm_mode):
    if not groups:
        return go.Figure()
    df   = g_age_idx[g_age_idx["index_group_name"].isin(groups)]
    agg  = (df.groupby(["age_bin", "index_group_name"], observed=True)["count"]
              .sum().reset_index())
    col_order = [g for g in INDEX_GROUPS if g in groups]
    piv  = pivot_normalize(agg, "age_bin", "index_group_name", "count",
                           norm_mode, idx_order=AGE_LABELS, col_order=col_order)
    return make_heatmap(piv, "YlOrRd", norm_mode, xangle=-15)


@app.callback(
    Output("heatmap-color", "figure"),
    [Input("group-filter", "value"), Input("norm2", "value")],
)
def update_color_heatmap(groups, norm_mode):
    if not groups:
        return go.Figure()
    df  = g_age_color[g_age_color["index_group_name"].isin(groups)]
    agg = (df.groupby(["age_bin", "perceived_colour_master_name"], observed=True)["count"]
             .sum().reset_index())
    # Order colours by total volume descending
    col_order = (agg.groupby("perceived_colour_master_name")["count"]
                    .sum().sort_values(ascending=False).index.tolist())
    piv = pivot_normalize(agg, "age_bin", "perceived_colour_master_name", "count",
                          norm_mode, idx_order=AGE_LABELS, col_order=col_order)
    return make_heatmap(piv, "RdPu", norm_mode, xangle=-25)


@app.callback(
    Output("trend-chart", "figure"),
    Input("group-filter", "value"),
)
def update_trend(groups):
    if not groups:
        return go.Figure()
    df  = g_monthly[g_monthly["index_group_name"].isin(groups)].sort_values("month")
    fig = go.Figure()
    for group in groups:
        sub = df[df["index_group_name"] == group]
        fig.add_trace(go.Scatter(
            x=sub["month"], y=sub["indexed"],
            mode="lines", name=group,
            line=dict(color=INDEX_COLORS.get(group, "#999"), width=2.5),
            hovertemplate=(f"<b>{group}</b><br>"
                           "%{{x|%b %Y}}<br>"
                           "Index: <b>%{{y:.0f}}</b><extra></extra>"),
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="#BBBBBB",
                  annotation_text="Baseline (first month = 100)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#AAA"))
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Sales index (100 = start)", gridcolor="#EEEEEE"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    return fig


@app.callback(
    Output("channel-chart", "figure"),
    Input("group-filter", "value"),
)
def update_channel(groups):
    if not groups:
        return go.Figure()
    df = ch_piv[ch_piv.index.isin(groups)].sort_values("Online%").copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Store",
        y=df.index, x=df["Store%"],
        orientation="h",
        marker_color="#1F4E79",
        hovertemplate="<b>%{y}</b><br>Store: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Online",
        y=df.index, x=df["Online%"],
        orientation="h",
        marker_color="#E10028",
        hovertemplate="<b>%{y}</b><br>Online: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(range=[0, 100], ticksuffix="%",
                   title="% of transactions", gridcolor="#EEEEEE"),
        yaxis=dict(title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


@app.callback(
    Output("bubble-chart", "figure"),
    Input("group-filter", "value"),
)
def update_bubble(groups):
    if not groups:
        return go.Figure()
    df = g_bubble[g_bubble["index_group_name"].isin(groups)].copy()

    # Scale price to a more readable number (original is normalised 0-1)
    df["price_display"] = (df["avg_price"] * 100).round(2)

    fig = px.scatter(
        df,
        x="price_display", y="volume",
        size="revenue",
        color="index_group_name",
        text="index_group_name",
        color_discrete_map=INDEX_COLORS,
        size_max=70,
        labels={
            "price_display": "Avg price (x100 scaled)",
            "volume":        "Transaction volume",
            "index_group_name": "Product line",
        },
        custom_data=["revenue"],
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(sizemin=12, opacity=0.85, line=dict(width=1.5, color="white")),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Avg price: %{x:.2f}<br>"
            "Volume: %{y:,}<br>"
            "Revenue: %{customdata[0]:,.0f}<extra></extra>"
        ),
    )
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#EEEEEE", title="Avg price (normalised scale x100)"),
        yaxis=dict(gridcolor="#EEEEEE", title="Transaction volume"),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=False)
