"""
H&M Fashion Intelligence Dashboard  v3
──────────────────────────────────────────────────────────────────────────────
Analyst-grade BI layout:
  • Four global slicers (product line, age group, year range)
  • KPI summary row (updates with every filter change)
  • Per-chart sort controls (channel bar, heatmap columns)
  • Crossfilter: click an age-row in the left heatmap → right heatmap drills in
  • Minimal text — charts and interactions do the explaining

Run: python app.py  →  http://127.0.0.1:8050

to share, in naother terminal, run:
ngrok http 8050
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback_context

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(__file__)
ART_PATH = os.path.join(BASE, "articles.csv")
CUS_PATH = os.path.join(BASE, "customers.csv")
TX_PATH  = os.path.join(BASE, "transactions_train.csv")

# ── palette ───────────────────────────────────────────────────────────────────
INDEX_COLORS = {
    "Baby/Children": "#FF9F9F",
    "Divided":       "#E8AC00",
    "Ladieswear":    "#E10028",
    "Menswear":      "#1F4E79",
    "Sport":         "#00875A",
}
INDEX_GROUPS = list(INDEX_COLORS.keys())
AGE_BINS     = [15, 25, 35, 45, 55, 65, 100]
AGE_LABELS   = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading data ...")
art  = pd.read_csv(ART_PATH, usecols=["article_id", "index_group_name",
                                       "perceived_colour_master_name"])
cust = pd.read_csv(CUS_PATH, usecols=["customer_id", "age"])
tx   = pd.read_csv(TX_PATH,  parse_dates=["t_dat"],
                   usecols=["t_dat", "customer_id", "article_id",
                            "price", "sales_channel_id"])
tx = tx.merge(art,  on="article_id",  how="left")
tx = tx.merge(cust, on="customer_id", how="left")
print(f"Rows: {len(tx):,}")

tx["age_bin"] = pd.cut(tx["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
tx["month"]   = tx["t_dat"].dt.to_period("M").dt.to_timestamp()
tx["year"]    = tx["t_dat"].dt.year.astype(int)

tx_c = tx.dropna(subset=["age_bin", "index_group_name"]).copy()
tx_c = tx_c[tx_c["index_group_name"].isin(INDEX_GROUPS)]

yr_min = int(tx_c["year"].min())
yr_max = int(tx_c["year"].max())

# top-12 master colours by transaction volume
top_colors = (tx_c.dropna(subset=["perceived_colour_master_name"])
                   .groupby("perceived_colour_master_name")
                   .size().nlargest(12).index.tolist())
tx_col = tx_c.dropna(subset=["perceived_colour_master_name"])
tx_col = tx_col[tx_col["perceived_colour_master_name"].isin(top_colors)]

# ── pre-aggregations ──────────────────────────────────────────────────────────
# Fine-grained: all grouping dimensions, year included → callbacks just filter.

g_age_idx = (tx_c.groupby(["age_bin", "index_group_name", "year"], observed=True)
               .size().reset_index(name="count"))

g_age_color = (tx_col.groupby(
                   ["age_bin", "perceived_colour_master_name",
                    "index_group_name", "year"], observed=True)
               .size().reset_index(name="count"))

g_monthly = (tx_c.groupby(["month", "index_group_name"])
               .size().reset_index(name="count"))
# year extracted from month in callbacks

g_channel = (tx_c.groupby(["index_group_name", "sales_channel_id", "year"])
               .size().reset_index(name="count"))

g_summ = (tx_c.groupby(["index_group_name", "year"])
            .agg(count=("article_id", "count"),
                 total_price=("price", "sum"))
            .reset_index())

print("Pre-aggregation done.")

# ── helpers ───────────────────────────────────────────────────────────────────

def norm_pivot(piv, mode):
    if mode == "row":
        return piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0) * 100
    if mode == "col":
        return piv.div(piv.sum(axis=0).replace(0, np.nan), axis=1) * 100
    return piv


def heatmap_fig(piv, colorscale, mode):
    sfx  = "%" if mode != "raw" else ""
    fmt  = ".1f" if mode != "raw" else ",.0f"
    text = [[f"{v:{fmt}}{sfx}" for v in row] for row in piv.values]
    fig  = go.Figure(go.Heatmap(
        z=piv.values,
        x=piv.columns.tolist(),
        y=[str(r) for r in piv.index],
        colorscale=colorscale,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> · %{x}: %{text}<extra></extra>",
        colorbar=dict(thickness=10),
    ))
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(tickangle=-30, title=None, tickfont=dict(size=10)),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def triggered_id():
    """Returns the component id that fired the current callback (compatible API)."""
    ctx = callback_context
    if not ctx.triggered:
        return None
    return ctx.triggered[0]["prop_id"].split(".")[0]


# ── layout primitives ─────────────────────────────────────────────────────────

def card(*children, **style_extra):
    s = {"background": "white", "borderRadius": "10px",
         "padding": "12px 16px", "boxShadow": "0 1px 5px rgba(0,0,0,0.07)"}
    s.update(style_extra)
    return html.Div(list(children), style=s)


def kpi_tile(kpi_id, label):
    return html.Div([
        html.Div("—", id=kpi_id,
                 style={"fontSize": "24px", "fontWeight": 800,
                        "color": "#E10028", "lineHeight": 1}),
        html.Div(label,
                 style={"fontSize": "10px", "color": "#999", "marginTop": "4px",
                        "textTransform": "uppercase", "letterSpacing": "0.6px"}),
    ], style={"background": "white", "borderRadius": "8px", "padding": "14px 20px",
              "boxShadow": "0 1px 4px rgba(0,0,0,0.07)", "flex": 1, "minWidth": "120px"})


def chart_header(title, *controls):
    """Title + inline controls (dropdowns / radios) in one row."""
    return html.Div([
        html.Span(title, style={"fontSize": "13px", "fontWeight": 700,
                                "color": "#222", "marginRight": "12px"}),
        *controls,
    ], style={"display": "flex", "alignItems": "center",
              "flexWrap": "wrap", "marginBottom": "8px", "gap": "8px"})


def mini_radio(id_, options, value):
    return dcc.RadioItems(id=id_, options=options, value=value, inline=True,
                          style={"fontSize": "11px", "color": "#555"},
                          inputStyle={"marginRight": "3px"},
                          labelStyle={"marginRight": "10px"})


def mini_drop(id_, options, value, width=160):
    return dcc.Dropdown(id=id_, options=options, value=value,
                        clearable=False,
                        style={"fontSize": "11px", "width": f"{width}px"})


# ── option sets ───────────────────────────────────────────────────────────────
NORM = [{"label": "% of age",      "value": "row"},
        {"label": "% of category", "value": "col"},
        {"label": "Count",         "value": "raw"}]

SORT_COLS = [{"label": "Sort cols: volume",    "value": "vol"},
             {"label": "Sort cols: A→Z",        "value": "alpha"}]

SORT_CH   = [{"label": "Most online first",   "value": "online"},
             {"label": "Most in-store first", "value": "store"},
             {"label": "Alphabetical",        "value": "name"}]

SORT_ROWS = [{"label": "Sort rows: age ↑",    "value": "age"},
             {"label": "Sort rows: volume ↓", "value": "vol"}]

# ── app ───────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="H&M Fashion Intelligence")

app.layout = html.Div([

    # ── Slicer bar ────────────────────────────────────────────────────────────
    html.Div([
        # Brand
        html.Div([
            html.Span("H&M", style={"color": "#E10028", "fontWeight": 900,
                                    "fontSize": "20px", "marginRight": "6px"}),
            html.Span("Fashion Intelligence",
                      style={"color": "#333", "fontWeight": 300, "fontSize": "18px"}),
        ], style={"whiteSpace": "nowrap"}),

        # Product line
        html.Div([
            html.Label("Product line",
                       style={"fontSize": "10px", "color": "#999", "display": "block",
                              "textTransform": "uppercase", "letterSpacing": "0.5px",
                              "marginBottom": "3px"}),
            dcc.Dropdown(
                id="slicer-lines",
                options=[{"label": g, "value": g} for g in INDEX_GROUPS],
                value=INDEX_GROUPS, multi=True, clearable=False,
                style={"fontSize": "12px", "minWidth": "260px"},
            ),
        ]),

        # Age group
        html.Div([
            html.Label("Age group",
                       style={"fontSize": "10px", "color": "#999", "display": "block",
                              "textTransform": "uppercase", "letterSpacing": "0.5px",
                              "marginBottom": "3px"}),
            dcc.Dropdown(
                id="slicer-age",
                options=[{"label": a, "value": a} for a in AGE_LABELS],
                value=AGE_LABELS, multi=True, clearable=False,
                style={"fontSize": "12px", "minWidth": "220px"},
            ),
        ]),

        # Year range
        html.Div([
            html.Label("Year range",
                       style={"fontSize": "10px", "color": "#999", "display": "block",
                              "textTransform": "uppercase", "letterSpacing": "0.5px",
                              "marginBottom": "3px"}),
            dcc.RangeSlider(
                id="slicer-year",
                min=yr_min, max=yr_max, step=1,
                value=[yr_min, yr_max],
                marks={y: str(y) for y in range(yr_min, yr_max + 1)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"minWidth": "200px"}),

        # Hint
        html.Div("Click an age row in the left heatmap to drill into colours →",
                 style={"fontSize": "10px", "color": "#BBB", "fontStyle": "italic",
                        "whiteSpace": "nowrap", "marginLeft": "auto"}),

    ], style={
        "display": "flex", "alignItems": "flex-end", "gap": "20px",
        "padding": "12px 24px",
        "background": "white",
        "borderBottom": "3px solid #E10028",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
        "position": "sticky", "top": 0, "zIndex": 999,
        "flexWrap": "wrap",
    }),

    # ── KPI row ───────────────────────────────────────────────────────────────
    html.Div([
        kpi_tile("kpi-tx",     "Transactions"),
        kpi_tile("kpi-price",  "Avg price ×100"),
        kpi_tile("kpi-seg",    "Top segment"),
        kpi_tile("kpi-online", "Online share"),
    ], style={"display": "flex", "gap": "10px", "padding": "12px 20px 0"}),

    # ── Charts ────────────────────────────────────────────────────────────────
    html.Div([

        # Row 1 — two heatmaps
        html.Div([
            card(
                chart_header(
                    "Purchases · Age × Product line",
                    mini_radio("norm1", NORM, "row"),
                    mini_drop("sort1-cols", SORT_COLS, "vol", 160),
                    mini_drop("sort1-rows", SORT_ROWS, "age", 155),
                ),
                dcc.Graph(id="heatmap-cat",
                          config={"displayModeBar": False},
                          style={"height": "275px"}),
                flex="1",
            ),
            card(
                chart_header(
                    "Colour palette · Age × Master tone",
                    mini_radio("norm2", NORM, "row"),
                    mini_drop("sort2-cols", SORT_COLS, "vol", 160),
                    mini_drop("sort2-rows", SORT_ROWS, "age", 155),
                ),
                dcc.Graph(id="heatmap-color",
                          config={"displayModeBar": False},
                          style={"height": "275px"}),
                flex="1",
            ),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # Row 2 — trend (full width)
        card(
            chart_header("Monthly sales · indexed to period start  (100 = baseline)"),
            dcc.Graph(id="trend-chart",
                      config={"displayModeBar": False},
                      style={"height": "240px"}),
            marginBottom="12px",
        ),

        # Row 3 — channel + bubble
        html.Div([
            card(
                chart_header(
                    "Channel mix · Store vs Online",
                    mini_drop("sort-ch", SORT_CH, "online", 175),
                ),
                dcc.Graph(id="channel-chart",
                          config={"displayModeBar": False},
                          style={"height": "240px"}),
                flex="1",
            ),
            card(
                chart_header("Revenue landscape · price × volume (bubble = revenue)"),
                dcc.Graph(id="bubble-chart",
                          config={"displayModeBar": False},
                          style={"height": "240px"}),
                flex="1",
            ),
        ], style={"display": "flex", "gap": "12px"}),

    ], style={"padding": "12px 20px", "background": "#F2F2F2",
              "minHeight": "calc(100vh - 120px)"}),

], style={"fontFamily": "'Segoe UI', system-ui, -apple-system, sans-serif",
          "margin": 0, "padding": 0})


# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    [Output("kpi-tx", "children"),
     Output("kpi-price", "children"),
     Output("kpi-seg", "children"),
     Output("kpi-online", "children")],
    [Input("slicer-lines", "value"),
     Input("slicer-age",   "value"),
     Input("slicer-year",  "value")],
)
def update_kpis(lines, ages, yr):
    lines = lines or INDEX_GROUPS
    ages  = ages  or AGE_LABELS
    y0, y1 = yr

    ds = g_summ[(g_summ["index_group_name"].isin(lines)) &
                (g_summ["year"].between(y0, y1))]
    total  = ds["count"].sum()
    price  = ds["total_price"].sum() / max(total, 1) * 100
    top    = ds.groupby("index_group_name")["count"].sum().idxmax() if not ds.empty else "—"

    ch = g_channel[(g_channel["index_group_name"].isin(lines)) &
                   (g_channel["year"].between(y0, y1))]
    ch_total  = ch["count"].sum()
    online_n  = ch[ch["sales_channel_id"] == 2]["count"].sum()
    online_pct = online_n / max(ch_total, 1) * 100

    return (f"{total/1e6:.1f}M",
            f"×{price:.2f}",
            top,
            f"{online_pct:.0f}%")


@app.callback(
    Output("heatmap-cat", "figure"),
    [Input("slicer-lines",  "value"),
     Input("slicer-age",    "value"),
     Input("slicer-year",   "value"),
     Input("norm1",         "value"),
     Input("sort1-cols",    "value"),
     Input("sort1-rows",    "value")],
)
def update_cat(lines, ages, yr, norm, col_sort, row_sort):
    lines = lines or INDEX_GROUPS
    ages  = ages  or AGE_LABELS
    y0, y1 = yr

    df  = g_age_idx[(g_age_idx["index_group_name"].isin(lines)) &
                    (g_age_idx["age_bin"].isin(ages)) &
                    (g_age_idx["year"].between(y0, y1))]
    agg = (df.groupby(["age_bin", "index_group_name"], observed=True)["count"]
             .sum().reset_index())
    piv = (agg.pivot(index="age_bin", columns="index_group_name", values="count")
              .fillna(0))

    # sort rows
    if row_sort == "vol":
        row_order = piv.sum(axis=1).sort_values(ascending=False).index.tolist()
    else:
        row_order = [a for a in AGE_LABELS if a in piv.index]
    piv = piv.reindex(row_order)

    # sort cols
    if col_sort == "alpha":
        piv = piv[sorted(piv.columns)]
    else:
        piv = piv[piv.sum().sort_values(ascending=False).index]

    return heatmap_fig(norm_pivot(piv, norm), "YlOrRd", norm)


@app.callback(
    Output("heatmap-color", "figure"),
    [Input("slicer-lines",  "value"),
     Input("slicer-age",    "value"),
     Input("slicer-year",   "value"),
     Input("norm2",         "value"),
     Input("sort2-cols",    "value"),
     Input("sort2-rows",    "value"),
     Input("heatmap-cat",   "clickData")],   # ← crossfilter trigger
)
def update_color(lines, ages, yr, norm, col_sort, row_sort, click):
    lines = lines or INDEX_GROUPS
    ages  = ages  or AGE_LABELS
    y0, y1 = yr

    # Crossfilter: if the left heatmap row was clicked, override the age filter
    if click and triggered_id() == "heatmap-cat":
        ages = [click["points"][0]["y"]]

    df  = g_age_color[(g_age_color["index_group_name"].isin(lines)) &
                      (g_age_color["age_bin"].isin(ages)) &
                      (g_age_color["year"].between(y0, y1))]
    agg = (df.groupby(["age_bin", "perceived_colour_master_name"], observed=True)["count"]
             .sum().reset_index())

    if agg.empty:
        return go.Figure()

    piv = (agg.pivot(index="age_bin", columns="perceived_colour_master_name",
                     values="count").fillna(0))

    # sort rows
    if row_sort == "vol":
        row_order = piv.sum(axis=1).sort_values(ascending=False).index.tolist()
    else:
        row_order = [a for a in AGE_LABELS if a in piv.index]
    piv = piv.reindex(row_order)

    # sort cols
    if col_sort == "alpha":
        piv = piv[sorted(piv.columns)]
    else:
        piv = piv[piv.sum().sort_values(ascending=False).index]

    return heatmap_fig(norm_pivot(piv, norm), "RdPu", norm)


@app.callback(
    Output("trend-chart", "figure"),
    [Input("slicer-lines", "value"),
     Input("slicer-year",  "value")],
)
def update_trend(lines, yr):
    lines = lines or INDEX_GROUPS
    y0, y1 = yr

    df = g_monthly[g_monthly["index_group_name"].isin(lines)].copy()
    df = df[df["month"].dt.year.between(y0, y1)].sort_values("month")

    fig = go.Figure()
    for grp in lines:
        sub = df[df["index_group_name"] == grp].copy()
        if sub.empty:
            continue
        base = sub.iloc[0]["count"]
        sub["idx"] = sub["count"] / base * 100 if base > 0 else sub["count"]
        fig.add_trace(go.Scatter(
            x=sub["month"], y=sub["idx"], mode="lines", name=grp,
            line=dict(color=INDEX_COLORS.get(grp, "#999"), width=2.5),
            hovertemplate=f"<b>{grp}</b>  %{{x|%b %Y}}  →  %{{y:.0f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dot", line_color="#CCC",
                  annotation_text="Baseline", annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#AAA"))
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Sales index", gridcolor="#EEE"),
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    return fig


@app.callback(
    Output("channel-chart", "figure"),
    [Input("slicer-lines", "value"),
     Input("slicer-year",  "value"),
     Input("sort-ch",      "value")],
)
def update_channel(lines, yr, sort_by):
    lines = lines or INDEX_GROUPS
    y0, y1 = yr

    ch  = g_channel[(g_channel["index_group_name"].isin(lines)) &
                    (g_channel["year"].between(y0, y1))]
    agg = ch.groupby(["index_group_name", "sales_channel_id"])["count"].sum().reset_index()
    piv = (agg.pivot(index="index_group_name", columns="sales_channel_id",
                     values="count").fillna(0))
    for col in [1, 2]:
        if col not in piv.columns:
            piv[col] = 0
    piv = piv.rename(columns={1: "Store", 2: "Online"})
    piv["total"]   = piv["Store"] + piv["Online"]
    piv["Online%"] = piv["Online"] / piv["total"] * 100
    piv["Store%"]  = piv["Store"]  / piv["total"] * 100

    if sort_by == "online":
        piv = piv.sort_values("Online%", ascending=True)
    elif sort_by == "store":
        piv = piv.sort_values("Store%", ascending=True)
    else:
        piv = piv.sort_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Store", y=piv.index, x=piv["Store%"],
        orientation="h", marker_color="#1F4E79",
        hovertemplate="<b>%{y}</b>  Store %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Online", y=piv.index, x=piv["Online%"],
        orientation="h", marker_color="#E10028",
        hovertemplate="<b>%{y}</b>  Online %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(range=[0, 100], ticksuffix="%",
                   title="% of transactions", gridcolor="#EEE"),
        yaxis=dict(title=None),
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


@app.callback(
    Output("bubble-chart", "figure"),
    [Input("slicer-lines", "value"),
     Input("slicer-year",  "value")],
)
def update_bubble(lines, yr):
    lines = lines or INDEX_GROUPS
    y0, y1 = yr

    ds  = g_summ[(g_summ["index_group_name"].isin(lines)) &
                 (g_summ["year"].between(y0, y1))]
    agg = ds.groupby("index_group_name").agg(
        count=("count", "sum"), total_price=("total_price", "sum")).reset_index()
    agg["avg_price"] = agg["total_price"] / agg["count"].clip(lower=1) * 100

    fig = px.scatter(
        agg, x="avg_price", y="count",
        size="total_price", color="index_group_name", text="index_group_name",
        color_discrete_map=INDEX_COLORS, size_max=70,
        custom_data=["total_price"],
        labels={"avg_price": "Avg price ×100", "count": "Transactions",
                "index_group_name": "Line"},
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(sizemin=12, opacity=0.85, line=dict(width=1.5, color="white")),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Price ×%{x:.2f}  |  Vol %{y:,}  |  Rev %{customdata[0]:,.0f}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
