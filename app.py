"""
H&M Fashion Intelligence — Cascade Storytelling Dashboard 
──────────────────────────────────────────────────────────────────────────────
Analytical flow:

  Step 1  WHO?
  Step 2  WHAT?
  Step 3  COLOUR?
  Step 4  PRICE?

Run:  python app.py   →   http://127.0.0.1:8050
"""

import os, json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import (Dash, dcc, html, Input, Output, State,
                  ALL, callback_context, no_update)

# ── constants ──────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(__file__)
ART_PATH = os.path.join(BASE, "articles.csv")
CUS_PATH = os.path.join(BASE, "customers.csv")
TX_PATH  = os.path.join(BASE, "transactions_train.csv")

RED  = "#E10028"
GREY = "#DDDDDD"
BG   = "#F2F2F2"

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

# Colour hex mapping for Step 3 bars
COLOUR_MAP = {
    "Black":        "#1a1a1a",
    "White":        "#f5f5f5",
    "Grey":         "#888888",
    "Dark Blue":    "#003087",
    "Blue":         "#4169E1",
    "Light Blue":   "#add8e6",
    "Red":          "#E10028",
    "Pink":         "#FFB6C1",
    "Beige":        "#f5f0e1",
    "Brown":        "#8B4513",
    "Dark Green":   "#1a5c2a",
    "Green":        "#3cb371",
    "Yellow":       "#FFD700",
    "Orange":       "#FF8C00",
    "Purple":       "#6A0DAD",
    "Lilac Purple": "#C8A2C8",
    "Turquoise":    "#40E0D0",
    "Khaki green":  "#8B864E",
    "Mole":         "#7d6b5d",
    "Rust":         "#B7410E",
    "Gold":         "#FFD700",
    "Silver":       "#C0C0C0",
    "Transparent":  "#e0e0e0",
    "Other":        "#d3d3d3",
}
# Light colours that need a border to be visible on white background
LIGHT_COLOURS = {"White", "Beige", "Yellow", "Pink", "Light Blue", "Silver", "Transparent", "Other"}

# ── load & merge ───────────────────────────────────────────────────────────────
print("Loading data ...")
art  = pd.read_csv(ART_PATH, usecols=["article_id", "index_group_name",
                                       "garment_group_name",
                                       "perceived_colour_master_name"])
cust = pd.read_csv(CUS_PATH, usecols=["customer_id", "age"])
tx   = pd.read_csv(TX_PATH, parse_dates=["t_dat"],
                   usecols=["t_dat", "customer_id", "article_id",
                            "price", "sales_channel_id"])
tx = tx.merge(art,  on="article_id",  how="left")
tx = tx.merge(cust, on="customer_id", how="left")
print(f"Rows after merge: {len(tx):,}")

tx["age_bin"] = pd.cut(tx["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
tx["month"]   = tx["t_dat"].dt.to_period("M").dt.to_timestamp()
tx["year"]    = tx["t_dat"].dt.year.astype(int)

tx_c = tx.dropna(subset=["age_bin", "index_group_name", "garment_group_name"]).copy()
tx_c = tx_c[tx_c["index_group_name"].isin(INDEX_GROUPS)]

yr_min = int(tx_c["year"].min())
yr_max = int(tx_c["year"].max())

# top-12 master colours by transaction volume
top_colors = (tx_c.dropna(subset=["perceived_colour_master_name"])
                   .groupby("perceived_colour_master_name")
                   .size().nlargest(12).index.tolist())
tx_col = (tx_c[tx_c["perceived_colour_master_name"].isin(top_colors)]
              .dropna(subset=["perceived_colour_master_name"]))

# ── pre-aggregations (all callbacks filter these, never rescan raw data) ───────

# Age tile counts (static, for tile subtitles)
age_total_counts = (tx_c.groupby("age_bin", observed=True).size()
                        .reindex(AGE_LABELS, fill_value=0))

# Step 2 — garment counts
g_garm = (tx_c.groupby(
    ["age_bin", "garment_group_name", "index_group_name", "year"], observed=True)
    .size().reset_index(name="count"))

# Step 3 — colour counts + median price (restricted to top_colors)
g_col = (tx_col.groupby(
    ["age_bin", "garment_group_name", "perceived_colour_master_name",
     "index_group_name", "year"], observed=True)
    .agg(count=("price", "size"), med_price=("price", "median"))
    .reset_index())
g_col["med_price"] = g_col["med_price"] * 100   # scale ×100 for display

# Step 4a — price percentiles WITHOUT colour dimension
def _price_pct(grp_cols, src=None):
    src = src if src is not None else tx_c
    pq  = (src.groupby(grp_cols, observed=True)["price"]
              .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
              .reset_index())
    pq.columns = list(grp_cols) + ["cnt","mean","std","pmin",
                                    "p10","p25","p50","p75","p90","pmax"]
    for col in ["p10","p25","p50","p75","p90"]:
        pq[col] = pq[col] * 100          # scale to ×100 for readability
    return pq

g_price_nc = _price_pct(
    ["age_bin", "garment_group_name", "index_group_name", "year"])
g_price_wc = _price_pct(
    ["age_bin", "garment_group_name", "perceived_colour_master_name",
     "index_group_name", "year"],
    src=tx_col)

# Trend — month × age × garment × line
g_trend = (tx_c.groupby(
    ["month", "age_bin", "garment_group_name", "index_group_name"], observed=True)
    .size().reset_index(name="count"))

# KPIs — age × line × year
g_kpi = (tx_c.groupby(["age_bin", "index_group_name", "year"], observed=True)
         .agg(count=("article_id", "count"), total_price=("price", "sum"))
         .reset_index())

print("Pre-aggregation done.")

# ── layout helpers ─────────────────────────────────────────────────────────────

def card(*children, **kw):
    s = {"background": "white", "borderRadius": "10px",
         "padding": "14px 18px", "boxShadow": "0 1px 5px rgba(0,0,0,0.07)"}
    s.update(kw)
    return html.Div(list(children), style=s)

def step_hdr(num, text):
    return html.Div([
        html.Span(f"STEP {num}",
                  style={"fontSize": "9px", "color": "#BBB",
                         "letterSpacing": "1.2px", "textTransform": "uppercase",
                         "marginRight": "6px"}),
        html.Span(text, style={"fontSize": "13px", "fontWeight": 700, "color": "#222"}),
    ], style={"marginBottom": "10px"})

def kpi_tile(kid, label):
    return html.Div([
        html.Div("—", id=kid,
                 style={"fontSize": "24px", "fontWeight": 800,
                        "color": RED, "lineHeight": 1}),
        html.Div(label, style={"fontSize": "10px", "color": "#999",
                               "marginTop": "4px", "textTransform": "uppercase",
                               "letterSpacing": "0.6px"}),
    ], style={"background": "white", "borderRadius": "8px", "padding": "14px 20px",
              "boxShadow": "0 1px 4px rgba(0,0,0,0.07)", "flex": 1})

def tile_style(selected):
    return {
        "padding": "14px 16px", "borderRadius": "8px", "cursor": "pointer",
        "border": f"2px solid {RED if selected else 'transparent'}",
        "background": RED if selected else "white",
        "color": "white" if selected else "#333",
        "textAlign": "center", "minWidth": "86px", "userSelect": "none",
        "boxShadow": ("0 2px 8px rgba(225,0,40,0.3)" if selected
                      else "0 1px 4px rgba(0,0,0,0.07)"),
        "transition": "all 0.15s",
    }

def empty_fig(msg="No data for this selection"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=12, color="#BBBBBB"))
    fig.update_layout(margin=dict(l=4,r=4,t=4,b=4),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig

TRANSITION = dict(duration=300, easing="cubic-in-out")

def hbar(cats, counts, selected=None, max_bars=18):
    """Horizontal ranked bar chart; selected bar is red, others grey."""
    df = (pd.DataFrame({"cat": cats, "n": counts})
            .nlargest(max_bars, "n")
            .sort_values("n"))
    colors = [RED if c == selected else GREY for c in df["cat"]]
    fig = go.Figure(go.Bar(
        x=df["n"], y=df["cat"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=4, r=8, t=4, b=4),
        xaxis=dict(title=None, gridcolor="#EEE", showgrid=True,
                   tickfont=dict(size=10)),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        bargap=0.25,
        transition=TRANSITION,
    )
    return fig

def colour_bar(cats, counts, med_prices, top_garments, selected=None, max_bars=12):
    """
    Horizontal bar chart for Step 3 — each bar is filled with the actual colour hex.
    Selected bar gets a dark border. Light colours get a grey border.
    Enriched hover shows count + median price + top garment.
    """
    df = (pd.DataFrame({
            "cat": cats, "n": counts,
            "med": med_prices, "top_gar": top_garments,
          })
          .nlargest(max_bars, "n")
          .sort_values("n"))

    bar_colors      = []
    line_colors     = []
    line_widths     = []

    for c in df["cat"]:
        hex_fill = COLOUR_MAP.get(c, "#AAAAAA")
        bar_colors.append(hex_fill)
        if c == selected:
            line_colors.append("#333333")
            line_widths.append(2)
        elif c in LIGHT_COLOURS:
            line_colors.append("#BBBBBB")
            line_widths.append(1)
        else:
            line_colors.append(hex_fill)
            line_widths.append(0)

    customdata = list(zip(df["med"].round(2), df["top_gar"]))

    fig = go.Figure(go.Bar(
        x=df["n"], y=df["cat"], orientation="h",
        marker_color=bar_colors,
        marker_line_color=line_colors,
        marker_line_width=line_widths,
        customdata=customdata,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Transactions: %{x:,}<br>"
            "Median price: x%{customdata[0]:.2f}<br>"
            "Top garment: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        margin=dict(l=4, r=8, t=4, b=4),
        xaxis=dict(title=None, gridcolor="#EEE", showgrid=True,
                   tickfont=dict(size=10)),
        yaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        bargap=0.25,
        transition=TRANSITION,
    )
    return fig

# ── app ────────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="H&M Fashion Intelligence")

ALL_AGES = ["All"] + AGE_LABELS

# Build age tile children with static counts baked in
def _age_tile_children(a):
    if a == "All":
        total = age_total_counts.sum()
        count_str = f"{total/1e6:.1f}M"
        sub = "all ages"
    else:
        total = age_total_counts.get(a, 0)
        count_str = f"{total/1e6:.1f}M" if total >= 1_000_000 else f"{total/1e3:.0f}k"
        sub = "yrs"
    return [
        html.Div(a, style={"fontSize": "16px", "fontWeight": 700}),
        html.Div(sub, style={"fontSize": "10px", "opacity": 0.6, "marginTop": "2px"}),
        html.Div(count_str, style={"fontSize": "9px", "opacity": 0.5, "marginTop": "1px"}),
    ]

app.layout = html.Div([

    # Cascade state: single source of truth for steps 1-3
    dcc.Store(id="cascade", data={"age": None, "garment": None, "color": None}),

    # ── top slicer bar ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("H&M", style={"color": RED, "fontWeight": 900,
                                    "fontSize": "20px", "marginRight": "6px"}),
            html.Span("Fashion Intelligence",
                      style={"color": "#333", "fontWeight": 300, "fontSize": "18px"}),
        ]),
        html.Div([
            html.Label("Product line",
                       style={"fontSize": "10px", "color": "#999", "display": "block",
                              "textTransform": "uppercase", "letterSpacing": "0.5px",
                              "marginBottom": "3px"}),
            dcc.Dropdown(id="sl-lines",
                         options=[{"label": g, "value": g} for g in INDEX_GROUPS],
                         value=INDEX_GROUPS, multi=True, clearable=False,
                         style={"fontSize": "12px", "minWidth": "280px"}),
        ]),
        html.Div([
            html.Label("Year range",
                       style={"fontSize": "10px", "color": "#999", "display": "block",
                              "textTransform": "uppercase", "letterSpacing": "0.5px",
                              "marginBottom": "3px"}),
            dcc.RangeSlider(id="sl-year", min=yr_min, max=yr_max, step=1,
                            value=[yr_min, yr_max],
                            marks={y: str(y) for y in range(yr_min, yr_max + 1)},
                            tooltip={"placement": "bottom", "always_visible": False}),
        ], style={"minWidth": "200px"}),
    ], style={
        "display": "flex", "alignItems": "flex-end", "gap": "24px",
        "padding": "12px 24px", "background": "white",
        "borderBottom": f"3px solid {RED}",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
        "position": "sticky", "top": 0, "zIndex": 999, "flexWrap": "wrap",
    }),

    # ── main ───────────────────────────────────────────────────────────────────
    html.Div([

        # KPI row
        html.Div([
            kpi_tile("kpi-tx",  "Transactions"),
            kpi_tile("kpi-med", "Median price x100"),
            kpi_tile("kpi-top", "Top product line"),
        ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"}),

        # Breadcrumb strip — sticky below header
        html.Div([
            html.Span("Path: ", style={"fontSize": "11px", "color": "#BBB",
                                       "marginRight": "4px"}),
            html.Span(id="bc-age", n_clicks=0,
                      style={"fontSize": "12px", "cursor": "pointer", "color": "#888"}),
            html.Span(id="bc-sep1", children=" > ",
                      style={"fontSize": "12px", "color": "#CCC", "display": "none"}),
            html.Span(id="bc-gar", n_clicks=0,
                      style={"fontSize": "12px", "cursor": "pointer",
                             "color": RED, "fontWeight": 600, "display": "none"}),
            html.Span(id="bc-sep2", children=" > ",
                      style={"fontSize": "12px", "color": "#CCC", "display": "none"}),
            html.Span(id="bc-col", n_clicks=0,
                      style={"fontSize": "12px", "cursor": "pointer",
                             "color": RED, "fontWeight": 600, "display": "none"}),
            html.Span(id="bc-end", children=" > price breakdown",
                      style={"fontSize": "12px", "color": "#BBB",
                             "fontStyle": "italic", "display": "none"}),
            html.Span(" (click a segment to reset from that point)",
                      style={"fontSize": "10px", "color": "#CCC",
                             "marginLeft": "8px", "fontStyle": "italic"}),
            # Reset all button
            html.Button("✕ Reset all", id="btn-reset-all", n_clicks=0,
                        style={
                            "marginLeft": "auto", "fontSize": "11px",
                            "color": "#999", "background": "none",
                            "border": "1px solid #DDD", "borderRadius": "4px",
                            "padding": "2px 10px", "cursor": "pointer",
                        }),
        ], style={
            "display": "flex", "alignItems": "center", "flexWrap": "wrap",
            "padding": "6px 14px", "background": "white", "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
            "marginBottom": "12px",
            "position": "sticky", "top": "74px", "zIndex": 998,
        }),

        # Step 1 — age tiles
        card(
            step_hdr(1, "Who is buying?"),
            html.Div([
                html.Div(
                    _age_tile_children(a),
                    id={"type": "age-tile", "index": a},
                    n_clicks=0,
                    style=tile_style(a == "All"),
                )
                for a in ALL_AGES
            ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
            marginBottom="12px",
        ),

        # Steps 2, 3, 4 — side by side
        html.Div([
            card(
                step_hdr(2, "What garment type?"),
                html.Div("Click a bar to filter colour and price",
                         style={"fontSize": "10px", "color": "#BBB",
                                "marginBottom": "6px", "fontStyle": "italic"}),
                dcc.Loading(type="dot", color=RED, children=[
                    dcc.Graph(id="garm-chart", config={"displayModeBar": False},
                              style={"height": "380px"}),
                ]),
                flex="1",
            ),
            card(
                step_hdr(3, "Which colour?"),
                html.Div("Click a bar to lock in the price view",
                         style={"fontSize": "10px", "color": "#BBB",
                                "marginBottom": "6px", "fontStyle": "italic"}),
                dcc.Loading(type="dot", color=RED, children=[
                    dcc.Graph(id="col-chart", config={"displayModeBar": False},
                              style={"height": "380px"}),
                ]),
                flex="1",
            ),
            card(
                step_hdr(4, "At what price?"),
                html.Div("Price distribution by product line for the active selection",
                         style={"fontSize": "10px", "color": "#BBB",
                                "marginBottom": "6px", "fontStyle": "italic"}),
                dcc.Loading(type="dot", color=RED, children=[
                    dcc.Graph(id="price-chart", config={"displayModeBar": False},
                              style={"height": "360px"}),
                ]),
                html.Div("Whiskers show the P10–P90 range. Extreme outliers (top/bottom 10%) "
                         "are excluded to focus on typical pricing.",
                         style={"fontSize": "9px", "color": "#CCC",
                                "marginTop": "6px", "fontStyle": "italic"}),
                flex="1",
            ),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # Trend — context panel
        card(
            html.Div("Sales trend — monthly index for current selection  "
                     "(100 = first month in range)",
                     style={"fontSize": "12px", "fontWeight": 600, "color": "#555",
                            "marginBottom": "8px"}),
            dcc.Loading(type="dot", color=RED, children=[
                dcc.Graph(id="trend-chart", config={"displayModeBar": False},
                          style={"height": "220px"}),
            ]),
        ),

    ], style={"padding": "12px 20px", "background": BG,
              "minHeight": "calc(100vh - 74px)"}),

], style={"fontFamily": "'Segoe UI', system-ui, -apple-system, sans-serif",
          "margin": 0, "padding": 0})


# ── callbacks ──────────────────────────────────────────────────────────────────

def _triggered():
    ctx = callback_context
    return ctx.triggered[0]["prop_id"] if ctx.triggered else ""


# 1. Cascade state — single callback handles every possible trigger ─────────────
@app.callback(
    Output("cascade", "data"),
    [Input({"type": "age-tile", "index": ALL}, "n_clicks"),
     Input("garm-chart",    "clickData"),
     Input("col-chart",     "clickData"),
     Input("bc-age",        "n_clicks"),
     Input("bc-gar",        "n_clicks"),
     Input("bc-col",        "n_clicks"),
     Input("btn-reset-all", "n_clicks")],
    State("cascade", "data"),
    prevent_initial_call=True,
)
def update_cascade(age_clicks, gar_click, col_click,
                   _bca, _bcg, _bcc, _reset, current):
    t = _triggered()
    c = current or {"age": None, "garment": None, "color": None}

    # Reset all
    if "btn-reset-all" in t:
        return {"age": None, "garment": None, "color": None}

    # Age tile clicked
    if "age-tile" in t:
        sel     = json.loads(t.split(".")[0])["index"]
        new_age = None if sel == "All" else sel
        return {"age": new_age, "garment": None, "color": None}

    # Garment bar clicked (toggle: click selected bar again to deselect)
    if "garm-chart.clickData" in t and gar_click:
        val     = gar_click["points"][0]["y"]
        new_gar = None if c.get("garment") == val else val
        return {**c, "garment": new_gar, "color": None}

    # Colour bar clicked (toggle)
    if "col-chart.clickData" in t and col_click:
        val     = col_click["points"][0]["y"]
        new_col = None if c.get("color") == val else val
        return {**c, "color": new_col}

    # Breadcrumb resets
    if "bc-age" in t:
        return {"age": None, "garment": None, "color": None}
    if "bc-gar" in t:
        return {**c, "garment": None, "color": None}
    if "bc-col" in t:
        return {**c, "color": None}

    return no_update


# 2. Age tile visual highlight ─────────────────────────────────────────────────
@app.callback(
    Output({"type": "age-tile", "index": ALL}, "style"),
    Input("cascade", "data"),
    State({"type": "age-tile", "index": ALL}, "id"),
)
def update_tile_styles(cascade, ids):
    sel = (cascade or {}).get("age")          # None = "All" selected
    return [
        tile_style((i["index"] == "All" and sel is None) or i["index"] == sel)
        for i in ids
    ]


# 3. Breadcrumb ────────────────────────────────────────────────────────────────
@app.callback(
    [Output("bc-age",  "children"), Output("bc-age",  "style"),
     Output("bc-sep1", "style"),
     Output("bc-gar",  "children"), Output("bc-gar",  "style"),
     Output("bc-sep2", "style"),
     Output("bc-col",  "children"), Output("bc-col",  "style"),
     Output("bc-end",  "style")],
    Input("cascade", "data"),
)
def update_breadcrumb(cascade):
    c   = cascade or {}
    age = c.get("age")
    gar = c.get("garment")
    col = c.get("color")

    shown = {"fontSize": "12px", "display": "inline"}
    hide  = {"display": "none"}

    age_style = {**shown, "cursor": "pointer",
                 "color": RED if age else "#888",
                 "fontWeight": 700 if age else 400}
    sep1  = {**shown, "color": "#CCC"} if gar else hide
    gar_s = {**shown, "cursor": "pointer", "color": RED,
             "fontWeight": 600} if gar else hide
    sep2  = {**shown, "color": "#CCC"} if col else hide
    col_s = {**shown, "cursor": "pointer", "color": RED,
             "fontWeight": 600} if col else hide
    end_s = {**shown, "color": "#BBB", "fontStyle": "italic"} if (age or gar or col) else hide

    return (age or "All ages", age_style,
            sep1,
            gar or "", gar_s,
            sep2,
            col or "", col_s,
            end_s)


# 4. KPIs ──────────────────────────────────────────────────────────────────────
@app.callback(
    [Output("kpi-tx",  "children"),
     Output("kpi-med", "children"),
     Output("kpi-top", "children")],
    [Input("cascade",  "data"),
     Input("sl-lines", "value"),
     Input("sl-year",  "value")],
)
def update_kpis(cascade, lines, yr):
    lines  = lines or INDEX_GROUPS
    c      = cascade or {}
    age    = c.get("age")
    y0, y1 = yr

    df = g_kpi[(g_kpi["index_group_name"].isin(lines)) &
               (g_kpi["year"].between(y0, y1))]
    if age:
        df = df[df["age_bin"] == age]

    total = int(df["count"].sum())
    top   = (df.groupby("index_group_name")["count"].sum().idxmax()
             if not df.empty else "—")

    pf = g_price_nc[(g_price_nc["index_group_name"].isin(lines)) &
                    (g_price_nc["year"].between(y0, y1))]
    if age:
        pf = pf[pf["age_bin"] == age]
    med = pf["p50"].median() if not pf.empty else 0.0

    tx_label = f"{total/1e6:.1f}M" if total >= 1_000_000 else f"{total:,}"
    return tx_label, f"x{med:.2f}", top


# 5. Step 2 — garment bar ──────────────────────────────────────────────────────
@app.callback(
    Output("garm-chart", "figure"),
    [Input("cascade",  "data"),
     Input("sl-lines", "value"),
     Input("sl-year",  "value")],
)
def update_garm(cascade, lines, yr):
    lines  = lines or INDEX_GROUPS
    c      = cascade or {}
    age    = c.get("age")
    y0, y1 = yr

    df = g_garm[(g_garm["index_group_name"].isin(lines)) &
                (g_garm["year"].between(y0, y1))]
    if age:
        df = df[df["age_bin"] == age]

    agg = df.groupby("garment_group_name")["count"].sum().reset_index()
    if agg.empty:
        return empty_fig()

    return hbar(agg["garment_group_name"], agg["count"],
                selected=c.get("garment"))


# 6. Step 3 — colour bar ───────────────────────────────────────────────────────
@app.callback(
    Output("col-chart", "figure"),
    [Input("cascade",  "data"),
     Input("sl-lines", "value"),
     Input("sl-year",  "value")],
)
def update_col(cascade, lines, yr):
    lines  = lines or INDEX_GROUPS
    c      = cascade or {}
    age    = c.get("age")
    gar    = c.get("garment")
    y0, y1 = yr

    df = g_col[(g_col["index_group_name"].isin(lines)) &
               (g_col["year"].between(y0, y1))]
    if age:
        df = df[df["age_bin"] == age]
    if gar:
        df = df[df["garment_group_name"] == gar]

    if df.empty:
        return empty_fig()

    # Aggregate: sum count, weighted-avg median price, top garment per colour
    agg_count = (df.groupby("perceived_colour_master_name")["count"]
                   .sum().reset_index(name="count"))
    agg_med   = (df.groupby("perceived_colour_master_name")
                   .apply(lambda g: np.average(g["med_price"], weights=g["count"]))
                   .reset_index(name="med_price"))
    agg_top   = (df.groupby(["perceived_colour_master_name", "garment_group_name"])["count"]
                   .sum().reset_index()
                   .sort_values("count", ascending=False)
                   .drop_duplicates("perceived_colour_master_name")
                   [["perceived_colour_master_name", "garment_group_name"]]
                   .rename(columns={"garment_group_name": "top_garment"}))

    agg = (agg_count
           .merge(agg_med,  on="perceived_colour_master_name")
           .merge(agg_top,  on="perceived_colour_master_name", how="left"))
    agg["top_garment"] = agg["top_garment"].fillna("—")

    return colour_bar(
        agg["perceived_colour_master_name"],
        agg["count"],
        agg["med_price"],
        agg["top_garment"],
        selected=c.get("color"),
        max_bars=12,
    )


# 7. Step 4 — price box plot ───────────────────────────────────────────────────
@app.callback(
    Output("price-chart", "figure"),
    [Input("cascade",  "data"),
     Input("sl-lines", "value"),
     Input("sl-year",  "value")],
)
def update_price(cascade, lines, yr):
    lines  = lines or INDEX_GROUPS
    c      = cascade or {}
    age    = c.get("age")
    gar    = c.get("garment")
    col    = c.get("color")
    y0, y1 = yr

    if col:
        df = g_price_wc[g_price_wc["perceived_colour_master_name"] == col].copy()
    else:
        df = g_price_nc.copy()

    df = df[(df["index_group_name"].isin(lines)) &
            (df["year"].between(y0, y1))]
    if age:
        df = df[df["age_bin"] == age]
    if gar:
        df = df[df["garment_group_name"] == gar]

    agg = (df.groupby("index_group_name")[["p10","p25","p50","p75","p90"]]
             .median().reset_index())
    if agg.empty:
        return empty_fig()

    fig = go.Figure()
    for _, row in agg.iterrows():
        grp = row["index_group_name"]
        fig.add_trace(go.Box(
            name=grp,
            median=[row["p50"]],
            q1=[row["p25"]],
            q3=[row["p75"]],
            lowerfence=[row["p10"]],
            upperfence=[row["p90"]],
            marker_color=INDEX_COLORS.get(grp, "#999"),
            line_color=INDEX_COLORS.get(grp, "#999"),
            showlegend=False,
            boxpoints=False,
        ))
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        yaxis=dict(title="Price (normalised x100)", gridcolor="#EEE",
                   tickfont=dict(size=11)),
        xaxis=dict(title=None, tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        transition=TRANSITION,
    )
    return fig


# 8. Trend — context panel ─────────────────────────────────────────────────────
@app.callback(
    Output("trend-chart", "figure"),
    [Input("cascade",  "data"),
     Input("sl-lines", "value"),
     Input("sl-year",  "value")],
)
def update_trend(cascade, lines, yr):
    lines  = lines or INDEX_GROUPS
    c      = cascade or {}
    age    = c.get("age")
    gar    = c.get("garment")
    y0, y1 = yr

    df = g_trend[(g_trend["index_group_name"].isin(lines)) &
                 (g_trend["month"].dt.year.between(y0, y1))]
    if age:
        df = df[df["age_bin"] == age]
    if gar:
        df = df[df["garment_group_name"] == gar]

    monthly = (df.groupby(["month", "index_group_name"])["count"]
                 .sum().reset_index())

    # Find overall peak month for annotation
    if not monthly.empty:
        peak_m = monthly.groupby("month")["count"].sum().idxmax()
    else:
        peak_m = None

    fig = go.Figure()
    for grp in lines:
        sub = monthly[monthly["index_group_name"] == grp].sort_values("month").copy()
        if sub.empty:
            continue
        base = sub.iloc[0]["count"]
        sub["idx"] = sub["count"] / base * 100 if base > 0 else sub["count"]
        fig.add_trace(go.Scatter(
            x=sub["month"], y=sub["idx"],
            mode="lines", name=grp,
            line=dict(color=INDEX_COLORS.get(grp, "#999"), width=2.5),
            hovertemplate=f"<b>{grp}</b>  %{{x|%b %Y}}: %{{y:.0f}}<extra></extra>",
        ))
        # Peak annotation dot
        if peak_m is not None:
            peak_row = sub[sub["month"] == peak_m]
            if not peak_row.empty:
                peak_y = float(peak_row.iloc[0]["idx"])
                fig.add_trace(go.Scatter(
                    x=[peak_m], y=[peak_y],
                    mode="markers+text",
                    marker=dict(color=RED, size=8, symbol="circle"),
                    text=[f"Peak<br>{peak_m.strftime('%b %Y')}"],
                    textposition="top center",
                    textfont=dict(size=9, color=RED),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    fig.add_hline(y=100, line_dash="dot", line_color="#DDD",
                  annotation_text="Baseline",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#BBB"))
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Sales index (100 = start of range)",
                   gridcolor="#EEE"),
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        transition=TRANSITION,
    )
    return fig


if __name__ == "__main__":
    app.run(debug=False)
