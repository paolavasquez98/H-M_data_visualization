# H&M Fashion Intelligence Dashboard

An interactive **cascade storytelling dashboard** built with [Plotly Dash](https://dash.plotly.com/). It explores the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset by guiding the user through four linked analytical steps:

| Step | Question | What you see |
|------|----------|--------------|
| **1 — WHO?** | Who is buying? | Age-group tiles (15–24, 25–34, …, 65+, or All) |
| **2 — WHAT?** | What garment type? | Horizontal bar chart of garment groups |
| **3 — COLOUR?** | Which colour? | Colour-coded bars for the top 12 colours |
| **4 — PRICE?** | At what price? | Box plots of price percentiles by product line |

Each selection narrows the next view. A breadcrumb trail and **Reset all** button let you step back at any point.

---

## What the app does

On startup, `app.py`:

1. Loads three CSV files from the project root.
2. Merges transactions with article metadata (product line, garment type, colour) and customer age.
3. Pre-aggregates counts and price statistics so the dashboard stays responsive during interaction.
4. Serves a single-page Dash app at **http://127.0.0.1:8050**.

### Global filters (top bar)

- **Product line** — Baby/Children, Divided, Ladieswear, Menswear, Sport (multi-select).
- **Year range** — Slider over the years present in the filtered data.

### KPI row

Updates with the current filters and cascade selection:

- Total transactions
- Median price (displayed as **×100** for readability — raw prices in the dataset are small decimals)
- Top product line by volume

### Cascade interaction

1. Click an **age tile** to filter by age group (or **All** for no age filter).
2. Click a **garment bar** in Step 2 to drill into that garment type (click again to deselect).
3. Click a **colour bar** in Step 3 to lock in the price view (click again to deselect).
4. Step 4 shows price distributions (P10–P90 whiskers) per product line for the active selection.

The **trend panel** at the bottom plots monthly sales indexed to 100 at the start of the selected year range, broken down by product line.

---

## Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- Enough disk space and RAM for the dataset (~3.5 GB on disk; several GB of RAM recommended during load)

---

## Data files

The app expects these files in the **same folder as `app.py`**:

| File | Approx. size | Required columns used |
|------|--------------|------------------------|
| `articles.csv` | ~34 MB | `article_id`, `index_group_name`, `garment_group_name`, `perceived_colour_master_name` |
| `customers.csv` | ~198 MB | `customer_id`, `age` |
| `transactions_train.csv` | ~3.2 GB | `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id` |

These files are **not tracked in git** (see `.gitignore`). Download them from the [Kaggle competition data page](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) and place them in the project root:

```
Project/
├── app.py
├── articles.csv
├── customers.csv
└── transactions_train.csv
```

---

## Installation

From the project directory, create a virtual environment (recommended) and install dependencies:

```bash
cd Project

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install dash pandas numpy plotly
```

Optional: pin versions for reproducibility:

```bash
pip install "dash>=2.14" "pandas>=2.0" "numpy>=1.24" "plotly>=5.18"
```

---

## Running the app

```bash
python app.py
```

You should see console output similar to:

```
Loading data ...
Rows after merge: ...
Pre-aggregation done.
```

Then open a browser at:

**http://127.0.0.1:8050**

> **Note:** The first launch can take **several minutes** while the 3 GB transaction file is read and aggregated. Subsequent page loads are fast; only restart the server if you change the data or code.

To stop the server, press `Ctrl+C` in the terminal.

### Troubleshooting

| Issue | What to try |
|-------|-------------|
| `FileNotFoundError` for a CSV | Ensure all three data files are in the project root next to `app.py`. |
| `ModuleNotFoundError: No module named 'dash'` | Activate your virtual environment and run `pip install dash pandas numpy plotly`. |
| Port already in use | Another process is using port 8050. Stop it or change the port in `app.py` (last line): `app.run(debug=False, port=8051)`. |
| Slow startup / high memory | Expected with the full transaction file. Close other heavy applications or use a machine with more RAM. |

---

## Project structure

```
Project/
├── app.py                      # Dash application (data load, layout, callbacks)
├── articles.csv                # Product catalogue (from kaggle, not in git)
├── customers.csv               # Customer demographics (from kaggle, not in git)
├── transactions_train.csv      # Purchase history (from kaggle, not in git)
└── README.md
```

---

## Tech stack

- **Dash** — web UI and callbacks
- **Plotly** — bar charts, box plots, trend lines
- **Pandas / NumPy** — data merging and pre-aggregation

---

This project is part of a **Data Visualization (DV)** course. The dashboard turns H&M transaction data into an exploratory narrative: start with *who* buys, then *what*, *which colour*, and finally *at what price* — supporting personalized fashion recommendation analysis and presentation.

