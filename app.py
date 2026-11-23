from datetime import date
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IslandHarvest · Sri Lanka Food Prices",
    page_icon=":material/monitoring:",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "wfp_food_prices_lka.csv"
MODEL_DIR = BASE_DIR / "models"
LOGO_PATH = BASE_DIR / "assets" / "logo.svg"

# Palette (kept in sync with .streamlit/config.toml)
ACCENT = "#059669"
ACCENT_DARK = "#047857"
AMBER = "#F59E0B"
MUTED_BAR = "#CBD5E1"
INK = "#0F172A"
MUTED_INK = "#64748B"
GRID = "#EEF2F6"


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, button, input, textarea, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-feature-settings: "cv11", "ss01";
    }

    .block-container {
        max-width: 1280px;
        padding-top: 4.5rem !important;
        padding-bottom: 3rem !important;
    }

    [data-testid="stAppDeployButton"] {display: none;}
    footer {visibility: hidden;}

    /* ---------- Page header ---------- */
    .page-head {margin: 0.5rem 0 1.5rem;}
    .eyebrow {
        display: inline-flex; align-items: center; gap: .4rem;
        font-size: .75rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
        color: #047857; margin-bottom: .55rem;
    }
    .eyebrow::before {
        content: ""; width: 6px; height: 6px; border-radius: 999px; background: #10B981;
    }
    .page-title {
        font-size: 2rem; font-weight: 700; letter-spacing: -0.025em; line-height: 1.15;
        color: #0F172A; margin: 0 0 .45rem;
    }
    .page-sub {font-size: 1rem; color: #64748B; line-height: 1.6; margin: 0; max-width: 72ch;}

    /* ---------- Cards (st.container(key="card-...")) ---------- */
    [class*="st-key-card"] {
        background: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-radius: 16px;
        padding: 1.25rem 1.35rem 1.1rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
    }
    .card-title {font-size: 1rem; font-weight: 600; color: #0F172A; margin: 0; letter-spacing: -0.01em;}
    .card-sub {font-size: .85rem; color: #64748B; margin: .2rem 0 .4rem; line-height: 1.5;}

    /* ---------- KPI tiles ---------- */
    .kpi-grid {
        display: grid; gap: 14px; margin: 0 0 1.25rem;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }
    .kpi {
        background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
    }
    .kpi-label {font-size: .8rem; font-weight: 500; color: #64748B; margin-bottom: .35rem;}
    .kpi-value {
        font-size: 1.6rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums; line-height: 1.2;
    }
    .kpi-hint {font-size: .78rem; color: #94A3B8; margin-top: .3rem;}
    .kpi-hint.up {color: #B45309;}
    .kpi-hint.down {color: #047857;}

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden;
        background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 20px;
        padding: 2.1rem 2.2rem; margin: .25rem 0 1.25rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
    }
    .hero::after {
        content: ""; position: absolute; right: -60px; top: -60px; width: 340px; height: 340px;
        background-image: radial-gradient(#D1FAE5 1.6px, transparent 1.6px);
        background-size: 16px 16px; opacity: .9;
        mask-image: radial-gradient(circle at 70% 30%, #000 0%, transparent 70%);
        -webkit-mask-image: radial-gradient(circle at 70% 30%, #000 0%, transparent 70%);
    }
    .hero-title {
        position: relative; z-index: 1;
        font-size: clamp(1.8rem, 3.2vw, 2.6rem); font-weight: 700; letter-spacing: -0.03em;
        line-height: 1.12; color: #0F172A; margin: .2rem 0 .7rem; max-width: 22ch;
    }
    .hero-title span {color: #047857;}
    .hero-copy {position: relative; z-index: 1; color: #475569; font-size: 1rem; line-height: 1.65; max-width: 62ch; margin: 0;}
    .chip-row {position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.2rem;}
    .chip {
        display: inline-flex; align-items: center; gap: .35rem;
        font-size: .8rem; font-weight: 500; color: #334155;
        background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 999px; padding: .3rem .7rem;
    }
    .chip b {color: #0F172A; font-weight: 600;}

    /* ---------- Info lists ---------- */
    .info-list {margin: .35rem 0 0; padding: 0; list-style: none;}
    .info-list li {
        display: grid; grid-template-columns: 130px 1fr; gap: 1rem;
        padding: .7rem 0; border-top: 1px solid #F1F5F9; font-size: .9rem; line-height: 1.55;
    }
    .info-list li:first-child {border-top: none;}
    .info-list .k {color: #64748B; font-weight: 500;}
    .info-list .v {color: #1E293B;}
    .info-list a {color: #047857; text-decoration: none; font-weight: 500;}

    /* ---------- Prediction result ---------- */
    .result {
        background: #0F172A; color: #E2E8F0; border-radius: 18px; padding: 1.6rem 1.7rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, .18);
    }
    .result-top {display: flex; justify-content: space-between; align-items: center; gap: 1rem;}
    .result-label {font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #94A3B8;}
    .result-badge {
        font-size: .75rem; font-weight: 600; color: #A7F3D0;
        background: rgba(16, 185, 129, .14); border: 1px solid rgba(16, 185, 129, .3);
        padding: .22rem .6rem; border-radius: 999px; white-space: nowrap;
    }
    .result-value {
        font-size: 2.7rem; font-weight: 700; letter-spacing: -0.03em; color: #FFFFFF;
        margin: .55rem 0 .15rem; font-variant-numeric: tabular-nums;
    }
    .result-value small {font-size: 1.1rem; font-weight: 600; color: #94A3B8; margin-right: .35rem;}
    .result-range {font-size: .85rem; color: #94A3B8;}
    .result-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: .9rem 1.2rem;
        margin-top: 1.3rem; padding-top: 1.2rem; border-top: 1px solid rgba(148, 163, 184, .2);
    }
    .result-grid .k {font-size: .75rem; color: #94A3B8; margin-bottom: .15rem;}
    .result-grid .v {font-size: .92rem; color: #F8FAFC; font-weight: 500;}

    .empty {
        border: 1.5px dashed #CBD5E1; border-radius: 18px; padding: 3rem 1.5rem; text-align: center;
        color: #64748B; background: #FFFFFF;
    }
    .empty-icon {
        width: 44px; height: 44px; border-radius: 12px; margin: 0 auto .9rem;
        display: flex; align-items: center; justify-content: center;
        background: #ECFDF5; color: #047857; font-size: 1.3rem; font-weight: 700;
    }
    .empty h4 {margin: 0 0 .35rem; color: #0F172A; font-size: 1rem; font-weight: 600;}
    .empty p {margin: 0 auto; max-width: 40ch; font-size: .9rem; line-height: 1.6;}

    .stat-line {display: flex; justify-content: space-between; padding: .55rem 0; border-top: 1px solid #F1F5F9; font-size: .88rem;}
    .stat-line:first-of-type {border-top: none;}
    .stat-line .k {color: #64748B;}
    .stat-line .v {color: #0F172A; font-weight: 600; font-variant-numeric: tabular-nums;}

    .app-footer {text-align: center; color: #94A3B8; font-size: .8rem; margin-top: 2.5rem;}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {font-weight: 600;}
    .stButton > button[kind="primary"] {box-shadow: 0 1px 2px rgba(4, 120, 87, .25);}

    @media (max-width: 800px) {
        .hero {padding: 1.5rem;}
        .info-list li {grid-template-columns: 1fr; gap: .2rem;}
        .result-grid {grid-template-columns: 1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# UI HELPERS
# --------------------------------------------------

def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div><p class="page-sub">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items) -> None:
    """items: list of (label, value, hint, hint_class)."""
    tiles = []
    for item in items:
        label, value, hint, cls = (list(item) + [None, None])[:4]
        hint_html = f'<div class="kpi-hint {cls or ""}">{hint}</div>' if hint else ""
        tiles.append(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{hint_html}</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


def card_header(title: str, subtitle: str | None = None) -> None:
    sub = f'<p class="card-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<p class="card-title">{title}</p>{sub}', unsafe_allow_html=True)


def card(key: str):
    return st.container(key=f"card-{key}")


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.properties(background="transparent", padding={"left": 4, "right": 12, "top": 10, "bottom": 4})
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED_INK,
            titleColor=MUTED_INK,
            gridColor=GRID,
            gridDash=[3, 3],
            domain=False,
            ticks=False,
            labelFont="Inter",
            titleFont="Inter",
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=500,
            labelPadding=8,
        )
        .configure_axisX(grid=False, domain=True, domainColor="#E2E8F0")
        .configure_text(font="Inter")
    )


# --------------------------------------------------
# DATA & MODEL LOADING
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


@st.cache_resource(show_spinner=False)
def load_artifact(file_name: str):
    """Load a saved model / preprocessor once and keep it in memory."""
    return joblib.load(MODEL_DIR / file_name)


# Test-set (2025) results from the notebook
MODEL_RESULTS = pd.DataFrame(
    [
        {"Model": "Stacking Ensemble", "MAE": 120.4602, "RMSE": 174.0408, "R2": 0.9290},
        {"Model": "XGBoost", "MAE": 124.3591, "RMSE": 180.6740, "R2": 0.9235},
        {"Model": "CatBoost", "MAE": 125.6818, "RMSE": 181.6840, "R2": 0.9226},
        {"Model": "LightGBM", "MAE": 126.9739, "RMSE": 177.3770, "R2": 0.9262},
        {"Model": "Random Forest", "MAE": 134.5089, "RMSE": 202.1905, "R2": 0.9041},
        {"Model": "Hybrid RF + ANN", "MAE": 142.5718, "RMSE": 214.4527, "R2": 0.8922},
    ]
)

MODEL_INFO = {
    "Stacking Ensemble": {
        "preprocessor": "stacking_preprocessor.pkl",
        "description": (
            "Random Forest, XGBoost, LightGBM and CatBoost base learners whose predictions "
            "are combined by a Linear Regression meta-learner."
        ),
        "training": "Base learners trained on 2004–2023, meta-learner fitted on 2024.",
    },
    "XGBoost": {
        "preprocessor": "preprocessor.pkl",
        "file": "xgboost_model.pkl",
        "description": "Gradient-boosted decision trees with regularisation, tuned with RandomizedSearchCV.",
        "training": "Trained on 2004–2024.",
    },
    "LightGBM": {
        "preprocessor": "preprocessor.pkl",
        "file": "lightgbm_model.pkl",
        "description": "Leaf-wise gradient boosting optimised for speed, tuned with RandomizedSearchCV.",
        "training": "Trained on 2004–2024.",
    },
    "CatBoost": {
        "preprocessor": "preprocessor.pkl",
        "file": "catboost_model.pkl",
        "description": "Gradient boosting with ordered boosting to reduce overfitting, tuned with RandomizedSearchCV.",
        "training": "Trained on 2004–2024.",
    },
    "Random Forest": {
        "preprocessor": "preprocessor.pkl",
        "file": "random_forest_model.pkl",
        "description": "Bagging ensemble of decision trees, used as the baseline model.",
        "training": "Trained on 2004–2024.",
    },
    "Hybrid RF + ANN": {
        "preprocessor": "stacking_preprocessor.pkl",
        "description": (
            "Weighted average of a tuned Random Forest and an MLP neural network, "
            "with weights selected on the 2024 validation set."
        ),
        "training": "Trained on 2004–2023, weights chosen on 2024.",
    },
}


def predict_price(model_name: str, input_df: pd.DataFrame) -> float:
    """Return a single price prediction (LKR) from the selected model."""
    info = MODEL_INFO[model_name]
    X = load_artifact(info["preprocessor"]).transform(input_df)

    if model_name == "Stacking Ensemble":
        base_models = load_artifact("stacking_base_models.pkl")
        meta_model = load_artifact("stacking_meta_model.pkl")
        base_preds = np.column_stack([m.predict(X) for _, m in base_models])
        return float(meta_model.predict(base_preds)[0])

    if model_name == "Hybrid RF + ANN":
        bundle = load_artifact("hybrid_rf_ann_model.pkl")
        rf_pred = bundle["rf_model"].predict(X)[0]
        ann_pred = bundle["ann_model"].predict(X)[0] if bundle["ann_weight"] else 0.0
        return float(bundle["rf_weight"] * rf_pred + bundle["ann_weight"] * ann_pred)

    return float(load_artifact(info["file"]).predict(X)[0])


df = load_data()
market_coordinates = df.groupby("market")[["latitude", "longitude"]].median()

min_year = int(df["date"].dt.year.min())
max_year = int(df["date"].dt.year.max())


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

def overview_page():
    best = MODEL_RESULTS.loc[MODEL_RESULTS["MAE"].idxmin()]

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Sri Lanka food price monitor</div>
            <div class="hero-title">Track food prices and <span>predict what comes next.</span></div>
            <p class="hero-copy">
                {len(df):,} market price records from the World Food Programme, covering
                {df['commodity'].nunique()} commodities across {df['market'].nunique()} markets.
                Explore the history, then estimate prices with six trained machine learning models.
            </p>
            <div class="chip-row">
                <span class="chip">Source <b>WFP · HDX</b></span>
                <span class="chip">Period <b>{min_year}–{max_year}</b></span>
                <span class="chip">Currency <b>LKR</b></span>
                <span class="chip">Best model <b>{best['Model']}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_row(
        [
            ("Price records", f"{len(df):,}", f"{min_year}–{max_year}"),
            ("Commodities", f"{df['commodity'].nunique()}", f"{df['category'].nunique()} categories"),
            ("Markets", f"{df['market'].nunique()}", f"{df['admin1'].nunique()} provinces"),
            ("Best test MAE", f"{best['MAE']:.0f} LKR", best["Model"]),
            ("Best test R²", f"{best['R2']:.3f}", "2025 hold-out set"),
        ]
    )

    left, right = st.columns([1.6, 1], gap="medium")

    with left:
        with card("overview-trend"):
            card_header("Average recorded price by year", "Mean across all commodities and units, in LKR.")
            yearly = (
                df.dropna(subset=["date", "price"])
                .groupby(df["date"].dt.year, as_index=False)["price"]
                .mean()
                .rename(columns={"date": "Year", "price": "Average Price (LKR)"})
            )
            yearly["Year"] = yearly["Year"].astype(int)

            base = alt.Chart(yearly).encode(
                x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0, values=list(range(min_year, max_year + 1, 3)))),
                y=alt.Y("Average Price (LKR):Q", title=None, axis=alt.Axis(format=",.0f", tickCount=5)),
                tooltip=[
                    alt.Tooltip("Year:O"),
                    alt.Tooltip("Average Price (LKR):Q", title="Mean price (LKR)", format=",.2f"),
                ],
            )
            area = base.mark_area(
                interpolate="monotone",
                line={"color": ACCENT, "strokeWidth": 2.2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[alt.GradientStop(color="rgba(5,150,105,0.22)", offset=0),
                           alt.GradientStop(color="rgba(5,150,105,0)", offset=1)],
                    x1=1, x2=1, y1=0, y2=1,
                ),
            )
            points = base.mark_circle(color=ACCENT, size=28, opacity=0.9)
            st.altair_chart(style_chart((area + points).properties(height=300)), use_container_width=True)

    with right:
        with card("overview-category"):
            card_header("Median price by category", "Typical price per record, in LKR.")
            cat = (
                df.groupby("category", as_index=False)["price"].median()
                .rename(columns={"price": "Median"})
                .sort_values("Median", ascending=False)
            )
            cat["category"] = cat["category"].str.capitalize()
            bars = (
                alt.Chart(cat)
                .mark_bar(color=ACCENT, cornerRadiusEnd=4, size=14)
                .encode(
                    x=alt.X("Median:Q", title=None, axis=alt.Axis(format=",.0f", tickCount=4)),
                    y=alt.Y("category:N", sort="-x", title=None, axis=alt.Axis(labelLimit=150)),
                    tooltip=[alt.Tooltip("category:N", title="Category"),
                             alt.Tooltip("Median:Q", title="Median (LKR)", format=",.2f")],
                )
                .properties(height=300)
            )
            st.altair_chart(style_chart(bars), use_container_width=True)

    st.write("")
    with card("overview-table"):
        card_header("Records by commodity category")
        cat_summary = (
            df.groupby("category", as_index=False)
            .agg(Records=("price", "count"), Commodities=("commodity", "nunique"),
                 Markets=("market", "nunique"), Median=("price", "median"))
            .sort_values("Records", ascending=False)
        )
        cat_summary["category"] = cat_summary["category"].str.capitalize()
        st.dataframe(
            cat_summary.rename(columns={"category": "Category", "Median": "Median price (LKR)"}),
            column_config={
                "Records": st.column_config.ProgressColumn(
                    "Records", format="%d", min_value=0, max_value=int(cat_summary["Records"].max())
                ),
                "Median price (LKR)": st.column_config.NumberColumn(format="%.2f"),
            },
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

def dataset_page():
    page_header(
        "Dataset",
        "Dataset summary",
        "What the WFP Sri Lanka food price dataset contains, how complete it is, "
        "and how it was used to train the models.",
    )

    kpi_row(
        [
            ("Rows", f"{len(df):,}", None),
            ("Columns", f"{df.shape[1]}", None),
            ("Date range", f"{min_year}–{max_year}", "Monthly observations"),
            ("Missing cells", f"{int(df.isna().sum().sum()):,}", "National Average location fields"),
            ("Duplicate rows", f"{int(df.duplicated().sum()):,}", None),
        ]
    )

    src_col, prep_col = st.columns(2, gap="medium")

    with src_col:
        with card("dataset-source"):
            card_header("Source")
            st.markdown(
                f"""
                <ul class="info-list">
                    <li><span class="k">Dataset</span><span class="v">WFP Food Prices for Sri Lanka</span></li>
                    <li><span class="k">Publisher</span><span class="v">World Food Programme via the Humanitarian Data Exchange.
                        <a href="https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka" target="_blank">View on HDX ↗</a></span></li>
                    <li><span class="k">Coverage</span><span class="v">{df['date'].min():%b %Y} to {df['date'].max():%b %Y}, {df['market'].nunique()} markets,
                        {df['commodity'].nunique()} commodities in {df['category'].nunique()} categories</span></li>
                    <li><span class="k">Target</span><span class="v"><code>price</code>, recorded price in Sri Lankan Rupees (LKR)</span></li>
                </ul>
                """,
                unsafe_allow_html=True,
            )

    with prep_col:
        with card("dataset-usage"):
            card_header("How it was used")
            st.markdown(
                """
                <ul class="info-list">
                    <li><span class="k">Cleaning</span><span class="v">Dates converted to datetime. No invalid dates and no missing,
                        zero or negative prices. The 194 rows with no province or district are the <em>National Average</em> series, so they were kept.</span></li>
                    <li><span class="k">Features</span><span class="v">year, month, quarter, commodity, market, unit, price type,
                        latitude, longitude, national-average flag</span></li>
                    <li><span class="k">Encoding</span><span class="v">Numeric features standardised, categorical features one-hot encoded</span></li>
                    <li><span class="k">Split</span><span class="v">Train 2004–2023 · Validation 2024 · Test 2025</span></li>
                </ul>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    with card("dataset-dictionary"):
        card_header("Column dictionary", "Type, completeness and an example value for every column.")
        column_descriptions = {
            "date": "Observation date (monthly)",
            "admin1": "Province",
            "admin2": "District",
            "market": "Market name",
            "market_id": "WFP market identifier",
            "latitude": "Market latitude",
            "longitude": "Market longitude",
            "category": "Commodity group",
            "commodity": "Food item",
            "commodity_id": "WFP commodity identifier",
            "unit": "Unit of measurement",
            "priceflag": "How the price was obtained (actual, aggregate, …)",
            "pricetype": "Retail or Wholesale",
            "currency": "Currency of the price",
            "price": "Price in LKR (target)",
            "usdprice": "Price converted to USD",
        }
        dictionary_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Description": [column_descriptions.get(c, "") for c in df.columns],
                "Type": [str(t) for t in df.dtypes],
                "Complete": (df.notna().mean() * 100).values,
                "Unique": df.nunique().values,
                "Example": [
                    str(df[c].dropna().iloc[0]).replace(" 00:00:00", "") if df[c].notna().any() else ""
                    for c in df.columns
                ],
            }
        )
        st.dataframe(
            dictionary_df,
            column_config={
                "Complete": st.column_config.ProgressColumn("Complete", format="%.1f%%", min_value=0, max_value=100),
                "Unique": st.column_config.NumberColumn(format="%d"),
            },
            use_container_width=True,
            hide_index=True,
        )

    st.write("")
    with card("dataset-charts"):
        card_header("What's in the data")
        tab_year, tab_price, tab_comm, tab_region, tab_type = st.tabs(
            ["Records per year", "Price distribution", "Top commodities", "By province", "Price type & unit"]
        )

        with tab_year:
            per_year = df.groupby(df["date"].dt.year).size().reset_index(name="Records").rename(columns={"date": "Year"})
            chart = (
                alt.Chart(per_year)
                .mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Records:Q", title=None, axis=alt.Axis(tickCount=5)),
                    tooltip=["Year:O", alt.Tooltip("Records:Q", format=",")],
                )
                .properties(height=300)
            )
            st.altair_chart(style_chart(chart), use_container_width=True)
            st.caption("Coverage grows sharply after 2020 as WFP added more markets and commodities.")

        with tab_price:
            use_log = st.toggle("Log scale", value=True, key="price_log_scale")
            price_df = df[["price"]].dropna()
            if use_log:
                price_df = price_df[price_df["price"] > 0].assign(value=lambda x: np.log10(x["price"]))
                x_title = "log10(price in LKR)"
            else:
                price_df = price_df.assign(value=price_df["price"])
                x_title = "Price (LKR)"
            hist = (
                alt.Chart(price_df)
                .mark_bar(color=ACCENT, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X("value:Q", bin=alt.Bin(maxbins=50), title=x_title),
                    y=alt.Y("count():Q", title=None),
                    tooltip=[alt.Tooltip("count():Q", title="Records", format=",")],
                )
                .properties(height=280)
            )
            st.altair_chart(style_chart(hist), use_container_width=True)

            stats = df.groupby("pricetype")["price"].describe().reset_index()
            overall = df["price"].describe().to_frame().T.assign(pricetype="All")
            stats = pd.concat([overall, stats], ignore_index=True).rename(
                columns={"pricetype": "Price type", "count": "Count", "mean": "Mean", "std": "Std dev",
                         "min": "Min", "50%": "Median", "max": "Max"}
            )[["Price type", "Count", "Mean", "Std dev", "Min", "25%", "Median", "75%", "Max"]]
            st.dataframe(
                stats,
                column_config={c: st.column_config.NumberColumn(format="%.2f")
                               for c in ["Mean", "Std dev", "Min", "25%", "Median", "75%", "Max"]}
                | {"Count": st.column_config.NumberColumn(format="%d")},
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Prices are right-skewed: a few items such as fish, meat and oils cost far more than staples.")

        with tab_comm:
            top_n = st.slider("Number of commodities", 5, 30, 15, key="top_commodities_n")
            top_comm = (
                df.groupby("commodity")
                .agg(Records=("price", "size"), Median=("price", "median"))
                .reset_index()
                .sort_values("Records", ascending=False)
                .head(top_n)
            )
            chart = (
                alt.Chart(top_comm)
                .mark_bar(color=ACCENT, cornerRadiusEnd=4, size=14)
                .encode(
                    x=alt.X("Records:Q", title=None),
                    y=alt.Y("commodity:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
                    tooltip=[alt.Tooltip("commodity:N", title="Commodity"), alt.Tooltip("Records:Q", format=","),
                             alt.Tooltip("Median:Q", title="Median price (LKR)", format=",.2f")],
                )
                .properties(height=max(240, top_n * 24))
            )
            st.altair_chart(style_chart(chart), use_container_width=True)

        with tab_region:
            region = (
                df.assign(admin1=df["admin1"].fillna("National Average"))
                .groupby("admin1")
                .agg(Records=("price", "size"), Markets=("market", "nunique"))
                .reset_index()
                .sort_values("Records", ascending=False)
            )
            chart = (
                alt.Chart(region)
                .mark_bar(color=ACCENT, cornerRadiusEnd=4, size=16)
                .encode(
                    x=alt.X("Records:Q", title=None),
                    y=alt.Y("admin1:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("admin1:N", title="Province"), alt.Tooltip("Records:Q", format=","),
                             alt.Tooltip("Markets:Q")],
                )
                .properties(height=max(240, len(region) * 30))
            )
            st.altair_chart(style_chart(chart), use_container_width=True)

        with tab_type:
            t1, t2 = st.columns(2, gap="large")
            for col, field, label in [(t1, "pricetype", "Price type"), (t2, "unit", "Unit")]:
                counts = df[field].value_counts().reset_index()
                counts.columns = [label, "Records"]
                counts["Share"] = counts["Records"] / counts["Records"].sum() * 100
                with col:
                    st.dataframe(
                        counts,
                        column_config={
                            "Share": st.column_config.ProgressColumn("Share", format="%.1f%%", min_value=0, max_value=100),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

    st.write("")
    with card("dataset-preview"):
        head_l, head_r = st.columns([3, 1], vertical_alignment="center")
        with head_l:
            card_header("Data preview", "First 200 rows of the raw file.")
        with head_r:
            st.download_button(
                "Download CSV",
                data=DATA_PATH.read_bytes(),
                file_name="wfp_food_prices_lka.csv",
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True,
            )
        preview = df.head(200).copy()
        preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(preview, use_container_width=True, hide_index=True, height=360)


# --------------------------------------------------
# PRICE PREDICTION
# --------------------------------------------------

def prediction_page():
    page_header(
        "Prediction",
        "Price prediction",
        "Choose a model and describe the item to get an estimated price in LKR. "
        "Turn on comparison to see what every trained model predicts for the same input.",
    )

    model_names = MODEL_RESULTS["Model"].tolist()
    metrics_by_model = MODEL_RESULTS.set_index("Model")

    form_col, result_col = st.columns([1, 1.05], gap="large")

    with form_col:
        with card("predict-form"):
            card_header("Inputs")
            selected_model = st.selectbox(
                "Model",
                model_names,
                index=0,
                format_func=lambda m: f"{m}  ·  MAE {metrics_by_model.loc[m, 'MAE']:.0f} LKR",
                key="pred_model",
            )

            commodity = st.selectbox("Commodity", sorted(df["commodity"].dropna().unique()), key="pred_commodity")

            commodity_rows = df[df["commodity"] == commodity]
            market = st.selectbox(
                "Market",
                sorted(commodity_rows["market"].dropna().unique()),
                key="pred_market",
                help="Only markets where this commodity has been recorded are listed.",
            )

            combo_rows = commodity_rows[commodity_rows["market"] == market]
            unit_options = sorted(combo_rows["unit"].dropna().unique()) or sorted(df["unit"].dropna().unique())
            type_options = sorted(combo_rows["pricetype"].dropna().unique()) or sorted(df["pricetype"].dropna().unique())

            sub1, sub2 = st.columns(2)
            with sub1:
                unit = st.selectbox("Unit", unit_options, key="pred_unit")
            with sub2:
                pricetype = st.selectbox("Price type", type_options, key="pred_pricetype")

            prediction_date = st.date_input(
                "Target date",
                value=date(2025, 9, 15),
                min_value=date(2004, 1, 1),
                max_value=date(2030, 12, 31),
                key="pred_date",
            )

            compare_all = st.toggle("Compare with all models", value=False, key="pred_compare")

            submitted = st.button(
                "Predict price", type="primary", icon=":material/bolt:",
                use_container_width=True, key="pred_submit",
            )

        info = MODEL_INFO[selected_model]
        sel = metrics_by_model.loc[selected_model]
        st.write("")
        with card("predict-model-info"):
            card_header(selected_model, info["description"])
            st.markdown(
                f"""
                <div class="stat-line"><span class="k">Training data</span><span class="v">{info['training']}</span></div>
                <div class="stat-line"><span class="k">Test MAE</span><span class="v">{sel['MAE']:.2f} LKR</span></div>
                <div class="stat-line"><span class="k">Test RMSE</span><span class="v">{sel['RMSE']:.2f} LKR</span></div>
                <div class="stat-line"><span class="k">Test R²</span><span class="v">{sel['R2']:.4f}</span></div>
                """,
                unsafe_allow_html=True,
            )

    with result_col:
        if not submitted:
            st.markdown(
                """
                <div class="empty">
                    <div class="empty-icon">₨</div>
                    <h4>No prediction yet</h4>
                    <p>Fill in the inputs and click <b>Predict price</b>. Turn on
                    <b>Compare with all models</b> to see every model side by side.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        year, month = prediction_date.year, prediction_date.month
        if market in market_coordinates.index:
            latitude = market_coordinates.loc[market, "latitude"]
            longitude = market_coordinates.loc[market, "longitude"]
        else:
            latitude, longitude = 0, 0
        latitude = 0 if pd.isna(latitude) else latitude
        longitude = 0 if pd.isna(longitude) else longitude

        input_data = pd.DataFrame(
            [{
                "year": year,
                "month": month,
                "quarter": (month - 1) // 3 + 1,
                "commodity": commodity,
                "market": market,
                "unit": unit,
                "pricetype": pricetype,
                "latitude": latitude,
                "longitude": longitude,
                "is_national_average": int(market == "National Average"),
            }]
        )

        try:
            with st.spinner(f"Running {selected_model}…"):
                predicted_price = predict_price(selected_model, input_data)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        low = max(predicted_price - sel["MAE"], 0)
        high = predicted_price + sel["MAE"]
        st.markdown(
            f"""
            <div class="result">
                <div class="result-top">
                    <span class="result-label">Estimated price</span>
                    <span class="result-badge">{selected_model}</span>
                </div>
                <div class="result-value"><small>LKR</small>{predicted_price:,.2f}</div>
                <div class="result-range">Typical range LKR {low:,.0f} – {high:,.0f} (± test MAE)</div>
                <div class="result-grid">
                    <div><div class="k">Commodity</div><div class="v">{commodity}</div></div>
                    <div><div class="k">Unit · Price type</div><div class="v">{unit} · {pricetype}</div></div>
                    <div><div class="k">Market</div><div class="v">{market}</div></div>
                    <div><div class="k">Target date</div><div class="v">{prediction_date.strftime("%d %b %Y")}</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        hist = combo_rows.dropna(subset=["date", "price"]).sort_values("date")
        st.write("")
        if not hist.empty:
            latest = hist.iloc[-1]
            diff = (predicted_price - latest["price"]) / latest["price"] * 100 if latest["price"] else 0
            kpi_row(
                [
                    (f"Last recorded · {latest['date']:%b %Y}", f"LKR {latest['price']:,.2f}", None),
                    ("Prediction vs last", f"{diff:+.1f}%", "higher" if diff > 0 else "lower",
                     "up" if diff > 0 else "down"),
                    ("Historical range", f"{hist['price'].min():,.0f} – {hist['price'].max():,.0f}", "LKR"),
                ]
            )
        if year > max_year:
            st.info(
                f"This date is after the last observation ({max_year}), so the estimate is an "
                "extrapolation and less reliable.",
                icon=":material/info:",
            )

        if compare_all:
            rows = []
            with st.spinner("Running all models…"):
                for name in model_names:
                    try:
                        value = predicted_price if name == selected_model else predict_price(name, input_data)
                        rows.append({"Model": name, "Predicted Price (LKR)": value})
                    except Exception as model_exc:
                        st.warning(f"{name} failed: {model_exc}")

            comparison = pd.DataFrame(rows).merge(MODEL_RESULTS[["Model", "MAE"]], on="Model")
            comparison["Selected"] = comparison["Model"] == selected_model

            with card("predict-compare"):
                spread = comparison["Predicted Price (LKR)"]
                card_header(
                    "All models",
                    f"Estimates range from LKR {spread.min():,.2f} to {spread.max():,.2f}. "
                    "A wide spread means the models disagree.",
                )
                bars = (
                    alt.Chart(comparison)
                    .mark_bar(cornerRadiusEnd=4, size=16)
                    .encode(
                        x=alt.X("Predicted Price (LKR):Q", title=None, axis=alt.Axis(format=",.0f", tickCount=4)),
                        y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=160)),
                        color=alt.condition(alt.datum.Selected, alt.value(ACCENT), alt.value(MUTED_BAR)),
                        tooltip=["Model:N", alt.Tooltip("Predicted Price (LKR):Q", format=",.2f"),
                                 alt.Tooltip("MAE:Q", title="Test MAE", format=".2f")],
                    )
                )
                labels = (
                    alt.Chart(comparison)
                    .mark_text(align="left", dx=6, fontSize=11, color=INK, fontWeight=500)
                    .encode(
                        x="Predicted Price (LKR):Q",
                        y=alt.Y("Model:N", sort="-x"),
                        text=alt.Text("Predicted Price (LKR):Q", format=",.0f"),
                    )
                )
                st.altair_chart(style_chart((bars + labels).properties(height=230)), use_container_width=True)


# --------------------------------------------------
# MARKET TRENDS
# --------------------------------------------------

def trends_page():
    page_header(
        "Market trends",
        "Market trends",
        "Inspect the price history and underlying WFP observations for any commodity and market.",
    )

    with card("trend-filters"):
        f1, f2, f3 = st.columns([1.2, 1.2, 1])
        with f1:
            selected_commodity = st.selectbox(
                "Commodity", sorted(df["commodity"].dropna().unique()), key="trend_commodity"
            )
        available_markets = sorted(df.loc[df["commodity"] == selected_commodity, "market"].dropna().unique())
        with f2:
            selected_market = st.selectbox("Market", available_markets, key="trend_market")
        combo = df[(df["commodity"] == selected_commodity) & (df["market"] == selected_market)]
        with f3:
            type_options = sorted(combo["pricetype"].dropna().unique())
            selected_type = st.selectbox("Price type", type_options, key="trend_pricetype")

    filtered_data = (
        combo[combo["pricetype"] == selected_type]
        .dropna(subset=["date", "price"])
        .sort_values("date")
        .copy()
    )

    if filtered_data.empty:
        st.info("No historical records found for this selection.")
        return

    st.write("")
    latest_row = filtered_data.iloc[-1]
    year_ago = filtered_data[filtered_data["date"] <= latest_row["date"] - pd.DateOffset(months=12)]
    if not year_ago.empty and year_ago.iloc[-1]["price"]:
        change = (latest_row["price"] - year_ago.iloc[-1]["price"]) / year_ago.iloc[-1]["price"] * 100
        change_hint, change_cls = f"{change:+.1f}% vs a year earlier", ("up" if change > 0 else "down")
    else:
        change_hint, change_cls = "No data a year earlier", None

    kpi_row(
        [
            (f"Latest · {latest_row['date']:%b %Y}", f"LKR {latest_row['price']:,.2f}", change_hint, change_cls),
            ("Average", f"LKR {filtered_data['price'].mean():,.2f}", f"{len(filtered_data):,} records"),
            ("Lowest", f"LKR {filtered_data['price'].min():,.2f}",
             f"{filtered_data.loc[filtered_data['price'].idxmin(), 'date']:%b %Y}"),
            ("Highest", f"LKR {filtered_data['price'].max():,.2f}",
             f"{filtered_data.loc[filtered_data['price'].idxmax(), 'date']:%b %Y}"),
        ]
    )

    with card("trend-chart"):
        unit_label = ", ".join(filtered_data["unit"].dropna().unique())
        card_header(f"{selected_commodity} · {selected_market}", f"{selected_type} price per {unit_label}, in LKR.")
        trend_df = (
            filtered_data.groupby("date", as_index=False)["price"].mean()
            .rename(columns={"date": "Date", "price": "Price (LKR)"})
        )
        base = alt.Chart(trend_df).encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Price (LKR):Q", title=None, axis=alt.Axis(format=",.0f", tickCount=5)),
            tooltip=[alt.Tooltip("Date:T", format="%b %Y"),
                     alt.Tooltip("Price (LKR):Q", title="Price (LKR)", format=",.2f")],
        )
        area = base.mark_area(
            interpolate="monotone",
            line={"color": ACCENT, "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="rgba(5,150,105,0.20)", offset=0),
                       alt.GradientStop(color="rgba(5,150,105,0)", offset=1)],
                x1=1, x2=1, y1=0, y2=1,
            ),
        )
        st.altair_chart(style_chart(area.properties(height=340)), use_container_width=True)

    st.write("")
    with card("trend-table"):
        card_header("Recorded observations", "Most recent first.")
        display_df = (
            filtered_data[["date", "price", "unit", "pricetype", "priceflag"]]
            .sort_values("date", ascending=False)
            .rename(columns={"date": "Date", "price": "Price (LKR)", "unit": "Unit",
                             "pricetype": "Price type", "priceflag": "Flag"})
        )
        st.dataframe(
            display_df,
            column_config={
                "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Price (LKR)": st.column_config.NumberColumn(format="%.2f"),
            },
            use_container_width=True,
            hide_index=True,
            height=320,
        )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluation_page():
    page_header(
        "Model evaluation",
        "Model evaluation",
        "Test-set (2025) regression metrics for the standalone tree models, the stacking ensemble "
        "and the hybrid RF + ANN model.",
    )

    results = MODEL_RESULTS.copy()
    best = results.loc[results["MAE"].idxmin()]
    runner_up = results.sort_values("MAE").iloc[1]

    kpi_row(
        [
            ("Best model", best["Model"], "Lowest test MAE"),
            ("Test MAE", f"{best['MAE']:.2f} LKR",
             f"{runner_up['MAE'] - best['MAE']:.2f} LKR better than {runner_up['Model']}", "down"),
            ("Test RMSE", f"{best['RMSE']:.2f} LKR", None),
            ("Test R²", f"{best['R2']:.4f}", "Share of variance explained"),
        ]
    )

    with card("eval-table"):
        card_header("Metrics summary", "Lower MAE and RMSE are better; higher R² is better.")
        table = results.sort_values("MAE").reset_index(drop=True)
        table.insert(0, "Rank", range(1, len(table) + 1))
        st.dataframe(
            table,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "MAE": st.column_config.NumberColumn("MAE (LKR)", format="%.2f"),
                "RMSE": st.column_config.NumberColumn("RMSE (LKR)", format="%.2f"),
                "R2": st.column_config.ProgressColumn("R²", format="%.4f", min_value=0.85, max_value=0.95),
            },
            use_container_width=True,
            hide_index=True,
        )

    def metric_bar_chart(col_name: str, fmt: str, lower_is_better: bool):
        best_val = results[col_name].min() if lower_is_better else results[col_name].max()
        plot_df = results.copy()
        plot_df["is_best"] = plot_df[col_name] == best_val
        sort_order = "x" if lower_is_better else "-x"
        bars = (
            alt.Chart(plot_df)
            .mark_bar(cornerRadiusEnd=4, size=18)
            .encode(
                x=alt.X(f"{col_name}:Q", title=None, scale=alt.Scale(zero=lower_is_better), axis=alt.Axis(tickCount=5)),
                y=alt.Y("Model:N", sort=sort_order, title=None, axis=alt.Axis(labelLimit=160)),
                color=alt.condition(alt.datum.is_best, alt.value(ACCENT), alt.value(MUTED_BAR)),
                tooltip=[alt.Tooltip("Model:N"), alt.Tooltip(f"{col_name}:Q", format=fmt)],
            )
        )
        labels = (
            alt.Chart(plot_df)
            .mark_text(align="left", dx=6, fontSize=11, color=INK, fontWeight=500)
            .encode(
                x=alt.X(f"{col_name}:Q"),
                y=alt.Y("Model:N", sort=sort_order),
                text=alt.Text(f"{col_name}:Q", format=fmt),
            )
        )
        return style_chart((bars + labels).properties(height=250))

    st.write("")
    with card("eval-charts"):
        card_header("Metric breakdown")
        tab1, tab2, tab3 = st.tabs(["MAE", "RMSE", "R²"])
        with tab1:
            st.altair_chart(metric_bar_chart("MAE", ".2f", True), use_container_width=True)
        with tab2:
            st.altair_chart(metric_bar_chart("RMSE", ".2f", True), use_container_width=True)
        with tab3:
            st.altair_chart(metric_bar_chart("R2", ".4f", False), use_container_width=True)

    st.caption(
        "Standalone models were trained on data through 2024. The Stacking and Hybrid models used a "
        "separate temporal split with base learners trained through 2023, so interpret direct comparisons with care."
    )


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large")

navigation = st.navigation(
    [
        st.Page(overview_page, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
        st.Page(dataset_page, title="Dataset", icon=":material/dataset:", url_path="dataset"),
        st.Page(prediction_page, title="Predict", icon=":material/online_prediction:", url_path="predict"),
        st.Page(trends_page, title="Market Trends", icon=":material/monitoring:", url_path="trends"),
        st.Page(evaluation_page, title="Models", icon=":material/leaderboard:", url_path="models"),
    ],
    position="top",
)
navigation.run()

st.markdown(
    '<div class="app-footer">Data: World Food Programme, via the Humanitarian Data Exchange · '
    "Prices in Sri Lankan Rupees (LKR)</div>",
    unsafe_allow_html=True,
)
