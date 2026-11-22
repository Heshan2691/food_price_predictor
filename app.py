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
    page_title="Sri Lanka Food Prices",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "wfp_food_prices_lka.csv"
MODEL_DIR = BASE_DIR / "models"

ACCENT = "#166553"
ACCENT_SECONDARY = "#D96B27"
ACCENT_SOFT = "#DFF0E9"
MUTED_BAR = "#9FB0AC"
INK = "#182226"
MUTED_INK = "#526066"
GRID = "#DCE3E0"


# --------------------------------------------------
# MODERN EDITORIAL STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #182226;
    }

    .stApp {
        background-color: #EEF1EF;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2.75rem;
        padding-right: 2.75rem;
        max-width: 1260px;
    }

    h1, h2, h3 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: #132125 !important;
        letter-spacing: -0.015em;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.25rem !important;
    }

    h2 {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }

    h3 {
        font-size: 1.12rem !important;
        font-weight: 600 !important;
    }

    /* Deep slate-pine sidebar */
    [data-testid="stSidebar"] {
        background-color: #162A2C;
        border-right: 1px solid #243E40;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.75rem;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #E6ECEA !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.2rem;
        transition: background-color 0.15s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #203A3D;
    }

    .sidebar-brand {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #F7F9F8;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }

    .sidebar-sub {
        font-size: 0.78rem;
        color: #9BB5B1;
        margin-bottom: 1.35rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #264245;
    }

    .sidebar-meta {
        font-size: 0.78rem;
        color: #9BB5B1;
        line-height: 1.7;
        border-top: 1px solid #264245;
        padding-top: 1rem;
        margin-top: 1.5rem;
    }

    .sidebar-meta strong {
        color: #F0F5F3;
        font-weight: 600;
    }

    /* Page header */
    .page-subtitle {
        font-size: 0.96rem;
        color: #4D5E63;
        margin-top: 0;
        margin-bottom: 1.5rem;
        max-width: 70ch;
        line-height: 1.55;
    }

    /* Elevated metric cards with subtle color top-accents */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D9E2DE;
        border-top: 3px solid #166553;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 2px rgba(22, 42, 44, 0.04);
    }

    div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] {
        border-top-color: #D96B27;
    }

    div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] {
        border-top-color: #2B5B84;
    }

    div[data-testid="column"]:nth-child(4) div[data-testid="stMetric"] {
        border-top-color: #5C6F68;
    }

    div[data-testid="stMetricLabel"] {
        color: #526066;
        font-size: 0.8rem;
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] {
        color: #132125;
        font-size: 1.55rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    /* Form card */
    [data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #D9E2DE;
        border-radius: 8px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 1px 2px rgba(22, 42, 44, 0.04);
    }

    /* High-contrast result card */
    .estimate-box {
        background-color: #162A2C;
        color: #F4F7F6;
        border-radius: 8px;
        border-top: 4px solid #D96B27;
        padding: 1.6rem 1.75rem;
        box-shadow: 0 4px 12px rgba(22, 42, 44, 0.12);
    }

    .estimate-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #A6C2BE;
        margin-bottom: 0.4rem;
    }

    .estimate-value {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2.45rem;
        font-weight: 700;
        color: #FFFFFF;
        font-variant-numeric: tabular-nums;
        line-height: 1.15;
    }

    .estimate-meta {
        font-size: 0.88rem;
        color: #D5E2DF;
        margin-top: 1rem;
        padding-top: 0.9rem;
        border-top: 1px solid #2A4649;
        line-height: 1.65;
    }

    .estimate-meta strong {
        color: #FFFFFF;
    }

    .placeholder-box {
        background-color: #E5ECE9;
        border: 1px dashed #9EB5AE;
        border-radius: 8px;
        padding: 2.25rem 1.6rem;
        color: #3B4F4C;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    /* Warm sand methodology / note block */
    .note-box {
        background-color: #FAF6F0;
        border: 1px solid #E8DCCB;
        border-left: 4px solid #D96B27;
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        font-size: 0.88rem;
        color: #3D3B37;
        line-height: 1.6;
    }

    .note-box h4 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9A4A18 !important;
        margin: 0 0 0.45rem 0 !important;
    }

    .note-box p {
        margin: 0 0 0.8rem 0;
    }

    .note-box p:last-child {
        margin-bottom: 0;
    }

    /* Primary buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #166553;
        color: #FFFFFF;
        border: 1px solid #166553;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.55rem 1.25rem;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #114F41;
        border-color: #114F41;
        color: #FFFFFF;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #E3EAE7;
        border-radius: 6px 6px 0 0;
        padding: 0.45rem 1rem;
        color: #2C3E42;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #166553 !important;
    }

    /* Dividers & footer cleanup */
    hr {
        border-color: #D5DFDB;
        margin: 1.75rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DATA & MODEL LOADING
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


@st.cache_resource
def load_models():
    base_models = joblib.load(MODEL_DIR / "stacking_base_models.pkl")
    meta_model = joblib.load(MODEL_DIR / "stacking_meta_model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "stacking_preprocessor.pkl")
    return base_models, meta_model, preprocessor


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.properties(background="#FFFFFF", padding={"left": 14, "right": 14, "top": 14, "bottom": 10})
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED_INK,
            titleColor=MUTED_INK,
            gridColor=GRID,
            domainColor="#C7D3CE",
            tickColor="#C7D3CE",
            labelFont="IBM Plex Sans",
            titleFont="IBM Plex Sans",
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight=500,
        )
        .configure_title(
            font="Source Serif 4",
            fontSize=15,
            color=INK,
            anchor="start",
        )
    )


df = load_data()
base_models, meta_model, preprocessor = load_models()
market_coordinates = df.groupby("market")[["latitude", "longitude"]].median()

min_year = int(df["date"].dt.year.min())
max_year = int(df["date"].dt.year.max())


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">Sri Lanka Food Prices</div>
        <div class="sidebar-sub">WFP Market Monitor &amp; Estimator</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Price Estimator",
            "Market Trends",
            "Model Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="sidebar-meta">
            <strong>Source:</strong> World Food Programme (WFP)<br>
            <strong>Coverage:</strong> {min_year}–{max_year}<br>
            <strong>Currency:</strong> LKR (Sri Lankan Rupee)<br>
            <strong>Active Model:</strong> Stacking Ensemble
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":
    st.title("Sri Lanka Food Prices")
    st.markdown(
        f'<p class="page-subtitle">'
        f"Historical retail and wholesale food price records from the World Food Programme "
        f"({min_year}–{max_year}), paired with a stacking ensemble for estimating commodity prices in LKR."
        f"</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price Observations", f"{len(df):,}")
    c2.metric("Commodities", f"{df['commodity'].nunique()}")
    c3.metric("Markets Tracked", f"{df['market'].nunique()}")
    c4.metric("Time Span", f"{min_year}–{max_year}")

    st.markdown("")

    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.subheader("Average Recorded Price by Year")
        yearly = (
            df.dropna(subset=["date", "price"])
            .groupby(df["date"].dt.year, as_index=False)["price"]
            .mean()
            .rename(columns={"date": "Year", "price": "Average Price (LKR)"})
        )
        yearly["Year"] = yearly["Year"].astype(int)

        area = (
            alt.Chart(yearly)
            .mark_area(
                line={"color": ACCENT, "strokeWidth": 2.5},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="#166553", offset=0),
                        alt.GradientStop(color="#FFFFFF", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=0,
                    y2=1,
                ),
                opacity=0.22,
            )
            .encode(
                x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "Average Price (LKR):Q",
                    title="Mean Price (LKR)",
                    axis=alt.Axis(format=",.0f"),
                ),
                tooltip=[
                    alt.Tooltip("Year:O", title="Year"),
                    alt.Tooltip("Average Price (LKR):Q", title="Mean Price (LKR)", format=",.2f"),
                ],
            )
            .properties(height=320)
        )
        points = (
            alt.Chart(yearly)
            .mark_circle(color=ACCENT_SECONDARY, size=48, stroke="#FFFFFF", strokeWidth=1.5)
            .encode(
                x="Year:O",
                y="Average Price (LKR):Q",
                tooltip=[
                    alt.Tooltip("Year:O", title="Year"),
                    alt.Tooltip("Average Price (LKR):Q", title="Mean Price (LKR)", format=",.2f"),
                ],
            )
        )
        st.altair_chart(style_chart(area + points), use_container_width=True)
        st.caption(
            "Unweighted mean across all recorded commodities and units per year. "
            "Use Market Trends to inspect individual items."
        )

    with right:
        st.subheader("About This Dataset")
        st.markdown(
            """
            <div class="note-box">
                <h4>Scope</h4>
                <p>
                    Compiled from World Food Programme (WFP) market monitoring across
                    Sri Lankan provinces, covering staple cereals, pulses, vegetables,
                    dairy, fish, and cooking oils.
                </p>
                <h4>Modeling Approach</h4>
                <p>
                    Six models were evaluated — Random Forest, XGBoost, LightGBM,
                    CatBoost, a Stacking Ensemble, and a Hybrid RF + ANN architecture.
                    The Stacking Ensemble achieved the lowest test MAE (120.46 LKR)
                    and powers the Price Estimator.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Records by Commodity Category")

    cat_summary = (
        df.groupby("category", as_index=False)
        .agg(
            Records=("price", "count"),
            Commodities=("commodity", "nunique"),
            Median_Price=("price", "median"),
        )
        .sort_values("Records", ascending=False)
    )
    cat_summary["category"] = cat_summary["category"].str.title()
    cat_summary = cat_summary.rename(
        columns={
            "category": "Category",
            "Median_Price": "Median Price (LKR)",
        }
    )

    st.dataframe(
        cat_summary.style.format(
            {
                "Records": "{:,}",
                "Commodities": "{:,}",
                "Median Price (LKR)": "{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# PRICE ESTIMATOR PAGE
# --------------------------------------------------

elif page == "Price Estimator":
    st.title("Price Estimator")
    st.markdown(
        '<p class="page-subtitle">'
        "Estimate a commodity price in LKR for a given market, unit, price type, and date "
        "using the trained Stacking Ensemble."
        "</p>",
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.15, 1], gap="large")

    with form_col:
        with st.form("prediction_form"):
            commodity = st.selectbox(
                "Commodity",
                sorted(df["commodity"].dropna().unique()),
            )
            market = st.selectbox(
                "Market",
                sorted(df["market"].dropna().unique()),
            )

            sub1, sub2 = st.columns(2)
            with sub1:
                unit = st.selectbox(
                    "Unit",
                    sorted(df["unit"].dropna().unique()),
                )
            with sub2:
                pricetype = st.selectbox(
                    "Price Type",
                    sorted(df["pricetype"].dropna().unique()),
                )

            prediction_date = st.date_input(
                "Target Date",
                value=date(2025, 9, 15),
                min_value=date(2004, 1, 1),
            )

            submitted = st.form_submit_button(
                "Estimate Price",
                use_container_width=True,
            )

    with result_col:
        if submitted:
            year = prediction_date.year
            month = prediction_date.month
            quarter = (month - 1) // 3 + 1
            is_national_average = int(market == "National Average")

            if market in market_coordinates.index:
                latitude = market_coordinates.loc[market, "latitude"]
                longitude = market_coordinates.loc[market, "longitude"]
            else:
                latitude = 0
                longitude = 0

            latitude = 0 if pd.isna(latitude) else latitude
            longitude = 0 if pd.isna(longitude) else longitude

            input_data = pd.DataFrame(
                [
                    {
                        "year": year,
                        "month": month,
                        "quarter": quarter,
                        "commodity": commodity,
                        "market": market,
                        "unit": unit,
                        "pricetype": pricetype,
                        "latitude": latitude,
                        "longitude": longitude,
                        "is_national_average": is_national_average,
                    }
                ]
            )

            try:
                processed_input = preprocessor.transform(input_data)
                base_predictions = [
                    model.predict(processed_input) for _, model in base_models
                ]
                stacking_features = np.column_stack(base_predictions)
                predicted_price = meta_model.predict(stacking_features)[0]

                st.markdown(
                    f"""
                    <div class="estimate-box">
                        <div class="estimate-label">Estimated Price</div>
                        <div class="estimate-value">LKR {predicted_price:,.2f}</div>
                        <div class="estimate-meta">
                            <strong>Commodity:</strong> {commodity} ({unit})<br>
                            <strong>Market:</strong> {market} &middot; {pricetype}<br>
                            <strong>Date:</strong> {prediction_date.strftime("%d %b %Y")}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Historical reference for the same commodity & market if available
                hist = df[
                    (df["commodity"] == commodity) & (df["market"] == market)
                ].sort_values("date")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    st.caption(
                        f"Reference: latest observed price for {commodity} in {market} was "
                        f"LKR {latest['price']:,.2f} on {latest['date'].strftime('%d %b %Y')} "
                        f"(historical range: LKR {hist['price'].min():,.2f} – {hist['price'].max():,.2f})."
                    )

                st.caption(
                    "Note: Estimates are based on historical patterns; extrapolations far beyond "
                    f"the observed data ({max_year}) carry higher uncertainty."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        else:
            st.markdown(
                """
                <div class="placeholder-box">
                    Select a commodity, market, unit, and date on the left, then click
                    <strong>Estimate Price</strong> to compute the stacking ensemble forecast.
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------
# MARKET TRENDS PAGE
# --------------------------------------------------

elif page == "Market Trends":
    st.title("Market Trends")
    st.markdown(
        '<p class="page-subtitle">'
        "Inspect historical price series and underlying WFP observations for any commodity and market."
        "</p>",
        unsafe_allow_html=True,
    )

    f1, f2 = st.columns(2)
    with f1:
        selected_commodity = st.selectbox(
            "Commodity",
            sorted(df["commodity"].dropna().unique()),
            key="trend_commodity",
        )

    available_markets = sorted(
        df.loc[df["commodity"] == selected_commodity, "market"].dropna().unique()
    )
    with f2:
        selected_market = st.selectbox(
            "Market",
            available_markets,
            key="trend_market",
        )

    filtered_data = (
        df[
            (df["commodity"] == selected_commodity)
            & (df["market"] == selected_market)
        ]
        .dropna(subset=["date", "price"])
        .sort_values("date")
        .copy()
    )

    if not filtered_data.empty:
        latest_row = filtered_data.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            f"Latest ({latest_row['date'].strftime('%b %Y')})",
            f"LKR {latest_row['price']:,.2f}",
        )
        c2.metric("Mean Price", f"LKR {filtered_data['price'].mean():,.2f}")
        c3.metric("Minimum", f"LKR {filtered_data['price'].min():,.2f}")
        c4.metric("Maximum", f"LKR {filtered_data['price'].max():,.2f}")

        st.subheader(f"{selected_commodity} — {selected_market}")

        trend_df = (
            filtered_data.groupby("date", as_index=False)["price"]
            .mean()
            .rename(columns={"date": "Date", "price": "Price (LKR)"})
        )

        trend_chart = (
            alt.Chart(trend_df)
            .mark_area(
                line={"color": ACCENT, "strokeWidth": 2.2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="#166553", offset=0),
                        alt.GradientStop(color="#FFFFFF", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=0,
                    y2=1,
                ),
                opacity=0.2,
            )
            .encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y(
                    "Price (LKR):Q",
                    title="Price (LKR)",
                    axis=alt.Axis(format=",.0f"),
                ),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date", format="%d %b %Y"),
                    alt.Tooltip("Price (LKR):Q", title="Price (LKR)", format=",.2f"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(style_chart(trend_chart), use_container_width=True)

        st.subheader("Recorded Observations")
        display_df = (
            filtered_data[["date", "price", "unit", "pricetype", "priceflag"]]
            .sort_values("date", ascending=False)
            .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
            .rename(
                columns={
                    "date": "Date",
                    "price": "Price (LKR)",
                    "unit": "Unit",
                    "pricetype": "Price Type",
                    "priceflag": "Flag",
                }
            )
        )
        st.dataframe(
            display_df.style.format({"Price (LKR)": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No historical records found for this selection.")


# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.markdown(
        '<p class="page-subtitle">'
        "Comparison of test-set regression metrics across standalone tree models, "
        "a stacking ensemble, and a hybrid RF + ANN architecture."
        "</p>",
        unsafe_allow_html=True,
    )

    results = pd.DataFrame(
        [
            {"Model": "Stacking", "MAE": 120.4602, "RMSE": 174.0408, "R2": 0.9290},
            {"Model": "XGBoost", "MAE": 124.3591, "RMSE": 180.6740, "R2": 0.9235},
            {"Model": "CatBoost", "MAE": 125.6818, "RMSE": 181.6840, "R2": 0.9226},
            {"Model": "LightGBM", "MAE": 126.9739, "RMSE": 177.3770, "R2": 0.9262},
            {"Model": "Random Forest", "MAE": 134.5089, "RMSE": 202.1905, "R2": 0.9041},
            {"Model": "Hybrid RF + ANN", "MAE": 142.5718, "RMSE": 214.4527, "R2": 0.8922},
        ]
    )

    best = results.loc[results["MAE"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best["Model"])
    c2.metric("Test MAE", f"{best['MAE']:.2f} LKR")
    c3.metric("Test RMSE", f"{best['RMSE']:.2f} LKR")
    c4.metric("Test R²", f"{best['R2']:.4f}")

    st.subheader("Metrics Summary")
    st.dataframe(
        results.style.format(
            {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}
        )
        .highlight_min(subset=["MAE", "RMSE"], color=ACCENT_SOFT)
        .highlight_max(subset=["R2"], color=ACCENT_SOFT),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Metric Breakdown")
    tab1, tab2, tab3 = st.tabs(["MAE (Lower is better)", "RMSE (Lower is better)", "R² (Higher is better)"])

    def metric_bar_chart(col_name: str, fmt: str, lower_is_better: bool):
        best_val = results[col_name].min() if lower_is_better else results[col_name].max()
        plot_df = results.copy()
        plot_df["is_best"] = plot_df[col_name] == best_val
        sort_order = "x" if lower_is_better else "-x"

        bars = (
            alt.Chart(plot_df)
            .mark_bar(cornerRadiusEnd=3, size=24)
            .encode(
                x=alt.X(f"{col_name}:Q", title=col_name),
                y=alt.Y("Model:N", sort=sort_order, title=None),
                color=alt.condition(
                    alt.datum.is_best,
                    alt.value(ACCENT),
                    alt.value(MUTED_BAR),
                ),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip(f"{col_name}:Q", format=fmt),
                ],
            )
        )
        labels = (
            alt.Chart(plot_df)
            .mark_text(align="left", dx=6, font="IBM Plex Sans", fontSize=11, color=INK)
            .encode(
                x=alt.X(f"{col_name}:Q"),
                y=alt.Y("Model:N", sort=sort_order),
                text=alt.Text(f"{col_name}:Q", format=fmt),
            )
        )
        return style_chart((bars + labels).properties(height=240))

    with tab1:
        st.altair_chart(metric_bar_chart("MAE", ".2f", True), use_container_width=True)

    with tab2:
        st.altair_chart(metric_bar_chart("RMSE", ".2f", True), use_container_width=True)

    with tab3:
        st.altair_chart(metric_bar_chart("R2", ".4f", False), use_container_width=True)

    st.caption(
        "Evaluation note: Standalone models were trained on data through 2024. "
        "The Stacking and Hybrid models used a separate temporal split with base learners "
        "trained through 2023; interpret direct comparisons accordingly."
    )
