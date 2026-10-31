"""
E-commerce Analytics & Intelligence
Streamlit app for customer segmentation, sales forecasting and order-level sales estimation.

Model hosting:
- Upload order_sales_prediction.joblib to a Hugging Face Model repository.
- Set HF_MODEL_REPO in Streamlit secrets or environment variables:
    HF_MODEL_REPO = "your-hf-username/ecommerce-analytics-models"

Expected repository files:
    data/customer_segments.csv
    data/monthly_sales_actuals.csv
    data/sales_forecast_6_months.csv
    data/sales_forecast_backtest.csv
    data/order_sales_predictions.csv
    data/ecommerce_cleaned.csv

The app also supports those CSVs under an "outputs/data/" folder.
"""

from pathlib import Path
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------
# Page configuration and theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Analytics & Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------
def get_setting(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


HF_MODEL_REPO = get_setting("HF_MODEL_REPO", "")
HF_MODEL_FILENAME = get_setting(
    "HF_MODEL_FILENAME", "order_sales_prediction.joblib"
)
HF_TOKEN = get_setting("HF_TOKEN", "") or None


def locate_data(filename: str):
    """Find a project CSV in common local project layouts."""
    candidates = [
        ROOT / "data" / filename,
        ROOT / "outputs" / "data" / filename,
        ROOT / "ecommerce_upgrade" / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


@st.cache_resource(show_spinner="Downloading and loading the sales model...")
def load_order_model(repo_id: str, filename: str, token: str | None):
    """Download the trusted model artifact from Hugging Face and load it."""
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=token,
    )
    artifact = joblib.load(model_path)

    # The training pipeline saved a dictionary, not the bare sklearn Pipeline.
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        feature_columns = artifact.get("features", [])
    else:
        # Also support a bare sklearn pipeline if the artifact is replaced later.
        pipeline = artifact
        feature_columns = []

    return pipeline, feature_columns


def money(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.2f}"


def metric_delta(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return None
    return f"{(current - previous) / abs(previous) * 100:+.1f}%"


def options_from_data(df, column, fallback):
    if df is not None and column in df.columns:
        values = df[column].dropna().astype(str).unique().tolist()
        if values:
            return sorted(values)
    return fallback


# ---------------------------------------------------------------------
# Load project data
# ---------------------------------------------------------------------
files = {
    "cleaned": "ecommerce_cleaned.csv",
    "segments": "customer_segments.csv",
    "monthly": "monthly_sales_actuals.csv",
    "forecast": "sales_forecast_6_months.csv",
    "backtest": "sales_forecast_backtest.csv",
    "order_predictions": "order_sales_predictions.csv",
}

data_paths = {key: locate_data(name) for key, name in files.items()}
data = {
    key: read_csv(str(path)) if path else None
    for key, path in data_paths.items()
}

cleaned = data["cleaned"]
segments = data["segments"]
monthly = data["monthly"]
forecast = data["forecast"]
backtest = data["backtest"]
order_predictions = data["order_predictions"]


# ---------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------
st.title("🛒 E-commerce Analytics & Intelligence")
st.markdown(
    "An end-to-end analytics application combining **customer intelligence**, "
    "**sales trends**, **forecasting**, and **machine-learning predictions**."
)

with st.sidebar:
    st.header("Project navigation")
    st.caption("Use the tabs to explore each analytical component.")
    st.divider()
    st.subheader("Data availability")
    for label, key in [
        ("Cleaned transactions", "cleaned"),
        ("Customer segments", "segments"),
        ("Monthly actuals", "monthly"),
        ("Six-month forecast", "forecast"),
        ("Forecast backtest", "backtest"),
        ("Order prediction results", "order_predictions"),
    ]:
        st.write(f"{'✅' if data[key] is not None else '⚠️'} {label}")

    st.divider()
    st.caption(
        "Forecast values are exploratory estimates. The available history "
        "contains only 36 monthly observations."
    )


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
overview_tab, product_tab, customer_tab, forecast_tab, predict_tab = st.tabs(
    [
        "Executive Overview",
        "Product Performance",
        "Customer Intelligence",
        "Sales Forecast",
        "Sales Prediction",
    ]
)


# ---------------------------------------------------------------------
# 1. Executive overview
# ---------------------------------------------------------------------
with overview_tab:
    st.subheader("Executive Sales Overview")

    if cleaned is None:
        st.warning(
            "Cleaned transaction data was not found. Add "
            "`ecommerce_cleaned.csv` to the repository's `data/` folder."
        )
    else:
        df = cleaned.copy()
        for col in ["Sales", "Profit", "Quantity", "Discount"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "OrderDate" in df:
            df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

        total_sales = df["Sales"].sum() if "Sales" in df else 0
        total_profit = df["Profit"].sum() if "Profit" in df else 0
        total_orders = df["OrderID"].nunique() if "OrderID" in df else len(df)
        total_customers = (
            df["CustomerName"].nunique() if "CustomerName" in df else 0
        )
        avg_order = total_sales / total_orders if total_orders else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total sales", money(total_sales))
        k2.metric("Total profit", money(total_profit))
        k3.metric("Orders", f"{total_orders:,}")
        k4.metric("Customers", f"{total_customers:,}")
        k5.metric("Average order sales", money(avg_order))

        if "OrderDate" in df and "Sales" in df:
            trend = (
                df.dropna(subset=["OrderDate"])
                .set_index("OrderDate")
                .resample("MS")["Sales"]
                .sum()
                .reset_index()
            )
            fig = px.line(
                trend,
                x="OrderDate",
                y="Sales",
                markers=True,
                title="Monthly sales trend",
                labels={"OrderDate": "Month", "Sales": "Sales"},
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        if "Category" in df and "Sales" in df:
            category_sales = (
                df.groupby("Category", as_index=False)["Sales"]
                .sum()
                .sort_values("Sales", ascending=False)
            )
            fig = px.bar(
                category_sales,
                x="Category",
                y="Sales",
                title="Sales by category",
                text_auto=".2s",
            )
            c1.plotly_chart(fig, use_container_width=True)

        if "Region" in df and "Sales" in df:
            region_sales = (
                df.groupby("Region", as_index=False)["Sales"]
                .sum()
                .sort_values("Sales", ascending=False)
            )
            fig = px.bar(
                region_sales,
                x="Region",
                y="Sales",
                title="Sales by region",
                text_auto=".2s",
            )
            c2.plotly_chart(fig, use_container_width=True)

        with st.expander("View cleaned transaction data"):
            st.dataframe(df.head(500), use_container_width=True)
            st.download_button(
                "Download cleaned data",
                df.to_csv(index=False).encode("utf-8"),
                "ecommerce_cleaned.csv",
                "text/csv",
            )


# ---------------------------------------------------------------------
# 2. Product performance
# ---------------------------------------------------------------------
with product_tab:
    st.subheader("Product Performance")

    if cleaned is None:
        st.info("Add ecommerce_cleaned.csv to enable product analysis.")
    else:
        df = cleaned.copy()
        for col in ["Sales", "Profit", "Quantity"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Category" in df:
            categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
            selected_category = st.selectbox("Filter category", categories)
            if selected_category != "All":
                df = df[df["Category"] == selected_category]

        if {"Category", "Sales"}.issubset(df.columns):
            category = (
                df.groupby("Category", as_index=False)
                .agg(Sales=("Sales", "sum"))
                .sort_values("Sales", ascending=False)
            )
            fig = px.bar(
                category,
                x="Category",
                y="Sales",
                color="Category",
                title="Sales by product category",
                text_auto=".2s",
            )
            st.plotly_chart(fig, use_container_width=True)

        if {"SubCategory", "Sales", "Profit"}.issubset(df.columns):
            subcategory = (
                df.groupby("SubCategory", as_index=False)
                .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
                .sort_values("Sales", ascending=False)
            )
            fig = px.bar(
                subcategory.head(15),
                x="SubCategory",
                y="Sales",
                title="Top sub-categories by sales",
                hover_data=["Profit"],
                text_auto=".2s",
            )
            st.plotly_chart(fig, use_container_width=True)

        if {"ProductName", "Sales", "Profit"}.issubset(df.columns):
            products = (
                df.groupby("ProductName", as_index=False)
                .agg(
                    Sales=("Sales", "sum"),
                    Profit=("Profit", "sum"),
                    Units=("Quantity", "sum") if "Quantity" in df else ("Sales", "count"),
                )
                .sort_values("Sales", ascending=False)
            )
            st.markdown("**Top 20 products by sales**")
            st.dataframe(products.head(20), use_container_width=True)


# ---------------------------------------------------------------------
# 3. Customer intelligence
# ---------------------------------------------------------------------
with customer_tab:
    st.subheader("Customer Intelligence — RFM + K-Means")

    if segments is None:
        st.info("Add customer_segments.csv to enable customer segmentation.")
    else:
        seg = segments.copy()
        if "CustomerSegment" not in seg.columns:
            st.error("Customer segments file does not contain CustomerSegment.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Customers segmented", f"{len(seg):,}")
            c2.metric("Number of segments", f"{seg['CustomerSegment'].nunique():,}")
            c3.metric(
                "Average monetary value",
                money(pd.to_numeric(seg.get("Monetary"), errors="coerce").mean())
                if "Monetary" in seg else "—",
            )

            counts = (
                seg["CustomerSegment"]
                .value_counts()
                .rename_axis("Customer segment")
                .reset_index(name="Customers")
            )
            fig = px.pie(
                counts,
                names="Customer segment",
                values="Customers",
                hole=0.45,
                title="Customer distribution by segment",
            )
            st.plotly_chart(fig, use_container_width=True)

            if {"Recency", "Monetary", "CustomerSegment"}.issubset(seg.columns):
                seg["Recency"] = pd.to_numeric(seg["Recency"], errors="coerce")
                seg["Monetary"] = pd.to_numeric(seg["Monetary"], errors="coerce")
                fig = px.scatter(
                    seg,
                    x="Recency",
                    y="Monetary",
                    color="CustomerSegment",
                    hover_name="CustomerName" if "CustomerName" in seg else None,
                    title="Customer recency vs monetary value",
                    labels={
                        "Recency": "Days since last purchase",
                        "Monetary": "Total sales value",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Segment-level profile**")
            numeric_cols = [
                c for c in ["Recency", "Frequency", "Monetary", "TotalProfit"]
                if c in seg.columns
            ]
            if numeric_cols:
                st.dataframe(
                    seg.groupby("CustomerSegment")[numeric_cols]
                    .mean()
                    .round(2)
                    .reset_index(),
                    use_container_width=True,
                )

            st.dataframe(seg, use_container_width=True)
            st.download_button(
                "Download customer segments",
                seg.to_csv(index=False).encode("utf-8"),
                "customer_segments.csv",
                "text/csv",
            )


# ---------------------------------------------------------------------
# 4. Sales forecast
# ---------------------------------------------------------------------
with forecast_tab:
    st.subheader("Monthly Sales Forecast")

    if monthly is not None:
        hist = monthly.copy()
        hist["MonthStart"] = pd.to_datetime(hist["MonthStart"], errors="coerce")
        hist["Sales"] = pd.to_numeric(hist["Sales"], errors="coerce")
        hist = hist.dropna(subset=["MonthStart", "Sales"]).sort_values("MonthStart")

        fig = px.line(
            hist,
            x="MonthStart",
            y="Sales",
            markers=True,
            title="Historical monthly sales",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add monthly_sales_actuals.csv to show historical sales.")

    if forecast is not None:
        fc = forecast.copy()
        fc["MonthStart"] = pd.to_datetime(fc["MonthStart"], errors="coerce")
        fc["PredictedSales"] = pd.to_numeric(
            fc["PredictedSales"], errors="coerce"
        )
        fc = fc.dropna(subset=["MonthStart", "PredictedSales"]).sort_values(
            "MonthStart"
        )

        st.markdown("**Six-month forecast**")
        fig = px.line(
            fc,
            x="MonthStart",
            y="PredictedSales",
            markers=True,
            title="Forecast sales for the next six months",
            labels={"PredictedSales": "Forecast sales", "MonthStart": "Month"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fc, use_container_width=True)
        st.download_button(
            "Download six-month forecast",
            fc.to_csv(index=False).encode("utf-8"),
            "sales_forecast_6_months.csv",
            "text/csv",
        )
        if "ForecastType" in fc.columns:
            st.caption(f"Forecast method: {fc['ForecastType'].iloc[0]}")

    if backtest is not None:
        bt = backtest.copy()
        date_col = "MonthStart" if "MonthStart" in bt.columns else "Date"
        if date_col in bt and {"Sales", "PredictedSales"}.issubset(bt.columns):
            bt[date_col] = pd.to_datetime(bt[date_col], errors="coerce")
            bt = bt.dropna(subset=[date_col])
            bt_long = bt.melt(
                id_vars=[date_col],
                value_vars=["Sales", "PredictedSales"],
                var_name="Series",
                value_name="Value",
            )
            fig = px.line(
                bt_long,
                x=date_col,
                y="Value",
                color="Series",
                markers=True,
                title="Random Forest backtest: actual vs predicted",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Forecast limitation: only 36 monthly observations are available. "
        "The six-month forecast uses the seasonal-naive baseline selected "
        "after comparison with the Random Forest lag model. Treat it as exploratory."
    )


# ---------------------------------------------------------------------
# 5. Sales prediction
# ---------------------------------------------------------------------
with predict_tab:
    st.subheader("Order-Level Sales Prediction")
    st.write(
        "Enter order attributes to estimate sales using the trained "
        "Random Forest regression pipeline."
    )

    if not HF_MODEL_REPO:
        st.warning(
            "The Hugging Face model repository is not configured. Add "
            "`HF_MODEL_REPO` to Streamlit secrets before using this predictor."
        )
        st.code(
            '[secrets.toml]\n'
            'HF_MODEL_REPO = "your-username/ecommerce-analytics-models"\n'
            'HF_MODEL_FILENAME = "order_sales_prediction.joblib"'
        )
    else:
        try:
            pipeline, saved_features = load_order_model(
                HF_MODEL_REPO, HF_MODEL_FILENAME, HF_TOKEN
            )
            st.success("Order sales model loaded successfully.")

            # Match the training pipeline's categorical options to the source data.
            reference = cleaned
            ship_modes = options_from_data(
                reference, "ShipMode",
                ["Standard Class", "Second Class", "First Class", "Same Day"],
            )
            segments_options = options_from_data(
                reference, "Segment", ["Consumer", "Corporate", "Home Office"]
            )
            regions = options_from_data(
                reference, "Region", ["Central", "East", "South", "West"]
            )
            categories = options_from_data(
                reference, "Category", ["Furniture", "Office Supplies", "Technology"]
            )
            subcategories = options_from_data(
                reference, "SubCategory", ["Accessories", "Binders", "Chairs", "Phones"]
            )
            product_names = options_from_data(
                reference, "ProductName", ["Other / Unspecified"]
            )

            with st.form("order_prediction_form"):
                a, b, c = st.columns(3)
                quantity = a.number_input("Quantity", min_value=0, value=1, step=1)
                discount = b.number_input(
                    "Discount", min_value=0.0, max_value=1.0,
                    value=0.0, step=0.05, format="%.2f"
                )
                ship_mode = c.selectbox("Ship mode", ship_modes)

                d, e, f = st.columns(3)
                customer_segment = d.selectbox("Customer type", segments_options)
                region = e.selectbox("Region", regions)
                category = f.selectbox("Category", categories)

                g, h, i = st.columns(3)
                subcategory = g.selectbox("Sub-category", subcategories)
                product_name = h.selectbox("Product", product_names)
                order_month = i.selectbox("Order month", list(range(1, 13)))

                order_year = st.selectbox(
                    "Order year", [2022, 2023, 2024, 2025, 2026, 2027],
                    index=4
                )
                submitted = st.form_submit_button(
                    "Predict order sales", type="primary", use_container_width=True
                )

            if submitted:
                # These are exactly the columns used in run_pipeline.py during training.
                input_row = pd.DataFrame([{
                    "Quantity": quantity,
                    "Discount": discount,
                    "OrderMonth": order_month,
                    "OrderYear": order_year,
                    "ShipMode": ship_mode,
                    "Segment": customer_segment,
                    "Region": region,
                    "Category": category,
                    "SubCategory": subcategory,
                    "ProductName": product_name,
                }])

                if saved_features:
                    missing = [c for c in saved_features if c not in input_row.columns]
                    if missing:
                        st.error(f"Model expects missing input features: {missing}")
                    else:
                        input_row = input_row[saved_features]

                prediction = float(pipeline.predict(input_row)[0])
                prediction = max(0.0, prediction)
                st.success(f"Estimated order sales: {prediction:,.2f}")
                st.caption(
                    "This is a model estimate based on the selected attributes, "
                    "not a guaranteed selling price or final invoice amount."
                )

        except Exception as exc:
            st.error("The model could not be downloaded or loaded.")
            st.exception(exc)
            st.info(
                "Check that the Hugging Face repository is public (or that "
                "HF_TOKEN is configured), and that the uploaded filename matches "
                "HF_MODEL_FILENAME."
            )

    if order_predictions is not None:
        with st.expander("Review held-out model predictions"):
            st.dataframe(order_predictions.head(500), use_container_width=True)
            if {"Sales", "PredictedSales"}.issubset(order_predictions.columns):
                actual = pd.to_numeric(order_predictions["Sales"], errors="coerce")
                predicted = pd.to_numeric(
                    order_predictions["PredictedSales"], errors="coerce"
                )
                comparison = pd.DataFrame(
                    {"Actual sales": actual, "Predicted sales": predicted}
                ).dropna()
                fig = px.scatter(
                    comparison,
                    x="Actual sales",
                    y="Predicted sales",
                    title="Actual vs predicted sales on held-out orders",
                    opacity=0.55,
                )
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "E-commerce Analytics & Intelligence | Python • Pandas • scikit-learn • "
    "Streamlit • Hugging Face"
)
