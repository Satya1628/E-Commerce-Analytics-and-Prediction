# E-commerce Analytics & Intelligence

An end-to-end e-commerce analytics project combining data cleaning,
exploratory data analysis, customer segmentation, sales prediction,
sales forecasting, and an interactive Streamlit dashboard.

## Live Application

🚀 **Try the app:**
https://e-commerce-analytics-predictive.streamlit.app/

## Project Overview

This project transforms e-commerce transaction data into actionable
business insights. It combines descriptive analytics with machine
learning to explore sales performance, understand customer behaviour,
estimate order sales, and examine future sales trends.

### Key Features

-   **Data Cleaning & EDA:** Handles duplicate records, checks data
    quality, and explores sales, profit, products, customers, and
    time-based trends.
-   **Customer Segmentation:** Applies RFM (Recency, Frequency,
    Monetary) analysis and K-Means clustering to group customers by
    purchasing behaviour.
-   **Order Sales Prediction:** Uses a Random Forest regression model to
    estimate sales for a new order based on its details.
-   **Sales Forecasting:** Compares a Random Forest lag-based approach
    with a seasonal-naive baseline and uses the selected baseline to
    generate an exploratory six-month forecast.
-   **Interactive Dashboard:** Provides an interactive Streamlit
    interface for exploring business performance, customer segments,
    forecasts, and sales predictions.

## Machine Learning

  -----------------------------------------------------------------------
  Component               Method                  Purpose
  ----------------------- ----------------------- -----------------------
  Customer segmentation   RFM + K-Means           Group customers by
                                                  purchasing behaviour

  Order sales prediction  Random Forest Regressor Estimate sales for an
                                                  order

  Monthly sales           Seasonal-naive          Produce an exploratory
  forecasting             baseline, compared with six-month forecast
                          Random Forest lag model 
  -----------------------------------------------------------------------

> **Note:** The dataset contains 36 monthly observations. The sales
> forecast is exploratory and should not be treated as a highly reliable
> long-term forecast. Model performance can also vary when new data is
> used.

## Technology Stack

-   **Language:** Python
-   **Data analysis:** Pandas, NumPy
-   **Machine learning:** Scikit-learn
-   **Visualization:** Plotly
-   **Application:** Streamlit
-   **Model persistence:** Joblib
-   **Dashboarding:** Power BI

## Repository Structure

``` text
Ecommerce-Analytics-Intelligence/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── ecommerce_cleaned.csv
│   ├── customer_segments.csv
│   ├── monthly_sales_actuals.csv
│   ├── sales_forecast_6_months.csv
│   ├── sales_forecast_backtest.csv
│   └── order_sales_predictions.csv
├── notebooks/
│   └── ecommerce_analytics_intelligence.ipynb
├── powerbi/
│   ├── DAX_measures.dax
│   ├── powerbi_theme.json
│   └── POWER_BI_BUILD_GUIDE.md
└── src/
    └── run_pipeline.py
```

*The repository structure may differ slightly depending on which project
artifacts are included.*

## Run Locally

1.  Clone the repository:

    ``` bash
    git clone https://github.com/Satya1628/E-commerce-Sales-Analysis.git
    cd E-commerce-Sales-Analysis
    ```

2.  Install the dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Start the Streamlit application:

    ``` bash
    streamlit run app.py
    ```

Some model artifacts may be hosted separately. If the application uses a
Hugging Face-hosted model, configure the required repository settings or
access token as environment variables or Streamlit secrets before
running it.

## Project Goals

-   Convert raw transaction data into useful business insights.
-   Identify customer groups using purchasing behaviour.
-   Build and evaluate a model for order-level sales estimation.
-   Compare forecasting approaches and communicate forecast limitations.
-   Present the analysis through an accessible interactive application.

## Author

**Satya**

-   GitHub: https://github.com/Satya1628
-   Live App: https://e-commerce-analytics-predictive.streamlit.app/
