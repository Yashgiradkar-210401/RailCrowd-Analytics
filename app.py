# app.py
import os
import csv
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Try importing PuLP (for optimization)
try:
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

# ---------------------------------------------
# 🧭 Page Configuration
# ---------------------------------------------
st.set_page_config(page_title="RailCrowd-Analytics", layout="wide", page_icon="🚆")

# ---------------------------------------------
# 💅 CSS Styling
# ---------------------------------------------
st.markdown("""
<style>
div.block-container {
    padding: 1rem 3rem 3rem 3rem;
    background-color: #F9FBFD;
}
/* Header */
.dashboard-header {
    background: linear-gradient(90deg, #007BFF, #00BFFF);
    padding: 30px 20px;
    border-radius: 20px;
    color: white;
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}
.dashboard-subtitle {
    text-align: center;
    font-size: 18px;
    color: #E3F2FD;
    margin-top: -10px;
    margin-bottom: 30px;
}
/* KPI Cards */
.kpi-box {
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-weight: 700;
    font-size: 22px;
    color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
}
.kpi-blue {background: linear-gradient(90deg, #2196F3, #21CBF3);}
.kpi-orange {background: linear-gradient(90deg, #FF9800, #FFC107);}
.kpi-green {background: linear-gradient(90deg, #4CAF50, #81C784);}
.kpi-red {background: linear-gradient(90deg, #f44336, #E57373);}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 🧭 Header
# ---------------------------------------------
st.markdown('<div class="dashboard-header">🚆 RailCrowd-Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Crowd Forecasting | Revenue Insights | Profitability | Coach Optimization</div>', unsafe_allow_html=True)

# ---------------------------------------------
# 📂 Load Dataset
# ---------------------------------------------
def load_dataset(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            sep = csv.Sniffer().sniff(first_line).delimiter if first_line else ","
        df = pd.read_csv(file_path, sep=sep, encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV (optional)", type=["csv"])
if uploaded_file:
    raw = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()
    sep = csv.Sniffer().sniff(raw[0]).delimiter if raw and len(raw[0]) > 0 else ","
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=sep, encoding="utf-8")
    st.sidebar.success("✅ File loaded successfully.")
else:
    # Default path (adjust as needed)
    file_path = os.path.join(os.path.dirname(__file__), "../data/IRCTC.csv")
    df = load_dataset(file_path)

if df is None or df.empty:
    st.warning("⚠️ No dataset found.")
    st.stop()

# ---------------------------------------------
# 🧹 Data Cleaning
# ---------------------------------------------
# Normalize columns
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

# Find source/destination/train/coach/date/occupancy columns heuristically
src_col = next((c for c in df.columns if "Source" in c or "source" in c), None)
dest_col = next((c for c in df.columns if "Destination" in c or "destination" in c), None)
train_col = next((c for c in df.columns if "Train" in c or "train" in c), None)
coach_col = next((c for c in df.columns if "Coach" in c), None)
occ_col = next((c for c in df.columns if "Occupancy" in c or "Crowd" in c), None)
date_col = next((c for c in df.columns if "Date" in c or "date" in c), None)

# Route column
if src_col and dest_col:
    df["Route"] = df[src_col].astype(str).str.strip() + " → " + df[dest_col].astype(str).str.strip()
else:
    df["Route"] = "Unknown Route"

# Coach type
df["Coach_Type"] = df[coach_col].astype(str).str.strip() if coach_col else "General"

# Occupancy handling
if occ_col:
    df["Occupancy_Percent"] = pd.to_numeric(df[occ_col], errors="coerce")
else:
    df["Occupancy_Percent"] = np.nan

mask = df["Occupancy_Percent"].isna()
if mask.any():
    df.loc[mask, "Occupancy_Percent"] = np.random.randint(40, 100, size=mask.sum())

# Synthetic finance columns (if not present)
if "Revenue" not in df.columns:
    df["Revenue"] = np.random.randint(50000, 200000, len(df))
if "Loss" not in df.columns:
    df["Loss"] = df["Revenue"] * (100 - df["Occupancy_Percent"]) / 100
if "Profit" not in df.columns:
    df["Profit"] = df["Revenue"] - df["Loss"]
if "Profit_Margin" not in df.columns:
    df["Profit_Margin"] = (df["Profit"] / df["Revenue"]) * 100

# Date parsing
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ---------------------------------------------
# 🧭 Sidebar Filters (All Functionalities)
# ---------------------------------------------
st.sidebar.header("🔍 Filter Options")

routes = st.sidebar.multiselect("Select Route(s)", sorted(df["Route"].unique()))
coach_types = st.sidebar.multiselect("Select Coach Type(s)", sorted(df["Coach_Type"].unique()))
selected_train = st.sidebar.multiselect("Select Train No/Name", sorted(df[train_col].dropna().astype(str).unique().tolist())) if train_col else []
selected_src = st.sidebar.multiselect("Select Source", sorted(df[src_col].dropna().astype(str).unique().tolist())) if src_col else []
selected_dest = st.sidebar.multiselect("Select Destination", sorted(df[dest_col].dropna().astype(str).unique().tolist())) if dest_col else []

if date_col:
    min_date, max_date = df[date_col].min(), df[date_col].max()
    # default date input requires date objects
    date_range = st.sidebar.date_input("Select Date Range", [min_date.date() if pd.notna(min_date) else pd.to_datetime("2020-01-01").date(),
                                                              max_date.date() if pd.notna(max_date) else pd.to_datetime("2020-12-31").date()])
else:
    date_range = None

# Apply filters
filtered_df = df.copy()
if routes:
    filtered_df = filtered_df[filtered_df["Route"].isin(routes)]
if coach_types:
    filtered_df = filtered_df[filtered_df["Coach_Type"].isin(coach_types)]
if selected_train and train_col:
    filtered_df = filtered_df[filtered_df[train_col].astype(str).isin(selected_train)]
if selected_src and src_col:
    filtered_df = filtered_df[filtered_df[src_col].astype(str).isin(selected_src)]
if selected_dest and dest_col:
    filtered_df = filtered_df[filtered_df[dest_col].astype(str).isin(selected_dest)]
if date_range and date_col:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df[date_col] >= start) & (filtered_df[date_col] <= end)]

# ---------------------------------------------
# 💡 KPI Cards
# ---------------------------------------------
st.markdown("### 💡 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='kpi-box kpi-orange'>Avg Occupancy<br>{filtered_df['Occupancy_Percent'].mean():.2f}%</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='kpi-box kpi-blue'>Total Revenue<br>₹{filtered_df['Revenue'].sum():,.0f}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='kpi-box kpi-green'>Total Profit<br>₹{filtered_df['Profit'].sum():,.0f}</div>", unsafe_allow_html=True)
col4.markdown(f"<div class='kpi-box kpi-red'>Avg Profit Margin<br>{filtered_df['Profit_Margin'].mean():.2f}%</div>", unsafe_allow_html=True)

# ---------------------------------------------
# Tabs
# ---------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview", "💰 Revenue Insights", "🚆 Occupancy Trends",
    "🧠 Predictive Analytics", "💵 Profitability & Efficiency",
    "📊 Coach Optimization", "📘 About Project"
])

# ---------- Tab 1 ----------
with tab1:
    st.subheader("🧾 Data Summary")
    st.dataframe(filtered_df.head(20))

# ---------- Tab 2: Revenue Insights ----------
with tab2:
    st.subheader("💰 Revenue by Route")
    fig_rev = px.bar(filtered_df.groupby("Route", as_index=False)["Revenue"].sum(),
                     x="Route", y="Revenue", color="Route", title="Total Revenue per Route")
    st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("### 💡 Route-Level Financial KPIs")
    colA, colB, colC = st.columns(3)
    colA.metric("Avg Revenue per Route", f"₹{filtered_df['Revenue'].mean():,.0f}")
    colB.metric("Avg Profit Margin", f"{filtered_df['Profit_Margin'].mean():.2f}%")
    colC.metric("Revenue Variance", f"₹{filtered_df['Revenue'].std():,.0f}")

    st.markdown("---")
    try:
        # forecast_df is generated in Predictive tab; show if available
        if 'forecast_df' in locals() and not forecast_df.empty:
            st.subheader("📊 Forecasted Financial Trends (Next Days)")
            # Add simple derived profit/loss projections
            ff = forecast_df.copy()
            ff["Predicted_Profit"] = ff["Predicted_Revenue"] * 0.25
            ff["Predicted_Loss"] = ff["Predicted_Revenue"] * 0.10
            fig_fin = px.line(ff, x="Date", y=["Predicted_Revenue", "Predicted_Profit", "Predicted_Loss"],
                              title="Forecasted Revenue, Profit & Loss Trends")
            st.plotly_chart(fig_fin, use_container_width=True)
    except Exception:
        st.info("Generate forecast data from Predictive Analytics tab (Tab: 🧠 Predictive Analytics).")

    st.markdown("---")
    st.subheader("🎯 Ticket Pricing Simulation")
    price_change = st.slider("Change Ticket Price (%)", -30, 30, 0)
    adjusted_revenue = filtered_df["Revenue"].sum() * (1 + price_change / 100)
    st.metric("Adjusted Revenue Estimate", f"₹{adjusted_revenue:,.0f}", f"{price_change}%")

# ---------- Tab 3: Occupancy Trends ----------
with tab3:
    st.subheader("🚆 Average Occupancy by Route")
    fig_occ = px.bar(filtered_df.groupby("Route", as_index=False)["Occupancy_Percent"].mean(),
                     x="Route", y="Occupancy_Percent", color="Route", title="Average Occupancy per Route")
    st.plotly_chart(fig_occ, use_container_width=True)

# ---------- Tab 4: Predictive Analytics ----------
with tab4:
    st.subheader("🧠 Predictive Analytics: Occupancy & Revenue Forecast")

    if date_col and pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
        routes_list = sorted(filtered_df["Route"].astype(str).unique())
        if not routes_list:
            st.info("No routes available in filtered data.")
        else:
            selected_route = st.selectbox("Select Route for Prediction", routes_list)
            forecast_days = st.slider("Select Forecast Horizon (days)", 7, 30, 14)

            route_df = filtered_df[filtered_df["Route"] == selected_route].sort_values(date_col)
            daily = route_df.groupby(pd.Grouper(key=date_col, freq="D")).agg(
                {"Occupancy_Percent": "mean", "Revenue": "sum"}).reset_index()

            if len(daily) > 8:
                daily["dayofweek"] = daily[date_col].dt.dayofweek
                for lag in [1, 7]:
                    daily[f"occ_lag_{lag}"] = daily["Occupancy_Percent"].shift(lag)
                    daily[f"rev_lag_{lag}"] = daily["Revenue"].shift(lag)
                daily = daily.dropna()

                # Features for models
                X_occ = daily[["dayofweek", "occ_lag_1", "occ_lag_7"]]
                y_occ = daily["Occupancy_Percent"]
                X_rev = daily[["dayofweek", "rev_lag_1", "rev_lag_7", "occ_lag_1", "occ_lag_7"]]
                y_rev = daily["Revenue"]

                # Train simple models (no shuffle since time series)
                try:
                    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_occ, y_occ, test_size=0.2, shuffle=False)
                    occ_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
                    occ_model.fit(X_train_o, y_train_o)

                    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_rev, y_rev, test_size=0.2, shuffle=False)
                    rev_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
                    rev_model.fit(X_train_r, y_train_r)

                    # Forecast loop (autoregressive using previous preds)
                    future = []
                    last_row = daily.iloc[-1:].copy()
                    for i in range(forecast_days):
                        next_date = last_row[date_col].values[0] + np.timedelta64(1, 'D')
                        dow = pd.Timestamp(next_date).dayofweek
                        pred_occ = occ_model.predict(np.array([[dow, last_row["Occupancy_Percent"].values[0], last_row["occ_lag_7"].values[0]]]))[0]
                        pred_rev = rev_model.predict(np.array([[dow, last_row["Revenue"].values[0], last_row["rev_lag_7"].values[0],
                                                               last_row["Occupancy_Percent"].values[0], last_row["occ_lag_7"].values[0]]]))[0]
                        future.append({"Date": pd.Timestamp(next_date), "Predicted_Occupancy": pred_occ, "Predicted_Revenue": pred_rev})
                        # shift last_row values
                        last_row["Occupancy_Percent"], last_row["Revenue"] = pred_occ, pred_rev
                        # create new lag columns for next iteration if needed
                        last_row["occ_lag_7"] = last_row.get("occ_lag_1", pred_occ)  # fallback
                        last_row["rev_lag_7"] = last_row.get("rev_lag_1", pred_rev)
                    forecast_df = pd.DataFrame(future)

                    # Show forecast chart
                    st.plotly_chart(px.line(forecast_df, x="Date", y=["Predicted_Occupancy", "Predicted_Revenue"],
                                            title=f"Forecast for {selected_route}"), use_container_width=True)

                    # Model evaluation (on holdout)
                    mae_occ = mean_absolute_error(y_test_o, occ_model.predict(X_test_o))
                    rmse_occ = np.sqrt(mean_squared_error(y_test_o, occ_model.predict(X_test_o)))
                    st.info(f"📊 Occupancy Model MAE: {mae_occ:.2f} | RMSE: {rmse_occ:.2f}")

                    # -------------------------------
                    # Instant Prediction Interface
                    # -------------------------------
                    st.markdown("---")
                    st.subheader("🎯 Instant Prediction Interface (Manual Inputs)")
                    with st.form("predict_form", clear_on_submit=False):
                        # route & coach are already selected above for context; we allow manual day/values
                        day_input = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)
                        occ_input = st.number_input("Last Day Occupancy (%)", 0.0, 150.0, 80.0)
                        occ_7_input = st.number_input("Occupancy 7 Days Ago (%)", 0.0, 150.0, 75.0)
                        rev_last_input = st.number_input("Last Day Revenue (₹)", 0.0, 5_000_000.0, 120000.0)
                        rev_7_input = st.number_input("Revenue 7 Days Ago (₹)", 0.0, 5_000_000.0, 100000.0)
                        submit_btn = st.form_submit_button("🚀 Predict Crowd & Revenue")

                    if submit_btn:
                        try:
                            pred_occ = occ_model.predict(np.array([[day_input, occ_input, occ_7_input]]))[0]
                            pred_rev = rev_model.predict(np.array([[day_input, rev_last_input, rev_7_input, occ_input, occ_7_input]]))[0]
                            colA, colB = st.columns(2)
                            colA.markdown(f"<div class='kpi-box kpi-blue'>🎫 Predicted Occupancy<br>{pred_occ:.2f}%</div>", unsafe_allow_html=True)
                            colB.markdown(f"<div class='kpi-box kpi-green'>💰 Predicted Revenue<br>₹{pred_rev:,.0f}</div>", unsafe_allow_html=True)
                            st.success("✅ Prediction generated successfully!")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")

                except Exception as e:
                    st.warning(f"Not enough historical data to build reliable models for this route. ({e})")
            else:
                st.info("Not enough daily history (needs > 8 days) for forecasting on this route.")
    else:
        st.warning("⚠️ Date column missing or invalid for forecasting. Please ensure your dataset has a valid date column.")

# ---------- Tab 5: Profitability ----------
with tab5:
    st.subheader("💵 Profitability & Efficiency Analysis")
    profit_route = filtered_df.groupby("Route", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    st.plotly_chart(px.bar(profit_route, x="Route", y="Profit", color="Route", title="Profit by Route"), use_container_width=True)
    st.markdown("#### 🏆 Top 5 Profitable Routes")
    st.dataframe(profit_route.head(5))
    st.markdown("#### ⚠️ Bottom 5 Underperforming Routes")
    st.dataframe(profit_route.tail(5))

# ---------- Tab 6: Optimization ----------
with tab6:
    st.subheader("📊 Coach Allocation Optimization")
    if not PULP_AVAILABLE:
        st.warning("⚠️ PuLP not installed. Run `pip install pulp` to enable optimization.")
    else:
        total_coaches = st.slider("Total Available Coaches", 5, 200, 20)
        route_summary = filtered_df.groupby("Route", as_index=False).agg({
            "Profit": "mean", "Revenue": "mean", "Loss": "mean", "Occupancy_Percent": "mean"
        }).reset_index(drop=True)
        if route_summary.empty:
            st.info("No route data available for optimization with the current filters.")
        else:
            model = LpProblem("Coach_Allocation_Optimization", LpMaximize)
            routes_list = route_summary["Route"].tolist()
            x = {r: LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i, r in enumerate(routes_list)}
            model += lpSum((route_summary.loc[i, "Profit"]) * x[r] for i, r in enumerate(routes_list))
            model += lpSum(x[r] for r in x) <= total_coaches
            model.solve()
            alloc = [{"Route": r, "Optimal_Coaches": int(x[r].value())} for r in x]
            df_alloc = pd.DataFrame(alloc)
            st.plotly_chart(px.bar(df_alloc, x="Route", y="Optimal_Coaches", title="Optimal Coach Allocation per Route"), use_container_width=True)
            st.dataframe(df_alloc)

# ---------- Tab 7: About ----------
with tab7:
    st.markdown("""
    ### 🎯 Project Summary
    **TrainCrowd AI** is a predictive analytics dashboard that forecasts train coach occupancy and revenue 
    using **Machine Learning (XGBoost)** and provides insights into route profitability, utilization, and 
    coach allocation optimization.

    ### 🧠 Key Features
    - Real-time crowd forecasting per route  
    - Revenue and profit analytics  
    - Optimization for coach allocation (using PuLP)  
    - Instant prediction interface for business scenarios  

    ### 💼 Business Value
    Helps rail operators:
    - Reduce overcrowding losses  
    - Optimize coach deployment  
    - Improve passenger experience  
    - Enhance operational efficiency and profitability  

    ### 🧰 Tech Stack
    **Python**, **Streamlit**, **Pandas**, **Plotly**, **XGBoost**, **PuLP**, **NumPy**

    ---
    © 2025 TrainCrowd AI | Developed by Yash — PGDM (Business Analytics), N.L. Dalmia Institute
    """, unsafe_allow_html=True)

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("© 2025 TrainCrowd AI | Predictive, Profitability & Optimization Dashboard")
