import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import streamlit as st
from statsmodels.tsa.api import SimpleExpSmoothing

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Ghana Biz Finance Dashboard",
    page_icon="📊",
    layout="wide"
)

DB_FILE = "gh_finance.db"

# -------------------------------------------------------------------
# 1. DATABASE MANAGEMENT (Persistent Local File)
# -------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_FILE)
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        type TEXT CHECK(type IN ('revenue', 'expense')),
        category TEXT NOT NULL,
        amount_ghs REAL NOT NULL
    )
    """)

    cursor.execute("SELECT COUNT(*) FROM transactions")
    if cursor.fetchone()[0] == 0:
        seed_data = [
            ("2026-03-01", "revenue", "Retail Sales & Mobile Money", 14500.0),
            ("2026-04-01", "revenue", "Retail Sales & Mobile Money", 15200.0),
            ("2026-05-01", "revenue", "Retail Sales & Mobile Money", 16000.0),
            ("2026-06-01", "revenue", "Retail Sales & Mobile Money", 15800.0),
            ("2026-07-01", "revenue", "Retail Sales & Mobile Money", 17100.0),
            ("2026-08-01", "revenue", "Retail Sales & Mobile Money", 18000.0),
            ("2026-03-05", "expense", "Wholesale Inventory", 6000.0),
            ("2026-03-15", "expense", "ECG Electricity Bill", 850.0),
            ("2026-04-05", "expense", "Wholesale Inventory", 6200.0),
            ("2026-04-18", "expense", "ECG Electricity Bill", 910.0),
            ("2026-05-05", "expense", "Wholesale Inventory", 6100.0),
            ("2026-05-12", "expense", "ECG Electricity Bill", 880.0),
            ("2026-06-05", "expense", "Wholesale Inventory", 6300.0),
            ("2026-06-20", "expense", "ECG Electricity Bill", 930.0),
            ("2026-07-05", "expense", "Wholesale Inventory", 6500.0),
            ("2026-07-15", "expense", "ECG Electricity Bill", 890.0),
            ("2026-08-05", "expense", "Wholesale Inventory", 6400.0),
            ("2026-08-14", "expense", "ECG Electricity Bill", 920.0),
            ("2026-08-22", "expense", "ECG Electricity Bill", 4850.0),
            ("2026-08-28", "expense", "Generator Fuel & Repairs", 7200.0)
        ]
        cursor.executemany("INSERT INTO transactions (date, type, category, amount_ghs) VALUES (?, ?, ?, ?)", seed_data)
        conn.commit()
    conn.close()

init_db()

def load_data():
    conn = get_db()
    df_data = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    return df_data

df = load_data()

# -------------------------------------------------------------------
# 2. ANALYTICS ENGINES
# -------------------------------------------------------------------
def detect_expense_anomalies(df_data, z_threshold):
    exp_df = df_data[df_data['type'] == 'expense'].copy()
    if exp_df.empty:
        return exp_df
    stats = exp_df.groupby('category')['amount_ghs'].agg(['mean', 'std']).reset_index()
    exp_df = pd.merge(exp_df, stats, on='category')
    exp_df['z_score'] = np.where(
        exp_df['std'] > 0,
        (exp_df['amount_ghs'] - exp_df['mean']) / exp_df['std'],
        0
    )
    exp_df['is_anomaly'] = exp_df['z_score'] > z_threshold
    return exp_df

def forecast_revenue(df_data, periods=3, smoothing_level=0.6):
    rev_df = df_data[df_data['type'] == 'revenue'].copy()
    if rev_df.empty or len(rev_df) < 2:
        return rev_df, pd.DataFrame()
    rev_df['date'] = pd.to_datetime(rev_df['date'])
    rev_df = rev_df.sort_values('date')

    model = SimpleExpSmoothing(rev_df['amount_ghs']).fit(smoothing_level=smoothing_level, optimized=False)
    forecast_values = model.forecast(periods)

    last_date = rev_df['date'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, periods + 1)]

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'amount_ghs': forecast_values,
        'type': 'Forecast'
    })

    rev_df['type'] = 'Actual'
    combined = pd.concat([rev_df[['date', 'amount_ghs', 'type']], forecast_df], ignore_index=True)
    return combined, forecast_df

def generate_ai_summary(rev_total, anomalies_df, forecast_df, api_key):
    anomaly_details = "\n".join(
        [f"- {row['date']}: GHS {row['amount_ghs']:,.2f} on '{row['category']}'"
         for _, row in anomalies_df.iterrows()]
    ) if not anomalies_df.empty else "No unusual expenses detected."

    forecast_details = "\n".join(
        [f"- {row['date'].strftime('%b %Y')}: GHS {row['amount_ghs']:,.2f}" 
         for _, row in forecast_df.iterrows()]
    ) if not forecast_df.empty else "Insufficient revenue data to forecast."

    prompt = f"""
    You are an expert financial advisor for a Ghanaian SME.
    Analyze the following financial metrics:

    - Total Revenue: GHS {rev_total:,.2f}
    - Anomaly Alerts:
    {anomaly_details}

    - Revenue Forecast:
    {forecast_details}

    Write a concise summary covering: Financial Overview, Key Risk Alerts (utilities/fuel), and Working Capital Recommendations for operating in Ghana.
    """

    if not api_key:
        return f"⚠️ **OpenAI API Key required.**\n\n*Draft Context Generated:* \n- Total Revenue: GHS {rev_total:,.2f}\n- Flagged Outliers: {len(anomalies_df)}"

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=12
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# -------------------------------------------------------------------
# 3. INTERACTIVE DASHBOARD & INPUT FORM
# -------------------------------------------------------------------
st.title("🇬🇭 Ghanaian SME Financial Intelligence Dashboard")

# --- SIDEBAR: INPUT FORM ---
st.sidebar.header("➕ Add New Transaction")
with st.sidebar.form("add_transaction_form", clear_on_submit=True):
    input_date = st.date_input("Date")
    input_type = st.selectbox("Type", ["expense", "revenue"])
    input_category = st.selectbox(
        "Category", 
        ["Wholesale Inventory", "ECG Electricity Bill", "GRA Taxes / Levy", 
         "Generator Fuel & Repairs", "Retail Sales & Mobile Money", "Other Overhead"]
    )
    input_amount = st.number_input("Amount (GHS)", min_value=1.0, value=500.0, step=50.0)
    submit_btn = st.form_submit_button("Save Entry")

if submit_btn:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO transactions (date, type, category, amount_ghs) VALUES (?, ?, ?, ?)",
        (str(input_date), input_type, input_category, input_amount)
    )
    conn.commit()
    conn.close()
    st.sidebar.success(f"Saved: {input_category} (GHS {input_amount:,.2f})")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Controls")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
z_thresh = st.sidebar.slider("Anomaly Sensitivity (Z-Score)", 1.0, 3.0, 1.8, 0.1)
forecast_months = st.sidebar.slider("Forecast Horizon (Months)", 1, 6, 3)

# --- CALCULATIONS & KPIs ---
expenses_analyzed = detect_expense_anomalies(df, z_thresh)
anomalies_only = expenses_analyzed[expenses_analyzed['is_anomaly']] if not expenses_analyzed.empty else pd.DataFrame()
rev_combined, forecast_only = forecast_revenue(df, periods=forecast_months)

total_revenue = df[df['type'] == 'revenue']['amount_ghs'].sum()
total_expenses = df[df['type'] == 'expense']['amount_ghs'].sum()
net_profit = total_revenue - total_expenses

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"GHS {total_revenue:,.2f}")
col2.metric("Total Expenses", f"GHS {total_expenses:,.2f}")
col3.metric("Net Margin", f"{(net_profit/total_revenue*100 if total_revenue>0 else 0):.1f}%", f"GHS {net_profit:,.2f}")
col4.metric("Flagged Outliers", len(anomalies_only), delta=f"{len(anomalies_only)} Alerts", delta_color="inverse")

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📉 Expense Outliers", "📈 Revenue Forecast", "🤖 AI Summary"])

with tab1:
    st.subheader("Expense Distribution")
    if not expenses_analyzed.empty:
        fig_exp = px.scatter(
            expenses_analyzed,
            x="date",
            y="amount_ghs",
            color="is_anomaly",
            size="amount_ghs",
            hover_data=["category", "z_score"],
            color_discrete_map={True: "#EF553B", False: "#636EFA"},
            title="Expense Outliers (Red = Anomaly)"
        )
        st.plotly_chart(fig_exp, width="stretch")

        st.write("📋 **Transaction History**")
        st.dataframe(expenses_analyzed[['date', 'category', 'amount_ghs', 'is_anomaly']], width="stretch")

with tab2:
    st.subheader("Revenue Projections")
    if not rev_combined.empty:
        fig_rev = px.line(
            rev_combined,
            x="date",
            y="amount_ghs",
            color="type",
            line_dash="type",
            markers=True,
            title=f"{forecast_months}-Month Revenue Forecast (GHS)"
        )
        st.plotly_chart(fig_rev, width="stretch")

with tab3:
    st.subheader("🤖 AI Insights Report")
    if st.button("Generate Insight Report", type="primary"):
        st.cache_data.clear()
    with st.spinner("Analyzing..."):
        summary = generate_ai_summary(total_revenue, anomalies_only, forecast_only, api_key)
        st.markdown(summary)

