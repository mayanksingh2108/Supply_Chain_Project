import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Supply Chain Intelligence System", 
    page_icon="📦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "primary": "#C470FF",     # Light Purple
    "secondary": "#00A1C2",   # Teal/Cyan
    "accent": "#FDFF70",      # Pale Yellow
    "danger": "#FF0000",      # Red
    "neutral": "#b3b3b3",     # Grey
    "success": "#00A86B"      # Jade Green
}

# ==========================================
# 2. CORE DATA FUNCTIONS (Cached)
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('SupplyChainData.csv', encoding='latin-1')
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    if 'delivery_status' in df.columns:
        df = df[df['delivery_status'] != 'shipping canceled']
    
    df['order_date_(dateorders)'] = pd.to_datetime(df['order_date_(dateorders)'], errors='coerce')
    df['shipping_date_(dateorders)'] = pd.to_datetime(df['shipping_date_(dateorders)'], errors='coerce')
    
    df['order_processing_time'] = (df['shipping_date_(dateorders)'] - df['order_date_(dateorders)']).dt.days
    df['delay'] = df['order_processing_time'] - df['days_for_shipment_(scheduled)']
    df['is_delayed'] = df['delay'] > 0
    df['order_month'] = df['order_date_(dateorders)'].dt.month
    df['order_hour'] = df['order_date_(dateorders)'].dt.hour
    
    if 'order_profit_per_order' in df.columns:
        df['profitability_flag'] = np.where(df['order_profit_per_order'] > 0, 'Profit', 
                                   np.where(df['order_profit_per_order'] < 0, 'Loss', 'Break_even'))
    return df

@st.cache_data
def get_frequency_mappings(df):
    cat_cols = ['type', 'category_name', 'customer_segment', 'department_name', 'order_region', 'shipping_mode']
    mappings = {}
    for col in cat_cols:
        if col in df.columns:
            mappings[col] = df[col].value_counts(normalize=True).to_dict()
    return mappings

@st.cache_resource
def load_model():
    model_path = 'rf_model_balanced.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    return None

df = load_data()
freq_mappings = get_frequency_mappings(df)
model = load_model()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
# Using the exact same emoji and text to ensure the page routing works perfectly
st.sidebar.image(
    "https://img.freepik.com/premium-photo/visual-supply-chain-management-app-interface-monitoring-managing-logistics-operations_1314467-61063.jpg",
    width=200
)
st.sidebar.title("Navigation")
st.markdown("""
<style>
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Go to:",
    ["📊 Performance Dashboard", "🔮 Prediction Model"]
)

# ==========================================
# 4. PAGE 1: DASHBOARD
# ==========================================
if page == "📊 Performance Dashboard":
    st.title("📦 Supply Chain Performance Dashboard")
    
    # --- Main Page Filters ---
    st.markdown("### Dashboard Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        regions = ['All'] + sorted(df['order_region'].dropna().unique().tolist())
        selected_region = st.selectbox("Select Order Region", regions)
    with f_col2:
        segments = ['All'] + sorted(df['customer_segment'].dropna().unique().tolist())
        selected_segment = st.selectbox("Select Customer Segment", segments)
    with f_col3:
        ship_modes = ['All'] + sorted(df['shipping_mode'].dropna().unique().tolist())
        selected_mode = st.selectbox("Select Shipping Mode", ship_modes)

    st.markdown("---")

    # --- Apply Filters ---
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['order_region'] == selected_region]
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['customer_segment'] == selected_segment]
    if selected_mode != 'All':
        filtered_df = filtered_df[filtered_df['shipping_mode'] == selected_mode]

    # --- 1) KPIs ---
    st.subheader("1. Key Performance Indicators (Baseline)")
    
    total_orders = len(filtered_df)
    late_deliveries = filtered_df['is_delayed'].sum()
    late_rate = (late_deliveries / total_orders) * 100 if total_orders > 0 else 0
    on_time_rate = 100 - late_rate

    total_profit = filtered_df[filtered_df['order_profit_per_order'] > 0]['order_profit_per_order'].sum()
    profit_at_risk = filtered_df[filtered_df['is_delayed'] == True]['order_profit_per_order'].sum()
    avg_order_profit = filtered_df['order_profit_per_order'].mean() if total_orders > 0 else 0

    delayed_orders = filtered_df[filtered_df['is_delayed'] == True]
    p90_delay = delayed_orders['delay'].quantile(0.9) if not delayed_orders.empty else 0

    def format_currency(value):
        if value >= 1e6: return f"${value/1e6:.1f}M"
        elif value >= 1e3: return f"${value/1e3:.1f}K"
        return f"${value:.2f}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders Analyzed", f"{total_orders:,}")
        st.metric("Total Profit (Profitable)", format_currency(total_profit))
    with col2:
        st.metric("Late Deliveries", f"{late_deliveries:,}")
        st.metric("Profit at Risk (Delayed)", format_currency(profit_at_risk), delta="Risk", delta_color="inverse")
    with col3:
        st.metric("Late Delivery Rate", f"{late_rate:.2f}%")
        st.metric("90th Percentile Delay", f"{int(p90_delay)} Days")
    with col4:
        st.metric("On-Time Delivery Rate", f"{on_time_rate:.2f}%")
        st.metric("Avg Order Profit", f"${avg_order_profit:.2f}")

    st.markdown("---")

    # --- 2) Profitability vs Delivery Time Analysis ---
    st.subheader("2. Profitability vs Delivery Time Analysis")
    if not filtered_df.empty:
        profit_metrics = filtered_df.groupby('delay').agg(
            total_profit=('order_profit_per_order', 'sum'),
            mean_profit=('order_profit_per_order', 'mean')
        ).reset_index()
        
        # Dual-axis chart
        fig_profit = make_subplots(specs=[[{"secondary_y": True}]])
        fig_profit.add_trace(
            go.Bar(x=profit_metrics['delay'], y=profit_metrics['total_profit'], name="Total Profit", marker_color=COLORS['primary']),
            secondary_y=False,
        )
        fig_profit.add_trace(
            go.Scatter(x=profit_metrics['delay'], y=profit_metrics['mean_profit'], name="Mean Profit", mode='lines+markers', line=dict(color=COLORS['accent'], width=3)),
            secondary_y=True,
        )
        
        fig_profit.update_layout(template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_profit.update_xaxes(title_text="Delay (Days)")
        fig_profit.update_yaxes(title_text="Total Profit ($)", secondary_y=False, color=COLORS['primary'])
        fig_profit.update_yaxes(title_text="Mean Profit ($)", secondary_y=True, color=COLORS['accent'])
        st.plotly_chart(fig_profit, use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    # --- 3) Bottleneck Detection ---
    with c1: 
        st.subheader("3. Bottleneck Detection")
        if not filtered_df.empty:
            delay_by_mode = filtered_df.groupby('shipping_mode').agg(
                total=('is_delayed', 'count'), late=('is_delayed', 'sum')
            ).reset_index()
            delay_by_mode['delay_pct'] = (delay_by_mode['late'] / delay_by_mode['total']) * 100
            
            fig_mode = px.bar(
                delay_by_mode.sort_values('delay_pct', ascending=True), 
                x='delay_pct', y='shipping_mode', orientation='h',
                text_auto='.1f', color_discrete_sequence=[COLORS["secondary"]]
            )
            fig_mode.update_layout(xaxis_title="Delay Percentage (%)", yaxis_title="", template="plotly_dark", title="Delay % by Shipping Mode")
            st.plotly_chart(fig_mode, use_container_width=True)

    # --- 4) Time Based Analysis ---
    with c2: 
        st.subheader("4. Time Based Analysis")
        if not filtered_df.empty:
            delay_by_month = filtered_df.groupby('order_month').agg(
                total=('is_delayed', 'count'), late=('is_delayed', 'sum')
            ).reset_index()
            delay_by_month['delay_pct'] = (delay_by_month['late'] / delay_by_month['total']) * 100
            month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            delay_by_month['month_name'] = delay_by_month['order_month'].map(month_names)
            
            fig_trend = px.line(
                delay_by_month, x='month_name', y='delay_pct', markers=True,
                color_discrete_sequence=[COLORS["primary"]]
            )
            fig_trend.update_layout(xaxis_title="Month", yaxis_title="Delay Percentage (%)", template="plotly_dark", title="Delay % Trend Over Months")
            st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # --- 5) Strategic Insights ---
    st.subheader("5. Strategic Insights")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.info(f"**Late Delivery Rate at {late_rate:.2f}%**\n\nMore than half of orders arrive late. This is a default experience, not an edge-case.")
    with ic2:
        st.error(f"**{format_currency(profit_at_risk)} Profit at Risk**\n\nDelayed orders generated significant profit that is under pressure from operational deterioration.")
    with ic3:
        st.warning(f"**90th Percentile Delay = {int(p90_delay)} Days**\n\nExtreme cases are contained within {int(p90_delay)} days, suggesting systemic, process-driven bottlenecks.")

# ==========================================
# 5. PAGE 2: PREDICTION MODEL
# ==========================================
elif page == "🔮 Prediction Model":
    st.title("🔮 Late Delivery Prediction Engine")
    st.markdown("Use this tool to predict if a new order is at risk of being delayed based on the Random Forest model.")
    st.markdown("---")

    if model is None:
        st.error("⚠️ Model not found! Please ensure 'rf_model_balanced.pkl' is in the same directory as this app.")
    else:
        st.subheader("Enter Order Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            inp_type = st.selectbox("Transaction Type", sorted(df['type'].dropna().unique()))
            inp_cat = st.selectbox("Category Name", sorted(df['category_name'].dropna().unique()))
            inp_sched = st.number_input("Scheduled Shipping Days", min_value=0, max_value=30, value=4)
            
        with col2:
            inp_seg = st.selectbox("Customer Segment", sorted(df['customer_segment'].dropna().unique()))
            inp_dept = st.selectbox("Department Name", sorted(df['department_name'].dropna().unique()))
            inp_month = st.slider("Order Month", 1, 12, 6)
            
        with col3:
            inp_reg = st.selectbox("Order Region", sorted(df['order_region'].dropna().unique()))
            inp_mode = st.selectbox("Shipping Mode", sorted(df['shipping_mode'].dropna().unique()))
            inp_hour = st.slider("Order Hour (0-23)", 0, 23, 12)

        if st.button("Predict Delivery Status", use_container_width=True, type="primary"):
            enc_type = freq_mappings['type'].get(inp_type, 0)
            enc_cat = freq_mappings['category_name'].get(inp_cat, 0)
            enc_seg = freq_mappings['customer_segment'].get(inp_seg, 0)
            enc_dept = freq_mappings['department_name'].get(inp_dept, 0)
            enc_reg = freq_mappings['order_region'].get(inp_reg, 0)
            enc_mode = freq_mappings['shipping_mode'].get(inp_mode, 0)

            input_features = pd.DataFrame([{
                'days_for_shipment_(scheduled)': inp_sched,
                'order_month': inp_month,
                'order_hour': inp_hour,
                'type_freq': enc_type,
                'category_name_freq': enc_cat,
                'customer_segment_freq': enc_seg,
                'department_name_freq': enc_dept,
                'order_region_freq': enc_reg,
                'shipping_mode_freq': enc_mode
            }])

            prediction = model.predict(input_features)[0]
            probability = model.predict_proba(input_features)[0][1]

            st.markdown("---")
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"🚨 **HIGH RISK OF DELAY** (Probability: {probability:.1%})")
                st.write("This order matches patterns of historically delayed shipments. Consider expediting fulfillment or proactively notifying the customer.")
            else:
                st.success(f"✅ **LIKELY ON TIME** (Probability of delay: {probability:.1%})")
                st.write("This order is predicted to ship and arrive within the scheduled timeframe.")