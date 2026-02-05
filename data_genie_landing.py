import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from difflib import get_close_matches
import io
import pymongo
import sqlalchemy
import requests
from datetime import datetime
from predictive_engine import EcomPredictor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from chatbot import render_chat_tab
import json
from bson import json_util

#  Initialize predictor ONCE (persists across reruns)
if 'predictor' not in st.session_state:
    st.session_state.predictor = EcomPredictor()  

# --- Initialize MongoDB session state variables ---
if 'mongo_conn_str' not in st.session_state:
    st.session_state.mongo_conn_str = ""
if 'mongo_db_name' not in st.session_state:
    st.session_state.mongo_db_name = ""
if 'mongo_collection_name' not in st.session_state:
    st.session_state.mongo_collection_name = ""
if 'mongo_query' not in st.session_state:
    st.session_state.mongo_query = "{}"
if 'mongo_df' not in st.session_state:
    st.session_state.mongo_df = pd.DataFrame()

# --- Initialize SQL session state variables ---
if 'sql_conn_str' not in st.session_state:
    st.session_state.sql_conn_str = ""
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = "SELECT * FROM sales_data"
if 'sql_df' not in st.session_state:
    st.session_state.sql_df = pd.DataFrame()

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Data Genie Pro", page_icon="🧠")
SAMPLE_CSV_PATH = r"C:\Users\Aman ur Rehman\Desktop\Data_genie\SuperMarket Analysis.csv"

# --------------------------
# 1. DATA LOADING
# --------------------------
@st.cache_data
def load_sample_data():
    try:
        # Try multiple encodings
        encodings = ['ISO-8859-1', 'latin1', 'cp1252', 'utf-8']
        for encoding in encodings:
            try:
                df = pd.read_csv(SAMPLE_CSV_PATH, encoding=encoding)
                if not df.empty:
                    df.columns = df.columns.str.strip()
                    return df
            except:
                continue
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

def load_from_sql(connection_string, query):
    try:
        engine = sqlalchemy.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        st.error(f"SQL Error: {str(e)}")
        return pd.DataFrame()

def load_from_mongo(connection_string, db_name, collection_name, query={}):
    try:
        client = pymongo.MongoClient(connection_string)
        db = client[db_name]
        collection = db[collection_name]
        
        # Debug: Print the query being executed
        print(f"Executing MongoDB query: {query}")
        
        # Handle different query types
        if isinstance(query, list):
            cursor = collection.aggregate(query)
        elif isinstance(query, dict):
            cursor = collection.find(query)
        else:
            st.error("Invalid query type")
            return pd.DataFrame()
            
        data = list(cursor)
        client.close()
        
        # Convert BSON to JSON-compatible format
        json_data = json.loads(json_util.dumps(data))
        return pd.DataFrame(json_data)
    except Exception as e:
        st.error(f"MongoDB Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def load_from_api(api_url, params={}):
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return pd.DataFrame()

# --------------------------
# 2. SMART COLUMN DETECTION
# --------------------------
expected_columns = {
    'order_id': ['order id', 'order number', 'order_no', 'invoiceno', 'invoice no', 'invoiceid', 'transaction id'],
    'order_date': ['invoicedate', 'invoice date', 'order date', 'date', 'orderdate', 'transaction date'],
    'product': ['stockcode', 'stock code', 'description', 'product', 'item name', 'sku', 'product line', 'item'],
    'quantity': ['quantity', 'qty', 'units', 'number sold'],
    'unit_price': ['unitprice', 'unit price', 'price', 'cost per unit'],
    'sales': ['sales', 'total', 'total sales', 'totalsales', 'revenue', 'amount', 'totalamount', 'income'],
    'cost': ['cost', 'purchase cost', 'expense', 'cogs', 'cost of goods'],
    'customer_id': ['customer id', 'client id', 'customerid', 'custid', 'client number'],
    'region': ['region', 'area', 'zone', 'location', 'country', 'territory', 'city', 'state'],
    'branch': ['branch', 'store', 'shop', 'outlet'],
    'customer_type': ['customer type', 'client type', 'customer segment', 'client category'],
    'gender': ['gender', 'sex', 'male/female'],
    'payment': ['payment', 'payment method', 'payment type', 'card type'],
    'rating': ['rating', 'score', 'customer rating', 'review score'],
    'gross_income': ['gross income', 'gross profit', 'total profit', 'net profit']
}

def get_column_mapping(df, expected_columns):
    mapping = {}
    df_cols = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    for key, synonyms in expected_columns.items():
        clean_synonyms = [syn.lower().replace(" ", "_") for syn in synonyms]
        
        # First check for exact matches
        exact_matches = set(df_cols) & set(clean_synonyms)
        if exact_matches:
            match = list(exact_matches)[0]
            mapping[key] = df.columns[df_cols.index(match)]
            continue
            
        # Then check for partial matches
        for col in df_cols:
            for syn in clean_synonyms:
                if syn in col or col in syn:
                    mapping[key] = df.columns[df_cols.index(col)]
                    break
            if key in mapping:
                break
                
        # Finally use fuzzy matching
        if key not in mapping:
            matches = get_close_matches(key, df_cols, n=1, cutoff=0.6)
            if matches:
                mapping[key] = df.columns[df_cols.index(matches[0])]
                
    return mapping
# --------------------------
# 3. DATA PROCESSING PIPELINE
# --------------------------
def auto_profile_and_clean(df, mapping):
    df = df.rename(columns={v: k for k, v in mapping.items() if v is not None})
    
    # Convert dictionary columns to string representations
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            df[col] = df[col].astype(str)
    
    # Convert list columns to string representations
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].astype(str)
    
    df = df.drop_duplicates()

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    for col in ['quantity', 'unit_price', 'sales', 'cost', 'rating', 'gross_income']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)

    return df

def transform_data(df):
    # Calculate sales if we have quantity and unit price but no sales column
    if 'quantity' in df.columns and 'unit_price' in df.columns and 'sales' not in df.columns:
        df['sales'] = df['quantity'] * df['unit_price']

    if 'total' in df.columns and 'sales' not in df.columns:
        df['sales'] = df['total']

    if 'sales' not in df.columns and 'gross_income' in df.columns:
        df['sales'] = df['gross_income']
        df = df.drop(columns=['gross_income'])

    # Calculate profit if we have sales and cost
    if 'sales' in df.columns and 'cost' in df.columns:
        df['profit'] = df['sales'] - df['cost']
        df['profit_margin'] = (df['profit'] / df['sales']).replace([np.inf, -np.inf], 0) * 100
    
    # Calculate gross income if not present
    if 'gross_income' not in df.columns and 'profit' in df.columns:
        df['gross_income'] = df['profit']

    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Add datetime features
    if 'order_date' in df.columns:
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month_name()
        df['weekday'] = df['order_date'].dt.day_name()
        df['week'] = df['order_date'].dt.to_period('W').apply(lambda r: r.start_time)
        df['hour'] = df['order_date'].dt.hour
    
    return df

def full_pipeline(user_df):
    col_map = get_column_mapping(user_df, expected_columns)
    cleaned_df = auto_profile_and_clean(user_df, col_map)
    transformed_df = transform_data(cleaned_df)
    return transformed_df

# --------------------------
# 4. CUSTOMER METRICS FALLBACK LOGIC
# --------------------------
def get_customer_metrics(df):
    metrics = {}
    
    # Unique customers fallback logic
    if 'customer_id' in df.columns:
        metrics['unique_customers'] = df['customer_id'].nunique()
    elif 'order_id' in df.columns:
        metrics['unique_customers'] = df['order_id'].nunique()
        st.warning("Using invoice count as customer proxy (no customer ID found)")
    else:
        metrics['unique_customers'] = len(df)
        st.warning("Using transaction count as customer proxy (no customer/invoice ID found)")
    
    # Customer activity metrics
    if 'order_date' in df.columns:
        if 'customer_id' in df.columns:
            metrics['repeat_customers'] = df.groupby('customer_id').size()[lambda x: x > 1].count()
            metrics['new_customers'] = metrics['unique_customers'] - metrics['repeat_customers']
        elif 'order_id' in df.columns:
            weekly_customers = df.groupby('week')['order_id'].nunique()
            metrics['repeat_customers'] = (weekly_customers > 1).sum()
            metrics['new_customers'] = len(weekly_customers) - metrics['repeat_customers']
    
    return metrics

def get_order_metrics(df):
    metrics = {}
    
    # BASIC ORDER COUNTS
    if 'order_id' in df.columns:
        metrics['total_orders'] = df['order_id'].nunique()
    else:
        metrics['total_orders'] = len(df)
    
    # TIME-BASED PATTERNS
    if 'order_date' in df.columns:
        # Daily orders
        daily_orders = df.resample('D', on='order_date').size()
        metrics['avg_daily_orders'] = daily_orders.mean()
        
        # Weekly change
        weekly_orders = df.resample('W', on='order_date').size()
        if len(weekly_orders) > 1:
            metrics['weekly_change_pct'] = (weekly_orders.pct_change()*100).iloc[-1]
        
        # Monthly growth
        monthly_orders = df.resample('M', on='order_date').size()
        if len(monthly_orders) > 1:
            metrics['monthly_growth_pct'] = (monthly_orders.pct_change()*100).iloc[-1]
    
    return metrics


def show_predictions(df):
    st.header("🔮 Predictive Insights")
    
    # Cache data to prevent reprocessing
    @st.cache_data
    def get_clean_data(df):
        return st.session_state.predictor.preprocess(df)
    
    clean_df = get_clean_data(df)
    
    tab1, tab2 = st.tabs(["Forecast", "Anomalies"])
    
    with tab1:
        model_type = st.radio("Select model", 
                            ["lightgbm", "prophet"],
                            horizontal=True)
        
        if st.button("Run Forecast"):
            with st.spinner("Training..."):
                try:
                    model = st.session_state.predictor.train_forecaster(clean_df, model_type)
                    st.success(f"{model_type.upper()} model trained!")
                    
                    # Enhanced visualization
                    if model_type == "lightgbm":
                        future_dates = pd.date_range(clean_df['date'].max(), periods=7)
                        X_pred = pd.DataFrame({
                            'day_of_week': future_dates.dayofweek,
                            'month': future_dates.month
                        })
                        preds = model.predict(X_pred)
                        fig = px.line(x=future_dates, y=preds, 
                                     labels={'x': 'Date', 'y': 'Forecasted Value'},
                                     title="7-Day Sales Forecast")
                        st.plotly_chart(fig)
                    
                    elif model_type == "prophet":
                        future = model.make_future_dataframe(periods=7)
                        forecast = model.predict(future)
                        fig = px.line(forecast, x='ds', y='yhat',
                                     title="Prophet Forecast with Uncertainty")
                        fig.add_vline(x=pd.Timestamp.today(), line_dash="dash")
                        st.plotly_chart(fig)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        if st.button("Scan for Anomalies"):
            with st.spinner("Analyzing..."):
                anomalies = st.session_state.predictor.detect_anomalies(clean_df)
                anomaly_dates = clean_df[anomalies == -1]['date'].dt.strftime('%Y-%m-%d').unique()
                
                if len(anomaly_dates) > 0:
                    st.warning(f"🚨 Anomalies detected on: {', '.join(anomaly_dates)}")
                    # Show anomaly points on original data
                    fig = px.scatter(clean_df, x='date', y='value', 
                                    color=(anomalies == -1),
                                    title="Anomaly Detection")
                    st.plotly_chart(fig)
                else:
                    st.success("No anomalies detected!")

# --------------------------
# 5. DASHBOARD COMPONENTS
# --------------------------

def render_executive_dashboard(df):
    st.title("🏠 Executive Dashboard")
    
    # Get date range info
    min_date = df['order_date'].min() if 'order_date' in df.columns else None
    max_date = df['order_date'].max() if 'order_date' in df.columns else None
    date_range_str = f"({min_date.strftime('%Y-%m-%d') if min_date else 'N/A'} to {max_date.strftime('%Y-%m-%d') if max_date else 'N/A'})"

    # Get appropriate metrics based on available data
    if 'customer_id' in df.columns:
        metrics = get_customer_metrics(df)
        show_customer_metrics = True
    else:
        metrics = get_order_metrics(df)
        show_customer_metrics = False
    
    # Detect rating scale (5 or 10)
    max_rating = df['rating'].max() if 'rating' in df.columns else 5
    rating_scale = 10 if max_rating > 5 else 5
    
    # KPI Cards
    cols = st.columns(4)
    with cols[0]:
        sales = df['sales'].sum() if 'sales' in df.columns else 0
        st.metric("Total Sales", f"${sales:,.2f}", f"All time {date_range_str}")
    
    with cols[1]:
        profit = df['profit'].sum() if 'profit' in df.columns else 0
        st.metric("Total Profit", f"${profit:,.2f}", f"All time {date_range_str}")
    
    with cols[2]:
        if show_customer_metrics:
            st.metric("Unique Customers", 
                     f"{metrics.get('unique_customers', 0):,}",
                     f"All time {date_range_str}")
        else:
            if 'total_orders' in metrics:
                st.metric("Total Orders", 
                         f"{metrics['total_orders']:,}",
                         f"All time {date_range_str}")
            else:
                st.metric("Transactions", 
                         len(df),
                         f"All time {date_range_str}")
    
    with cols[3]:
        if 'rating' in df.columns:
            avg_rating = df['rating'].mean() 
            normalized_rating = (avg_rating/2) if rating_scale == 10 else avg_rating
            st.metric(f"Average Rating", 
                     f"{normalized_rating:.1f}/5",
                     f"Based on {len(df)} ratings")
        else:
            st.metric("Average Rating", "N/A")
    
    # Time-based metrics with clear time frames
    if 'order_date' in df.columns:
        st.subheader("Time-based Performance")
        time_cols = st.columns(3)
        
        with time_cols[0]:
            if 'avg_daily_orders' in metrics:
                st.metric("Avg Daily Orders", 
                         f"{metrics['avg_daily_orders']:,.1f}",
                         f"Across {len(df.resample('D', on='order_date'))} days")
        
        with time_cols[1]:
            if 'weekly_change_pct' in metrics:
                last_2_weeks = df.resample('W', on='order_date').size().iloc[-2:]
                week_str = f"Week {last_2_weeks.index[-2].week} to {last_2_weeks.index[-1].week}"
                st.metric("Weekly Change", 
                         f"{metrics['weekly_change_pct']:.1f}%",
                         f"{week_str}",
                         delta_color="inverse")
        
        with time_cols[2]:
            if 'monthly_growth_pct' in metrics:
                last_2_months = df.resample('M', on='order_date').size().iloc[-2:]
                month_str = f"{last_2_months.index[-2].strftime('%b')} to {last_2_months.index[-1].strftime('%b')}"
                st.metric("Monthly Growth", 
                         f"{metrics['monthly_growth_pct']:.1f}%",
                         f"{month_str}")
    
    # Main Visualizations
    col1, col2 = st.columns(2)
    with col1:
        if 'product' in df.columns and 'sales' in df.columns:
            top_products = df.groupby('product')['sales'].sum().nlargest(5).reset_index()
            fig = px.bar(top_products,
                         x='product', y='sales', 
                         title=f"Top Products by Revenue {date_range_str}",
                         color='sales',
                         color_continuous_scale='Bluered')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'order_date' in df.columns and 'sales' in df.columns:
            weekly_sales = df.resample('W', on='order_date')['sales'].sum().reset_index()
            num_weeks = len(weekly_sales)
            fig = px.line(weekly_sales,
                         x='order_date', y='sales', 
                         title=f"Weekly Sales Trend - Last {num_weeks} weeks {date_range_str}",
                         markers=True,
                         labels={'order_date': 'Week Starting'})
            fig.update_traces(hovertemplate="<b>Week %{x|%Y-%m-%d}</b><br>Sales: %{y:$,.0f}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Geographic Analysis
    if 'branch' in df.columns or 'city' in df.columns:
        st.subheader(f"Sales by Geographic Dimension {date_range_str}")
        
        geo_cols = []
        if 'branch' in df.columns:
            geo_cols.append('branch')
        if 'city' in df.columns:
            geo_cols.append('city')
        if 'region' in df.columns:
            geo_cols.append('region')
        if 'country' in df.columns:
            geo_cols.append('country')
        
        tabs = st.tabs([col.title() for col in geo_cols])
        
        for i, geo_col in enumerate(geo_cols):
            with tabs[i]:
                geo_sales = df.groupby(geo_col)['sales'].sum().reset_index()
                fig_bar = px.bar(geo_sales,
                                x=geo_col, y='sales',
                                title=f"Sales by {geo_col.title()} {date_range_str}",
                                color=geo_col)
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gender Analysis
    if 'gender' in df.columns and 'sales' in df.columns:
        st.subheader(f"Sales by Gender {date_range_str}")
        gender_sales = df.groupby('gender')['sales'].sum().reset_index()
        fig = px.pie(gender_sales,
                    values='sales', names='gender',
                    title=f"Sales Distribution by Gender {date_range_str}")
        st.plotly_chart(fig, use_container_width=True)

def render_sales_analytics(df):
    st.title("📈 Sales Analytics")
    
    # Time-based trends
    if 'month' in df.columns and 'sales' in df.columns:
        monthly_sales = df.groupby(['year', 'month'])['sales'].sum().reset_index()
        fig = px.line(monthly_sales,
                     x='month', y='sales', color='year',
                     title="Monthly Sales Trend (by Year)",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Product performance
    if 'product' in df.columns and 'quantity' in df.columns and 'sales' in df.columns:
        product_perf = df.groupby('product').agg({'quantity':'sum', 'sales':'sum'}).reset_index()
        product_perf['avg_price'] = product_perf['sales'] / product_perf['quantity']
        
        fig = px.scatter(product_perf,
                        x='quantity', y='sales', 
                        size='avg_price', hover_name='product',
                        color='avg_price',
                        title="Product Performance: Volume vs Revenue",
                        labels={'quantity': 'Units Sold', 'sales': 'Total Revenue'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment method analysis
    if 'payment' in df.columns and 'sales' in df.columns:
        payment_sales = df.groupby('payment')['sales'].sum().reset_index()
        fig = px.bar(payment_sales,
                    x='payment', y='sales',
                    title="Sales by Payment Method",
                    color='payment')
        st.plotly_chart(fig, use_container_width=True)

def render_inventory(df):
    st.title("📦 Product Performance")
    
    # Product quantity sold analysis (not stock level)
    if 'product' in df.columns and 'quantity' in df.columns:
        top_sellers = df.groupby('product')['quantity'].sum().nlargest(10).reset_index()
        fig = px.bar(top_sellers,
                    x='product', y='quantity', 
                    title="Top 10 Products by Units Sold",
                    color='quantity',
                    labels={'quantity': 'Units Purchased'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Profit margin analysis
    if 'product' in df.columns and 'profit_margin' in df.columns:
        profit_margins = df.groupby('product')['profit_margin'].mean().reset_index()
        fig = px.bar(profit_margins.nlargest(10, 'profit_margin'),
                    x='product', y='profit_margin',
                    title="Top Products by Profit Margin (%)",
                    labels={'profit_margin': 'Profit Margin %'})
        st.plotly_chart(fig, use_container_width=True)

def render_customer_insights(df):
    st.title("👥 Customer Insights")
    
    # Get customer metrics with fallback logic
    if 'customer_id' in df.columns:
        customer_metrics = get_customer_metrics(df)
        
        # Customer metrics cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unique Customers", customer_metrics.get('unique_customers', 0))
        with col2:
            st.metric("Repeat Customers", customer_metrics.get('repeat_customers', 'N/A'))
        with col3:
            st.metric("New Customers", customer_metrics.get('new_customers', 'N/A'))
    else:
        st.warning("No customer ID found - showing order patterns")
    
    # Customer segmentation (sales only)
    if 'customer_type' in df.columns and 'sales' in df.columns:
        customer_sales = df.groupby('customer_type')['sales'].sum().reset_index()
        
        fig = px.bar(customer_sales,
                    x='customer_type', y='sales',
                    title="Sales by Customer Type",
                    color='customer_type')
        st.plotly_chart(fig, use_container_width=True)
    
    # Rating analysis
    if 'rating' in df.columns:
        # Detect rating scale (5 or 10)
        max_rating = df['rating'].max()
        rating_scale = 10 if max_rating > 5 else 5
        
        st.subheader("Customer Rating Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            rating_dist = df['rating'].value_counts().reset_index()
            fig = px.pie(rating_dist,
                        values='count', names='rating',
                        title=f"Rating Distribution (Out of {rating_scale})")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average rating by customer type (if available)
            if 'customer_type' in df.columns:
                avg_rating = df.groupby('customer_type')['rating'].mean().reset_index()
                fig = px.bar(avg_rating,
                            x='customer_type', y='rating',
                            title=f"Average Rating by Customer Type (Out of {rating_scale})",
                            color='customer_type')
                st.plotly_chart(fig, use_container_width=True)
    
    # Order Patterns Analysis
    if 'order_date' in df.columns:
        st.subheader("Order Patterns Analysis")
        
        # Create tabs for different time views
        tab1, tab2, tab3 = st.tabs(["Daily", "Weekly", "Monthly"])
        
        with tab1:
            daily_orders = df.resample('D', on='order_date').size().reset_index(name='count')
            fig = px.line(daily_orders,
                         x='order_date', y='count',
                         title="Daily Order Volume",
                         labels={'count': 'Orders'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'order_id' in df.columns:
                weekly_orders = df.resample('W', on='order_date')['order_id'].nunique().reset_index(name='count')
            else:
                weekly_orders = df.resample('W', on='order_date').size().reset_index(name='count')
            
            fig = px.bar(weekly_orders,
                        x='order_date', y='count',
                        title="Weekly Order Count",
                        labels={'count': 'Orders'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            monthly_orders = df.resample('M', on='order_date').size().reset_index(name='count')
            fig = px.area(monthly_orders,
                         x='order_date', y='count',
                         title="Monthly Order Trend",
                         labels={'count': 'Orders'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Customer activity by time
    if 'hour' in df.columns:
        if 'customer_id' in df.columns:
            activity_data = df.groupby('hour')['customer_id'].nunique().reset_index(name='count')
            title = "Customer Activity by Hour"
            y_label = "Unique Customers"
        elif 'order_id' in df.columns:
            activity_data = df.groupby('hour')['order_id'].nunique().reset_index(name='count')
            title = "Order Activity by Hour"
            y_label = "Unique Orders"
        else:
            activity_data = df.groupby('hour').size().reset_index(name='count')
            title = "Transaction Activity by Hour"
            y_label = "Transactions"
        
        fig = px.line(activity_data,
                     x='hour', y='count',
                     title=title,
                     labels={'count': y_label})
        st.plotly_chart(fig, use_container_width=True)

def render_predictive_tab(df):
    st.title("🔮 Predictive Analytics")

    # Column selection
    with st.expander("🔍 Select Columns", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox(
                "Date Column",
                options=df.columns,
                index=df.columns.get_loc(
                    st.session_state.predictor._detect_column(df, ['date', 'time', 'day']) or 0
                )
            )
        with col2:
            value_col = st.selectbox(
                "Value Column",
                options=df.select_dtypes(include=['number']).columns,
                index=df.select_dtypes(include=['number']).columns.get_loc(
                    st.session_state.predictor._detect_column(df, ['sales', 'amount', 'revenue']) or 0
                )
            )

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio(
            "Forecasting Model",
            ["Random Forest (Fast)", "Prophet (Detailed)"],
            index=0
        )
    with col2:
        horizon = st.slider(
            "Forecast Horizon (Days)", 
            min_value=7, 
            max_value=30, 
            value=14
        )

    # Process data with selected columns
    processed_df = df.rename(columns={date_col: 'date', value_col: 'value'})

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Training model..."):
            try:
                model_name = "rf" if "Random Forest" in model_type else "prophet"
                forecast = st.session_state.predictor.get_forecast(
                    processed_df,
                    periods=horizon
                )

                # Show forecast as a table
                st.subheader("📈 Forecast Results")
                forecast_table = pd.DataFrame({
                    "Date": forecast.index.strftime('%Y-%m-%d (%a)'),
                    "Predicted Value": forecast.values.round(2)
                })
                st.write("### Forecast Table")
                st.dataframe(forecast_table, hide_index=True)

                # Highlight top forecasted day
                max_row = forecast_table.loc[forecast_table['Predicted Value'].idxmax()]
                st.success(f"📈 Highest predicted value on **{max_row['Date']}**: {max_row['Predicted Value']}")

                # Show anomalies with chart
                st.subheader("🚨 Anomaly Detection")
                anomalies = st.session_state.predictor.detect_anomalies(processed_df)
                anomaly_df = processed_df.copy()
                anomaly_df['is_anomaly'] = (anomalies == -1)

                if anomaly_df['is_anomaly'].any():
                    st.warning(f"Found {anomaly_df['is_anomaly'].sum()} anomalies")
                    fig = px.scatter(
                        anomaly_df,
                        x='date',
                        y='value',
                        color='is_anomaly',
                        color_discrete_map={True: "red", False: "blue"},
                        labels={"date": "Date", "value": "Value", "is_anomaly": "Anomaly"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(
                        anomaly_df[anomaly_df['is_anomaly']].sort_values('date', ascending=False),
                        hide_index=True
                    )
                    st.info(f"Anomalies are highlighted in red on the chart and table above.")
                else:
                    st.success("No anomalies detected")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.error("Please verify your column selections and data types")



# --------------------------
# 6. MAIN APP
# --------------------------
def main():
    # Data Loading
    st.sidebar.title("Data Source")
    data_source = st.sidebar.radio("", ["Sample Data", "Upload CSV", "SQL Database", "MongoDB", "REST API"])
    
    raw_df = pd.DataFrame()
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose CSV", type=["csv"])
        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    elif data_source == "Sample Data":
        raw_df = load_sample_data()
    elif data_source == "SQL Database":
        st.sidebar.subheader("SQL Connection")

        db_type = st.sidebar.selectbox("Database Type", ["MySQL", "PostgreSQL", "SQL Server", "Oracle"])
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        host = st.sidebar.text_input("Host", "localhost")
        port = st.sidebar.text_input("Port", "3306")
        database = st.sidebar.text_input("Database Name")
        query = st.sidebar.text_area("SQL Query", st.session_state.sql_query)

        # Update session state
        st.session_state.sql_query = query

        if st.sidebar.button("Connect"):
            try:
                if db_type == "MySQL":
                    conn_str = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
                elif db_type == "PostgreSQL":
                    conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                elif db_type == "SQL Server":
                    conn_str = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
                elif db_type == "Oracle":
                    conn_str = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={database}"
                else:
                    st.error("Unsupported database type")
                    conn_str = None

                if conn_str:
                    st.session_state.sql_conn_str = conn_str
                    st.session_state.raw_df = load_from_sql(conn_str, query)
                    st.success(f"✅ Loaded {len(st.session_state.raw_df)} records from SQL database")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")

        # ✅ Ensure raw_df is assigned outside button scope for rendering tabs
        if "raw_df" in st.session_state and not st.session_state.raw_df.empty:
            raw_df = st.session_state.raw_df

            

    elif data_source == "MongoDB":
        st.sidebar.subheader("MongoDB Connection")
        
        # Use session state variables with defaults
        conn_str = st.sidebar.text_input(
            "Connection String", 
            value=st.session_state.mongo_conn_str,
            key="mongo_conn_str_input"
        )
        db_name = st.sidebar.text_input(
            "Database Name", 
            value=st.session_state.mongo_db_name,
            key="mongo_db_name_input"
        )
        orders_collection = st.sidebar.text_input(
            "Orders Collection", 
            value=st.session_state.mongo_collection_name,
            key="mongo_orders_collection"
        )
        query = st.sidebar.text_area(
            "Query (JSON)", 
            value=st.session_state.mongo_query,
            key="mongo_query_input"
        )
        
        # Update params in session state
        st.session_state.mongo_conn_str = conn_str
        st.session_state.mongo_db_name = db_name
        st.session_state.mongo_collection_name = orders_collection
        st.session_state.mongo_query = query
        
        if st.sidebar.button("Connect"):
            try:
                # Use bson.json_util for MongoDB-specific parsing
                if query and query.strip():
                    try:
                        parsed_query = json_util.loads(query)
                    except Exception as e:
                        st.error(f"Invalid query format: {str(e)}")
                        parsed_query = {}
                else:
                    parsed_query = {}
                
                # Load data and store in session state
                st.session_state.mongo_df = load_from_mongo(
                    conn_str, 
                    db_name, 
                    orders_collection, 
                    parsed_query
                )
                st.success(f"Loaded {len(st.session_state.mongo_df)} documents")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        # ALWAYS use session state if available
        if not st.session_state.mongo_df.empty:
            raw_df = st.session_state.mongo_df
    elif data_source == "REST API":
        st.sidebar.subheader("API Connection")
        api_url = st.sidebar.text_input("API URL")
        params = st.sidebar.text_area("Query Parameters (JSON)", "{}")
        
        if st.sidebar.button("Fetch Data"):
            try:
                params = eval(params) if params else {}
                raw_df = load_from_api(api_url, params)
            except Exception as e:
                st.error(f"Invalid parameters format: {str(e)}")
    
    # Data Processing
    if not raw_df.empty:
        df = full_pipeline(raw_df)
        
        # Show raw data preview
        with st.expander("Show Raw Data"):
            st.dataframe(raw_df.head())
        
        # Show processed data preview
        with st.expander("Show Processed Data"):
            st.dataframe(df.head())
        
        # Dashboard Selection
        menu_options = [
            "🏠 Executive Dashboard",
            "📈 Sales Analytics", 
            "📦 Product Performance",
            "👥 Customer Insights",
            "🔮 Predictive Analytics",
            "💬 AI Assistant" 
        ]
        
        with st.sidebar:
            st.title("Navigation")
            selected = option_menu(
                menu_title=None,
                options=menu_options,
                default_index=0
            )
        
        # Dashboard Rendering
        if selected == "🏠 Executive Dashboard":
            render_executive_dashboard(df)
        elif selected == "📈 Sales Analytics":
            render_sales_analytics(df)
        elif selected == "📦 Product Performance":
            render_inventory(df)
        elif selected == "👥 Customer Insights":
            render_customer_insights(df)
        elif selected == "🔮 Predictive Analytics":
            render_predictive_tab(df)
        elif selected == "💬 AI Assistant":  
            class DataProcessor:
                def __init__(self):
                    pass
                    
                def full_pipeline(self, df):
                    return full_pipeline(df)
                    
                def get_customer_metrics(self, df):
                    return get_customer_metrics(df)
                    
                def get_order_metrics(self, df):
                    return get_order_metrics(df)
    
            render_chat_tab(
                df=df,
                data_processor=DataProcessor(),
                predictor=st.session_state.predictor
            )
    else:
        st.warning("Please load data to begin analysis")

    # Debug test (indented at main() level, not inside else)
    if st.sidebar.checkbox("🛠️ Run predictor test", help="Debug the predictive engine"):
        st.write("## 🧪 Predictor Test")
        with st.spinner("Testing..."):
            try:
                predictor = EcomPredictor()
                df_test = pd.DataFrame({
                    'order_date': pd.date_range('2023-01-01', periods=100),
                    'sales': np.random.randint(100, 1000, 100)
                })
                model = predictor.train_forecaster(df_test)
                st.success(f"Test passed! Model: {type(model).__name__}")
                st.balloons()
            except Exception as e:
                st.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()