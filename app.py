import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
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
import time

# Initialize predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = EcomPredictor()

# Initialize all session state variables at the start
if 'selected_kpis' not in st.session_state:
    st.session_state.selected_kpis = {
        "sales": True,
        "profit": True,
        "costs": False,
        "customers": True,
        "products": True,
        "inventory": False
    }

# Initialize session state variables
if 'user_column_map' not in st.session_state:
    st.session_state.user_column_map = {}
if 'fallback_column_map' not in st.session_state:
    st.session_state.fallback_column_map = {
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

if 'column_mapping_complete' not in st.session_state:
    st.session_state.column_mapping_complete = False

# Initialize other session state variables
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
if 'sql_conn_str' not in st.session_state:
    st.session_state.sql_conn_str = ""
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = "SELECT * FROM sales_data"
if 'sql_df' not in st.session_state:
    st.session_state.sql_df = pd.DataFrame()

# Page configuration
st.set_page_config(layout="wide", page_title="Data Genie Pro", page_icon="🧠")
SAMPLE_CSV_PATH = r"C:\Users\Aman ur Rehman\Desktop\Data_genie\SuperMarket Analysis.csv"

# --------------------------
# 1. DATA LOADING FUNCTIONS
# --------------------------
@st.cache_data
def load_sample_data():
    try:
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
        if not df.empty:
            df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"SQL Error: {str(e)}")
        return pd.DataFrame()

# remove later

    st.write("🧪 SQL DataFrame Preview:", raw_df.head())
    st.write("🧪 SQL DataFrame dtypes:", raw_df.dtypes)

def load_from_mongo(connection_string, db_name, collection_name, query={}):
    try:
        client = pymongo.MongoClient(connection_string)
        db = client[db_name]
        collection = db[collection_name]
        
        if isinstance(query, list):
            cursor = collection.aggregate(query)
        elif isinstance(query, dict):
            cursor = collection.find(query)
        else:
            st.error("Invalid query type")
            return pd.DataFrame()
            
        data = list(cursor)
        client.close()
        
        json_data = json.loads(json_util.dumps(data))
        return pd.DataFrame(json_data)
    except Exception as e:
        st.error(f"MongoDB Error: {str(e)}")
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
# 2. COLUMN MAPPING INTERFACE
# --------------------------
def render_column_mapping_interface(df):
    st.header("📝 Assign Column Meanings")
    st.info("Describe what each column represents. Leave blank to use auto-detection (if available).")

    user_map = st.session_state.get('user_column_map', {})
    fallback_map = st.session_state.get('fallback_column_map', {})

    cols = st.columns(3)
    col_idx = 0

    for i, column in enumerate(df.columns):
        with cols[col_idx]:
            # Sample value
            sample = str(df[column].iloc[0]) if len(df) > 0 else "N/A"
            sample = sample[:20] + "..." if len(sample) > 20 else sample

            # Detect fallback meaning
            fallback = None
            for meaning, aliases in fallback_map.items():
                if column.lower() in [alias.lower() for alias in aliases]:
                    fallback = meaning
                    break

            # Render input box with fallback suggestion as default
            user_input = st.text_input(
                f"**{column}** (Sample: {sample})",
                value=user_map.get(column, fallback if fallback else ""),
                key=f"col_input_{i}"
            )

            # Save either user input or fallback
            if user_input.strip():
                user_map[column] = user_input.strip().lower()
            elif fallback:
                user_map[column] = fallback.lower()

        col_idx = (col_idx + 1) % 3

    # Save Mapping Button
    if st.button("✅ Save Mapping", type="primary"):
        if any(user_map.values()):
            # Update fallback map
            for col, meaning in user_map.items():
                if meaning in fallback_map:
                    if col not in fallback_map[meaning]:
                        fallback_map[meaning].append(col)
                else:
                    fallback_map[meaning] = [col]

            st.session_state.user_column_map = {
                k: v for k, v in user_map.items() if v
            }
            st.session_state.fallback_column_map = fallback_map
            st.session_state.column_mapping_complete = True

            st.success("Mapping saved successfully!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Please provide at least one column mapping.")

    # Reset Button
    if st.button("🔄 Reset Mappings"):
        st.session_state.user_column_map = {}
        st.session_state.column_mapping_complete = False
        st.rerun()
# --------------------------
# 3. DATA PROCESSING PIPELINE
# --------------------------
def auto_profile_and_clean(df):
    # ✅ Step 0: Deduplicate column names
    if df.columns.duplicated().any():
        counts = {}
        new_cols = []
        for col in df.columns:
            if col not in counts:
                counts[col] = 0
                new_cols.append(col)
            else:
                counts[col] += 1
                new_cols.append(f"{col}_{counts[col]}")
        df.columns = new_cols

    # ✅ Step 1: Convert dict/list columns to strings
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].astype(str)

    # ✅ Step 2: Drop duplicates
    df = df.drop_duplicates()

    # ✅ Step 3: Convert date/time columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ✅ Step 4: Convert numeric columns
    numeric_cols = ['quantity', 'unit_price', 'sales', 'cost', 'rating', 'profit', 'amount', 'total']
    for col in df.columns:
        if any(n in col.lower() for n in numeric_cols):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ✅ Step 5: Fill missing values
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)

    return df


def transform_data(df):
    try:
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

        # Handle date processing with error checking
        if 'order_date' in df.columns:
            try:
                df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
                
                # Only proceed if we have valid dates
                if not df['order_date'].isnull().all():
                    df['year'] = df['order_date'].dt.year
                    df['month'] = df['order_date'].dt.month_name()
                    df['weekday'] = df['order_date'].dt.day_name()
                    
                    # Only add week if we have enough data points
                    if len(df) >= 7:  # At least a week's worth of data
                        df['week'] = df['order_date'].dt.to_period('W').apply(lambda r: r.start_time)
                        
                    df['hour'] = df['order_date'].dt.hour
                else:
                    st.warning("Date column contains no valid dates")
            except Exception as e:
                st.warning(f"Could not process dates: {str(e)}")
    
    except Exception as e:
        st.error(f"Error transforming data: {str(e)}")
        # Return the partially transformed dataframe
        return df
    
    return df

def full_pipeline(user_df, column_map=None):
    final_map = {}

    # Step 1: Use column_map if passed
    if column_map:
        for col, meaning in column_map.items():
            if col in user_df.columns:
                final_map[col] = meaning.replace(" ", "_")

    # Step 2: Apply fallback for unmapped columns
    for col in user_df.columns:
        if col not in final_map:
            for standard_name, aliases in st.session_state.fallback_column_map.items():
                if col.lower() in [a.lower() for a in aliases]:
                    final_map[col] = standard_name
                    break

    # Step 3: Apply renaming
    if final_map:
        user_df = user_df.rename(columns=final_map)

    

    # Step 4: Clean and transform safely
    try:
        cleaned_df = auto_profile_and_clean(user_df)
        transformed_df = transform_data(cleaned_df)

        # ✅ Debug: Show final shape
        st.write("✅ Final DF shape:", transformed_df.shape)
        return transformed_df

    except Exception as e:
        st.error(f" Error in processing pipeline: {e}")
        return pd.DataFrame()



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


# --------------------------
# 4. DASHBOARD COMPONENTS
# --------------------------
def render_executive_dashboard(df):
    st.title("🏠 Executive Dashboard")
    
    # Dynamic column detection
    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
    customer_col = next((col for col in df.columns if 'customer' in col.lower()), None)
    order_col = next((col for col in df.columns if 'order' in col.lower() or 'invoice' in col.lower()), None)
    
    # Get date range info
    date_range_str = ""
    if date_col:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        date_range_str = f"({min_date.strftime('%Y-%m-%d') if pd.notnull(min_date) else 'N/A'} to {max_date.strftime('%Y-%m-%d') if pd.notnull(max_date) else 'N/A'})"

    # Get appropriate metrics
    if customer_col:
        metrics = get_customer_metrics(df)
        show_customer_metrics = True
    else:
        metrics = get_order_metrics(df)
        show_customer_metrics = False
    
    # Detect rating scale
    rating_col = next((col for col in df.columns if 'rating' in col.lower()), None)
    rating_scale = 5
    if rating_col:
        max_rating = df[rating_col].max()
        rating_scale = 10 if max_rating > 5 else 5
    
    # KPI Cards
    cols = st.columns(4)
    with cols[0]:
        sales_col = next((col for col in df.columns if 'sales' in col.lower() or 'revenue' in col.lower()), None)
        sales = df[sales_col].sum() if sales_col else 0
        st.metric("Total Sales", f"${sales:,.2f}", f"All time {date_range_str}")
    
    with cols[1]:
        profit_col = next((col for col in df.columns if 'profit' in col.lower()), None)
        profit = df[profit_col].sum() if profit_col else 0
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
        if rating_col:
            avg_rating = df[rating_col].mean() 
            normalized_rating = (avg_rating/2) if rating_scale == 10 else avg_rating
            st.metric(f"Average Rating", 
                     f"{normalized_rating:.1f}/5",
                     f"Based on {len(df)} ratings")
        else:
            st.metric("Average Rating", "N/A")
    
    # Time-based metrics
    if date_col:
        st.subheader("Time-based Performance")
        time_cols = st.columns(3)
        
        with time_cols[0]:
            if 'avg_daily_orders' in metrics:
                days = len(df.resample('D', on=date_col))
                st.metric("Avg Daily Orders", 
                         f"{metrics['avg_daily_orders']:,.1f}",
                         f"Across {days} days")
        
        with time_cols[1]:
            if 'weekly_change_pct' in metrics:
                weekly_data = df.resample('W', on=date_col).size()
                if len(weekly_data) > 1:
                    week_str = f"Week {weekly_data.index[-2].week} to {weekly_data.index[-1].week}"
                    st.metric("Weekly Change", 
                             f"{metrics['weekly_change_pct']:.1f}%",
                             f"{week_str}",
                             delta_color="inverse")
        
        with time_cols[2]:
            if 'monthly_growth_pct' in metrics:
                monthly_data = df.resample('M', on=date_col).size()
                if len(monthly_data) > 1:
                    month_str = f"{monthly_data.index[-2].strftime('%b')} to {monthly_data.index[-1].strftime('%b')}"
                    st.metric("Monthly Growth", 
                             f"{metrics['monthly_growth_pct']:.1f}%",
                             f"{month_str}")
    
    # Main Visualizations
    col1, col2 = st.columns(2)
    with col1:
        product_col = next((col for col in df.columns if 'product' in col.lower() or 'item' in col.lower()), None)
        if product_col and sales_col:
            top_products = df.groupby(product_col)[sales_col].sum().nlargest(5).reset_index()
            fig = px.bar(top_products,
                         x=product_col, y=sales_col, 
                         title=f"Top Products by Revenue {date_range_str}",
                         color=sales_col,
                         color_continuous_scale='Bluered')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if date_col and sales_col:
            weekly_sales = df.resample('W', on=date_col)[sales_col].sum().reset_index()
            num_weeks = len(weekly_sales)
            fig = px.line(weekly_sales,
                         x=date_col, y=sales_col, 
                         title=f"Weekly Sales Trend - Last {num_weeks} weeks {date_range_str}",
                         markers=True,
                         labels={date_col: 'Week Starting'})
            fig.update_traces(hovertemplate=f"<b>Week %{{x|%Y-%m-%d}}</b><br>Sales: %{{y:$,.0f}}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Geographic Analysis
    geo_cols = []
    for geo_type in ['branch', 'city', 'region', 'country']:
        if geo_type in df.columns:
            geo_cols.append(geo_type)
    
    if geo_cols and sales_col:
        st.subheader(f"Sales by Geographic Dimension {date_range_str}")
        tabs = st.tabs([col.title() for col in geo_cols])
        
        for i, geo_col in enumerate(geo_cols):
            with tabs[i]:
                geo_sales = df.groupby(geo_col)[sales_col].sum().reset_index()
                fig_bar = px.bar(geo_sales,
                                x=geo_col, y=sales_col,
                                title=f"Sales by {geo_col.title()} {date_range_str}",
                                color=geo_col)
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gender Analysis
    gender_col = next((col for col in df.columns if 'gender' in col.lower() or 'sex' in col.lower()), None)
    if gender_col and sales_col:
        st.subheader(f"Sales by Gender {date_range_str}")
        gender_sales = df.groupby(gender_col)[sales_col].sum().reset_index()
        fig = px.pie(gender_sales,
                    values=sales_col, names=gender_col,
                    title=f"Sales Distribution by Gender {date_range_str}")
        st.plotly_chart(fig, use_container_width=True)


def render_sales_analytics(df):
    st.title("📈 Sales Analytics")
    
    # Time-based trends
    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
    
    if date_col and 'month' in df.columns and 'sales' in df.columns:
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
    
    # Product quantity sold analysis
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
    
    # Customer metrics cards
    cols = st.columns(3)
    with cols[0]:
        if 'customer_id' in df.columns:
            st.metric("Unique Customers", df['customer_id'].nunique())
    
    with cols[1]:
        if 'customer_id' in df.columns:
            repeat_customers = df.groupby('customer_id').size()[lambda x: x > 1].count()
            st.metric("Repeat Customers", repeat_customers)
    
    with cols[2]:
        if 'customer_id' in df.columns:
            st.metric("New Customers", df['customer_id'].nunique() - repeat_customers)
    
    # Customer segmentation
    if 'customer_type' in df.columns and 'sales' in df.columns:
        customer_sales = df.groupby('customer_type')['sales'].sum().reset_index()
        fig = px.bar(customer_sales,
                    x='customer_type', y='sales',
                    title="Sales by Customer Type",
                    color='customer_type')
        st.plotly_chart(fig, use_container_width=True)
    
    # Rating analysis
    if 'rating' in df.columns:
        st.subheader("Customer Rating Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            rating_dist = df['rating'].value_counts().reset_index()
            fig = px.pie(rating_dist,
                        values='count', names='rating',
                        title="Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'customer_type' in df.columns:
                avg_rating = df.groupby('customer_type')['rating'].mean().reset_index()
                fig = px.bar(avg_rating,
                            x='customer_type', y='rating',
                            title="Average Rating by Customer Type")
                st.plotly_chart(fig, use_container_width=True)

def render_predictive_tab(df):
    st.title("🔮 Predictive Analytics")

    # Column selection
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    with st.expander("🔍 Select Columns", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox(
                "Date Column",
                options=date_cols if date_cols else df.columns,
                index=0
            )
        with col2:
            value_col = st.selectbox(
                "Value Column",
                options=numeric_cols if numeric_cols else df.columns,
                index=0
            )

    # Model configuration
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

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                processed_df = df[[date_col, value_col]].copy()
                processed_df.columns = ['date', 'value']
                
                # Generate forecast
                model_name = "rf" if "Random Forest" in model_type else "prophet"
                forecast = st.session_state.predictor.get_forecast(
                    processed_df,
                    periods=horizon
                )

                # Show results
                st.subheader("📈 Forecast Results")
                forecast_table = pd.DataFrame({
                    "Date": forecast.index.strftime('%Y-%m-%d (%a)'),
                    "Predicted Value": forecast.values.round(2)
                })
                st.dataframe(forecast_table, hide_index=True)

                # Highlight best day
                max_row = forecast_table.loc[forecast_table['Predicted Value'].idxmax()]
                st.success(f"📈 Highest predicted value on **{max_row['Date']}**: {max_row['Predicted Value']}")

                # Anomaly detection
                st.subheader("🚨 Anomaly Detection")
                anomalies = st.session_state.predictor.detect_anomalies(processed_df)
                if anomalies is not None:
                    anomaly_df = processed_df.copy()
                    anomaly_df['is_anomaly'] = (anomalies == -1)
                    
                    if anomaly_df['is_anomaly'].any():
                        st.warning(f"Found {anomaly_df['is_anomaly'].sum()} anomalies")
                        fig = px.scatter(
                            anomaly_df,
                            x='date',
                            y='value',
                            color='is_anomaly',
                            color_discrete_map={True: "red", False: "blue"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("No anomalies detected")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# --------------------------
# 5. MAIN APP
# --------------------------
def main():
    st.sidebar.title("Data Source")
    data_source = st.sidebar.radio("", ["Sample Data", "Upload CSV", "SQL Database", "MongoDB", "REST API"])
    
    raw_df = pd.DataFrame()

    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose CSV", type=["csv"])

        if uploaded_file:
            uploaded_filename = uploaded_file.name

            # Detect file change → reset mapping
            if st.session_state.get("last_uploaded_file") != uploaded_filename:
                st.session_state.column_mapping_complete = False
                st.session_state.last_uploaded_file = uploaded_filename
                st.session_state.raw_df = pd.read_csv(uploaded_file)

            raw_df = st.session_state.raw_df

            # Show mapping interface if not completed
            if not st.session_state.column_mapping_complete:
                st.header("Data Understanding")
                st.dataframe(raw_df.head(3))
                render_column_mapping_interface(raw_df)
                st.stop()  # Pause execution until mapping is complete

            
            
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
                
                if conn_str:
                    st.session_state.sql_conn_str = conn_str
                    st.session_state.raw_df = load_from_sql(conn_str, query)
                    st.success(f"✅ Loaded {len(st.session_state.raw_df)} records from SQL database")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")

        if "raw_df" in st.session_state and not st.session_state.raw_df.empty:
            raw_df = st.session_state.raw_df
            
    elif data_source == "MongoDB":
        st.sidebar.subheader("MongoDB Connection")
        
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
        
        st.session_state.mongo_conn_str = conn_str
        st.session_state.mongo_db_name = db_name
        st.session_state.mongo_collection_name = orders_collection
        st.session_state.mongo_query = query
        
        if st.sidebar.button("Connect"):
            try:
                if query and query.strip():
                    try:
                        parsed_query = json_util.loads(query)
                    except Exception as e:
                        st.error(f"Invalid query format: {str(e)}")
                        parsed_query = {}
                else:
                    parsed_query = {}
                
                st.session_state.mongo_df = load_from_mongo(
                    conn_str, 
                    db_name, 
                    orders_collection, 
                    parsed_query
                )
                st.success(f"Loaded {len(st.session_state.mongo_df)} documents")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
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
    
    if not raw_df.empty:
        # Show mapping interface if not completed
        if not st.session_state.column_mapping_complete:
            st.header("Data Understanding")
            st.dataframe(raw_df.head(3))
            render_column_mapping_interface(raw_df)
            st.stop()  # Pause until mapping is complete


        
        # Process data with user mapping
        try:
            


            df = full_pipeline(raw_df, column_map=st.session_state.user_column_map)

            # ✅ 
            if "selected_kpis" not in st.session_state:
                st.session_state.selected_kpis = {
                    "sales": True,
                    "products": True,
                    "customers": True
                }


            
            with st.expander("Show Processed Data", expanded=False):
                st.dataframe(df.head())
                
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
                
                # KEEP THIS CONFIGURATION SECTION
                with st.expander("⚙️ Configuration", expanded=False):
                    if st.button("Modify Column Mapping"):
                        st.session_state.column_mapping_complete = False
                        st.rerun()
                    
                    st.write("**Active Analysis:**")
                    for kpi, enabled in st.session_state.selected_kpis.items():
                        if enabled:
                            st.write(f"- {kpi.capitalize()}")

            # Dashboard Rendering
            try:
                if selected == "🏠 Executive Dashboard":
                    render_executive_dashboard(df)
                    
                elif selected == "📈 Sales Analytics":
                    if st.session_state.selected_kpis.get("sales", False):
                        render_sales_analytics(df)
                    else:
                        st.warning("Enable 'Sales Analysis' in configuration to view this dashboard")
                        
                elif selected == "📦 Product Performance":
                    if st.session_state.selected_kpis.get("products", False):
                        render_inventory(df)
                    else:
                        st.warning("Enable 'Product Performance' in configuration to view this dashboard")
                        
                elif selected == "👥 Customer Insights":
                    if st.session_state.selected_kpis.get("customers", False):
                        render_customer_insights(df)
                    else:
                        st.warning("Enable 'Customer Behavior' in configuration to view this dashboard")
                        
                elif selected == "🔮 Predictive Analytics":
                    render_predictive_tab(df)  # Always available
                    
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
                    
            except Exception as e:
                st.error(f"Error rendering dashboard: {str(e)}")
                st.error("Please check your data and column mappings")
                if st.button("Reset View"):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Data processing failed: {str(e)}")
            if st.button("Reset Column Mapping"):
                st.session_state.column_mapping_complete = False
                st.rerun()
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