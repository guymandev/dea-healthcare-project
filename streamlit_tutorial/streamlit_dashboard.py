import streamlit as st
import pandas as pd
import plotly.express as px
 
st.set_page_config(page_title="Sales Dashboard", layout="wide")
 
# Sample dataset (replace with real sales data)
@st.cache_data
def load_data():
    url = " https://raw.githubusercontent.com/yannie28/Global-Superstore/master/Global_Superstore(CSV).csv"
    return pd.read_csv(url)
 
df = load_data()
 
# Sidebar filters
st.sidebar.header("Filters")
year = st.sidebar.selectbox("Select Year", sorted(df['Order Date'].str[:4].unique()))
region = st.sidebar.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
 
# Filter data
filtered_df = df[(df['Order Date'].str[:4] == year) & (df['Region'].isin(region))]
 
# KPIs
total_sales = int(filtered_df["Sales"].sum())
total_profit = int(filtered_df["Profit"].sum())
total_customers = filtered_df["Customer ID"].nunique()
 
st.title("Sales Dashboard")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Sales", f"${total_sales:,}")
kpi2.metric("Total Profit", f"${total_profit:,}")
kpi3.metric("Unique Customers", total_customers)
 
st.markdown("---")
 
# Charts
col1, col2 = st.columns([2, 1])
 
with col1:
    sales_trend = filtered_df.groupby(filtered_df["Order Date"].str[:7])["Sales"].sum().reset_index()
    fig = px.line(sales_trend, x="Order Date", y="Sales", title="Monthly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)
 
with col2:
    top_products = filtered_df.groupby("Product Name")["Sales"].sum().nlargest(5).reset_index()
    fig = px.bar(top_products, x="Sales", y="Product Name", orientation="h", title="Top 5 Products")
    st.plotly_chart(fig, use_container_width=True)
