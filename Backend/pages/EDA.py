import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from statistics import mean
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from textwrap import wrap
from numerize import numerize

from MyUtils.HideStDefaults import hideNavBar
from MyUtils.Metrics import displayMetrics
from MyUtils.searchAndSelectFile import selectDataset

hideNavBar()
st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(216, 198, 188, 1);
   border: 2px solid rgba(32, 48, 84, 1);
   border-radius: 5px;
}
</style>
"""
, unsafe_allow_html=True)

df = selectDataset()

#st.title("Sales Performance Dashboard")
#displayMetrics(df)

#taking the input from the user
df['Year'] = df['Year'].astype(str)
df[['Sales','Quantity','Rating']] = df[['Sales','Quantity','Rating']].apply(pd.to_numeric)
col1,col2,col3,col4 = st.columns(4,gap="small")
with col1:
    input_year = st.multiselect(label="Choose Year", options=df["Year"].unique())
with col2:
    input_Quarter = st.multiselect(label="Choose Quarter", options=df["Quarter"].unique())
with col3:
    input_month = st.multiselect(label="Choose Month", options=df["Month"].unique())
with col4:
    input_state = st.multiselect(label="Choose State", options=df["State"].unique())
if not input_year:
  input_year=df['Year'].unique()
if not input_Quarter:
  input_Quarter=df['Quarter'].unique()
if not input_month:
  input_month=df['Month'].unique()
if not input_state:
  input_state=df['State'].unique()
filtered_df = df[ df['Year'].isin(input_year) &  df['Quarter'].isin(input_Quarter) & df['Month'].isin(input_month) & df['State'].isin(input_state)] 
#kpi df
df_kpi=filtered_df.groupby(['Year','Quarter','Month']).agg({'Gross Sales':sum,'Sales':mean,'Profit':sum,'Rating':mean}).reset_index()
df_kpi.rename(columns={'Gross Sales':'Total Sales','Sales':'Average Monthly Revenue per Customer',},inplace=True)
sales_kpi=numerize.numerize(float(df_kpi['Total Sales'].sum()),2)
AVMR_kpi=round( df_kpi['Average Monthly Revenue per Customer'].mean() , 2 )
profit_kpi= numerize.numerize(float(df_kpi['Profit'].sum()),2)
Rating_kpi=round(df_kpi['Rating'].mean(), 2 )
col4, col5, col6, col7 = st.columns(4,gap="small")

with col4:
    st.metric(
        label="Sales",
        value=sales_kpi,
        
    )

with col5:
    st.metric(
        label="Average Monthly Revenue per Customer",
        value=AVMR_kpi,
        
    )

with col6:
    st.metric(
        label="Profit",
        value=profit_kpi,
        
    )

with col7:
    st.metric(
        label="Rating",
        value=Rating_kpi,
        
    )

col8, col9 = st.columns(2,gap="small")


with col8:
    filtered_df_product=filtered_df.groupby('Category').agg({'Sales':sum}).reset_index()

    fig_filtered_df_product = px.pie(filtered_df_product,
                        names='Category', values="Sales",
                        template="plotly_white", title="Product Category Distribution",width=600, height=400,color_discrete_sequence=px.colors.sequential.RdBu,hole=0.4)
    fig_filtered_df_product.update_layout(title_x=0.3)
    st.plotly_chart(fig_filtered_df_product)
    
#Region Sales

with col9:
    filtered_df_subcategory=filtered_df.groupby('Sub Category').agg({'Quantity':sum}).reset_index()
    fig_filtered_df_subcategory = px.bar(filtered_df_subcategory,
                        y="Sub Category", x="Quantity",
                        template="plotly_white", title="Order Distribution across Sub-Categories ",width=600, height=400,color_discrete_sequence=['#B2182B'],orientation='h')
    fig_filtered_df_subcategory.update_layout(xaxis={'categoryorder': 'total ascending'})
    fig_filtered_df_subcategory .update_layout(title_x=0.3,xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
    st.plotly_chart(fig_filtered_df_subcategory)
    #rating
col10, col11 = st.columns(2,gap="small")
with col10:
    filtered_df_rating=filtered_df.groupby('Category').agg({'Rating':mean}).reset_index()
    fig_filtered_df_rating = px.bar(filtered_df_rating,
                        x="Category", y="Rating",
                        template="plotly_white", title="Rating distribution across Categories",width=600, height=400,color_discrete_sequence=['#B2182B'])
    fig_filtered_df_rating .update_layout(title_x=0.3,xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    fig_filtered_df_rating.update_yaxes(range=[3.5, 5])  
    st.plotly_chart(fig_filtered_df_rating)

with col11:
    filtered_df_weekly_sales=filtered_df.groupby('Week number').agg({'Sales':sum}).reset_index()
    fig_weekly_sales = px.line(filtered_df_weekly_sales,
                        x="Week number", y="Sales",
                        template="simple_white", title="Weekly Total Sales",labels={"Week number":"Week","Sales":"Sales"},width=600, height=400,color_discrete_sequence=['#B2182B'])
    fig_weekly_sales.update_layout(title_x=0.4,xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
    st.plotly_chart(fig_weekly_sales)
