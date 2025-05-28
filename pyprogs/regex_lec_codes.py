import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "plotly_white"

df = pd.read_csv("D:\\enosh_regex\\Datasets\\supply_chain.csv")

#print(df.describe())
fig=px.scatter(df,x='Price', y='Revenue generated', color='Product type', hover_data=['Number of products sold'], trendline = 'ols')

'''
fig = px.scatter(
    df,
    x='Price',  # Case-sensitive column name
    y='Revenue generated',
    color='Product type',
    hover_data=['Number of products sold'],
    trendline='ols'
)
fig.show()

sales_data = df.groupby('Product type')['Number of products sold'].sum().reset_index()

pie_chart = px.pie(
    sales_data,
    values='Number of products sold',
    names='Product type',
    title='Sales by Product Type',
    hole=0.5,
    color_discrete_sequence=px.colors.qualitative.Pastel
)

pie_chart.update_traces(
    textposition='inside',
    textinfo='percent+label'  
)

pie_chart.show()
total_revenue = df.groupby('Shipping carriers')['Revenue generated'].sum().reset_index()
fix=go.figure()
fig.add_trace(go.Bar(x=total_revenue['Shipping carriers'],
                     y=total_revenue['Revenue generated']))
fig.update_layout(title='Total Revenue by Shipping carriers',
                  xaxis_title='Shipping carriers',
                  yaxis_title='Revenue generated')
fig.show()
'''
avg_lead_time=df.groupby('Product type')['Lead time'].mean().reset.index

avg_manufacturing_costs = df.groupby('Product type ')['Manufacturing costs'].mean().reset_index()

result = pd.merge(avg_lead_time, avg_manufacturing_costs, on='Product type')

result.rename(columns={'Lead time': 'Average Lead Time', 'Manufacturing costs' : 'Average Manufacturing Costs'}, inplace=True)

print(result)

