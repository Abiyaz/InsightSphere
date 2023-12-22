import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import os
st.title("Data Analysis with InsightSphere! :coffee:")
if st.button("Refresh"):
    st.rerun()
tab1, tab2 = st.tabs(["Health", "Sales"])
with tab1:
    st.header("Heart Failure Analysis")
    df = pd.read_csv("./Resources/heart_failure_clinical_records_dataset.csv")
    option = st.selectbox(
    'Select an event',
    ('diabetes', 'smoking', 'anaemia', 'high_blood_pressure'))
    disease_count=df[option].value_counts()
    values = ['-Ve', '+Ve']
    fig = px.pie(disease_count, values='count', title=f'Distribution of {option}',names=values, height=500, width=600)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
    st.plotly_chart(fig, use_container_width=True)
    theage_most_haveanamia=df['age'][df[option]==1].value_counts()
    # st.header('Diabetes count across age')
    # st.bar_chart(theage_most_haveanamia, y='count',use_container_width=True)
    fig = px.bar(theage_most_haveanamia, title=f'{option} count across age')
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
    st.plotly_chart(fig, user_container_width=True)
    fig = px.imshow(df.corr(),title="Correlation Plot of the Heat Failure Prediction")
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
    st.plotly_chart(fig)
    fig = px.box(df , y = 'age' , x = 'DEATH_EVENT' , title='Distribution of age')
    st.plotly_chart(fig)


with tab2:
    st.header("Sales Analysis")
    df = pd.read_csv("./Resources/train.csv")
    df.drop_duplicates(inplace =True)
    city_sales = df.groupby('City')['Sales'].sum().round().reset_index().sort_values('Sales', ascending=False).head(10)
    fig = px.bar(city_sales, title="Top 10 Cities Sales", x="City", y="Sales")
    st.plotly_chart(fig)
    # Sales Per each Category
    Sales_by_category = df.groupby('Category')['Sales'].sum().reset_index()

    # Sort Values by Sub-Category
    Sales_by_category = Sales_by_category.sort_values(by='Sales', ascending=False)
    fig = px.pie(Sales_by_category, values='Sales', title='Sales by category',names='Category', height=500, width=600)
    st.plotly_chart(fig)
    sub_category_count = df.groupby('Sub-Category')['Sales'].sum().reset_index()

    # Sort Values by Sub-Category
    top_sub_category_count = sub_category_count.sort_values(by='Sales', ascending=True)
    fig = px.bar(top_sub_category_count, x='Sales', y='Sub-Category', title='Top sub categories', orientation='h')
    st.plotly_chart(fig)


    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

    yearly_sales = df.groupby(df['Order Date'].dt.year)['Sales'].sum().reset_index()

    yearly_sales = yearly_sales.rename(columns={'Order Date':"Year", "Date":"Total Sales"})
    fig = px.bar(yearly_sales, x='Year', y='Sales', title='Total sales per year')
    st.plotly_chart(fig)

    # line chart
    fig = px.line(yearly_sales, x='Year', y='Sales', title = "Total sales per year (Line)", markers=True)
    st.plotly_chart(fig)

    # fig = go.Figure(data=go.Scatter(x=yearly_sales['Year'], y=yearly_sales['Sales'], mode='markers'))
    # st.plotly_chart(fig)

    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

    # Filter Data according to Year
    year_sale = df[df['Order Date'].dt.year == 2018]


    # Calculate Monthly Sales of Year 2018
    monthly_sale = year_sale.resample("M", on='Order Date')['Sales'].sum()


    # Rename the columns
    monthly_sale = monthly_sale.reset_index()
    monthly_sale = monthly_sale.rename(columns={'Order Date': "Month", 'Sales':'Total Monthly Sales'})
    fig = px.line(monthly_sale, x='Month', y='Total Monthly Sales', title='Total Monthly sales', markers=True)
    st.plotly_chart(fig)



# model_url = "https://clarifai.com/facebook/nougat/models/nougat-base"
# # image_url = "https://s3.amazonaws.com/samples.clarifai.com/featured-models/image-captioning-statue-of-liberty.jpeg"
# image_url = "/home/abiyaz/Downloads/Latest_Downloads/2022_9$largeimg_1999179093.jpeg"
# # image_url = "https://englishtribuneimages.blob.core.windows.net/gallary-content/2022/9/2022_9$largeimg_1999179093.jpeg"
# model_prediction = Model(model_url).predict_by_filepath(image_url, input_type="image")

# # # Get the output
# print(model_prediction.outputs[0].data.text.raw)




