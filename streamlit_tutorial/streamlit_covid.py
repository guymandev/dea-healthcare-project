import streamlit as st
import pandas as pd
import plotly.express as px
import requests
 
st.title("COVID-19 Live Cases Dashboard")
 
@st.cache_data
def load_data():
    url = " https://disease.sh/v3/covid-19/historical/all?lastdays=all"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame({
        "date": list(data["cases"].keys()),
        "cases": list(data["cases"].values()),
        "deaths": list(data["deaths"].values()),
        "recovered": list(data["recovered"].values())
    })
    return df
 
df = load_data()
 
# Plot cases over time
fig = px.line(df, x="date", y=["cases", "deaths", "recovered"],
              title="Global COVID-19 Cases Over Time")
st.plotly_chart(fig, use_container_width=True)