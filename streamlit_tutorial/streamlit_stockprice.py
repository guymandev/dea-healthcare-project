import streamlit as st
import pandas as pd
import yfinance as yf
 
st.title("Stock Price Viewer")
 
# User selects stock
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
 
if st.button("Get Data"):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo")  # 1 month data
    st.line_chart(df["Close"])
 
    # Download option
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker}_data.csv",
        mime="text/csv"
    )