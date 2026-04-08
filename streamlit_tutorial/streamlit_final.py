import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Stock Dashboard (Offline AI)", page_icon="📈", layout="wide")

# ------------------------------
# AUTH UI: Remember session?
# ------------------------------
with st.sidebar:
    st.markdown("### 🔐 Sign-in Options")
    remember_session = st.checkbox("Remember me (stay signed in)", value=False,
                                   help="If unchecked, cookie expiry is 0 days and you will need to log in again next visit.")

# ------------------------------
# USER AUTHENTICATION CONFIG (v0.2.3 style)
# ------------------------------
names = ["Ghost User", "Alice Trader"]
usernames = ["ghost", "alice"]
passwords = ["mypassword", "alice123"]

hashed = stauth.Hasher(passwords).generate()
credentials = {
    "usernames": {
        usernames[i]: {"name": names[i], "password": hashed[i]}
        for i in range(len(usernames))
    }
}

cookie_expiry_days = 30 if remember_session else 0

authenticator = stauth.Authenticate(
    credentials,
    "stock_dash_cookie",     # cookie name
    "super_secret_key",      # signature key
    cookie_expiry_days       # expiry days (0 => no persistence)
)

name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False:
    st.error("Username/password is incorrect")
elif auth_status is None:
    st.warning("Please enter your username and password")
else:
    # Logged in
    with st.sidebar:
        authenticator.logout("Logout", "sidebar")
        st.success(f"Welcome {name} 👋")

    st.title("📈 Stock Dashboard — Offline AI Forecasts")
    st.caption(f"Signed in as **{username}** • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ------------------------------
    # Controls
    # ------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Controls")
        default_choices = ["AAPL", "MSFT", "TSLA"]
        tickers = st.multiselect(
            "Choose one or more tickers",
            options=default_choices + ["GOOGL", "AMZN", "NVDA", "META"],
            default=["AAPL"]
        )
        start_date = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End date", value=pd.to_datetime(datetime.today().date()))
        forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=14, step=1)

    # Guard
    if not tickers:
        st.info("Select at least one ticker from the sidebar.")
        st.stop()

    # ------------------------------
    # Data download
    # ------------------------------
    with st.spinner("Downloading market data..."):
        data = yf.download(tickers, start=start_date, end=end_date)

    if data.empty:
        st.warning("No data returned. Try different tickers or date range.")
        st.stop()

    # Normalize columns
    # If multiple tickers → MultiIndex columns; if single → flat columns
    if isinstance(data.columns, pd.MultiIndex):
        # Long format: Date | Ticker | [Open, High, Low, Close, Adj Close, Volume]
        long_df = data.stack(level=1).rename_axis(["Date", "Ticker"]).reset_index()
    else:
        # Single ticker → add a Ticker column to unify downstream logic
        t = tickers[0]
        tmp = data.copy()
        tmp["Ticker"] = t
        tmp["Date"] = tmp.index
        long_df = tmp.reset_index(drop=True)

    # ------------------------------
    # Overview: comparison chart
    # ------------------------------
    st.subheader("📊 Closing Prices — Comparison")
    if "Close" not in long_df.columns:
        st.error("Downloaded data is missing 'Close' column. Try another symbol or range.")
        st.stop()

    fig_comp = px.line(
        long_df.sort_values("Date"),
        x="Date", y="Close", color="Ticker",
        title="Closing Price Trend",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # KPIs for the latest available date per ticker
    latest_rows = long_df.sort_values("Date").groupby("Ticker").tail(1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers loaded", f"{latest_rows['Ticker'].nunique()}")
    c2.metric("Most Recent Date", latest_rows["Date"].max().date().isoformat())
    c3.metric("Min/Max Close (latest)", 
              f"${latest_rows['Close'].min():,.2f} / ${latest_rows['Close'].max():,.2f}")

    st.markdown("---")

    # ------------------------------
    # Per-ticker tabs with forecast
    # ------------------------------
    tabs = st.tabs([f"{t}" for t in sorted(set(long_df['Ticker']))])

    for tab, t in zip(tabs, sorted(set(long_df["Ticker"]))):
        with tab:
            st.markdown(f"### {t} — Details & Forecast")

            df_t = long_df[long_df["Ticker"] == t].dropna(subset=["Close"]).copy()
            df_t.sort_values("Date", inplace=True)

            # Show recent table
            with st.expander("Recent rows"):
                st.dataframe(df_t.tail(20), use_container_width=True)

            # Price line
            fig_line = px.line(df_t, x="Date", y="Close", title=f"{t} Closing Price")
            st.plotly_chart(fig_line, use_container_width=True)

            # Simple MAs
            with st.expander("📐 Moving Averages"):
                ma1 = st.number_input(f"{t} — MA 1 (days)", min_value=2, max_value=200, value=20, step=1, key=f"ma1_{t}")
                ma2 = st.number_input(f"{t} — MA 2 (days)", min_value=2, max_value=400, value=50, step=1, key=f"ma2_{t}")

            df_t["MA1"] = df_t["Close"].rolling(ma1).mean()
            df_t["MA2"] = df_t["Close"].rolling(ma2).mean()

            fig_ma = px.line(df_t, x="Date", y=["Close", "MA1", "MA2"],
                             title=f"{t} Close & Moving Averages")
            st.plotly_chart(fig_ma, use_container_width=True)

            # Offline ML forecast (Linear Regression on index)
            # Prepare numeric feature
            df_t["DayNum"] = np.arange(len(df_t))
            X = df_t[["DayNum"]]
            y = df_t["Close"]

            if len(df_t) >= 5:
                model = LinearRegression()
                model.fit(X, y)

                # In-sample error (demo)
                preds = model.predict(X)
                mae = mean_absolute_error(y, preds)

                # Future forecast
                fut_idx = np.arange(len(df_t), len(df_t) + forecast_days)
                fut_preds = model.predict(fut_idx.reshape(-1, 1))
                forecast_df = pd.DataFrame({
                    "Date": pd.date_range(df_t["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days),
                    "Forecast": fut_preds
                })

                st.markdown(f"**MAE (in-sample, demo)**: `{mae:.2f}`")
                fig_fc = px.line(forecast_df, x="Date", y="Forecast", title=f"{t} — {forecast_days}-Day Forecast")
                fig_fc.add_scatter(x=df_t["Date"], y=df_t["Close"], mode="lines", name="Historical Close")
                st.plotly_chart(fig_fc, use_container_width=True)

                # Download forecast CSV
                csv = forecast_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"Download {t} forecast CSV",
                    data=csv,
                    file_name=f"{t}_forecast_{forecast_days}d.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Not enough data points to train a model (need at least 5).")

    st.markdown("---")
    st.caption("Note: Forecasts use a simple linear regression on time index (offline). For educational use only.")