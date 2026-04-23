import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from model_runner import get_latest_model_prediction
from blockchain_logger import (
    submit_prediction_to_blockchain,
    get_latest_prediction,
    get_prediction_count
)
from bitcoin_forecast_models import download_market_data

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bitcoin Forecast Logger",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Load market data
# ---------------------------------------------------------
@st.cache_data
def load_market_data():
    df = download_market_data(start="2020-01-01")
    return df

df = load_market_data()

# ---------------------------------------------------------
# Cache model prediction
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_prediction(model_choice):
    return get_latest_model_prediction(model_choice)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center;">
    <h4>Done by Kishen</h4>
    <p style="font-size:12px;">FinTech | Data Science | Blockchain</p>
    <a href="https://github.com/Kishendas123" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="28">
    </a>
    &nbsp;&nbsp;
    <a href="https://www.linkedin.com/in/kishen-das/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="28">
    </a>
</div>
<hr>
""", unsafe_allow_html=True)

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose forecasting model",
    ["Random Forest", "LSTM", "ARIMA"]
)

history_window = st.sidebar.slider(
    "Recent price history to display (days)",
    min_value=30,
    max_value=180,
    value=90,
    step=10
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🪙 Bitcoin Price Prediction & On-Chain Forecast Logger")

st.markdown("""
### Forecast the **next-day Bitcoin closing price**
This app compares **Random Forest**, **LSTM**, and **ARIMA** models to estimate the next-day BTC closing price.

It also lets you log the prediction on-chain through a smart contract, creating a **transparent and timestamped forecast record**.
""")

st.info(
    "The forecast shown in this app is a **next-day BTC price prediction**. "
    "The direction (Up / Down) is derived by comparing the predicted next-day price against the latest observed BTC price."
)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "⛓ Blockchain", "ℹ️ About"])

# =========================================================
# TAB 1: FORECAST
# =========================================================
with tab1:
    st.subheader("Recent Bitcoin Price Trend")

    recent_df = df.tail(history_window).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_df.index, recent_df["BTC Close"], linewidth=2, label="BTC Close")
    ax.set_title(f"BTC Closing Price - Last {history_window} Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Generate Forecast")

    st.markdown("""
**What this does:**  
When you click **Generate Prediction**, the selected model estimates the **next-day BTC closing price**.

The app then shows:
- the predicted price
- whether the model expects BTC to go **Up** or **Down**
- a confidence proxy based on the size of the predicted move
- a prediction interval where available
""")

    if st.button("Generate Prediction"):
        with st.spinner("Running model and generating next-day forecast..."):
            prediction = cached_prediction(model_choice)

        st.session_state["latest_prediction"] = prediction

    if "latest_prediction" in st.session_state:
        prediction = st.session_state["latest_prediction"]

        st.subheader("Latest Model Forecast")

        latest_btc_price = df["BTC Close"].iloc[-1]
        forecast_date = df.index[-1] + pd.Timedelta(days=1)

        col0, col1, col2, col3 = st.columns(4)

        with col0:
            st.metric(
                label="Latest BTC Close",
                value=f"${latest_btc_price:,.2f}"
            )

        with col1:
            st.metric(
                label="Predicted Next-Day BTC Price",
                value=f"${prediction['predicted_price']:,.2f}"
            )

        with col2:
            direction_emoji = "📈" if prediction["predicted_direction"] == "Up" else "📉"
            st.metric(
                label="Predicted Direction",
                value=f"{direction_emoji} {prediction['predicted_direction']}"
            )

        with col3:
            st.metric(
                label="Confidence",
                value=f"{prediction['confidence_pct']:.2f}%"
            )

        st.write(f"**Model Used:** {prediction['model_name']}")
        st.write(f"**Forecast Date:** {forecast_date.date()}")

        if prediction["lower_pi"] is not None and prediction["upper_pi"] is not None:
            st.write(
                f"**Prediction Interval:** "
                f"${prediction['lower_pi']:,.2f} to ${prediction['upper_pi']:,.2f}"
            )

        st.caption("Confidence is a simple proxy based on predicted move size, not a formal probability.")

        if prediction["predicted_direction"] == "Up":
            st.success("📈 The model expects Bitcoin to close higher the next day.")
        else:
            st.error("📉 The model expects Bitcoin to close lower the next day.")

        # -----------------------------------------------------
        # Forecast Chart
        # -----------------------------------------------------
        st.subheader("Forecast vs Recent BTC History")

        recent_plot_df = df.tail(60).copy()
        last_date = recent_plot_df.index[-1]
        next_day_price = prediction["predicted_price"]

        forecast_df = pd.DataFrame(
            {"BTC Close": [next_day_price]},
            index=[last_date + pd.Timedelta(days=1)]
        )

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(
            recent_plot_df.index,
            recent_plot_df["BTC Close"],
            label="Historical BTC Close",
            linewidth=2
        )
        ax2.scatter(
            forecast_df.index,
            forecast_df["BTC Close"],
            color="red",
            s=100,
            label="Predicted Next-Day Price"
        )

        if prediction["lower_pi"] is not None and prediction["upper_pi"] is not None:
            ax2.vlines(
                forecast_df.index[0],
                prediction["lower_pi"],
                prediction["upper_pi"],
                linestyles="dashed",
                linewidth=2,
                label="Prediction Interval"
            )

        ax2.set_title("Recent BTC History + Next-Day Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price (USD)")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

# =========================================================
# TAB 2: BLOCKCHAIN
# =========================================================
with tab2:
    st.subheader("Why Log the Forecast On-Chain?")

    st.markdown("""
A blockchain log works like a **tamper-resistant forecast notebook**.

When you log a forecast on-chain, the app stores:
- the model name
- the predicted BTC value
- the confidence score
- the timestamp
- the submitting wallet

This means the forecast becomes a **transparent historical record** that can be checked later.
""")

    if "latest_prediction" in st.session_state:
        if st.button("Log Latest Prediction On-Chain"):
            pred = st.session_state["latest_prediction"]

            with st.spinner("Submitting forecast to the smart contract..."):
                tx_hash = submit_prediction_to_blockchain(
                    model_name=pred["model_name"],
                    predicted_price=pred["predicted_price"],
                    confidence_pct=pred["confidence_pct"]
                )

            st.success("Prediction successfully logged on-chain.")
            st.code(tx_hash, language="text")
    else:
        st.warning("Generate a prediction first before logging it on-chain.")

    st.subheader("Latest On-Chain Forecast")

    st.markdown("""
This section reads the most recent forecast stored in the smart contract.

It lets users verify that a forecast was written on-chain and see the latest logged values.
""")

    if st.button("Refresh Blockchain Record"):
        latest_onchain = get_latest_prediction()
        count = get_prediction_count()

        readable_time = pd.to_datetime(latest_onchain["timestamp"], unit="s")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Stored Forecast Count", count)
            st.write(f"**Model:** {latest_onchain['model_name']}")
            st.write(f"**Predicted Value:** {latest_onchain['predicted_value']}")
            st.write(f"**Confidence:** {latest_onchain['confidence']}%")

        with col2:
            st.write(f"**Submitted By:** {latest_onchain['submitted_by']}")
            st.write(f"**Timestamp:** {readable_time}")

with tab2:
    st.subheader("Blockchain Logging")
    st.info(
        "Blockchain logging is available in the local development version of this project. "
        "The cloud deployment focuses on the forecasting interface and model outputs."
    )

    st.markdown("""
This project also includes a local blockchain forecast logger built with Hardhat and Solidity.
In the local version, predictions can be written to a smart contract as a timestamped record.
""")

# =========================================================
# TAB 3: ABOUT
# =========================================================
with tab3:
    st.subheader("How This App Works")

    with st.expander("Click to understand how the prediction system works", expanded=True):
        st.markdown("""
### 📊 Data Collection
- Bitcoin (BTC), NASDAQ, and Gold data are retrieved using **Yahoo Finance**
- Historical daily closing prices are used for modeling

---

### 🤖 Forecasting Models
This app compares three models:

**1. Random Forest**
- Uses engineered features (returns, moving averages, RSI, etc.)
- Captures non-linear relationships

**2. LSTM (Deep Learning)**
- Uses sequential price data
- Learns time-based patterns in BTC prices

**3. ARIMA**
- Time-series statistical model
- Focuses on trend and autocorrelation

---

### 📈 What is being predicted?
- The model predicts the **next-day Bitcoin closing price**
- Direction is derived:
  - 📈 Up → predicted price > latest price  
  - 📉 Down → predicted price < latest price  

---

### 🎯 Confidence Score
- A simple proxy based on the size of predicted price movement
- Not a probability, but an indicator of prediction strength

---

### 🔗 Blockchain Logging
- When you log a prediction:
  - Model name, predicted price, confidence, and timestamp are stored
- This creates a **transparent and tamper-resistant forecast record**

---

### 🚀 Why this matters
- Combines **Machine Learning + Blockchain**
- Enables **verifiable and trackable financial predictions**
""")

    st.markdown("---")


    st.caption("Built using Python, Streamlit, Web3.py, TensorFlow, Scikit-learn and Hardhat")