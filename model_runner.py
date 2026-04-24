import streamlit as st

from bitcoin_forecast_models import (
    download_market_data,
    build_features,
    run_random_forest,
    run_lstm_price_only,
    get_model_prediction
)


@st.cache_resource
def load_models():
    df = download_market_data(start="2020-01-01")
    df_clean = build_features(df)

    if df_clean.empty:
        raise ValueError("No usable training data after feature engineering.")

    rf_output = run_random_forest(df_clean)
    lstm_output = run_lstm_price_only(df, time_steps=30)

    return df, df_clean, rf_output, lstm_output


def get_latest_model_prediction(model_choice: str):
    df, df_clean, rf_output, lstm_output = load_models()

    if model_choice == "Random Forest":
        return get_model_prediction(
            model_choice="Random Forest",
            df=df,
            rf_model=rf_output["model"],
            df_clean=df_clean
        )

    elif model_choice == "LSTM":
        return get_model_prediction(
            model_choice="LSTM",
            df=df,
            lstm_model=lstm_output["model"],
            scaler_lstm=lstm_output["scaler"],
            time_steps=30,
            lstm_residual_std=lstm_output["residual_std"]
        )

    elif model_choice == "ARIMA":
        return get_model_prediction(
            model_choice="ARIMA",
            df=df,
            arima_order=(2, 1, 0)
        )

    else:
        raise ValueError("Invalid model choice")