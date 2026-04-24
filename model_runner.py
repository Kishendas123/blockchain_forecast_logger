import joblib
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model

from bitcoin_forecast_models import (
    download_market_data,
    build_features,
    run_random_forest,
    run_lstm_price_only,
    get_model_prediction
)

MODEL_DIR = Path("models")


@st.cache_resource
def load_pretrained_assets():
    df = joblib.load(MODEL_DIR / "market_df.pkl")
    df_clean = joblib.load(MODEL_DIR / "df_clean.pkl")
    rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")

    lstm_model = load_model(MODEL_DIR / "lstm_model.keras")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    lstm_residual_std = joblib.load(MODEL_DIR / "lstm_residual_std.pkl")

    return df, df_clean, rf_model, lstm_model, lstm_scaler, lstm_residual_std


def get_latest_model_prediction(model_choice: str):
    df, df_clean, rf_model, lstm_model, lstm_scaler, lstm_residual_std = load_pretrained_assets()

    if model_choice == "Random Forest":
        return get_model_prediction(
            model_choice="Random Forest",
            df=df,
            rf_model=rf_model,
            df_clean=df_clean
        )

    elif model_choice == "LSTM":
        return get_model_prediction(
            model_choice="LSTM",
            df=df,
            lstm_model=lstm_model,
            scaler_lstm=lstm_scaler,
            time_steps=30,
            lstm_residual_std=lstm_residual_std
        )

    elif model_choice == "ARIMA":
        return get_model_prediction(
            model_choice="ARIMA",
            df=df,
            arima_order=(2, 1, 0)
        )

    else:
        raise ValueError("Invalid model choice")


def retrain_and_save_models():
    df = download_market_data(start="2020-01-01")
    df_clean = build_features(df)

    if df_clean.empty:
        raise ValueError("No usable training data after feature engineering.")

    rf_output = run_random_forest(df_clean)
    lstm_output = run_lstm_price_only(df, time_steps=30)

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(rf_output["model"], MODEL_DIR / "rf_model.pkl")
    joblib.dump(df, MODEL_DIR / "market_df.pkl")
    joblib.dump(df_clean, MODEL_DIR / "df_clean.pkl")

    lstm_output["model"].save(MODEL_DIR / "lstm_model.keras")
    joblib.dump(lstm_output["scaler"], MODEL_DIR / "lstm_scaler.pkl")
    joblib.dump(lstm_output["residual_std"], MODEL_DIR / "lstm_residual_std.pkl")

    load_pretrained_assets.clear()

    return "Models retrained and saved successfully."