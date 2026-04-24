import joblib
from pathlib import Path

from bitcoin_forecast_models import (
    download_market_data,
    build_features,
    run_random_forest,
    run_lstm_price_only
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Downloading live market data...")
df = download_market_data(start="2020-01-01")

print("Building features...")
df_clean = build_features(df)

print("Training Random Forest...")
rf_output = run_random_forest(df_clean)

print("Training LSTM...")
lstm_output = run_lstm_price_only(df, time_steps=30)

print("Saving models...")
joblib.dump(rf_output["model"], MODEL_DIR / "rf_model.pkl")
joblib.dump(df, MODEL_DIR / "market_df.pkl")
joblib.dump(df_clean, MODEL_DIR / "df_clean.pkl")

lstm_output["model"].save(MODEL_DIR / "lstm_model.keras")
joblib.dump(lstm_output["scaler"], MODEL_DIR / "lstm_scaler.pkl")
joblib.dump(lstm_output["residual_std"], MODEL_DIR / "lstm_residual_std.pkl")

print("Models saved successfully.")