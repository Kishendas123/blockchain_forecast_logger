

# =========================================================
# 0. Imports
# =========================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. Shared helpers
# =========================================================
def evaluate_model(y_true, y_pred, model_name="Model"):
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        raise ValueError(f"{model_name}: all values are NaN after filtering.")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1:
        direction_accuracy = np.mean(
            (np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).astype(int)
        ) * 100
    else:
        direction_accuracy = np.nan

    print("-" * 60)
    print(f"{model_name} Evaluation")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Directional Accuracy: {direction_accuracy:.2f}%")
    print(f"R-squared: {r2:.4f}")
    print("-" * 60)

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Directional Accuracy": direction_accuracy,
        "R2": r2
    }

def compute_direction_and_confidence(predicted_price, last_price):
    predicted_direction = "Up" if predicted_price > last_price else "Down"
    confidence_pct = min(abs(predicted_price - last_price) / last_price * 100, 100)
    return predicted_direction, confidence_pct


def format_prediction_output(model_name, predicted_price, last_price, lower_pi=None, upper_pi=None):
    predicted_direction, confidence_pct = compute_direction_and_confidence(
        predicted_price, last_price
    )

    return {
        "model_name": model_name,
        "predicted_price": float(predicted_price),
        "predicted_direction": predicted_direction,
        "confidence_pct": float(confidence_pct),
        "lower_pi": None if lower_pi is None else float(lower_pi),
        "upper_pi": None if upper_pi is None else float(upper_pi)
    }


def get_adf_summary(series):
    result = adfuller(series.dropna())
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "is_stationary": bool(result[1] < 0.05)
    }


# =========================================================
# 2. Download data from yfinance
# =========================================================
def download_market_data(start="2020-01-01"):
    tickers = ["BTC-USD", "^IXIC", "GC=F"]

    data = yf.download(
        tickers,
        start=start,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    # Flatten MultiIndex columns
    data.columns = [f"{ticker} {field}" for field, ticker in data.columns]

    rename_map = {
        "BTC-USD Open": "BTC Open",
        "BTC-USD High": "BTC High",
        "BTC-USD Low": "BTC Low",
        "BTC-USD Close": "BTC Close",
        "BTC-USD Volume": "BTC Volume",

        "^IXIC Open": "NASDAQ Open",
        "^IXIC High": "NASDAQ High",
        "^IXIC Low": "NASDAQ Low",
        "^IXIC Close": "NASDAQ Close",
        "^IXIC Volume": "NASDAQ Volume",

        "GC=F Open": "Gold Open",
        "GC=F High": "Gold High",
        "GC=F Low": "Gold Low",
        "GC=F Close": "Gold Close",
        "GC=F Volume": "Gold Volume",
    }

    df = data.rename(columns=rename_map)

    # Keep only columns that actually exist
    desired_cols = [
        "BTC Open", "BTC High", "BTC Low", "BTC Close", "BTC Volume",
        "NASDAQ Close", "NASDAQ Volume",
        "Gold Close", "Gold Volume"
    ]

    existing_cols = [col for col in desired_cols if col in df.columns]
    df = df[existing_cols].copy()

    # Sort by date
    df = df.sort_index()

    # BTC must exist
    df = df[df["BTC Close"].notna()].copy()

    # Fill non-BTC columns where possible
    fill_cols = [col for col in ["NASDAQ Close", "NASDAQ Volume", "Gold Close", "Gold Volume"] if col in df.columns]
    if fill_cols:
        df[fill_cols] = df[fill_cols].ffill().bfill()

    print("downloaded df shape:", df.shape)
    print("downloaded df columns:", df.columns.tolist())
    print(df.head())

    return df

# =========================================================
# 3. Feature engineering for Random Forest
# =========================================================
def build_features(df):
    df_feat = df.copy()

    print("build_features input shape:", df_feat.shape)
    print("build_features input nulls:")
    print(df_feat.isna().sum())

    # Returns
    df_feat["BTC Return"] = df_feat["BTC Close"].pct_change()
    df_feat["NASDAQ Return"] = df_feat["NASDAQ Close"].pct_change()

    # Gold return only if Gold exists and is usable
    if "Gold Close" in df_feat.columns and df_feat["Gold Close"].notna().sum() > 0:
        df_feat["Gold Return"] = df_feat["Gold Close"].pct_change()
    else:
        print("Gold data unavailable - skipping Gold Return feature.")

    # SMA
    df_feat["SMA_7"] = df_feat["BTC Close"].rolling(window=7).mean()
    df_feat["SMA_30"] = df_feat["BTC Close"].rolling(window=30).mean()
    df_feat["SMA_ratio"] = df_feat["SMA_7"] / df_feat["SMA_30"]

    # EMA
    df_feat["EMA_7"] = df_feat["BTC Close"].ewm(span=7, adjust=False).mean()
    df_feat["EMA_30"] = df_feat["BTC Close"].ewm(span=30, adjust=False).mean()
    df_feat["EMA_ratio"] = df_feat["EMA_7"] / df_feat["EMA_30"]
    df_feat["EMA_signal"] = (df_feat["EMA_7"] > df_feat["EMA_30"]).astype(int)

    # Bollinger Bands
    rolling_mean = df_feat["BTC Close"].rolling(20).mean()
    rolling_std = df_feat["BTC Close"].rolling(20).std()

    df_feat["Bollinger_Mid"] = rolling_mean
    df_feat["Bollinger_Upper"] = rolling_mean + (2 * rolling_std)
    df_feat["Bollinger_Lower"] = rolling_mean - (2 * rolling_std)

    df_feat["BB_position"] = (
        (df_feat["BTC Close"] - df_feat["Bollinger_Lower"]) /
        (df_feat["Bollinger_Upper"] - df_feat["Bollinger_Lower"])
    )
    df_feat["BB_width"] = (
        (df_feat["Bollinger_Upper"] - df_feat["Bollinger_Lower"]) /
        df_feat["Bollinger_Mid"]
    )
    df_feat["BB_breakout"] = (
        (df_feat["BTC Close"] > df_feat["Bollinger_Upper"]) |
        (df_feat["BTC Close"] < df_feat["Bollinger_Lower"])
    ).astype(int)

    # RSI
    delta = df_feat["BTC Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, 1e-10)

    rs = avg_gain / avg_loss
    df_feat["RSI"] = 100 - (100 / (1 + rs))
    df_feat["RSI_norm"] = df_feat["RSI"] / 100
    df_feat["RSI_signal"] = ((df_feat["RSI"] > 70) | (df_feat["RSI"] < 30)).astype(int)
    df_feat["RSI_direction"] = (df_feat["RSI"] > 50).astype(int)

    # Drop NaNs caused by rolling indicators
    df_feat.dropna(inplace=True)

    # Columns to drop only if they exist
    drop_cols = [
        "NASDAQ Close",
        "Gold Close",
        "SMA_7",
        "SMA_30",
        "EMA_7",
        "EMA_30",
        "Bollinger_Mid",
        "Bollinger_Upper",
        "Bollinger_Lower",
        "RSI",
        "Gold Volume"
    ]

    existing_drop_cols = [col for col in drop_cols if col in df_feat.columns]
    df_clean = df_feat.drop(columns=existing_drop_cols, errors="ignore").copy()

    # Target = next-day BTC close
    df_clean["Target"] = df_clean["BTC Close"].shift(-1)
    df_clean.dropna(inplace=True)

    print("build_features output shape:", df_clean.shape)
    print("build_features output nulls:")
    print(df_clean.isna().sum())

    return df_clean

# =========================================================
# 4. Random Forest block
# =========================================================
def run_random_forest(df_clean):
    df_rf = df_clean.copy()

    df_rf["lag_1"] = df_rf["BTC Close"].shift(1)
    df_rf["lag_2"] = df_rf["BTC Close"].shift(2)
    df_rf["lag_3"] = df_rf["BTC Close"].shift(3)
    df_rf["lag_7"] = df_rf["BTC Close"].shift(7)

    df_rf.dropna(inplace=True)

    y_rf = df_rf["Target"]
    X_rf = df_rf.drop(columns=["Target"])

    train_size = int(len(df_rf) * 0.8)

    X_train_rf = X_rf.iloc[:train_size]
    X_test_rf = X_rf.iloc[train_size:]

    y_train_rf = y_rf.iloc[:train_size]
    y_test_rf = y_rf.iloc[train_size:]
    

    param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4]
    }

    print("df_rf shape:", df_rf.shape)
    print("X_train_rf shape:", X_train_rf.shape)
    print("y_train_rf shape:", y_train_rf.shape)

    if len(X_train_rf) < 10:
        raise ValueError(f"Not enough training samples for Random Forest: {len(X_train_rf)}")

    rf_model = RandomForestRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=1
    )

    grid_search.fit(X_train_rf, y_train_rf)
    best_rf = grid_search.best_estimator_

    print("Best RF Parameters:", grid_search.best_params_)

    best_rf.fit(X_train_rf, y_train_rf)

    train_pred_rf = best_rf.predict(X_train_rf)
    test_pred_rf = best_rf.predict(X_test_rf)

    last_price_test_rf = X_test_rf["BTC Close"].values
    pred_direction_rf, confidence_rf = zip(*[
        compute_direction_and_confidence(p, lp)
        for p, lp in zip(test_pred_rf, last_price_test_rf)
    ])

    tree_preds = np.column_stack([tree.predict(X_test_rf) for tree in best_rf.estimators_])
    lower_rf = np.percentile(tree_preds, 10, axis=1)
    upper_rf = np.percentile(tree_preds, 90, axis=1)

    train_metrics = evaluate_model(y_train_rf, train_pred_rf, "Random Forest - Train")
    test_metrics = evaluate_model(y_test_rf, test_pred_rf, "Random Forest - Test")

    results_rf = pd.DataFrame({
        "Actual Price": y_test_rf.values,
        "Predicted Price": test_pred_rf,
        "Last Input Price": last_price_test_rf,
        "Predicted Direction": pred_direction_rf,
        "Confidence %": confidence_rf,
        "Lower PI": lower_rf,
        "Upper PI": upper_rf
    }, index=y_test_rf.index)

    return {
        "model": best_rf,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "results": results_rf,
        "X_test": X_test_rf,
        "y_test": y_test_rf,
        "train_pred": train_pred_rf,
        "test_pred": test_pred_rf
    }


def predict_with_random_forest(best_rf, df_clean):
    df_rf_latest = df_clean.copy()

    df_rf_latest["lag_1"] = df_rf_latest["BTC Close"].shift(1)
    df_rf_latest["lag_2"] = df_rf_latest["BTC Close"].shift(2)
    df_rf_latest["lag_3"] = df_rf_latest["BTC Close"].shift(3)
    df_rf_latest["lag_7"] = df_rf_latest["BTC Close"].shift(7)

    df_rf_latest.dropna(inplace=True)

    X_rf_latest = df_rf_latest.drop(columns=["Target"]).iloc[[-1]]
    last_price = X_rf_latest["BTC Close"].iloc[0]

    predicted_price = best_rf.predict(X_rf_latest)[0]

    tree_preds = np.array([tree.predict(X_rf_latest)[0] for tree in best_rf.estimators_])
    lower_pi = np.percentile(tree_preds, 10)
    upper_pi = np.percentile(tree_preds, 90)

    return format_prediction_output(
        model_name="Random Forest",
        predicted_price=predicted_price,
        last_price=last_price,
        lower_pi=lower_pi,
        upper_pi=upper_pi
    )


# =========================================================
# 5. LSTM block (price-only)
# =========================================================
def create_sequences(data, time_steps=30):
    X, y = [], []

    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])

    return np.array(X), np.array(y)


def run_lstm_price_only(df, time_steps=30):
    df_lstm = df[["BTC Close"]].copy().sort_index()

    scaler_lstm = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler_lstm.fit_transform(df_lstm[["BTC Close"]])

    X_lstm, y_lstm = create_sequences(scaled_prices, time_steps=time_steps)

    train_size = int(len(X_lstm) * 0.8)

    X_train_lstm = X_lstm[:train_size]
    X_test_lstm = X_lstm[train_size:]

    y_train_lstm = y_lstm[:train_size]
    y_test_lstm = y_lstm[train_size:]

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_lstm,
        y_train_lstm,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_lstm, y_test_lstm),
        callbacks=[early_stopping],
        verbose=0
    )

    train_pred_lstm = model.predict(X_train_lstm, verbose=0)
    test_pred_lstm = model.predict(X_test_lstm, verbose=0)

    train_pred_lstm = scaler_lstm.inverse_transform(train_pred_lstm).flatten()
    test_pred_lstm = scaler_lstm.inverse_transform(test_pred_lstm).flatten()

    y_train_actual = scaler_lstm.inverse_transform(y_train_lstm.reshape(-1, 1)).flatten()
    y_test_actual = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

    lstm_index = df_lstm.index[time_steps:]
    train_index = lstm_index[:train_size]
    test_index = lstm_index[train_size:]

    last_input_test = scaler_lstm.inverse_transform(X_test_lstm[:, -1, :]).flatten()
    pred_direction_lstm, confidence_lstm = zip(*[
        compute_direction_and_confidence(p, lp)
        for p, lp in zip(test_pred_lstm, last_input_test)
    ])

    train_residuals = y_train_actual - train_pred_lstm
    residual_std = np.std(train_residuals)

    lower_lstm = test_pred_lstm - 1.96 * residual_std
    upper_lstm = test_pred_lstm + 1.96 * residual_std

    train_metrics = evaluate_model(y_train_actual, train_pred_lstm, "LSTM Price-Only - Train")
    test_metrics = evaluate_model(y_test_actual, test_pred_lstm, "LSTM Price-Only - Test")

    derived_direction_accuracy = np.mean(
        (y_test_actual > last_input_test) == (test_pred_lstm > last_input_test)
    ) * 100
    print(f"LSTM Derived Direction Accuracy: {derived_direction_accuracy:.2f}%")
    print("-" * 60)

    results_lstm = pd.DataFrame({
        "Actual Price": y_test_actual,
        "Predicted Price": test_pred_lstm,
        "Last Input Price": last_input_test,
        "Predicted Direction": pred_direction_lstm,
        "Confidence %": confidence_lstm,
        "Lower PI": lower_lstm,
        "Upper PI": upper_lstm
    }, index=test_index)

    return {
        "model": model,
        "history": history,
        "scaler": scaler_lstm,
        "residual_std": residual_std,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "results": results_lstm,
        "train_index": train_index,
        "test_index": test_index,
        "y_train_actual": y_train_actual,
        "y_test_actual": y_test_actual,
        "train_pred": train_pred_lstm,
        "test_pred": test_pred_lstm
    }


def predict_with_lstm(lstm_model, scaler_lstm, df, time_steps=30, residual_std=None):
    df_lstm_latest = df[["BTC Close"]].copy().sort_index()
    df_lstm_latest.dropna(inplace=True)

    latest_prices = df_lstm_latest[["BTC Close"]].tail(time_steps).values

    if len(latest_prices) < time_steps:
        raise ValueError(f"Not enough BTC Close data. Need at least {time_steps} rows.")

    if np.isnan(latest_prices).any():
        raise ValueError("Latest BTC Close window contains NaN values.")

    last_price = latest_prices[-1][0]

    latest_scaled = scaler_lstm.transform(latest_prices)

    if np.isnan(latest_scaled).any():
        raise ValueError("Scaled latest LSTM input contains NaN values.")

    X_latest = latest_scaled.reshape(1, time_steps, 1)

    predicted_scaled = lstm_model.predict(X_latest, verbose=0)

    if np.isnan(predicted_scaled).any():
        raise ValueError("LSTM predicted NaN for the latest input sequence.")

    predicted_price = scaler_lstm.inverse_transform(predicted_scaled)[0][0]

    lower_pi, upper_pi = None, None
    if residual_std is not None and not np.isnan(residual_std):
        lower_pi = predicted_price - 1.96 * residual_std
        upper_pi = predicted_price + 1.96 * residual_std

    return format_prediction_output(
        model_name="LSTM",
        predicted_price=predicted_price,
        last_price=last_price,
        lower_pi=lower_pi,
        upper_pi=upper_pi
    )

# =========================================================
# 6. ARIMA block (rolling forecast)
# =========================================================
def run_arima_rolling(df, order=(2, 1, 0)):
    btc = df[["BTC Close"]].copy().sort_index()

    train_size = int(len(btc) * 0.8)
    train = btc.iloc[:train_size].copy()
    test = btc.iloc[train_size:].copy()

    history = train["BTC Close"].tolist()
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)[0]
        predictions.append(yhat)

        # update with actual next observation
        history.append(test["BTC Close"].iloc[t])

    forecast = pd.Series(predictions, index=test.index, name="Predicted Price")

    comparison_df = pd.concat([test["BTC Close"], forecast], axis=1)
    comparison_df.columns = ["Actual Price", "Predicted Price"]
    comparison_df.dropna(inplace=True)

    last_input_price = comparison_df["Actual Price"].shift(1).dropna()
    aligned_actual = comparison_df["Actual Price"].iloc[1:]
    aligned_pred = comparison_df["Predicted Price"].iloc[1:]

    pred_direction, confidence_pct = zip(*[
        compute_direction_and_confidence(p, lp)
        for p, lp in zip(aligned_pred.values, last_input_price.values)
    ])

    metrics = evaluate_model(
        comparison_df["Actual Price"].values,
        comparison_df["Predicted Price"].values,
        f"ARIMA Rolling {order}"
    )

    adf_original = get_adf_summary(train["BTC Close"])
    adf_diff = get_adf_summary(train["BTC Close"].diff().dropna())

    results_arima = pd.DataFrame({
        "Actual Price": aligned_actual.values,
        "Predicted Price": aligned_pred.values,
        "Last Input Price": last_input_price.values,
        "Predicted Direction": pred_direction,
        "Confidence %": confidence_pct
    }, index=aligned_actual.index)

    return {
        "order": order,
        "train": train,
        "test": test,
        "forecast": forecast,
        "metrics": metrics,
        "results": results_arima,
        "adf_original": adf_original,
        "adf_diff": adf_diff
    }


def predict_with_arima(df, order=(2, 1, 0)):
    df_arima = df[["BTC Close"]].copy().sort_index()
    series = df_arima["BTC Close"]
    last_price = series.iloc[-1]

    model = ARIMA(series, order=order)
    fit = model.fit()

    forecast_obj = fit.get_forecast(steps=1)
    predicted_price = forecast_obj.predicted_mean.iloc[0]
    conf_int = forecast_obj.conf_int()

    lower_pi = conf_int.iloc[0, 0]
    upper_pi = conf_int.iloc[0, 1]

    return format_prediction_output(
        model_name="ARIMA",
        predicted_price=predicted_price,
        last_price=last_price,
        lower_pi=lower_pi,
        upper_pi=upper_pi
    )


# =========================================================
# 7. Unified router for app / Streamlit
# =========================================================
def get_model_prediction(
    model_choice,
    df,
    rf_model=None,
    lstm_model=None,
    scaler_lstm=None,
    time_steps=30,
    lstm_residual_std=None,
    arima_order=(2, 1, 0),
    df_clean=None
):
    if model_choice == "Random Forest":
        if rf_model is None or df_clean is None:
            raise ValueError("Random Forest model and df_clean are required.")
        return predict_with_random_forest(rf_model, df_clean)

    elif model_choice == "LSTM":
        if lstm_model is None or scaler_lstm is None:
            raise ValueError("LSTM model and scaler_lstm are required.")
        return predict_with_lstm(
            lstm_model=lstm_model,
            scaler_lstm=scaler_lstm,
            df=df,
            time_steps=time_steps,
            residual_std=lstm_residual_std
        )

    elif model_choice == "ARIMA":
        return predict_with_arima(df=df, order=arima_order)

    else:
        raise ValueError("Invalid model choice.")


# =========================================================
# 8. Plot helpers
# =========================================================
def plot_rf_results(results_rf):
    plt.figure(figsize=(12, 6))
    plt.plot(results_rf.index, results_rf["Actual Price"], label="Actual", linewidth=2)
    plt.plot(results_rf.index, results_rf["Predicted Price"], label="RF Predicted", linestyle="--")
    plt.fill_between(
        results_rf.index,
        results_rf["Lower PI"],
        results_rf["Upper PI"],
        alpha=0.2,
        label="RF Prediction Interval"
    )
    plt.title("Bitcoin Price Prediction with Random Forest")
    plt.xlabel("Date")
    plt.ylabel("BTC Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_lstm_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_lstm_results(results_lstm):
    plt.figure(figsize=(12, 6))
    plt.plot(results_lstm.index, results_lstm["Actual Price"], label="Actual", linewidth=2)
    plt.plot(results_lstm.index, results_lstm["Predicted Price"], label="LSTM Predicted", linestyle="--")
    plt.fill_between(
        results_lstm.index,
        results_lstm["Lower PI"],
        results_lstm["Upper PI"],
        alpha=0.2,
        label="LSTM Approx Interval"
    )
    plt.title("Bitcoin Price Prediction with LSTM (Price-Only)")
    plt.xlabel("Date")
    plt.ylabel("BTC Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_arima_results(arima_output):
    test = arima_output["test"]
    forecast = arima_output["forecast"]

    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test["BTC Close"], label="Actual", linewidth=2)
    plt.plot(forecast.index, forecast, label="ARIMA Rolling Predicted", linestyle="--")
    plt.title(f"Bitcoin Price Prediction with Rolling ARIMA {arima_output['order']}")
    plt.xlabel("Date")
    plt.ylabel("BTC Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# 9. Main runnable demo
# =========================================================
def main():
    print("Downloading market data...")
    df = download_market_data(start="2020-01-01")
    print("Data shape:", df.shape)

    print("\nBuilding engineered features for Random Forest...")
    df_clean = build_features(df)
    print("Feature shape:", df_clean.shape)

    print("\nRunning Random Forest...")
    rf_output = run_random_forest(df_clean)

    print("\nRunning LSTM (price-only)...")
    lstm_output = run_lstm_price_only(df, time_steps=30)

    print("\nRunning Rolling ARIMA...")
    arima_output = run_arima_rolling(df, order=(2, 1, 0))

    print("\nADF Summary for ARIMA")
    print("Original series:", arima_output["adf_original"])
    print("Differenced series:", arima_output["adf_diff"])

    print("\nLatest model predictions")
    rf_pred = get_model_prediction(
        model_choice="Random Forest",
        df=df,
        rf_model=rf_output["model"],
        df_clean=df_clean
    )
    lstm_pred = get_model_prediction(
        model_choice="LSTM",
        df=df,
        lstm_model=lstm_output["model"],
        scaler_lstm=lstm_output["scaler"],
        time_steps=30,
        lstm_residual_std=lstm_output["residual_std"]
    )
    arima_pred = get_model_prediction(
        model_choice="ARIMA",
        df=df,
        arima_order=(2, 1, 0)
    )

    print("Random Forest:", rf_pred)
    print("LSTM:", lstm_pred)
    print("ARIMA:", arima_pred)

    comparison_df = pd.DataFrame([
        rf_output["test_metrics"],
        lstm_output["test_metrics"],
        arima_output["metrics"]
    ])
    print("\nModel comparison")
    print(comparison_df)

    # Save result tables
    rf_output["results"].to_csv("rf_results.csv")
    lstm_output["results"].to_csv("lstm_results.csv")
    arima_output["results"].to_csv("arima_results.csv")
    comparison_df.to_csv("model_comparison.csv", index=False)

    print("\nSaved:")
    print("- rf_results.csv")
    print("- lstm_results.csv")
    print("- arima_results.csv")
    print("- model_comparison.csv")

    # Optional plots
    plot_rf_results(rf_output["results"])
    plot_lstm_history(lstm_output["history"])
    plot_lstm_results(lstm_output["results"])
    plot_arima_results(arima_output)


if __name__ == "__main__":
    main()
