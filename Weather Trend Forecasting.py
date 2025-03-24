
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import folium

# Ensure directories exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load Dataset
file_path = r"C:\Users\nithi\Downloads\GlobalWeatherRepository.csv"
df = pd.read_csv(file_path)

# Data Cleaning & Preprocessing
df.rename(columns={"lastupdated": "last_updated"}, inplace=True)
df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
df["temperature_celsius"] = pd.to_numeric(df["temperature_celsius"], errors="coerce")
df.dropna(subset=["last_updated", "temperature_celsius"], inplace=True)

# Save cleaned data
df.to_csv(f"{output_dir}/cleaned_weather_data.csv", index=False)

# Basic EDA
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["last_updated"], y=df["temperature_celsius"])
plt.title("Temperature Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.savefig(f"{output_dir}/temperature_trend.png")
plt.show()

# Advanced EDA: Anomaly Detection
df["temp_zscore"] = (df["temperature_celsius"] - df["temperature_celsius"].mean()) / df["temperature_celsius"].std()
df["is_anomaly"] = df["temp_zscore"].apply(lambda x: 1 if abs(x) > 2.5 else 0)

plt.figure(figsize=(12, 6))
sns.scatterplot(x=df["last_updated"], y=df["temperature_celsius"], hue=df["is_anomaly"], palette=["blue", "red"])
plt.title("Anomaly Detection in Temperature Data")
plt.savefig(f"{output_dir}/temperature_anomalies.png")
plt.show()

# Feature Importance using RandomForest
features = ["humidity", "wind_kph", "pressure_in"]
df.dropna(subset=features, inplace=True)
X = df[features]
y = df["temperature_celsius"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
feature_importance.to_csv(f"{output_dir}/feature_importance.csv")

plt.figure(figsize=(8, 4))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance")
plt.savefig(f"{output_dir}/feature_importance.png")
plt.show()

# Forecasting Models

# Convert to daily temperature
df_daily = df[["last_updated", "temperature_celsius"]].set_index("last_updated").resample("D").mean().dropna()

def arima_forecasting():
    model = ARIMA(df_daily, order=(1, 1, 1)).fit()
    forecast = model.forecast(steps=30)
    future_dates = pd.date_range(start=df_daily.index[-1], periods=30, freq="D")

    forecast_df = pd.DataFrame({"date": future_dates, "predicted_temperature": forecast})
    forecast_df.to_csv(f"{output_dir}/arima_forecast.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df_daily.index, df_daily, label="Actual", color="blue")
    plt.plot(future_dates, forecast, label="ARIMA Forecast", color="red", linestyle="dashed")
    plt.title("ARIMA Weather Forecast")
    plt.legend()
    plt.savefig(f"{output_dir}/arima_forecast.png")
    plt.show()

def lstm_forecasting():
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_daily.values.reshape(-1, 1))

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 10
    X, y = create_sequences(df_scaled, seq_length)
    X_train, y_train = X[:-30], y[:-30]
    X_test, y_test = X[-30:], y[-30:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

    predictions = scaler.inverse_transform(model.predict(X_test))
    future_dates = pd.date_range(start=df_daily.index[-30], periods=30, freq="D")

    forecast_df = pd.DataFrame({"date": future_dates, "predicted_temperature": predictions.flatten()})
    forecast_df.to_csv(f"{output_dir}/lstm_forecast.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df_daily.index, df_daily, label="Actual", color="blue")
    plt.plot(future_dates, predictions, label="LSTM Forecast", color="green", linestyle="dashed")
    plt.title("LSTM Weather Forecast")
    plt.legend()
    plt.savefig(f"{output_dir}/lstm_forecast.png")
    plt.show()

# Run forecasting models
arima_forecasting()
lstm_forecasting()

