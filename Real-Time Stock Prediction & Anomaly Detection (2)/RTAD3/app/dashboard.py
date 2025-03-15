

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# PostgreSQL Config
POSTGRES_CONFIG = {
    "dbname": "finance_db",
    "user": "user",
    "password": "password",
    "host": "postgres",
    "port": "5432"
}

# Paths for Model & Scaler
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.npy"

# Connect to PostgreSQL
engine = create_engine(f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}/{POSTGRES_CONFIG['dbname']}")

# Load Model & Scaler
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("🚨 No trained model found! Train LSTM first.")
        return None, None

    from tensorflow.keras.metrics import MeanSquaredError
    model = load_model(
            MODEL_PATH,
            custom_objects={'mse': MeanSquaredError()}
        )
    scaler = np.load(SCALER_PATH, allow_pickle=True).item()
    return model, scaler

# Function to Fetch Data
def get_latest_stock_data():
    query = "SELECT * FROM stock_data ORDER BY date DESC LIMIT 50"
    df = pd.read_sql(query, con=engine)
    df["close"] = df["close"].astype(float)
    return df

# Function to Detect Anomalies
def detect_anomalies(model, scaler, df):
    data = df["close"].values.reshape(-1, 1)
    normalized_data = scaler.transform(data)
    
    seq_length = 10
    X = np.array([normalized_data[i:i+seq_length] for i in range(len(normalized_data) - seq_length)])

    preds = model.predict(X)
    preds = scaler.inverse_transform(preds)

    actual_prices = df["close"].values[seq_length:]
    mean_price = np.mean(actual_prices)
    std_dev = np.std(actual_prices)
    threshold = 3 * std_dev  # Anomaly threshold

    anomalies = []
    for i, (actual, pred) in enumerate(zip(actual_prices, preds.flatten())):
        error = abs(actual - pred)
        if error > threshold:
            anomalies.append({
                "date": df.iloc[i + seq_length]["date"],
                "actual_price": actual,
                "predicted_price": float(pred),
                "error": float(error)
            })
    
    return anomalies, preds.flatten()

# Streamlit UI
st.title("📊 Real-Time Stock Prediction & Anomaly Detection")

model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    df = get_latest_stock_data()
    anomalies, predictions = detect_anomalies(model, scaler, df)

    st.subheader("📈 Stock Price Trends")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["close"], label="Actual Prices", color="blue")
    ax.plot(df["date"].iloc[10:], predictions, label="Predicted Prices", linestyle="dashed", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🚨 Anomalies Detected")
    if anomalies:
        st.warning(f"{len(anomalies)} anomalies found!")
        anomaly_df = pd.DataFrame(anomalies)
        st.dataframe(anomaly_df)
    else:
        st.success("✅ No anomalies detected.")

else:
    st.error("❌ Model is not available. Train the model first.")
