

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from kafka import KafkaConsumer, KafkaProducer
import tensorflow as tf
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine
import numpy as np
import json
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# PostgreSQL Config
POSTGRES_CONFIG = {
    "dbname": "finance_db",
    "user": "user",
    "password": "password",
    "host": "postgres",
    "port": "5432"
}

# Kafka Config
KAFKA_BROKER = "ed-kafka:29092"
TOPIC = "raw"
ANOMALY_TOPIC = "anomalies"

# Paths for Model & Scaler
MODEL_PATH = "/app/models/lstm_model.h5"
SCALER_PATH = "/app/models/scaler.npy"

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Initialize Spark session
spark = SparkSession.builder.appName("KafkaSparkConsumer").getOrCreate()

# Define schema
schema = StructType([
    StructField("symbol", StringType(), True),
    StructField("date", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", IntegerType(), True)
])

# Connect to PostgreSQL
engine = create_engine(f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}/{POSTGRES_CONFIG['dbname']}")

# Kafka Consumer
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

batch = []
BATCH_SIZE = 250  # Train LSTM after 100 records

def save_to_postgres(data):
    df = pd.DataFrame(data)
    df.to_sql("stock_data", con=engine, if_exists="append", index=False)

def load_model_and_scaler():
    """Load the trained LSTM model and the MinMaxScaler"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("🚨 No trained model found! Train LSTM first.")
        return None, None
    
    model = load_model(MODEL_PATH)
    scaler = np.load(SCALER_PATH, allow_pickle=True).item()
    return model, scaler


def train_lstm():
    print("🚀 Training LSTM Model with Spark...")

    # Load Data from PostgreSQL
    df = pd.read_sql("SELECT * FROM stock_data ORDER BY date", con=engine)
    df["close"] = df["close"].astype(float)
    data = df["close"].values

    # Normalize Data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare Sequences
    seq_length = 10
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    X, y = np.array(X), np.array(y)

    # Define LSTM Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss="mse", optimizer="adam")
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    # Save model
    model.save(MODEL_PATH)
    np.save(SCALER_PATH, scaler)
    print(f"✅ LSTM Model Saved in {MODEL_PATH}")

# Read and process data
for message in consumer:
    record = message.value
    batch.append(record)
    print(f"Consumed: {record}")

    if len(batch) >= BATCH_SIZE:
        print("🔥 Batch full. Saving to PostgreSQL and training LSTM...")
        save_to_postgres(batch)
        train_lstm()
        # detect_anomalies()  # Detect anomalies after training
        batch = []  # Reset batch after training
