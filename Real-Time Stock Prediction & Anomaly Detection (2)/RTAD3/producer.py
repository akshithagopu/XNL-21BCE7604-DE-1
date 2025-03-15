from kafka import KafkaProducer
import json
import yfinance as yf
import time

KAFKA_BROKER = "localhost:9092" # 9092 when sending from outside docker 
TOPIC = "raw"

def fetch_stock_data(symbol="GOOG"):
    stock = yf.Ticker(symbol)
    data = stock.history(period="7d")
    if not data.empty:
        return {
            "symbol": symbol,
            "date": str(data.index[-1]),
            "open": data["Open"].iloc[-1],
            "high": data["High"].iloc[-1],
            "low": data["Low"].iloc[-1],
            "close": data["Close"].iloc[-1],
            "volume": int(data["Volume"].iloc[-1])
        }
    return None

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

while True:
    data = fetch_stock_data()
    if data:
        print(f"Producing: {data}")
        producer.send(TOPIC, data)
    time.sleep(0)  # Fetch data every 5 seconds

