# XNL-21BCE7604-DE-1
Real-Time Stock Prediction &amp; Anomaly Detection
# XNL Innovations - Full-Stack Data Engineering & Data Science Challenge

## Project Overview

XNL Innovations is a global leader in DCIM software, banking software, data analysis, and cloud services. This project is part of the **Full-Stack Data Engineering & Data Science Challenge**, focusing on building an AI-powered real-time analytics platform.

## Features & Requirements

- **Real-Time Data Ingestion**: Handles high-velocity data (1M events/sec) from IoT sensors, social media feeds, and transactional databases.
- **ETL/ELT Pipelines**: Supports both batch and streaming data processing.
- **Real-Time Processing**: Uses Apache Flink, Spark Streaming, or Kafka Streams.
- **Anomaly Detection & Predictive Analytics**: Implements Isolation Forest, DBSCAN, and LSTMs.
- **Data Governance & Security**: Ensures GDPR & CCPA compliance with role-based access control.
- **Data Visualization**: Real-time interactive dashboards using Apache Superset or Power BI.
- **CI/CD Pipeline**: Automates testing and deployment with GitHub Actions.

## Architecture & Technologies

- **Data Ingestion**: Apache Kafka, AWS Kinesis
- **ETL Pipelines**: Apache Airflow, Prefect
- **Storage**: S3, Snowflake, BigQuery
- **Processing**: Apache Flink, Spark Streaming
- **Anomaly Detection**: ML models using TensorFlow, Scikit-learn
- **Visualization**: Apache Superset, Power BI
- **CI/CD**: GitHub Actions, Kubernetes

## Line Graph Implementation

A **line graph** is used to visualize real-time data trends, such as:

- Anomaly detection patterns
- Predictive analytics insights
- Streaming data fluctuations

### Tools Used:

- **Matplotlib** (Python) for static graphs
- **Plotly/D3.js** for interactive graphs
- **Apache Superset/Power BI** for real-time dashboards

### Example Code:

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(100).cumsum()
plt.plot(data, label='Real-time Data Trend')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

## CI/CD Pipeline Setup

The project uses **GitHub Actions** to automate testing, building, and deployment.

### CI/CD Workflow:

1. **Code Push & PR** → Trigger GitHub Actions
2. **Unit & Integration Tests** → PyTest, Great Expectations
3. **Build & Containerization** → Docker, Kubernetes
4. **Deployment** → EKS/GKE using Helm Charts
5. **Monitoring & Logging** → Prometheus, Grafana

### `.github/workflows/main.yml` Sample:

```yaml
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v2
      - name: Install Dependencies
        run: pip install -r requirements.txt
      - name: Run Tests
        run: pytest
      - name: Build & Push Docker Image
        run: |
          docker build -t my-app:latest .
          docker push my-app:latest
      - name: Deploy to Kubernetes
        run: kubectl apply -f k8s/
```
## Contributors & References

- [XNL Innovations](https://www.xnlinnovations.com)
- [Apache Kafka](https://kafka.apache.org/)
- [Apache Superset](https://superset.apache.org/)
- [GitHub Actions](https://docs.github.com/en/actions)

