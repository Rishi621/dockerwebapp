import time
import os
import requests
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.85"))

def query_prometheus(query):
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10
        )
        data = response.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
        return None
    except Exception as e:
        print(f"[{datetime.now()}] Query failed: {e}")
        return None

def get_metrics():
    metrics = {}
    metrics["cpu"] = query_prometheus('avg(rate(container_cpu_usage_seconds_total{name=~".*application.*"}[1m]))')
    metrics["memory"] = query_prometheus('avg(container_memory_usage_bytes{name=~".*application.*"}) / 1024 / 1024')
    metrics["network_rx"] = query_prometheus('sum(rate(container_network_receive_bytes_total{name=~".*application.*"}[1m]))')
    return metrics

def detect_anomaly(history):
    if len(history) < 10:
        return False, 0.0

    data = np.array(history)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data)
    scores = model.decision_function(data)
    latest_score = scores[-1]
    is_anomaly = latest_score < -ANOMALY_THRESHOLD
    return is_anomaly, latest_score

def main():
    print("Anomaly Detector started...")
    history = []

    while True:
        metrics = get_metrics()
        if None not in metrics.values():
            point = [metrics["cpu"] or 0, metrics["memory"] or 0, metrics["network_rx"] or 0]
            history.append(point)
            if len(history) > 50:
                history.pop(0)

            is_anomaly, score = detect_anomaly(history)
            print(f"[{datetime.now()}] Metrics: {metrics} | Score: {score:.4f} | Anomaly: {is_anomaly}")

            if is_anomaly:
                print("🚨 ANOMALY DETECTED! Triggering remediation...")
                # Here you can call Jenkins webhook or execute remediation
                # Example: requests.post("http://jenkins:8080/generic-webhook-trigger/invoke?token=YOUR_TOKEN")
        else:
            print(f"[{datetime.now()}] Waiting for metrics...")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
