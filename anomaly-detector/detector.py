import time
import os
import sys
import requests
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://34.233.128.68:9090")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.85"))

print(f"[{datetime.now()}] Anomaly Detector starting...", flush=True)
print(f"[{datetime.now()}] Prometheus URL: {PROMETHEUS_URL}", flush=True)
print(f"[{datetime.now()}] Check Interval: {CHECK_INTERVAL}s", flush=True)

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
        print(f"[{datetime.now()}] Query failed: {e}", flush=True)
        return None

def get_metrics():
    metrics = {
        "cpu": query_prometheus('avg(rate(container_cpu_usage_seconds_total{name=~".*application.*"}[1m]))'),
        "memory": query_prometheus('avg(container_memory_usage_bytes{name=~".*application.*"}) / 1024 / 1024'),
        "network_rx": query_prometheus('sum(rate(container_network_receive_bytes_total{name=~".*application.*"}[1m]))')
    }
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
    history = []
    print(f"[{datetime.now()}] Entering main loop...", flush=True)

    while True:
        try:
            metrics = get_metrics()
            print(f"[{datetime.now()}] Raw metrics: {metrics}", flush=True)

            if None not in metrics.values():
                point = [
                    metrics["cpu"] or 0,
                    metrics["memory"] or 0,
                    metrics["network_rx"] or 0
                ]
                history.append(point)
                if len(history) > 50:
                    history.pop(0)

                is_anomaly, score = detect_anomaly(history)
                print(f"[{datetime.now()}] Score: {score:.4f} | Anomaly: {is_anomaly} | History size: {len(history)}", flush=True)

                if is_anomaly:
                    print(f"[{datetime.now()}] 🚨 ANOMALY DETECTED!", flush=True)
            else:
                print(f"[{datetime.now()}] Waiting for valid metrics from Prometheus...", flush=True)

        except Exception as e:
            print(f"[{datetime.now()}] Unexpected error: {e}", flush=True)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
