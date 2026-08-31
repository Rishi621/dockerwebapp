import time
import os
import sys
import requests
import numpy as np
import subprocess
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "15"))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.05"))
SERVICE_NAME = "myapp_application"

print(f"[{datetime.now()}] Anomaly Detector starting (Pure IsolationForest + Normalization)...", flush=True)
print(f"[{datetime.now()}] Prometheus URL: {PROMETHEUS_URL}", flush=True)
print(f"[{datetime.now()}] Check Interval: {CHECK_INTERVAL}s", flush=True)
print(f"[{datetime.now()}] Anomaly Threshold: {ANOMALY_THRESHOLD}", flush=True)

def query_prometheus(query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=10)
        data = response.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
        return None
    except Exception as e:
        print(f"[{datetime.now()}] Query failed: {e}", flush=True)
        return None

def get_metrics():
    return {
        "cpu": query_prometheus('avg(rate(container_cpu_usage_seconds_total{image=~".*appimage.*"}[1m]))'),
        "memory": query_prometheus('avg(container_memory_usage_bytes{image=~".*appimage.*"}) / 1024 / 1024'),
        "network_rx": query_prometheus('sum(rate(container_network_receive_bytes_total{image=~".*appimage.*"}[1m]))')
    }

def detect_anomaly(history):
    if len(history) < 8:
        return False, 0.0

    data = np.array(history)

    # Normalize features (this is the key improvement)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    model = IsolationForest(
        contamination=0.20,      # Higher contamination = more sensitive
        random_state=42,
        n_estimators=100
    )
    model.fit(data_scaled)

    scores = model.decision_function(data_scaled)
    latest_score = scores[-1]

    is_anomaly = latest_score < -ANOMALY_THRESHOLD
    return is_anomaly, latest_score

def restart_service():
    try:
        print(f"[{datetime.now()}] 🔄 Executing self-healing: Restarting {SERVICE_NAME}...", flush=True)
        result = subprocess.run(
            ["docker", "service", "update", "--force", SERVICE_NAME],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print(f"[{datetime.now()}] ✅ Successfully restarted {SERVICE_NAME}", flush=True)
        else:
            print(f"[{datetime.now()}] ❌ Failed: {result.stderr}", flush=True)
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error: {e}", flush=True)

def main():
    history = []
    last_action_time = 0
    cooldown = 180

    print(f"[{datetime.now()}] Entering main loop...", flush=True)

    while True:
        try:
            metrics = get_metrics()
            print(f"[{datetime.now()}] Raw metrics: {metrics}", flush=True)

            if None not in metrics.values():
                point = [metrics["cpu"] or 0, metrics["memory"] or 0, metrics["network_rx"] or 0]
                history.append(point)
                if len(history) > 50:
                    history.pop(0)

                is_anomaly, score = detect_anomaly(history)
                print(f"[{datetime.now()}] Score: {score:.4f} | Anomaly: {is_anomaly} | History size: {len(history)}", flush=True)

                if is_anomaly:
                    now = time.time()
                    if now - last_action_time > cooldown:
                        print(f"[{datetime.now()}] 🚨 ANOMALY DETECTED! Triggering self-healing...", flush=True)
                        restart_service()
                        last_action_time = now
                    else:
                        print(f"[{datetime.now()}] 🚨 Anomaly detected but in cooldown.", flush=True)
            else:
                print(f"[{datetime.now()}] Waiting for complete metrics...", flush=True)

        except Exception as e:
            print(f"[{datetime.now()}] Unexpected error: {e}", flush=True)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
