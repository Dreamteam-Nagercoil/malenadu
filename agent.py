import requests
import pandas as pd
import joblib
import json
import sseclient
import threading
import uvicorn

from fastapi import FastAPI
from typing import Dict

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL_RLS = "http://localhost:3000"
BASE_URL_OUT = "http://localhost:5000"

MACHINE_IDS = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

# -----------------------------
# FASTAPI RECEIVER
# -----------------------------
app = FastAPI()

latest_data: Dict[str, dict] = {}

@app.post("/pm/{machine_id}")
async def receive_data(machine_id: str, payload: dict):
    latest_data[machine_id] = payload
    print(f"📥 Received {machine_id} | Risk={payload['risk']:.2f}")
    return {"status": "ok"}

@app.get("/pm")
async def get_all():
    return latest_data

@app.get("/pm/{machine_id}")
async def get_machine(machine_id: str):
    return latest_data.get(machine_id, {"error": "No data yet"})


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="warning")


# -----------------------------
# LOAD MODELS
# -----------------------------
models = {}
for m in MACHINE_IDS:
    models[m] = joblib.load(f"model_{m}.pkl")

# -----------------------------
# LOAD BASELINES
# -----------------------------
with open("baselines.json") as f:
    baselines = json.load(f)


# -----------------------------
# SEND METRICS
# -----------------------------
def send_metrics(machine_id, payload):
    try:
        url = f"{BASE_URL_OUT}/pm/{machine_id}"
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        print(f"[{machine_id}] Send error:", e)


# -----------------------------
# RISK FUNCTION
# -----------------------------
def compute_risk(z_temp, z_vib, z_curr, error, threshold):
    z_temp_score = min(1.0, abs(z_temp) / 3)
    z_vib_score  = min(1.0, abs(z_vib) / 3)
    z_curr_score = min(1.0, abs(z_curr) / 3)
    ml_score     = min(1.0, error / threshold)

    return (
        0.3 * z_temp_score +
        0.3 * z_vib_score +
        0.2 * z_curr_score +
        0.2 * ml_score
    )


# -----------------------------
# MONITOR FUNCTION
# -----------------------------
def monitor(machine_id):
    print(f"🔌 Monitoring {machine_id}")

    model = models[machine_id]
    mean  = baselines[machine_id]["mean"]
    std   = baselines[machine_id]["std"]

    url = f"{BASE_URL_RLS}/stream/{machine_id}"
    client = sseclient.SSEClient(url)

    for event in client:
        try:
            if not event.data:
                continue

            data = json.loads(event.data)

            rpm  = data["rpm"]
            temp = data["temperature_C"]
            vib  = data["vibration_mm_s"]
            curr = data["current_A"]

            # Z-score
            z_temp = (temp - mean["temperature_C"]) / std["temperature_C"]
            z_vib  = (vib  - mean["vibration_mm_s"]) / std["vibration_mm_s"]
            z_curr = (curr - mean["current_A"])     / std["current_A"]

            # ML
            X = pd.DataFrame([{
                "rpm": rpm,
                "temperature_C": temp,
                "vibration_mm_s": vib
            }])

            predicted = model.predict(X)[0]
            error = abs(curr - predicted)

            # Risk
            risk = compute_risk(z_temp, z_vib, z_curr, error, threshold=2.0)

            print(f"[{machine_id}] Risk={risk:.2f}")

            payload = {
                "machine_id": machine_id,
                "timestamp": data["timestamp"],
                "z_scores": {
                    "temperature": z_temp,
                    "vibration": z_vib,
                    "current": z_curr
                },
                "ml": {
                    "predicted_current": predicted,
                    "actual_current": curr,
                    "error": error
                },
                "risk": risk,
                "raw": data
            }

            # async send
            threading.Thread(
                target=send_metrics,
                args=(machine_id, payload),
                daemon=True
            ).start()

        except Exception as e:
            print(f"[{machine_id}] Error:", e)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Start FastAPI server
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print("🚀 FastAPI running on http://localhost:5000")

    # Start monitoring
    threads = []
    for m in MACHINE_IDS:
        t = threading.Thread(target=monitor, args=(m,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()