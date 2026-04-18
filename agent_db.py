import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Dict

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from supabase import Client, create_client

# -----------------------------
# CONFIG
# -----------------------------
SUPABASE_URL = "http://100.125.109.107:54321"
SUPABASE_KEY = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"
MACHINE_IDS = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

LEARNING_ENABLED = False
THRESHOLD_FILE = "thresholds.json"

# -----------------------------
# GLOBALS & CLIENTS
# -----------------------------
stop_event = threading.Event()
app = FastAPI()
latest_data: Dict[str, dict] = {}
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Metrics tracking
metrics = {"total": 0, "correct": 0}
# History for smoothing (moving average of risk)
history = defaultdict(lambda: deque(maxlen=10))
# Track the last timestamp processed to avoid double-counting
last_processed_timestamp = defaultdict(lambda: None)

# -----------------------------
# THRESHOLD LOGIC
# -----------------------------
default_thresholds = {
    "CNC_01": {"warning": 0.4, "fault": 0.7},
    "CNC_02": {"warning": 0.5, "fault": 0.60},
    "PUMP_03": {"warning": 0.6, "fault": 0.90},
    "CONVEYOR_04": {"warning": 0.8, "fault": 0.9},
}


def load_thresholds():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            return json.load(f)
    return default_thresholds.copy()


thresholds = load_thresholds()

# -----------------------------
# LOAD MODELS & BASELINES
# -----------------------------
# Ensure these files exist in your local directory
models = {m: joblib.load(f"model_{m}.pkl") for m in MACHINE_IDS}
with open("baselines.json") as f:
    baselines = json.load(f)


# -----------------------------
# UTILS
# -----------------------------
def compute_risk(z_temp, z_vib, z_curr, error):
    risk = (
        0.3 * min(1, abs(z_temp) / 3)
        + 0.3 * min(1, abs(z_vib) / 3)
        + 0.3 * min(1, abs(z_curr) / 3)
        + 0.1 * min(1, error / 2.0)
    )
    return min(risk, 0.95)


def predict_label(machine_id, risk):
    t = thresholds[machine_id]
    if risk > (t["fault"] + 0.05):
        return "fault"
    elif risk > t["warning"]:
        return "warning"
    return "running"


# -----------------------------
# SUPABASE MONITORING TASK
# -----------------------------
def monitor(machine_id):
    print(f"📡 Polling Supabase for {machine_id}")

    model = models[machine_id]
    base = baselines[machine_id]

    while not stop_event.is_set():
        try:
            # Query the latest record from Supabase
            res = (
                supabase.table("machine_telemetry")
                .select("*")
                .eq("machine_id", machine_id)
                .order("recorded_at", desc=True)
                .limit(1)
                .execute()
            )

            if not res.data:
                time.sleep(2)
                continue

            row = res.data[0]
            ts = row["recorded_at"]

            # Only process if it's a new timestamp
            if ts == last_processed_timestamp[machine_id]:
                time.sleep(1)  # Wait for new data
                continue

            last_processed_timestamp[machine_id] = ts

            # Data Extraction (Adjust keys to match your Supabase column names)
            rpm = row["rpm"]
            temp = row["temperature_c"]
            vib = row["vibration_mm_s"]
            curr = row.get("current_a", 0)  # Assuming current might be in table
            status = row.get("status", "running")

            # Z-Score Calculation
            z_temp = (temp - base["mean"]["temperature_C"]) / base["std"][
                "temperature_C"
            ]
            z_vib = (vib - base["mean"]["vibration_mm_s"]) / base["std"][
                "vibration_mm_s"
            ]
            z_curr = (curr - base["mean"]["current_A"]) / base["std"]["current_A"]

            # ML Error Calculation
            X = pd.DataFrame(
                [{"rpm": rpm, "temperature_C": temp, "vibration_mm_s": vib}]
            )
            predicted_curr = model.predict(X)[0]
            error = abs(curr - predicted_curr)

            # Compute and Smooth Risk
            risk = compute_risk(z_temp, z_vib, z_curr, error)

            # Manual Machine Bias
            if machine_id == "CONVEYOR_04":
                risk *= 0.85
            elif machine_id == "CNC_02":
                risk *= 1.1

            history[machine_id].append(risk)
            avg_risk = sum(history[machine_id]) / len(history[machine_id])

            # Prediction logic
            prediction = predict_label(machine_id, avg_risk)

            # Accuracy Metrics
            is_correct = prediction == status
            metrics["total"] += 1
            if is_correct:
                metrics["correct"] += 1
            acc = metrics["correct"] / metrics["total"]

            # Update Global State for FastAPI
            latest_data[machine_id] = {
                "machine_id": machine_id,
                "risk": round(avg_risk, 3),
                "prediction": prediction,
                "actual": status,
                "accuracy": round(acc, 2),
                "timestamp": ts,
            }

            print(
                f"[{machine_id}] Risk: {avg_risk:.2f} | Pred: {prediction} | {'✅' if is_correct else '❌'}"
            )

        except Exception as e:
            print(f"Error monitoring {machine_id}: {e}")

        time.sleep(2)  # Polling interval


# -----------------------------
# API ROUTES
# -----------------------------
@app.get("/stream")
async def stream_all():
    async def gen():
        while not stop_event.is_set():
            yield f"data: {json.dumps(latest_data)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Start FastAPI in background
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=5000), daemon=True
    ).start()

    # Start Supabase Pollers
    for m in MACHINE_IDS:
        threading.Thread(target=monitor, args=(m,), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        print("Shutdown complete.")
