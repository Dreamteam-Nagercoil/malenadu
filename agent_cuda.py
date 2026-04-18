import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Dict

# GPU Specific Imports
import cupy as cp
import joblib
import pandas as pd
import sseclient
import uvicorn
from cuml import ForestInference
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL_RLS = "http://localhost:3000"
MACHINE_IDS = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

# 🔥 Toggle learning
LEARNING_ENABLED = False

# 🔥 Threshold file
THRESHOLD_FILE = "thresholds.json"

# -----------------------------
# GLOBALS
# -----------------------------
stop_event = threading.Event()
app = FastAPI()
latest_data: Dict[str, dict] = {}

# -----------------------------
# METRICS
# -----------------------------
metrics = {"total": 0, "correct": 0, "per_machine": {}}

# -----------------------------
# DEFAULT THRESHOLDS
# -----------------------------
default_thresholds = {
    "CNC_01": {"warning": 0.4, "fault": 0.7},
    "CNC_02": {"warning": 0.5, "fault": 0.60},
    "PUMP_03": {"warning": 0.6, "fault": 0.90},
    "CONVEYOR_04": {"warning": 0.8, "fault": 0.9},
}


# -----------------------------
# LOAD / SAVE THRESHOLDS
# -----------------------------
def load_thresholds():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            print("📂 Loaded saved thresholds")
            return json.load(f)
    print("⚠️ Using default thresholds")
    return default_thresholds.copy()


def save_thresholds(thresholds):
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(thresholds, f, indent=4)


# Initialize thresholds
thresholds = load_thresholds()

# -----------------------------
# HISTORY (SMOOTHING)
# -----------------------------
history = defaultdict(lambda: deque(maxlen=10))


# -----------------------------
# API
# -----------------------------
@app.get("/pm")
async def get_all():
    return latest_data


@app.get("/metrics")
async def get_metrics():
    return metrics


@app.get("/stream")
async def stream_all():
    async def gen():
        while not stop_event.is_set():
            yield f"data: {json.dumps(latest_data)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream")


# -----------------------------
# LOAD MODELS (GPU CONVERSION)
# -----------------------------
models = {}
for m in MACHINE_IDS:
    # Load the scikit-learn model from disk
    cpu_model = joblib.load(f"model_{m}.pkl")
    # Wrap and move the model to GPU VRAM using Forest Inference Library (FIL)
    models[m] = ForestInference.load_from_sklearn(cpu_model)
    print(f"✅ Model {m} moved to GPU")

# -----------------------------
# LOAD BASELINES
# -----------------------------
with open("baselines.json") as f:
    baselines = json.load(f)


# -----------------------------
# RISK FUNCTION
# -----------------------------
def compute_risk(z_temp, z_vib, z_curr, error):
    risk = (
        0.3 * min(1, abs(z_temp) / 3)
        + 0.3 * min(1, abs(z_vib) / 3)
        + 0.3 * min(1, abs(z_curr) / 3)
        + 0.1 * min(1, error / 2.0)
    )
    return min(risk, 0.95)


# -----------------------------
# PREDICTION
# -----------------------------
def predict_label(machine_id, risk):
    t = thresholds[machine_id]

    if risk > (t["fault"] + 0.05):
        return "fault"
    elif risk > t["warning"]:
        return "warning"
    else:
        return "running"


# -----------------------------
# MONITOR
# -----------------------------
def monitor(machine_id):
    print(f"🔌 Monitoring {machine_id} on NVIDIA GPU")

    model = models[machine_id]
    mean = baselines[machine_id]["mean"]
    std = baselines[machine_id]["std"]

    client = sseclient.SSEClient(f"{BASE_URL_RLS}/stream/{machine_id}")

    for event in client:
        if stop_event.is_set():
            break

        try:
            if not event.data:
                continue

            data = json.loads(event.data)

            rpm = data["rpm"]
            temp = data["temperature_C"]
            vib = data["vibration_mm_s"]
            curr = data["current_A"]
            status = data.get("status", "unknown")

            # Z-score (Standardization)
            z_temp = (temp - mean["temperature_C"]) / std["temperature_C"]
            z_vib = (vib - mean["vibration_mm_s"]) / std["vibration_mm_s"]
            z_curr = (curr - mean["current_A"]) / std["current_A"]

            # GPU DATA PREP
            # Convert inputs to float32 CuPy array (GPU resident)
            X_gpu = cp.array([[rpm, temp, vib]], dtype=cp.float32)

            # GPU PREDICTION
            # Inference happens on CUDA cores
            predicted_gpu = model.predict(X_gpu)
            predicted = float(predicted_gpu[0])  # Pull back single scalar for logic

            error = abs(curr - predicted)

            # Risk
            risk = compute_risk(z_temp, z_vib, z_curr, error)

            # Machine tuning
            if machine_id == "CONVEYOR_04":
                risk *= 0.85
            elif machine_id == "CNC_02":
                risk *= 1.1

            # Smoothing
            history[machine_id].append(risk)
            avg_risk = sum(history[machine_id]) / len(history[machine_id])

            # Prediction
            prediction = predict_label(machine_id, avg_risk)

            # Persistence-based fault detection (CNC_02)
            if machine_id == "CNC_02":
                recent = list(history[machine_id])
                if len(recent) >= 5 and all(r > 0.85 for r in recent[-5:]):
                    prediction = "fault"

            # Evaluation
            is_correct = prediction == status

            metrics["total"] += 1
            if is_correct:
                metrics["correct"] += 1

            acc = metrics["correct"] / metrics["total"]

            # LEARNING (OPTIONAL)
            if LEARNING_ENABLED:
                t = thresholds[machine_id]

                if not is_correct:
                    if status == "fault":
                        t["fault"] -= 0.02
                    elif status == "running" and prediction == "fault":
                        t["fault"] += 0.02
                    elif status == "warning" and avg_risk < t["warning"]:
                        t["warning"] -= 0.02
                    elif status == "running" and prediction == "warning":
                        t["warning"] += 0.02

                t["fault"] = max(0.6, min(0.95, t["fault"]))
                t["warning"] = max(0.3, min(0.9, t["warning"]))

                if metrics["total"] % 20 == 0:
                    save_thresholds(thresholds)

            # Print
            print(
                f"[{machine_id}] Risk={avg_risk:.2f} | "
                f"Pred={prediction} | Actual={status} | "
                f"{'✅' if is_correct else '❌'} | Acc={acc:.2f}"
            )

            # Store
            latest_data[machine_id] = {
                "machine_id": machine_id,
                "risk": avg_risk,
                "prediction": prediction,
                "actual": status,
                "accuracy": acc,
                "learning": LEARNING_ENABLED,
            }

        except Exception as e:
            print(f"[{machine_id}] Error:", e)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        threading.Thread(
            target=lambda: uvicorn.run(app, host="0.0.0.0", port=5000), daemon=True
        ).start()

        print("🚀 GPU-Accelerated Server → http://localhost:5000")
        print(f"🧠 Learning Mode: {'ON' if LEARNING_ENABLED else 'OFF'}")

        for m in MACHINE_IDS:
            threading.Thread(target=monitor, args=(m,), daemon=True).start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        stop_event.set()
        save_thresholds(thresholds)
        print("💾 Thresholds saved")
