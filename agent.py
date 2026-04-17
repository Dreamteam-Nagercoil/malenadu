import requests
import pandas as pd
import joblib
import json
import sseclient
import threading
import uvicorn
import time
import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Dict

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL_RLS = "http://localhost:3000"
BASE_URL_OUT = "http://localhost:5000"

MACHINE_IDS = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

# -----------------------------
# GLOBAL STOP FLAG
# -----------------------------
stop_event = threading.Event()

# -----------------------------
# FASTAPI SERVER
# -----------------------------
app = FastAPI()

latest_data: Dict[str, dict] = {}

@app.post("/pm/{machine_id}")
async def receive_data(machine_id: str, payload: dict):
    latest_data[machine_id] = payload
    print(f"📥 {machine_id} | Risk={payload['risk']:.2f}")
    return {"status": "ok"}

@app.get("/pm")
async def get_all():
    return latest_data

@app.get("/pm/{machine_id}")
async def get_machine(machine_id: str):
    return latest_data.get(machine_id, {"error": "No data yet"})

# -----------------------------
# 🔥 ALL MACHINES STREAM
# -----------------------------
async def event_generator():
    while not stop_event.is_set():
        try:
            yield f"data: {json.dumps(latest_data)}\n\n"
            await asyncio.sleep(1)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(1)

@app.get("/stream")
async def stream():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )

# -----------------------------
# 🔥 PER-MACHINE STREAM
# -----------------------------
async def machine_stream_generator(machine_id):
    while not stop_event.is_set():
        try:
            data = latest_data.get(machine_id, {})
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(1)

@app.get("/stream/{machine_id}")
async def stream_machine(machine_id: str):
    return StreamingResponse(
        machine_stream_generator(machine_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )

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

    try:
        client = sseclient.SSEClient(url)

        for event in client:
            if stop_event.is_set():
                print(f"🛑 Stopping {machine_id}")
                break

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

                # ML prediction
                X = pd.DataFrame([{
                    "rpm": rpm,
                    "temperature_C": temp,
                    "vibration_mm_s": vib
                }])

                predicted = model.predict(X)[0]
                error = abs(curr - predicted)

                # Risk
                risk = compute_risk(z_temp, z_vib, z_curr, error, threshold=2.0)

                print(f"[{machine_id}] Risk={risk:.2f} | Err={error:.2f}")

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
                print(f"[{machine_id}] Processing error:", e)

    except Exception as e:
        print(f"[{machine_id}] Stream error:", e)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    threads = []

    try:
        # Start FastAPI server
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        print("🚀 FastAPI running on http://localhost:5000")
        print("📡 All machines stream → http://localhost:5000/stream")
        print("📡 Per machine stream → http://localhost:5000/stream/{machine_id}")

        # Start monitoring
        for m in MACHINE_IDS:
            t = threading.Thread(target=monitor, args=(m,), daemon=True)
            t.start()
            threads.append(t)

        # Keep alive
        while not stop_event.is_set():
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")

        stop_event.set()

        for t in threads:
            t.join(timeout=2)

        print("✅ Shutdown complete")