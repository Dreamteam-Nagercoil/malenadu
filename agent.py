import requests
import pandas as pd
import joblib
import json
import sseclient
import threading

BASE_URL_RLS = "http://localhost:3000"
BASE_URL_OUT = "http://localhost:5000"


MACHINE_IDS = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

# -----------------------------
# LOAD MODELS
# -----------------------------
models = {}
for m in MACHINE_IDS:
    models[m] = joblib.load(f"models/model_{m}.pkl")

# -----------------------------
# LOAD BASELINES (NEW)
# -----------------------------
with open("baselines.json") as f:
    baselines = json.load(f)


# -----------------------------
# Sender
# -----------------------------
def send_metrics(machine_id, payload):
    try:
        url = f"{BASE_URL_OUT}/pm/{machine_id}"
        requests.post(url, json=payload, timeout=1)
    except Exception as e:
        print(f"[{machine_id}] Output send error:", e)

# -----------------------------
# RISK FUNCTION
# -----------------------------
def compute_risk(z_temp, z_vib, z_curr, error, threshold):
    z_temp_score = min(1.0, abs(z_temp) / 3)
    z_vib_score  = min(1.0, abs(z_vib) / 3)
    z_curr_score = min(1.0, abs(z_curr) / 3)
    ml_score     = min(1.0, error / threshold)

    risk = (
        0.3 * z_temp_score +
        0.3 * z_vib_score +
        0.2 * z_curr_score +
        0.2 * ml_score
    )
    return risk


# -----------------------------
# MONITOR FUNCTION
# -----------------------------
for event in client.events():
    try:
        if not event.data:
            continue

        data = json.loads(event.data)

        rpm  = data["rpm"]
        temp = data["temperature_C"]
        vib  = data["vibration_mm_s"]
        curr = data["current_A"]

        # -----------------------------
        # Z-SCORE
        # -----------------------------
        z_temp = (temp - mean["temperature_C"]) / std["temperature_C"]
        z_vib  = (vib  - mean["vibration_mm_s"]) / std["vibration_mm_s"]
        z_curr = (curr - mean["current_A"])     / std["current_A"]

        # -----------------------------
        # ML PREDICTION
        # -----------------------------
        X = pd.DataFrame([{
            "rpm": rpm,
            "temperature_C": temp,
            "vibration_mm_s": vib
        }])

        predicted = model.predict(X)[0]
        error = abs(curr - predicted)

        # -----------------------------
        # RISK SCORE
        # -----------------------------
        risk = compute_risk(z_temp, z_vib, z_curr, error, threshold=2.0)

        print(
            f"[{machine_id}] "
            f"Risk={risk:.2f} | "
            f"TempZ={z_temp:.2f} VibZ={z_vib:.2f} | "
            f"Err={error:.2f}"
        )

        # -----------------------------
        # BUILD OUTPUT JSON
        # -----------------------------
        output_payload = {
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

            "raw": {
                "rpm": rpm,
                "temperature_C": temp,
                "vibration_mm_s": vib,
                "current_A": curr
            }
        }

        # -----------------------------
        # SEND TO OUTPUT SERVER
        # -----------------------------
        send_metrics(machine_id, output_payload)

    except Exception as e:
        print("Processing error:", e)

# -----------------------------
# RUN MULTI-MACHINE
# -----------------------------
if __name__ == "__main__":
    threads = []

    for m in MACHINE_IDS:
        t = threading.Thread(target=monitor, args=(m,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()