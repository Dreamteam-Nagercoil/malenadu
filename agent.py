import requests
import pandas as pd
import joblib
import json
import sseclient
import numpy as np

BASE_URL_RTS = "http://localhost:3000"
#BASE_URL = "http://localhost:5000"

MACHINE_IDS = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

# -----------------------------
# Load models + baselines
# -----------------------------
models = {}
baselines = {}

for m in MACHINE_IDS:
    models[m] = joblib.load(f"models/model_{m}.pkl")

    # Load history for baseline
    data = requests.get(f"{BASE_URL_RTS}/history/{m}").json()
    df = pd.DataFrame(data["readings"])
    df = df[df["status"] == "running"]

    baselines[m] = {
        "mean": df[["temperature_C", "vibration_mm_s", "current_A"]].mean(),
        "std": df[["temperature_C", "vibration_mm_s", "current_A"]].std()
    }

# -----------------------------
# Risk function
# -----------------------------
def compute_risk(z_temp, z_vib, z_curr, error, threshold):
    z_temp = min(1, abs(z_temp)/3)
    z_vib  = min(1, abs(z_vib)/3)
    z_curr = min(1, abs(z_curr)/3)
    ml     = min(1, error/threshold)

    return 0.3*z_temp + 0.3*z_vib + 0.2*z_curr + 0.2*ml


# -----------------------------
# Alert function
# -----------------------------
def send_alert(machine_id, reason, data):
    payload = {
        "machine_id": machine_id,
        "reason": reason,
        "reading": data
    }
    requests.post(f"{BASE_URL_RTS}/alert", json=payload)


# -----------------------------
# Monitor one machine
# -----------------------------
def monitor(machine_id):
    print(f"🔌 Connecting to {machine_id}")

    model = models[machine_id]
    mean  = baselines[machine_id]["mean"]
    std   = baselines[machine_id]["std"]

    response = requests.get(f"{BASE_URL_RTS}/stream/{machine_id}", stream=True)
    client = sseclient.SSEClient(response)

    for event in client.events():
        try:
            data = json.loads(event.data)

            rpm = data["rpm"]
            temp = data["temperature_C"]
            vib = data["vibration_mm_s"]
            curr = data["current_A"]

            # -----------------------------
            # Z-SCORE
            # -----------------------------
            z_temp = (temp - mean["temperature_C"]) / std["temperature_C"]
            z_vib  = (vib  - mean["vibration_mm_s"]) / std["vibration_mm_s"]
            z_curr = (curr - mean["current_A"]) / std["current_A"]

            # -----------------------------
            # ML Prediction
            # -----------------------------
            X = pd.DataFrame([{
                "rpm": rpm,
                "temperature_C": temp,
                "vibration_mm_s": vib
            }])

            pred = model.predict(X)[0]
            error = abs(curr - pred)

            # -----------------------------
            # Risk Score
            # -----------------------------
            risk = compute_risk(z_temp, z_vib, z_curr, error, threshold=2.0)

            print(f"[{machine_id}] Risk={risk:.2f} | Error={error:.2f}")

            # -----------------------------
            # ALERT
            # -----------------------------
            if risk > 0.7:
                reason = (
                    f"High risk detected. "
                    f"Temp z={z_temp:.2f}, Vib z={z_vib:.2f}, "
                    f"Error={error:.2f}"
                )
                send_alert(machine_id, reason, data)

        except Exception as e:
            print("Error:", e)


# -----------------------------
# RUN ALL MACHINES
# -----------------------------
if __name__ == "__main__":
    import threading

    threads = []

    for m in MACHINE_IDS:
        t = threading.Thread(target=monitor, args=(m,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()