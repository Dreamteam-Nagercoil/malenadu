import requests
import pandas as pd
import joblib
import json
import sseclient
import threading

BASE_URL = "http://localhost:3000"

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
# ALERT FUNCTION (Node server format)
# -----------------------------
def send_alert(machine_id, reason, data):
    payload = {
        "machine_id": machine_id,
        "reason": reason,
        "reading": data
    }
    try:
        res = requests.post(f"{BASE_URL}/alert", json=payload)
        print(f"🚨 ALERT SENT [{machine_id}]")
    except Exception as e:
        print("Alert error:", e)


# -----------------------------
# MONITOR FUNCTION
# -----------------------------
def monitor(machine_id):
    print(f"🔌 Monitoring {machine_id}")

    model = models[machine_id]

    mean = baselines[machine_id]["mean"]
    std  = baselines[machine_id]["std"]

    response = requests.get(f"{BASE_URL}/stream/{machine_id}", stream=True)
    client = sseclient.SSEClient(response)

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
            # ALERT
            # -----------------------------
            if risk > 0.7:
                reason = (
                    f"High risk detected. "
                    f"Temp deviation={z_temp:.2f}, "
                    f"Vibration deviation={z_vib:.2f}, "
                    f"Prediction error={error:.2f}"
                )

                send_alert(machine_id, reason, data)

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