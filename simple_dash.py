import requests
import time
from collections import defaultdict, deque

BASE_URL = "http://localhost:5000/pm"

RISK_THRESHOLD = 0.7
WARNING_THRESHOLD = 0.4

# Store last few risk values per machine
history = defaultdict(lambda: deque(maxlen=5))


def classify_risk(risk):
    if risk >= RISK_THRESHOLD:
        return "🔴 HIGH RISK"
    elif risk >= WARNING_THRESHOLD:
        return "🟡 WARNING"
    else:
        return "🟢 NORMAL"


def is_persistent(machine_id):
    """Check if anomaly persists for last 5 readings"""
    values = history[machine_id]
    if len(values) < 5:
        return False
    return all(r >= RISK_THRESHOLD for r in values)


def is_multisignal_confirmed(info):
    """Check if both Z-score and ML agree"""
    z = info.get("z_scores", {})
    ml = info.get("ml", {})

    z_flag = (
        abs(z.get("temperature", 0)) > 2 or
        abs(z.get("vibration", 0)) > 2 or
        abs(z.get("current", 0)) > 2
    )

    ml_flag = ml.get("error", 0) > 2.0

    return z_flag and ml_flag


def run_monitor():
    print("📡 Smart Monitoring System Started...\n")

    while True:
        try:
            response = requests.get(BASE_URL)
            data = response.json()

            if not data:
                print("No data yet...\n")
                time.sleep(1)
                continue

            print("\n==============================")
            print("📊 MACHINE STATUS")
            print("==============================")

            global_alert = False
            confirmed_alert = False

            for machine_id, info in data.items():
                risk = info.get("risk", 0)

                # Store history
                history[machine_id].append(risk)

                status = classify_risk(risk)

                persistent = is_persistent(machine_id)
                confirmed = is_multisignal_confirmed(info)

                print(f"{machine_id} → Risk: {risk:.2f} | {status}")

                # Show validation info
                if risk >= RISK_THRESHOLD:
                    print(f"   ↳ Persistence: {'YES' if persistent else 'NO'}")
                    print(f"   ↳ Multi-signal: {'YES' if confirmed else 'NO'}")

                # Global anomaly
                if risk >= RISK_THRESHOLD:
                    global_alert = True

                    # Only count as "true anomaly" if validated
                    if persistent or confirmed:
                        confirmed_alert = True

            print("\n------------------------------")

            if confirmed_alert:
                print("🚨 CONFIRMED ANOMALY (Reliable detection)")
            elif global_alert:
                print("⚠️ POSSIBLE ANOMALY (Needs validation)")
            else:
                print("✅ System Stable")

            print("------------------------------")

            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            time.sleep(2)


if __name__ == "__main__":
    run_monitor()