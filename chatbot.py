import json
import time

import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from supabase import Client, create_client

app = Flask(__name__)
# Corrected CORS configuration
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIG ---
SUPABASE_URL = "http://100.125.109.107:54321"
SUPABASE_KEY = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"
MODEL_NAME = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Machine specific settings from your original script
THRESHOLDS = {
    "CNC_01": {"warning": 0.4, "fault": 0.7},
    "CNC_02": {"warning": 0.70, "fault": 0.85},
    "PUMP_03": {"warning": 0.54, "fault": 0.9},
    "CONVEYOR_04": {"warning": 0.78, "fault": 0.9},
}

# State tracking for AI messages
last_explanations = {}


def talk_to_qwen(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 100, "temperature": 0.1},
            },
            timeout=40,
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"AI Error: {e}")
        return ""


def get_realtime_data():
    output = []
    machine_ids = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

    for m_id in machine_ids:
        try:
            # Fetch latest telemetry
            tel = (
                supabase.table("machine_telemetry")
                .select("*")
                .eq("machine_id", m_id)
                .order("recorded_at", desc=True)
                .limit(1)
                .execute()
            )
            # Fetch latest AI prediction
            pred = (
                supabase.table("model_predictions")
                .select("*")
                .eq("machine_id", m_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            if tel.data and pred.data:
                tel_data = tel.data[0]
                risk = round(float(pred.data[0]["risk_score"]), 2)
                limit = THRESHOLDS[m_id]

                # Logic for Alerting
                is_anomaly = risk >= limit["warning"] or tel_data["status"] in [
                    "fault",
                    "warning",
                ]

                if is_anomaly:
                    alert_flag = 1
                    # Only call AI if we don't already have an explanation to save time
                    if not last_explanations.get(m_id):
                        prompt = f"Analyze {m_id}: Risk {risk}, Status {tel_data['status']}, Temp {tel_data['temperature_c']}C. Explain cause in 10 words."
                        last_explanations[m_id] = talk_to_qwen(prompt)
                    explanation = (
                        last_explanations[m_id]
                        or "High risk detected. Inspect hardware."
                    )
                else:
                    alert_flag = 0
                    explanation = "System operating within normal parameters."
                    last_explanations[m_id] = None

                output.append(
                    {
                        "machine_id": m_id,
                        "alert": alert_flag,
                        "risk": risk,
                        "message": explanation,
                        "telemetry": {
                            "temperature": tel_data["temperature_c"],
                            "rpm": tel_data["rpm"],
                            "vibration": tel_data["vibration_mm_s"],
                            "status": tel_data["status"],
                        },
                    }
                )
            else:
                # Placeholder if Supabase has no data for this specific machine
                output.append(
                    {
                        "machine_id": m_id,
                        "alert": 0,
                        "risk": 0,
                        "message": "No data in Supabase",
                        "telemetry": {
                            "temperature": 0,
                            "rpm": 0,
                            "vibration": 0,
                            "status": "offline",
                        },
                    }
                )
        except Exception as e:
            print(f"Error fetching {m_id}: {e}")

    return output


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")
    answer = talk_to_qwen(f"User: {user_msg}\nAssistant:")
    return jsonify({"reply": answer or "I'm having trouble connecting to my brain."})


@app.route("/alert", methods=["GET"])
def stream_alerts():
    def event_stream():
        while True:
            data = get_realtime_data()
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(2)  # Refresh every 2 seconds

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("--- NeuralGuard Backend Active (All 4 Machines) ---")
    app.run(host="0.0.0.0", port=6969, threaded=True)
