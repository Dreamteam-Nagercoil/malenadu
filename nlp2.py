import json
import time

import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from supabase import Client, create_client

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
SUPABASE_URL = "http://100.125.109.107:54321"
SUPABASE_KEY = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"
MODEL_NAME = "qwen3:4b"  # Ensure this matches 'ollama list'

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

THRESHOLDS = {
    "CNC_01": {"warning": 0.4, "fault": 0.7},
    "CNC_02": {"warning": 0.70, "fault": 0.85},  # Lowered CNC_02 warning
    "PUMP_03": {"warning": 0.54, "fault": 0.9},
    "CONVEYOR_04": {"warning": 0.78, "fault": 0.9},
}

# --- STATE TRACKING ---
last_risks = {}
last_explanations = {}
last_telemetry = {}


def talk_to_qwen(prompt, system=None):
    payload = {
        "model": MODEL_NAME,
        "messages": [],
        "stream": False,
        "options": {"num_predict": 150, "temperature": 0.1},
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=10,
        )
        return response.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        return ""


def get_alert_data():
    output = []
    machine_ids = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

    for m_id in machine_ids:
        try:
            tel = (
                supabase.table("machine_telemetry")
                .select("*")
                .eq("machine_id", m_id)
                .order("recorded_at", desc=True)
                .limit(1)
                .execute()
            )
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

                # Logic: If risk is high OR status is not 'running', trigger alert
                is_anomaly = risk >= limit["warning"] or tel_data["status"] in [
                    "fault",
                    "warning",
                ]

                if is_anomaly:
                    alert_flag = 1
                    # Only call AI if risk changed significantly or we have no message
                    prev_risk = last_risks.get(m_id, 0)
                    if abs(prev_risk - risk) > 0.05 or not last_explanations.get(m_id):
                        prompt = f"Analyze {m_id}: Risk {risk}, Status {tel_data['status']}, Temp {tel_data['temperature_c']}C, Vibration {tel_data['vibration_mm_s']}mm/s. Explain cause and fix in 15 words."
                        ai_msg = talk_to_qwen(prompt)

                        # Fallback if AI is empty or failing
                        if not ai_msg:
                            ai_msg = f"High risk ({risk}) detected. Unusual {tel_data['status']} state with vibration at {tel_data['vibration_mm_s']}mm/s. Inspect hardware."

                        last_explanations[m_id] = ai_msg

                    explanation = last_explanations[m_id]
                else:
                    alert_flag = 0
                    explanation = "no anomalies"
                    last_explanations[m_id] = None

                last_risks[m_id] = risk
                last_telemetry[m_id] = tel_data

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
                output.append(
                    {"machine_id": m_id, "alert": 0, "risk": 0, "message": "no data"}
                )
        except Exception as e:
            output.append(
                {
                    "machine_id": m_id,
                    "alert": 0,
                    "risk": 0,
                    "message": f"Error: {str(e)}",
                }
            )

    return output


@app.route("/alert", methods=["GET"])
def stream_alerts():
    def event_stream():
        while True:
            yield f"data: {json.dumps(get_alert_data())}\n\n"
            time.sleep(2)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")

    # Build live context from last known telemetry + risks
    context_lines = []
    for m_id in ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]:
        tel = last_telemetry.get(m_id)
        risk = last_risks.get(m_id, "unknown")
        if tel:
            context_lines.append(
                f"{m_id}: status={tel['status']}, temp={tel['temperature_c']}C, "
                f"vib={tel['vibration_mm_s']}mm/s, rpm={tel['rpm']}, risk={risk}"
            )
        else:
            context_lines.append(f"{m_id}: no data yet")

    system_prompt = (
        "You are NeuralGuard, an AI assistant for a predictive maintenance dashboard. "
        "You monitor 4 machines: CNC_01, CNC_02, PUMP_03, CONVEYOR_04. "
        "Current live telemetry:\n" + "\n".join(context_lines) + "\n"
        "Answer concisely and technically. If asked about a machine, use the telemetry above."
    )

    answer = talk_to_qwen(user_msg, system=system_prompt)
    return jsonify({"reply": answer or "I'm having trouble connecting to my brain."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969, threaded=True)
