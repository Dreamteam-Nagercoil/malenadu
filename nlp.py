import json

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from supabase import Client, create_client

app = Flask(__name__)

CORS(app)


# --- CONFIG ---

SUPABASE_URL = "http://100.125.109.107:54321"

SUPABASE_KEY = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Risk Thresholds

THRESHOLDS = {
    "CNC_01": {"warning": 0.4, "fault": 0.7},
    "CNC_02": {"warning": 0.9, "fault": 0.85},
    "PUMP_03": {"warning": 0.54, "fault": 0.9},
    "CONVEYOR_04": {"warning": 0.78, "fault": 0.9},
}


def talk_to_mistral(system_prompt, user_content):

    try:
        # Increased timeout to 60s to allow GPU to process more complex descriptions

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"<s>[INST] {system_prompt} [/INST] {user_content}</s>",
                "stream": False,
                "options": {"num_predict": 100},  # Allows for longer, detailed output
            },
            timeout=60,
        )

        return response.json().get("response", "").strip()

    except Exception as e:
        return f"Mistral Error: {str(e)}"


@app.route("/alert", methods=["GET"])
def get_alerts():

    output = []

    machine_ids = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

    for m_id in machine_ids:
        try:
            # Fetch latest telemetry and prediction

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

            # Logic Gate: If data exists and risk crosses warning threshold

            if tel.data and pred.data:
                risk = pred.data[0]["risk_score"]

                limit = THRESHOLDS[m_id]

                if risk >= limit["warning"]:
                    # MISTRAL PROMPT REFINEMENT: Asking for "Crux" + "Data Context"

                    system_msg = "You are a senior industrial systems engineer. Explain the specific anomaly based on the sensor values provided. Be technical yet clear. Mention the impact on the machine."

                    # Passing more data to Mistral so it can give a better answer

                    user_msg = (
                        f"Machine: {m_id}\n"
                        f"Current Risk: {risk}\n"
                        f"Telemetry: Temp={tel.data[0]['temperature_c']}C, RPM={tel.data[0]['rpm']}, "
                        f"Vibration={tel.data[0]['vibration_mm_s']}mm/s, Status={tel.data[0]['status']}"
                    )

                    explanation = talk_to_mistral(system_msg, user_msg)

                    # Ensure explanation isn't empty

                    if not explanation:
                        explanation = f"High Risk Detected ({risk}). Sensors show abnormal {tel.data[0]['status']} state."

                    output.extend([1, m_id, explanation])

                else:
                    # Risk is below warning

                    output.extend([0, m_id, "no anomalies"])

            else:
                # No data found for this machine yet

                output.extend([0, m_id, "no anomalies"])

        except Exception as e:
            print(f"Error processing {m_id}: {e}")

            output.extend([0, m_id, "error processing data"])

    return jsonify(output)


@app.route("/chat", methods=["POST"])
def chat():

    user_msg = request.json.get("message")

    schema = "Tables: machine_telemetry (id, machine_id, temperature_c, vibration_mm_s, rpm, status), model_predictions (id, machine_id, risk_score, prediction)"

    system_msg = f"You are an industrial data assistant. Access context: {schema}. Help the operator understand their factory."

    answer = talk_to_mistral(system_msg, user_msg)

    return jsonify({"reply": answer})


if __name__ == "__main__":
    print("--- FULL INDUSTRIAL NLP SERVER ACTIVE ---")

    app.run(port=6969, debug=False, threaded=True, host="0.0.0.0")
