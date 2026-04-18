import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime, timedelta

import requests
from flask import Flask, Response, jsonify, request, send_file  # Added send_file
from flask_cors import CORS
from supabase import Client, create_client

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
SUPABASE_URL = "http://100.125.109.107:54321"
SUPABASE_KEY = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"
MODEL_NAME = "llama3.2:3b"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

THRESHOLDS = {
    "CNC_01": {"warning": 0.4, "fault": 0.7},
    "CNC_02": {"warning": 0.70, "fault": 0.85},
    "PUMP_03": {"warning": 0.54, "fault": 0.9},
    "CONVEYOR_04": {"warning": 0.78, "fault": 0.9},
}

# --- GLOBALS ---
last_risks = {}
last_explanations = {}
last_telemetry = {}
maintenance_db = []
slot_counter = 1


# --- UTILS ---
def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def talk_to_qwen(prompt, system=None):
    payload = {
        "model": MODEL_NAME,
        "messages": [],
        "stream": False,
        "options": {
            "num_predict": 200,
            "temperature": 0.1,
            "num_ctx": 2048,
        },
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            "http://localhost:11434/api/chat", json=payload, timeout=60
        )
        response.raise_for_status()
        msg = response.json().get("message", {})
        content = msg.get("content", "").strip()
        thinking = msg.get("thinking", "").strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content if content else thinking
    except Exception as e:
        return f"Model Error: {str(e)}"


# --- CORE LOGIC ---
def get_alert_data():
    global maintenance_db, slot_counter
    output = []
    machine_ids = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

    for m_id in machine_ids:
        try:
            # 1. Fetch Telemetry & Predictions
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

                # 2. Determine Anomaly State
                is_anomaly = risk >= limit["warning"] or tel_data["status"] in [
                    "fault",
                    "warning",
                ]

                if is_anomaly:
                    alert_flag = 1

                    # --- AUTO-BOOKING LOGIC (Triggers on Fault) ---
                    if risk >= limit["fault"]:
                        already_scheduled = any(
                            s["machine_id"] == m_id for s in maintenance_db
                        )
                        if not already_scheduled:
                            start = datetime.now(UTC) + timedelta(hours=1)
                            auto_slot = {
                                "id": slot_counter,
                                "machine_id": m_id,
                                "priority": "critical",
                                "reason": f"AUTO: Risk {risk} hit fault threshold {limit['fault']}",
                                "slot_start": start.isoformat() + "Z",
                                "slot_end": (start + timedelta(hours=1)).isoformat()
                                + "Z",
                            }
                            maintenance_db.append(auto_slot)
                            slot_counter += 1

                    # 3. AI Explanation
                    prev_risk = last_risks.get(m_id, 0)
                    if abs(prev_risk - risk) > 0.05 or not last_explanations.get(m_id):
                        prompt = f"Analyze {m_id}: Risk {risk}, Status {tel_data['status']}, Temp {tel_data['temperature_c']}C. Explain cause/fix in 15 words."
                        ai_msg = talk_to_qwen(prompt)
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
                            "current": tel_data.get("current_a", 0),
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


# --- ROUTES ---
@app.route("/alert", methods=["GET"])
def stream_alerts():
    def event_stream():
        while True:
            yield f"data: {json.dumps(get_alert_data())}\n\n"
            time.sleep(2)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/chat", methods=["POST"])
def chat():
    global slot_counter
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "No message received."})

    # Command Interceptor
    if user_msg.lower().startswith("schedule "):
        parts = user_msg.split(" ", 3)
        if len(parts) >= 3:
            m_id, priority = parts[1].upper(), parts[2].lower()
            reason = parts[3] if len(parts) > 3 else "AI Requested"
            if m_id in THRESHOLDS and priority in ["critical", "high", "normal", "low"]:
                start = datetime.now(UTC) + timedelta(hours=2)
                new_slot = {
                    "id": slot_counter,
                    "machine_id": m_id,
                    "priority": priority,
                    "reason": reason,
                    "slot_start": start.isoformat() + "Z",
                    "slot_end": (start + timedelta(hours=1)).isoformat() + "Z",
                }
                maintenance_db.append(new_slot)
                slot_counter += 1
                return jsonify(
                    {"reply": f"CONFIRMED: Maintenance for {m_id} scheduled."}
                )

    # Contextual AI Reply
    context = [
        f"{m}: risk={last_risks.get(m, 0)}, analysis={last_explanations.get(m, 'Clear')}"
        for m in THRESHOLDS
    ]
    sys_prompt = (
        f"You are NeuralGuard. Data: {' | '.join(context)}. Be concise. No markdown."
    )
    answer = talk_to_qwen(user_msg, system=sys_prompt)
    return jsonify({"reply": answer})


@app.route("/schedule", methods=["GET"])
def get_schedule():
    priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
    sorted_slots = sorted(
        maintenance_db,
        key=lambda x: (priority_order.get(x["priority"], 9), x["slot_start"]),
    )
    return jsonify({"count": len(maintenance_db), "slots": sorted_slots})


@app.route("/schedule", methods=["POST"])
def add_schedule():
    global slot_counter
    data = request.json
    start_time = datetime.now(UTC) + timedelta(hours=1)
    new_slot = {
        "id": slot_counter,
        "machine_id": data.get("machine_id"),
        "priority": data.get("priority", "normal"),
        "reason": data.get("reason", "Manual booking"),
        "slot_start": start_time.isoformat() + "Z",
        "slot_end": (start_time + timedelta(hours=1)).isoformat() + "Z",
    }
    maintenance_db.append(new_slot)
    slot_counter += 1
    return jsonify({"success": True, "slot": new_slot})


@app.route("/schedule/<int:slot_id>", methods=["DELETE"])
def cancel_schedule(slot_id):
    global maintenance_db
    maintenance_db = [s for s in maintenance_db if s["id"] != slot_id]
    return jsonify({"success": True})


# --- NEW: PDF REPORT ROUTE ---
@app.route("/generate-report", methods=["GET"])
def generate_report():
    try:
        # 1. Run the Node.js script
        # Ensure generateReport.js is in the same directory
        print("Executing generateReport.js...")
        result = subprocess.run(
            ["node", "generateReport.js"], capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Node Error: {result.stderr}")
            return jsonify({"success": False, "error": "PDF Generation failed"}), 500

        # 2. Identify file path (Report defaults to current date)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"report_{date_str}.pdf"
        filepath = os.path.join(os.getcwd(), filename)

        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify(
                {"success": False, "error": f"File {filename} not found"}
            ), 404

    except Exception as e:
        print(f"System Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969, threaded=True)
