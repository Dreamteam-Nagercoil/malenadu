"""
NeuralGuard Flask Backend — port 6969

Endpoints:
  GET  /alert                       SSE stream: pushes machine state to dashboard
  POST /schedule                    Schedule maintenance (priority queue logic)
  GET  /schedule                    Get all scheduled slots (sorted by time)
  DELETE /schedule/<id>             Cancel a slot
  GET  /schedule/next/<machine_id>  Next pending slot for a machine
"""

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# ─── Config ──────────────────────────────────────────────────────────────────

SIM_URL = "http://127.0.0.1:8000"  # FastAPI simulator

# Working hours for scheduling (24h)
WORK_START = 6  # 06:00
WORK_END = 22  # 22:00
SLOT_MINS = 30  # each maintenance slot is 30 minutes

# How soon (hours) each priority must be scheduled
PRIORITY_DELAY = {
    "critical": 1,
    "high": 4,
    "normal": 8,
    "low": 24,
}

# ─── App ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# In-memory schedule: list of dicts
# { id, machine_id, reason, priority, risk_score, slot_start, slot_end, booked_at, status }
schedule: list[dict] = []
schedule_lock = threading.Lock()
_schedule_id_counter = 1

# Cached latest machine state (updated by background poller)
machine_cache: dict = {}

# ─── Background: poll simulator /status ──────────────────────────────────────


def poll_simulator():
    while True:
        try:
            r = requests.get(f"{SIM_URL}/status", timeout=2)
            if r.ok:
                machine_cache.update(r.json())
        except Exception:
            pass
        time.sleep(1)


threading.Thread(target=poll_simulator, daemon=True).start()

# ─── SSE alert stream ─────────────────────────────────────────────────────────


def build_sse_payload() -> list[dict]:
    """
    Build the list of machine objects the dashboard expects:
    [{ machine_id, telemetry, alert, risk, message }, ...]
    """
    machines = []
    for machine_id, state in machine_cache.items():
        live = state.get("live", {})
        progress = state.get("progress", 0.0)
        failure_label = state.get("failure_label")

        # Risk score: 0.0–1.0 derived from failure progress
        # If no failure, low background noise risk based on sensor deviation
        risk = (
            round(progress, 3) if failure_label else _background_risk(live, machine_id)
        )

        # Human-readable message
        message = _generate_insight(
            machine_id, live, risk, failure_label, state.get("failure_mode")
        )

        machines.append(
            {
                "machine_id": machine_id,
                "telemetry": {
                    "temperature": live.get("temperature_C"),
                    "vibration": live.get("vibration_mm_s"),
                    "rpm": live.get("rpm"),
                    "current": live.get("current_A"),
                    "status": live.get("status", "running"),
                },
                "alert": risk > 0.3,
                "risk": risk,
                "message": message,
            }
        )
    return machines


def _background_risk(live: dict, machine_id: str) -> float:
    """Very low risk score when no failure is injected."""
    return round(0.02 + abs(hash(machine_id + str(int(time.time() / 10)))) % 5 / 100, 3)


def _generate_insight(machine_id, live, risk, failure_label, failure_mode) -> str:
    if not live:
        return "Awaiting telemetry..."
    if not failure_label:
        return f"All sensors nominal. Risk: {risk * 100:.0f}%."

    pct = risk * 100
    temp = live.get("temperature_C", 0)
    vib = live.get("vibration_mm_s", 0)
    curr = live.get("current_A", 0)
    rpm = live.get("rpm", 0)

    insights = {
        "bearing_wear": f"Vibration elevated at {vib:.2f} mm/s. Bearing degradation pattern. Risk: {pct:.0f}%.",
        "overheating": f"Thermal runaway detected — {temp:.1f}°C. Cooling failure likely. Risk: {pct:.0f}%.",
        "electrical_fault": f"Current surge at {curr:.1f}A. Insulation or winding fault suspected. Risk: {pct:.0f}%.",
        "mechanical_imbalance": f"High vibration ({vib:.2f} mm/s) with RPM instability ({rpm:.0f}). Mass imbalance. Risk: {pct:.0f}%.",
        "rpm_runaway": f"RPM spike to {rpm:.0f}. Speed controller failure. Risk: {pct:.0f}%.",
    }
    return insights.get(failure_mode, f"{failure_label} detected. Risk: {pct:.0f}%.")


@app.route("/alert")
def alert_stream():
    def generate():
        while True:
            if machine_cache:
                payload = build_sse_payload()
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ─── Scheduling logic ─────────────────────────────────────────────────────────


def _find_free_slot(earliest: datetime) -> datetime:
    """
    Find the first free 30-min slot starting at or after `earliest`,
    within working hours, not already booked by any machine.
    """
    candidate = _snap_to_slot(earliest)

    with schedule_lock:
        booked_starts = {
            datetime.fromisoformat(s["slot_start"].replace("Z", "+00:00"))
            for s in schedule
            if s["status"] == "scheduled"
        }

    # Search up to 7 days ahead
    limit = candidate + timedelta(days=7)
    while candidate < limit:
        if WORK_START <= candidate.hour < WORK_END and candidate not in booked_starts:
            return candidate
        candidate += timedelta(minutes=SLOT_MINS)
        # Skip outside working hours
        if candidate.hour >= WORK_END:
            candidate = candidate.replace(
                hour=WORK_START, minute=0, second=0
            ) + timedelta(days=1)

    return candidate  # fallback


def _snap_to_slot(dt: datetime) -> datetime:
    """Round up to the next 30-minute boundary."""
    mins = dt.minute
    if mins < 30:
        snapped = dt.replace(minute=30, second=0, microsecond=0)
    else:
        snapped = (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return snapped


def _book_slot(machine_id: str, reason: str, priority: str, risk_score: float) -> dict:
    global _schedule_id_counter

    now = datetime.now(timezone.utc)
    min_delay_hours = PRIORITY_DELAY[priority]
    earliest = now + timedelta(hours=min_delay_hours)

    # Check if machine already has a pending slot — cancel it first if priority escalated
    with schedule_lock:
        existing = next(
            (
                s
                for s in schedule
                if s["machine_id"] == machine_id and s["status"] == "scheduled"
            ),
            None,
        )
        if existing:
            existing_prio_idx = list(PRIORITY_DELAY.keys()).index(existing["priority"])
            new_prio_idx = list(PRIORITY_DELAY.keys()).index(priority)
            if new_prio_idx <= existing_prio_idx:
                # New booking is same or higher priority — cancel old one
                existing["status"] = "cancelled"
            else:
                # Old booking is higher priority — return it unchanged
                return existing

    slot_start = _find_free_slot(earliest)
    slot_end = slot_start + timedelta(minutes=SLOT_MINS)

    entry = {
        "id": _schedule_id_counter,
        "machine_id": machine_id,
        "reason": reason,
        "priority": priority,
        "risk_score": round(risk_score, 3),
        "slot_start": slot_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slot_end": slot_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "booked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "scheduled",
    }

    with schedule_lock:
        schedule.append(entry)
        _schedule_id_counter += 1

    return entry


# ─── Schedule REST endpoints ──────────────────────────────────────────────────


@app.route("/schedule", methods=["POST"])
def post_schedule():
    data = request.json or {}
    machine_id = data.get("machine_id", "").upper()
    priority = data.get("priority", "normal").lower()
    reason = data.get("reason", "Maintenance required")
    risk_score = float(data.get("risk_score", 0.0))

    valid_machines = list(machine_cache.keys()) or [
        "CNC_01",
        "HYD_02",
        "COMP_03",
        "CONV_04",
    ]
    if machine_id not in valid_machines:
        return jsonify({"error": f"Unknown machine '{machine_id}'"}), 400
    if priority not in PRIORITY_DELAY:
        return jsonify(
            {"error": f"Invalid priority. Use: {list(PRIORITY_DELAY.keys())}"}
        ), 400

    slot = _book_slot(machine_id, reason, priority, risk_score)
    return jsonify({"success": True, "slot": slot})


@app.route("/schedule", methods=["GET"])
def get_schedule():
    with schedule_lock:
        active = sorted(
            [s for s in schedule if s["status"] == "scheduled"],
            key=lambda s: s["slot_start"],
        )
    return jsonify({"count": len(active), "slots": active})


@app.route("/schedule/<int:slot_id>", methods=["DELETE"])
def cancel_schedule(slot_id):
    with schedule_lock:
        slot = next((s for s in schedule if s["id"] == slot_id), None)
        if not slot:
            return jsonify({"error": "Slot not found"}), 404
        if slot["status"] != "scheduled":
            return jsonify({"error": f"Slot is already {slot['status']}"}), 400
        slot["status"] = "cancelled"
    return jsonify({"success": True, "cancelled_id": slot_id})


@app.route("/schedule/next/<machine_id>", methods=["GET"])
def next_slot(machine_id):
    machine_id = machine_id.upper()
    with schedule_lock:
        slots = sorted(
            [
                s
                for s in schedule
                if s["machine_id"] == machine_id and s["status"] == "scheduled"
            ],
            key=lambda s: s["slot_start"],
        )
    if not slots:
        return jsonify({"machine_id": machine_id, "next_slot": None})
    return jsonify({"machine_id": machine_id, "next_slot": slots[0]})


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NeuralGuard backend starting on http://127.0.0.1:6969")
    print("Endpoints:")
    print("  GET    /alert                     SSE stream")
    print("  POST   /schedule                  Book a slot")
    print("  GET    /schedule                  All slots")
    print("  DELETE /schedule/<id>             Cancel slot")
    print("  GET    /schedule/next/<machine_id>")
    app.run(host="127.0.0.1", port=6969, debug=False, threaded=True)
