from flask import Flask, jsonify, send_from_directory, request
import csv
import os
import math
from docking_env import OrbitalDockingEnv

app = Flask(__name__, static_folder='static_viz')

# --- MANUAL FLIGHT (BROWSER "Fly Manually" BUTTON) ---
# The dashboard's manual mode calls /api/manual_reset and /api/manual_step,
# but those routes never existed server-side, so every keypress during a
# manual flight silently 404'd (visible only in the browser console) and
# nothing ever moved, docked, or ended. This holds one live env per process
# so the browser can drive it step by step over HTTP.
_manual_env = None
_manual_cumulative_reward = 0.0

def _manual_state_payload(raw_state, info, action_taken, reward, terminated, truncated):
    return {
        "step": _manual_env._elapsed_steps,
        "x": float(raw_state[0]), "y": float(raw_state[1]),
        "vx": float(raw_state[2]), "vy": float(raw_state[3]),
        "theta_deg": float(raw_state[4]), "omega_dps": float(raw_state[5]),
        # action_taken is now the [X, Y, ROT] list, not a scalar — same
        # fix as record_ai_flight.py's action_taken field.
        "action_taken": list(action_taken),
        "reward": float(reward),
        "cumulative_reward": float(_manual_cumulative_reward),
        "fuel_kg": float(info.get("fuel_kg", 0.0)),
        "outcome": info.get("outcome"),
        "terminated": bool(terminated or truncated),
    }

@app.route('/api/manual_reset', methods=['POST'])
def manual_reset():
    global _manual_env, _manual_cumulative_reward
    _manual_env = OrbitalDockingEnv()
    _manual_env.reset()
    _manual_cumulative_reward = 0.0
    raw_state = _manual_env.state
    return jsonify(_manual_state_payload(raw_state, {"fuel_kg": _manual_env.fuel},
                                          action_taken=[0, 0, 0], reward=0.0,
                                          terminated=False, truncated=False))

@app.route('/api/manual_step', methods=['POST'])
def manual_step():
    global _manual_env, _manual_cumulative_reward
    if _manual_env is None:
        return jsonify({"error": "No active manual session. Call /api/manual_reset first."}), 400

    body = request.get_json(silent=True) or {}
    # --- FIX: action-space mismatch with docking_env.py ---
    # This used to treat the action as a single scalar 0..6 and validate it
    # against `_manual_env.action_space.n` — but action_space is
    # MultiDiscrete([3, 3, 3]) (three independent [X, Y, ROT] channels,
    # each 0=off/1=positive/2=negative), which has no `.n` attribute at
    # all (that's a Discrete-space attribute). Every call to this route hit
    # an AttributeError before the env was ever stepped, so every keypress
    # in the browser's manual-fly mode silently 404/500'd. The frontend
    # must now POST {"action": [x, y, rot]} instead of {"action": <int>}.
    raw_action = body.get("action", [0, 0, 0])
    try:
        action = [int(a) for a in raw_action]
        if len(action) != 3 or any(a not in (0, 1, 2) for a in action):
            action = [0, 0, 0]
    except (TypeError, ValueError):
        action = [0, 0, 0]

    obs, reward, terminated, truncated, info = _manual_env.step(action)
    _manual_cumulative_reward += reward
    raw_state = _manual_env.state
    return jsonify(_manual_state_payload(raw_state, info, action, reward, terminated, truncated))

def _no_store(response):
    response.headers['Cache-Control'] = 'no-store, must-revalidate'
    return response

@app.route('/')
def serve_dashboard():
    return send_from_directory('static_viz', 'viz_dashboard.html')

@app.route('/docking')
def serve_docking():
    return _no_store(send_from_directory('static_viz', 'docking_view.html'))

@app.route('/ai_flight_data.json')
def serve_flight_data():
    path = os.path.join('static_viz', 'ai_flight_data.json')
    if not os.path.exists(path):
        return jsonify({"error": "No flight recorded yet. Run record_ai_flight.py first."}), 404
    return _no_store(send_from_directory('static_viz', 'ai_flight_data.json'))

# --- NEW ROUTE: SERVE SPECIFIC HISTORICAL FLIGHTS ---
@app.route('/flights/<filename>')
def serve_historical_flight(filename):
    path = os.path.join('static_viz', filename)
    if not os.path.exists(path):
        return jsonify({"error": "Historical flight not found"}), 404
    return _no_store(send_from_directory('static_viz', filename))

@app.route('/mission-control')
def serve_mission_control():
    return _no_store(send_from_directory('static_viz', 'mission_control.html'))

@app.route('/mission_log.json')
def serve_mission_log():
    path = os.path.join('static_viz', 'mission_log.json')
    if not os.path.exists(path):
        return jsonify({"training": {"stages": {}}, "flights": []}), 200
    return _no_store(send_from_directory('static_viz', 'mission_log.json'))

@app.route('/api/trajectory')
def get_trajectory():
    data = {
        "iss": [],
        "chaser": [],
        "launch_path": []
    }
    
    for i in range(101):
        angle = (i / 100) * 2 * math.pi
        r_iss = 6371 + 400
        data["iss"].append({
            "x": r_iss * math.cos(angle),
            "y": 1500 * math.sin(angle), 
            "z": r_iss * math.sin(angle)
        })
        
    for i in range(101):
        angle = (i / 100) * 2 * math.pi
        r_chaser = 6371 + 600
        data["chaser"].append({
            "x": r_chaser * math.cos(angle + 1),
            "y": 2500 * math.sin(angle), 
            "z": r_chaser * math.sin(angle)
        })

    for i in range(20):
        t = i / 19.0
        r_launch = 6371 + (400 * t) 
        data["launch_path"].append({
            "x": r_launch * math.cos(0.2 * t),
            "y": 1000 * t,
            "z": r_launch * math.sin(0.2 * t)
        })

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)