import gymnasium as gym
from stable_baselines3 import PPO
from docking_env import OrbitalDockingEnv
import json
import os
import time

MISSION_LOG_PATH = "static_viz/mission_log.json"

def _load_mission_log(path=MISSION_LOG_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"training": {"stages": {}}, "flights": []}

def _save_mission_log(data, path=MISSION_LOG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data["updated_at"] = time.time()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)

# We define the stages here so the recording script knows exactly what 
# environment settings to use for each saved model.
STAGES = [
    {
        "name": "stage1_fundamentals",
        "spawn_range": (-20.0, -10.0),
        "spawn_lateral": 5.0,
        "tolerance_scale": 4.0,
        "max_episode_steps": 1000,
    },
    {
        # Kept in sync with train_agent_curriculum.py's stage list.
        "name": "stage1b_midrange",
        "spawn_range": (-35.0, -18.0),
        "spawn_lateral": 12.0,
        "tolerance_scale": 3.2,
        "max_episode_steps": 1400,
    },
    {
        # Kept in sync with train_agent_curriculum.py's new stage1c_distance.
        "name": "stage1c_distance",
        "spawn_range": (-60.0, -30.0),
        "spawn_lateral": 20.0,
        "tolerance_scale": 3.2,
        "max_episode_steps": 1800,
    },
    {
        "name": "stage2_braking",
        "spawn_range": (-60.0, -30.0),
        "spawn_lateral": 20.0,
        "tolerance_scale": 2.5,
        "max_episode_steps": 1800,
    },
    {
        # Kept in sync with train_agent_curriculum.py's stage list.
        "name": "stage2b_extended",
        "spawn_range": (-80.0, -40.0),
        "spawn_lateral": 27.0,
        "tolerance_scale": 2.0,
        "max_episode_steps": 2800,
    },
    {
        "name": "stage3_approach",
        "spawn_range": (-100.0, -50.0),
        "spawn_lateral": 35.0,
        "tolerance_scale": 1.5,
        "max_episode_steps": 4000,  # synced with train_agent_curriculum.py's time-budget fix
    },
    {
        # Kept in sync with train_agent_curriculum.py's stage list.
        "name": "stage3b_precision",
        "spawn_range": (-125.0, -60.0),
        "spawn_lateral": 42.0,
        "tolerance_scale": 1.2,
        "max_episode_steps": 5000,
    },
    {
        "name": "stage4_full_mission",
        "spawn_range": (-150.0, -70.0),
        "spawn_lateral": 50.0,
        "tolerance_scale": 1.0,
        "max_episode_steps": 6000,  # synced with train_agent_curriculum.py's time-budget fix
    },
    {
        "name": "final",
        "spawn_range": (-150.0, -70.0), # Final model tested at full difficulty
        "spawn_lateral": 50.0,
        "tolerance_scale": 1.0,
        "max_episode_steps": 6000,  # synced with train_agent_curriculum.py's time-budget fix
    }
]

def record_curriculum_history(flights_per_stage=5, search_attempts_per_stage=40):
    mission_log = _load_mission_log()
    os.makedirs("static_viz", exist_ok=True)
    
    total_saved = 0
    
    for stage_idx, stage in enumerate(STAGES):
        model_name = f"ppo_orbital_docker_{stage['name']}.zip"
        
        if not os.path.exists(model_name):
            print(f"Skipping {stage['name']} - Model file not found.")
            continue
            
        print(f"\n==================================================")
        print(f"Loading Model: {model_name}")
        print(f"Environment: Spawn {stage['spawn_range']}, Tolerance x{stage['tolerance_scale']}")
        print(f"==================================================")
        
        env = OrbitalDockingEnv(
            spawn_range=stage["spawn_range"],
            spawn_lateral=stage["spawn_lateral"],
            tolerance_scale=stage["tolerance_scale"],
            max_episode_steps=stage["max_episode_steps"]
        )
        
        model = PPO.load(model_name)
        flight_pool = []
        
        # 1. Scout attempts
        for attempt in range(1, search_attempts_per_stage + 1):
            obs, _ = env.reset()
            flight_data = []
            cumulative_reward = 0.0
            
            for step in range(env.max_episode_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward
                raw_state = env.state

                flight_data.append({
                    "step": step,
                    "x": float(raw_state[0]), "y": float(raw_state[1]),
                    "vx": float(raw_state[2]), "vy": float(raw_state[3]),
                    "theta_deg": float(raw_state[4]), "omega_dps": float(raw_state[5]),
                    # --- FIX: int() on a 3-element list ---
                    # docking_env.py's action_space is MultiDiscrete([3,3,3]),
                    # so info["actual_action"] is a 3-item list like [1,0,2],
                    # not a single number. `int([1,0,2])` raised
                    # "int() argument must be ... not 'list'" on the very
                    # first env.step() of the very first scouting attempt of
                    # the very first stage — this script never got past step
                    # 1, so no new flight files or mission_log entries were
                    # ever produced after docking_env.py's action space
                    # changed from scalar Discrete to MultiDiscrete. Store
                    # the action as the list it actually is.
                    "action_taken": info.get("actual_action", action.tolist() if hasattr(action, "tolist") else action),
                    "reward": float(reward), "cumulative_reward": float(cumulative_reward),
                    "fuel_kg": float(info.get("fuel_kg", 0.0)),
                    "outcome": info.get("outcome"),
                })

                if terminated or truncated:
                    outcome = info.get('outcome', 'unknown')
                    break
            
            print(f"\rScouting attempt {attempt}/{search_attempts_per_stage}... Outcome: {outcome.upper()}", end="")
            
            if outcome != "out_of_bounds":
                flight_pool.append({"outcome": outcome, "reward": cumulative_reward, "frames": flight_data})
                
        print("\nSorting and saving best flights for this stage...")
        
        # 2. Sort by priority
        priority_map = {"docked": 1, "soft_crash": 2, "hard_crash": 3, "hull_strike": 4, "truncated": 5}
        flight_pool.sort(key=lambda f: (priority_map.get(f["outcome"], 99), -f["reward"]))
        
        best_flights = flight_pool[:flights_per_stage]
        
        # 3. Save to disk
        for i, flight in enumerate(best_flights):
            timestamp = time.time()
            flight_id = int(timestamp * 1000)
            
            payload = {"generated_at": timestamp, "frames": flight["frames"]}

            # --- FIX: "latest" pointer file only ever updated for the
            # 'final' stage. If you've only trained up through, say,
            # stage2_braking (no 'final' model exists yet), this file
            # never gets created — every summarize_flight.py call needs
            # you to guess the exact timestamped filename instead. Now
            # this always points at the #1-ranked flight of whichever
            # stage record_ai_flight.py most recently processed, so
            # `static_viz/ai_flight_data.json` is always a valid,
            # predictable "look at what just happened" target regardless
            # of curriculum progress.
            if i == 0:
                with open("static_viz/ai_flight_data.json", "w") as f:
                    json.dump(payload, f)

            flight_filename = f"ai_flight_{flight_id}.json"
            with open(f"static_viz/{flight_filename}", "w") as f:
                json.dump(payload, f)

            last_frame = flight["frames"][-1]
            # Prefix the outcome with the stage name so it looks nice in the dropdown
            display_outcome = f"[{stage['name'].split('_')[0].upper()}] {flight['outcome']}"
            
            mission_log.setdefault("flights", []).append({
                "id": timestamp,
                "generated_at": timestamp,
                "model_path": model_name,
                "steps": len(flight["frames"]),
                "outcome": display_outcome,
                "cumulative_reward": flight["reward"],
                "fuel_kg_remaining": last_frame.get("fuel_kg", 0.0),
                "filename": flight_filename
            })
            
            total_saved += 1
            print(f"  -> Saved Rank #{i+1}: {flight['outcome'].upper()} (Reward: {flight['reward']:.2f})")
            time.sleep(0.05) 

    mission_log["flights"] = mission_log["flights"][-200:]
    _save_mission_log(mission_log)
    print(f"\nAll done! Saved {total_saved} total historical flights across all stages. Refresh your browser.")

if __name__ == "__main__":
    # This will pull the best 5 flights from EACH stage model it finds, searching up to 40 times per stage.
    record_curriculum_history(flights_per_stage=5, search_attempts_per_stage=40)