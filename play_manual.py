import pygame
import json
import time
import os
import numpy as np
from docking_env import OrbitalDockingEnv

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

# --- KEY MAPPINGS ---
# --- FIX: action-space mismatch with docking_env.py ---
# This used to send a single scalar 0-6 (matching the OLD Discrete(8)
# scheme still documented in physics_config.py's "action_space" dict).
# docking_env.py's action_space is now MultiDiscrete([3, 3, 3]) — three
# independent channels [X-thrust, Y-thrust, Rotation], each 0=off /
# 1=positive / 2=negative, fired simultaneously. Passing a bare scalar
# crashes the instant a thruster key is pressed, because docking_env.py's
# step() does `act_x, act_y, act_rot = action`, and you cannot unpack 3
# values out of a single int (numpy raises "iteration over a 0-d array").
# That's why manual flight crashed on the very first keypress.
#
# Building a 3-element array below also finally allows the SIMULTANEOUS
# multi-axis control (e.g. forward + strafe + rotate at once) that
# train_agent_curriculum.py's comments say a real manual dock needs —
# the old elif-chain could only ever fire one axis per frame.
#
#   X: W = +X (forward)     S = -X (brake)
#   Y: D = +Y (right)       A = -Y (left)
# ROT: Q = CCW               E = CW

def manual_flight():
    pygame.init()
    # Create a simple radar window
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Manual Override - BAS Docking")
    clock = pygame.time.Clock()

    env = OrbitalDockingEnv()
    obs, _ = env.reset()
    flight_data = []
    cumulative_reward = 0.0

    print("\n" + "="*40)
    print("MANUAL OVERRIDE ENGAGED")
    print("Controls:")
    print("  W / S : Thrust Forward / Brake")
    print("  A / D : Thrust Left / Right")
    print("  Q / E : Rotate CCW / CW")
    print("Close the window to abort.")
    print("="*40 + "\n")

    running = True
    step = 0

    while running and step < env.max_episode_steps:
        screen.fill((5, 10, 20)) # Deep space background

        # 1. Capture Keyboard Input -> build the [X, Y, ROT] MultiDiscrete action
        act_x, act_y, act_rot = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: act_x = 1
        elif keys[pygame.K_s]: act_x = 2
        if keys[pygame.K_d]: act_y = 1
        elif keys[pygame.K_a]: act_y = 2
        if keys[pygame.K_q]: act_rot = 1
        elif keys[pygame.K_e]: act_rot = 2
        action = np.array([act_x, act_y, act_rot])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. Step the Physics Environment
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        raw_state = env.state
        step += 1

        # 3. Record the frame for the Browser HUD
        flight_data.append({
            "step": step,
            "x": float(raw_state[0]), "y": float(raw_state[1]),
            "vx": float(raw_state[2]), "vy": float(raw_state[3]),
            "theta_deg": float(raw_state[4]), "omega_dps": float(raw_state[5]),
            "action_taken": action.tolist(),
            "reward": float(reward), "cumulative_reward": float(cumulative_reward),
            "fuel_kg": float(info.get("fuel_kg", 0.0)),
            "outcome": info.get("outcome")
        })

        # 4. Draw a crude radar screen so you can see what you are doing
        center_x, center_y = 300, 300
        scale = 2.0  # pixels per meter

        # Draw Target Station at (0,0)
        pygame.draw.circle(screen, (62, 224, 200), (center_x, center_y), 6)
        
        # Draw Chaser Ship
        chaser_x = int(center_x + raw_state[0] * scale)
        chaser_y = int(center_y + raw_state[1] * scale)
        pygame.draw.circle(screen, (255, 180, 84), (chaser_x, chaser_y), 4)
        
        # Draw Velocity Vector (Line showing where you are drifting)
        vel_end_x = int(chaser_x + raw_state[2] * 50)
        vel_end_y = int(chaser_y + raw_state[3] * 50)
        pygame.draw.line(screen, (255, 180, 84), (chaser_x, chaser_y), (vel_end_x, vel_end_y), 1)

        # Draw HUD text
        font = pygame.font.SysFont("monospace", 14)
        dist_text = font.render(f"Range: {((raw_state[0]**2 + raw_state[1]**2)**0.5):.1f}m", True, (255, 255, 255))
        fuel_text = font.render(f"Fuel: {info.get('fuel_kg', 0.0):.1f}kg", True, (255, 255, 255))
        screen.blit(dist_text, (10, 10))
        screen.blit(fuel_text, (10, 30))

        pygame.display.flip()
        
        # Lock framerate to 10 FPS so 1 frame = self.dt (0.1s) in docking_env.py
        clock.tick(10) 

        if terminated or truncated:
            print(f"\nMission Ended. Outcome: {info.get('outcome')}")
            print(f"Total RL Reward for your flight: {cumulative_reward:.2f}")
            break

    pygame.quit()

    # 5. Save the flight so you can watch it in your web UI!
    # --- FIX: THE OVERWRITE FLAW ---
    # This used to only ever write static_viz/ai_flight_data.json, so every
    # manual flight erased the previous one with no way to compare runs.
    # Now it saves a timestamped copy AND registers it in mission_log.json,
    # same as record_ai_flight.py does for AI flights, so manual attempts
    # show up in the dashboard's flight-history dropdown too and nothing
    # gets silently lost.
    os.makedirs("static_viz", exist_ok=True)
    timestamp = time.time()
    payload = {"generated_at": timestamp, "frames": flight_data}

    # Still write the default "latest" file for backward compatibility
    with open("static_viz/ai_flight_data.json", "w") as f:
        json.dump(payload, f)

    flight_id = int(timestamp * 1000)
    flight_filename = f"manual_flight_{flight_id}.json"
    with open(f"static_viz/{flight_filename}", "w") as f:
        json.dump(payload, f)

    last_frame = flight_data[-1] if flight_data else {}
    mission_log = _load_mission_log()
    mission_log.setdefault("flights", []).append({
        "id": timestamp,
        "generated_at": timestamp,
        "model_path": "MANUAL",
        "steps": len(flight_data),
        "outcome": f"[MANUAL] {last_frame.get('outcome', 'unresolved')}",
        "cumulative_reward": cumulative_reward,
        "fuel_kg_remaining": last_frame.get("fuel_kg", 0.0),
        "filename": flight_filename
    })
    mission_log["flights"] = mission_log["flights"][-200:]
    _save_mission_log(mission_log)

    print(f"\nManual flight recorded to {flight_filename} (and ai_flight_data.json)!")
    print("Open your browser HUD, refresh, and pick it from the Flight Log dropdown.")

if __name__ == "__main__":
    manual_flight()