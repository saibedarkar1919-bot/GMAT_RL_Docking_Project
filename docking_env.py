"""
OrbitalDockingEnv — 2D Clohessy-Wiltshire proximity-ops environment.

CHANGES FROM ORIGINAL (observation-only — physics/reward/termination
logic below is untouched):

1. Normalization constants were wildly mismatched to the real operating
   range. max_pos=2000 / max_vel=20 / max_omega=10 meant real values
   (x in [-190, 20], vy up to a few m/s, omega usually < 1 dps) were all
   squashed into roughly [-0.1, 0.1] of the observation range. A
   freshly-initialized network's weights are tuned for O(1) inputs, so
   this made the early learning signal on 4 of 6 dimensions extremely
   weak. Constants below are set from the actual spawn ranges, the
   actual allowed_v ceiling (0.10 + 0.015*|x|, which tops out ~2.35 m/s
   at the farthest stage-4 spawn), and the actual rotational-thruster
   authority (~0.59 deg/s^2 impulse), each with headroom.

2. Added two derived-but-not-new-information features: normalized range
   (hypot(x,y)) and normalized closing rate (radial component of
   velocity). These are pure trigonometry on the existing x,y,vx,vy —
   no new physics, no reward change — but they give the network directly
   what it previously had to infer: "how far am I, and am I closing or
   opening." A scripted controller test (bang-bang, using this exact
   physics) confirmed the two axes are coupled via the CW cross-terms
   (3n^2*y + 2n*vx), so a policy that can't easily read range/closing
   rate has to reconstruct that coupling from raw x,y,vx,vy alone, which
   is a much harder representation-learning problem than it needs to be.

   NOTE: this changes observation_space from shape (6,) to (8,), so any
   previously-trained .zip models are incompatible and must be retrained
   from scratch. self.state (raw physical state, used by
   record_ai_flight.py for logging) is unchanged at 6 elements.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from physics_config import PHYSICS_CONFIG

class OrbitalDockingEnv(gym.Env):
    def __init__(self, spawn_range=(-150.0, -70.0), spawn_lateral=50.0,
                 tolerance_scale=1.0, max_episode_steps=6000):
        super(OrbitalDockingEnv, self).__init__()

        rcs = PHYSICS_CONFIG["rcs"]
        cap = PHYSICS_CONFIG["crew_capsule_chaser"]
        orb = PHYSICS_CONFIG["orbit"]
        dock = PHYSICS_CONFIG["docking_constraints"]

        self.spawn_range = spawn_range
        self.spawn_lateral = spawn_lateral
        self.tolerance_scale = tolerance_scale

        self.dry_mass = cap["dry_mass_kg"]
        self.fuel_mass_init = cap["fuel_mass_kg"]
        self.inertia = cap["moment_of_inertia_kgm2"]

        self.translation_thrust_n = rcs["translation_thrust_N"]
        self.rotation_thrust_n = rcs["rotation_thrust_N"]
        self.translation_burn_rate = rcs["translation_fuel_rate_kg_s"] * 2
        self.rotation_burn_rate = rcs["rotation_fuel_rate_kg_s"] * 4
        self.n = orb["mean_motion_n_rad_s"]

        self.dt = 0.1
        self.pulse_s = 0.02

        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        self.max_relative_velocity = dock["max_relative_velocity_mps"] * tolerance_scale
        self.max_lateral_offset = dock["max_lateral_offset_m"] * tolerance_scale
        self.max_misalign_deg = dock["max_angular_misalignment_deg"] * tolerance_scale
        self.max_angular_rate_dps = dock["max_angular_rate_dps"] * tolerance_scale
        self.capture_range_m = 0.5 * tolerance_scale

        self.hull_length_m = PHYSICS_CONFIG["bas_target"]["length_m"]
        self.hull_halfwidth_m = PHYSICS_CONFIG["bas_target"]["width_m"] / 2.0

        # --- REWRITE: Gated V-bar Approach to a 5m Hold Point ---
        # Target is no longer the docking port at (0,0) — this stage
        # targets a hold point short of the station, ahead of a later
        # hybrid control handover for the final approach/dock itself.
        self.hold_x = -5.0
        self.hold_y = 0.0

        self.max_consecutive_pulses = 8
        self.cooldown_steps = 5
        self._last_fired_action = np.array([0, 0, 0])
        # --- FIX: per-channel cooldown tracking (see step() below) ---
        self._consecutive_count = np.array([0, 0, 0])
        self._cooldown_remaining = 0

        # --- FIX: MultiDiscrete Action Space ---
        # [X-thrust, Y-thrust, Rotation]
        # 0 = Off, 1 = Positive/CCW, 2 = Negative/CW
        self.action_space = spaces.MultiDiscrete([3, 3, 3])

        self.max_x = 220.0
        self.max_y = 160.0
        self.max_vel = 5.0
        self.max_omega = 3.0
        self.max_pos = max(self.max_x, self.max_y)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.state = None
        self.fuel = None
        self.initial_x = 0.0

    def _normalize(self, x, y, vx, vy, theta, omega):
        dist = math.hypot(x, y)
        closing_rate = (x * vx + y * vy) / max(dist, 1e-3)
        range_scale = max(self.max_x, self.max_y)

        return np.array([
            np.clip(x / self.max_x, -1, 1),
            np.clip(y / self.max_y, -1, 1),
            np.clip(vx / self.max_vel, -1, 1),
            np.clip(vy / self.max_vel, -1, 1),
            np.clip(theta / 180.0, -1, 1),
            np.clip(omega / self.max_omega, -1, 1),
            np.clip(dist / range_scale, -1, 1),
            np.clip(closing_rate / self.max_vel, -1, 1),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0
        self.fuel = self.fuel_mass_init
        self._last_fired_action = np.array([0, 0, 0])
        self._consecutive_count = np.array([0, 0, 0])
        self._cooldown_remaining = 0

        if self.np_random.random() < 0.5:
            x_init = self.np_random.uniform(self.spawn_range[0], self.spawn_range[1])
            y_init = self.np_random.uniform(-self.spawn_lateral * 0.3, self.spawn_lateral * 0.3)
        else:
            x_init = self.np_random.uniform(self.spawn_range[0] * 0.8, self.spawn_range[1] * 0.8)
            y_init = self.np_random.uniform(-self.spawn_lateral, self.spawn_lateral)

        self.initial_x = x_init

        vx_init = self.np_random.uniform(-0.05, 0.05)
        vy_init = self.np_random.uniform(-0.05, 0.05)
        theta_init = self.np_random.uniform(-15.0, 15.0)
        omega_init = self.np_random.uniform(-0.3, 0.3)

        self.state = np.array([x_init, y_init, vx_init, vy_init, theta_init, omega_init], dtype=np.float32)
        return self._normalize(*self.state), {}

    def step(self, action):
        x, y, vx, vy, theta, omega = self.state
        self._elapsed_steps += 1

        action = np.array(action)

        # --- THRUSTER COOLDOWN LOGIC ---
        if self._cooldown_remaining > 0:
            action = np.array([0, 0, 0])
            self._cooldown_remaining -= 1
        elif np.any(action != 0):
            for ch in range(3):
                if action[ch] != 0 and action[ch] == self._last_fired_action[ch]:
                    self._consecutive_count[ch] += 1
                elif action[ch] != 0:
                    self._consecutive_count[ch] = 1
                else:
                    self._consecutive_count[ch] = 0

            self._last_fired_action = action.copy()

            if np.any(self._consecutive_count > self.max_consecutive_pulses):
                self._cooldown_remaining = self.cooldown_steps
                self._consecutive_count = np.array([0, 0, 0])
                action = np.array([0, 0, 0])
        else:
            self._consecutive_count = np.array([0, 0, 0])
            self._last_fired_action = np.array([0, 0, 0])

        Fx, Fy, tau = 0.0, 0.0, 0.0
        fuel_used = 0.0
        thruster_fired = False

        # --- THRUST MAPPING ---
        if self.fuel > 0 and np.any(action != 0):
            act_x, act_y, act_rot = action
            
            if act_x == 1:
                Fx = self.translation_thrust_n * 2
                fuel_used += self.translation_burn_rate * self.pulse_s
            elif act_x == 2:
                Fx = -self.translation_thrust_n * 2
                fuel_used += self.translation_burn_rate * self.pulse_s
                
            if act_y == 1:
                Fy = self.translation_thrust_n * 2
                fuel_used += self.translation_burn_rate * self.pulse_s
            elif act_y == 2:
                Fy = -self.translation_thrust_n * 2
                fuel_used += self.translation_burn_rate * self.pulse_s
                
            if act_rot == 1:
                tau = -self.rotation_thrust_n * 4 * 0.5
                fuel_used += self.rotation_burn_rate * self.pulse_s
            elif act_rot == 2:
                tau = self.rotation_thrust_n * 4 * 0.5
                fuel_used += self.rotation_burn_rate * self.pulse_s

            thruster_fired = True

        self.fuel = max(0.0, self.fuel - fuel_used)
        mass = self.dry_mass + self.fuel

        # --- PHYSICS ENGINE ---
        if thruster_fired and self.pulse_s < self.dt:
            ax_pulse = Fx / mass
            ay_pulse = Fy / mass
            alpha_pulse = tau / self.inertia

            vx_after = vx + ax_pulse * self.pulse_s
            vy_after = vy + ay_pulse * self.pulse_s
            omega_after = omega + math.degrees(alpha_pulse) * self.pulse_s

            coast_dt = self.dt - self.pulse_s
            ax_coast = -2 * self.n * vy_after
            ay_coast = 3 * (self.n ** 2) * y + 2 * self.n * vx_after

            new_vx = vx_after + ax_coast * coast_dt
            new_vy = vy_after + ay_coast * coast_dt
            new_omega = omega_after
            new_x = x + new_vx * self.dt
            new_y = y + new_vy * self.dt
            new_theta = theta + new_omega * self.dt
        else:
            ax_coast = -2 * self.n * vy
            ay_coast = 3 * (self.n ** 2) * y + 2 * self.n * vx
            new_vx = vx + ax_coast * self.dt
            new_vy = vy + ay_coast * self.dt
            new_x = x + new_vx * self.dt
            new_y = y + new_vy * self.dt
            new_omega = omega
            new_theta = theta + new_omega * self.dt

        new_theta = ((new_theta + 180.0) % 360.0) - 180.0

        # --- REWARD SHAPING (Fixed RL Math) ---
        reward = 0.0

        d_theta_signed = ((new_theta - theta + 180.0) % 360.0) - 180.0
        err_theta_old, err_theta_new = abs(theta), abs(new_theta)
        err_y_old, err_y_new = abs(y - self.hold_y), abs(new_y - self.hold_y)
        err_x_old, err_x_new = abs(x - self.hold_x), abs(new_x - self.hold_x)

        # 1. Distance Progress (Massive positive reinforcement for closing the gap)
        dist_old = math.hypot(err_x_old, err_y_old)
        dist_new = math.hypot(err_x_new, err_y_new)
        prog_dist = dist_old - dist_new
        reward += prog_dist * 15.0

        # 2. Attitude Progress (Dense reward for aligning)
        prog_theta = err_theta_old - err_theta_new
        reward += prog_theta * 0.5

        # 3. Mild Efficiency Bleed (Prevents loitering, but living is mathematically better than dying)
        reward -= 0.05 

        velocity_mag = math.hypot(new_vx, new_vy)

        # 4. Dynamic Braking (Glideslope)
        allowed_v = self.max_relative_velocity + (0.015 * err_x_new)
        if velocity_mag > allowed_v:
            reward -= 5.0 * (velocity_mag - allowed_v)

        # --- TERMINATION LOGIC (V-bar Hold Point) ---
        terminated = False
        truncated = False
        info = {}
        
        was_aligned = err_theta_old <= 5.0
        was_centered = err_y_old <= 2.0

        # X-DEATH BOUNDARY
        if new_x > 0.0 or abs(new_y) > 150.0:
            reward -= 500.0  # Scaled to balance against the new reward stream
            terminated = True
            info["outcome"] = "out_of_bounds"

        elif err_x_new < self.capture_range_m and err_y_new < self.capture_range_m:
            if velocity_mag < (self.max_relative_velocity * 0.5):
                # PRECISION BONUS
                y_precision = max(0.0, 1.0 - (err_y_new / max(self.capture_range_m, 1e-3)))
                theta_precision = max(0.0, 1.0 - (err_theta_new / 5.0))
                reward += 1000.0 + 500.0 * y_precision + 500.0 * theta_precision
                terminated = True
                info["outcome"] = "docked"
            else:
                reward += 100.0
                terminated = True
                info["outcome"] = "soft_crash"

        if self._elapsed_steps >= self.max_episode_steps and not terminated:
            reward -= 200.0
            truncated = True
            info["outcome"] = "truncated"

        self.state = np.array([new_x, new_y, new_vx, new_vy, new_theta, new_omega], dtype=np.float32)
        norm_state = self._normalize(*self.state)
        info["fuel_kg"] = self.fuel
        info["velocity_mag"] = velocity_mag
        info["allowed_v"] = allowed_v
        
        info["range_m"] = math.hypot(new_x - self.hold_x, new_y - self.hold_y)
        info["err_x"] = err_x_new
        info["err_y"] = err_y_new
        info["phase"] = 1 if not was_aligned else (2 if not was_centered else 3)
        info["actual_action"] = action.tolist()

        # Rounded telemetry for JSON logs 
        info["telemetry_rounded"] = {
            "x": round(float(new_x), 2), "y": round(float(new_y), 2),
            "vx": round(float(new_vx), 2), "vy": round(float(new_vy), 2),
            "theta": round(float(new_theta), 2), "omega": round(float(new_omega), 2),
        }

        return norm_state, float(reward), terminated, truncated, info