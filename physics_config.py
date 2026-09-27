# GMAT_RL_Docking_Project/physics_config.py
#SAIIIIIAJFNESKJN
# COORDINATE CONVENTION: docking_env.py's (x, y) state is measured relative
# to the BAS docking port, which is the origin (0, 0) by definition.
# docking_port_offset_m below is (YES) ONLY for drawing the BAS hull relative to
# that same origin (hull center is at -docking_port_offset_m along the
# station's long axis). Any renderer must read this value rather than ( WE CAN CHANFGE IT)
# hardcoding its own offset — that mismatch was the root cause of the
# "AI thinks it's docking at (0,0) but the hull is drawn 14m over" bug.

PHYSICS_CONFIG = {
    "orbit": {
        "altitude_km": 400.0,
        "earth_radius_m": 6371000.0,
        "orbital_radius_m": 6771000.0,
        "mu_earth_m3_s2": 3.986004418e14,
        "mean_motion_n_rad_s": 1.1330e-3,
        "orbital_period_s": 5548.0,
    },
    "bas_target": {
        "name": "BAS",
        "dry_mass_kg": 82000.0,
        "fuel_mass_kg": 3000.0,
        "length_m": 38.0,
        "width_m": 6.0,
        "solar_array_span_m": 46.0,
        "moment_of_inertia_kgm2": 1.01e7,
        "docking_port_offset_m": [17.0, 0.0],
        "attitude_fixed": True,
    },
    "crew_capsule_chaser": {
        "name": "SAI",
        "dry_mass_kg": 4200.0,
        "fuel_mass_kg": 1500.0,
        "length_m": 5.0,
        "width_m": 4.0,
        "moment_of_inertia_kgm2": 19475.0,
        "docking_port_offset_m": [2.5, 0.0],
    },
    "rcs": {
        "num_translation_thrusters": 8,
        "translation_thrust_N": 400.0,
        "num_rotation_thrusters": 8,
        "rotation_thrust_N": 100.0,
        "specific_impulse_s": 230.0,
        "translation_fuel_rate_kg_s": 0.1773,
        "rotation_fuel_rate_kg_s": 0.0443,
    },
    "action_space": {
        "type": "Discrete",
        "n": 8,
        "actions": {
            0: "NOOP",
            1: "TRANSLATE_PLUS_X",
            2: "TRANSLATE_MINUS_X",
            3: "TRANSLATE_PLUS_Y",
            4: "TRANSLATE_MINUS_Y",
            5: "ROTATE_CCW",
            6: "ROTATE_CW",
            7: "DOCK_ATTEMPT",
        },
    },
    "docking_constraints": {
        "max_relative_velocity_mps": 0.10,
        "max_relative_velocity_lateral_mps": 0.05,
        "max_angular_misalignment_deg": 5.0,
        "max_angular_rate_dps": 0.5,
        "max_lateral_offset_m": 0.15,
        "capture_corridor_half_angle_deg": 10.0,
    },
    "failure_conditions": {
        "max_relative_velocity_collision_mps": 0.5,
        "max_range_m": 2000.0,
        "min_range_m": 0.0,
    },
}
