# 2D Orbital Docking Reinforcement Learning Environment

A custom 2D Clohessy-Wiltshire relative-motion orbital mechanics environment for training a spacecraft (chaser) to execute proximity operations and docking via Proximal Policy Optimization (PPO).

## Technical Architecture
* **Environment:** `gymnasium`-based custom environment (`OrbitalDockingEnv`) modeling Clohessy-Wiltshire equations of motion, RCS fuel consumption, and attitude dynamics.
* **Action Space:** `MultiDiscrete([3, 3, 3])` for simultaneous X-axis translation, Y-axis translation, and rotational thruster control.
* **Agent:** PPO implementation using `stable-baselines3` with an 8-stage curriculum learning pipeline.
* **Visualization:** Python Flask backend and static HTML/JS web dashboard for trajectory and telemetry playback.

## Project Structure
* `docking_env.py`: Core environment physics, reward functions, and observation normalization.
* `physics_config.py`: Orbital constants, spacecraft mass/inertia specs, and RCS parameters.
* `train_agent_curriculum.py`: Multi-stage curriculum training script with PlateauGuard and Outcome callbacks.
* `evaluate_model.py`: Monte Carlo evaluation script for randomized initial conditions.
* `play_manual.py`: Human-in-the-loop keyboard override using Pygame.
* `record_ai_flight.py`: Evaluates and records trained models into JSON telemetry for web visualization.
* `server_gmat_viz.py`: Flask server hosting the mission control dashboard.

## Setup & Installation
Requires Python 3.10+ (Standard virtual environment, no Docker required).

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install gymnasium stable-baselines3 pygame flask numpy
