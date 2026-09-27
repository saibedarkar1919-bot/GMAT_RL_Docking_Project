import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from docking_env import OrbitalDockingEnv

def evaluate():
    # Load the final trained model
    model_path = "ppo_orbital_docker_final.zip"
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create a full-difficulty evaluation environment with randomized resets
    env = Monitor(OrbitalDockingEnv(
        spawn_range=(-150.0, -70.0),
        spawn_lateral=50.0,
        tolerance_scale=1.0,
        max_episode_steps=6000
    ))

    print("\nRunning Monte Carlo Evaluation across 100 randomized episodes...")
    
    outcomes = {"docked": 0, "soft_crash": 0, "hard_crash": 0, "out_of_bounds": 0, "truncated": 0}
    total_rewards = []

    for ep in range(100):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        outcome = "truncated"

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                outcome = info.get("outcome", "truncated")
                done = True

        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_rewards.append(ep_reward)

    print("\n" + "="*40)
    print("MONTE CARLO EVALUATION RESULTS (100 Runs)")
    print("="*40)
    print(f"Successful Docks:    {outcomes['docked']}/100 ({outcomes['docked']}%)")
    print(f"Soft Crashes:        {outcomes['soft_crash']}/100")
    print(f"Hard Crashes:        {outcomes['hard_crash']}/100")
    print(f"Out of Bounds:       {outcomes['out_of_bounds']}/100")
    print(f"Timeouts (Truncated):{outcomes['truncated']}/100")
    print(f"Mean Reward:         {sum(total_rewards)/len(total_rewards):.2f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()