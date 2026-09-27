import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

# ---------------------------------------------------------
# We are importing your custom environment from docking_env.py
# ---------------------------------------------------------
from docking_env import OrbitalDockingEnv 

class TrainingDashboardCallback(BaseCallback):
    """
    Custom callback to print episode outcomes so you don't 
    have to stare at JSON logs to know what happened.
    """
    def __init__(self, verbose=0):
        super(TrainingDashboardCallback, self).__init__(verbose)
        self.episode_count = 0
        self.out_of_bounds = 0
        self.crashes = 0
        self.docks = 0
        self.timeouts = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals["dones"][0]:
            self.episode_count += 1
            info = self.locals["infos"][0]
            outcome = info.get("outcome", "unknown")
            
            # Track statistics
            if outcome == "out_of_bounds":
                self.out_of_bounds += 1
            # docking_env.py actually emits 'soft_crash'/'hard_crash'/'hull_strike',
            # never 'collision' — the old check here silently never matched.
            elif outcome in ("soft_crash", "hard_crash", "hull_strike"):
                self.crashes += 1
            elif outcome == "docked":
                self.docks += 1
            elif outcome == "truncated":
                self.timeouts += 1

            # Print an update every 50 episodes
            if self.episode_count % 50 == 0:
                print(f"--- EPISODE {self.episode_count} ---")
                print(f"Recent Outcome: {outcome.upper()}")
                print(f"Total Docks so far: {self.docks}")
                print(f"Total Out of Bounds: {self.out_of_bounds}")
                print(f"Total Crashes: {self.crashes}")
                print(f"Total Timeouts: {self.timeouts}")
                print("----------------------\n")
        return True

def main():
    print("Initializing Environment...")
    env = OrbitalDockingEnv()
    
    # Check if a model already exists and delete it to start totally fresh
    model_path = "orbital_agent.zip"
    if os.path.exists(model_path):
        print(f"Found old model {model_path}. Deleting it to start fresh with new brain.")
        os.remove(model_path)

    print("Creating Newborn PPO Agent...")
    # --- FIX: gamma/n_steps ---
    # Default env here (no args) spawns at full stage-4 difficulty
    # (-150 to -70m) from episode 1, with max_episode_steps now 6000 (see
    # docking_env.py's default). The default PPO gamma=0.99 gives an
    # effective ~100-step credit-assignment horizon, so the docking/crash
    # reward landing 1000+ steps into an episode was almost entirely
    # discounted away before it could influence early-episode actions.
    # gamma=0.999 stretches that horizon to ~1000 steps. n_steps raised to
    # 2048 (SB3's own default) so each update sees more of an episode
    # before bootstrapping. See docking_env.py and train_agent_curriculum.py
    # for the matching observation-scaling and reward-normalization fixes —
    # this task is fundamentally easier to learn starting from the
    # curriculum script rather than full difficulty cold, so consider using
    # train_agent_curriculum.py instead of this file where possible.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0, # We set this to 0 so it doesn't spam standard metrics
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01 # Encourages the agent to explore all thruster buttons
    )

    print("Starting Training! Watch the dashboard...")
    callback = TrainingDashboardCallback()
    
    # Train for 500,000 steps
    model.learn(total_timesteps=5000000, callback=callback)

    print("Training Complete. Saving Agent.")
    model.save("orbital_agent")

if __name__ == "__main__":
    main()