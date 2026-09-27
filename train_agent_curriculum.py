import gymnasium as gym
import json
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

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

class OutcomeDashboardCallback(BaseCallback):
    def __init__(self, stage_name="", print_every=50, save_every=10, verbose=0):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.print_every = print_every
        self.save_every = save_every
        self.episode_count = 0
        self.docks = 0
        self.soft_crashes = 0
        self.hard_crashes = 0
        self.hull_strikes = 0
        self.out_of_bounds = 0
        self.timeouts = 0
        self.episode_log = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.episode_count += 1
            outcome = info.get("outcome", "unknown")
            if outcome == "docked":
                self.docks += 1
            elif outcome == "soft_crash":
                self.soft_crashes += 1
            elif outcome == "hard_crash":
                self.hard_crashes += 1
            elif outcome == "hull_strike":
                self.hull_strikes += 1
            elif outcome == "out_of_bounds":
                self.out_of_bounds += 1
            elif outcome == "truncated":
                self.timeouts += 1

            self.episode_log.append({"episode": self.episode_count, "outcome": outcome})
            self.episode_log = self.episode_log[-500:]

            if self.episode_count % self.save_every == 0:
                self._save()

            if self.episode_count % self.print_every == 0:
                print(f"--- [{self.stage_name}] EPISODE {self.episode_count} "
                      f"(last outcome: {outcome.upper()}) ---")
                print(f"Docks: {self.docks} | Soft crashes: {self.soft_crashes} | "
                      f"Hard crashes: {self.hard_crashes} | Hull strikes: {self.hull_strikes} | "
                      f"Out of bounds: {self.out_of_bounds} | Timeouts: {self.timeouts}")
        return True

    def _save(self):
        data = _load_mission_log()
        data.setdefault("training", {}).setdefault("stages", {})[self.stage_name] = {
            "episode_count": self.episode_count,
            "docks": self.docks,
            "soft_crashes": self.soft_crashes,
            "hard_crashes": self.hard_crashes,
            "hull_strikes": self.hull_strikes,
            "out_of_bounds": self.out_of_bounds,
            "timeouts": self.timeouts,
            "episode_log": self.episode_log,
        }
        _save_mission_log(data)

    def finalize(self):
        self._save()


class PlateauGuardCallback(BaseCallback):
    """
    --- FIX: auto-stop when more training stops making sense ---
    Two full curriculum runs burned real wall-clock time producing
    trend lines that were readable well before the stage finished: reward
    monotonically getting worse across stage transitions, or hundreds of
    episodes at 100% timeout with zero docks and zero crashes of any kind.
    Both were visible at well under half of the stage's timestep budget,
    but nothing in the original code stopped the run early or flagged it.

    This callback checks, every `check_every_episodes` episodes once
    `min_fraction_before_check` of the stage's total timestep budget has
    elapsed:
      1. Has this stage produced at least `min_docks_required` dock(s)
         yet? If yes -> stage is demonstrably solvable, let it keep going.
      2. If not: has reward meaningfully improved? Compare the mean
         reward of the most recent `reward_window` completed episodes
         against the mean reward of the FIRST `reward_window` episodes of
         this stage. If it hasn't improved by at least
         `min_improvement`, the stage is plateaued with no docks to show
         for it -> stop this stage's training immediately (return False)
         and set self.aborted = True so main() can halt the whole
         curriculum instead of wasting budget on stages that are only
         going to be harder.

    This is a heuristic, not a proof — a stage that is learning very
    slowly but would eventually succeed could in principle trigger this.
    That tradeoff is intentional: given repeated evidence of runs that
    ran to completion doing nothing useful, stopping early and letting a
    human decide whether to loosen the thresholds and retry a single
    stage is a better default than silently burning the rest of a
    3-6 million step budget on a curriculum that's already failed.
    """
    def __init__(self, stage_name, stage_timesteps,
                 min_fraction_before_check=0.5, check_every_episodes=25,
                 reward_window=50, min_improvement=50.0,
                 min_docks_required=1, persistence=2, verbose=1):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.stage_timesteps = stage_timesteps
        self.min_fraction_before_check = min_fraction_before_check
        self.check_every_episodes = check_every_episodes
        self.reward_window = reward_window
        self.min_improvement = min_improvement
        self.min_docks_required = min_docks_required
        # --- FIX: require the plateau to persist, not just appear once ---
        # A run that aborted with "change=+41.0, needed >= +50.0" was
        # genuinely improving — just not past an arbitrary bar on ONE
        # check. A single borderline reading is noise as much as it is
        # signal; requiring the same failing verdict on `persistence`
        # consecutive checks (spaced check_every_episodes apart) before
        # acting on it filters out exactly that false-positive without
        # losing the ability to catch a real, sustained plateau — a
        # genuinely stuck stage will keep failing the check every time,
        # a genuinely improving one will clear the bar before the streak
        # completes.
        self.persistence = persistence
        self.consecutive_fails = 0

        self.episode_count = 0
        self.docks = 0
        self.episode_rewards = []
        self.baseline_reward = None
        self.aborted = False
        self.abort_reason = ""
        self.start_timesteps = 0

    def _on_training_start(self) -> None:
        # --- FIX: stage-local step counting ---
        # self.model.num_timesteps is CUMULATIVE across the whole curriculum
        # (it carries over between stages since reset_num_timesteps=False).
        # Recording it here at the start of THIS stage's learn() call lets
        # us compute how many steps have elapsed within this stage alone,
        # instead of comparing an inherited multi-stage total against a
        # single stage's budget (which made the checkpoint trigger
        # immediately on every stage after the first).
        self.start_timesteps = self.model.num_timesteps

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        just_reached_checkpoint = False

        for done, info in zip(dones, infos):
            if not done:
                continue
            self.episode_count += 1
            if info.get("outcome") == "docked":
                self.docks += 1
            # --- FIX: stage-local reward history ---
            # SB3's model.ep_info_buffer persists across stages (it isn't
            # cleared when a new learn() call starts with
            # reset_num_timesteps=False), so it was mixing the tail of the
            # PREVIOUS stage's episodes into this stage's "baseline",
            # making an intentionally-harder new stage look like a false
            # regression. Monitor already stamps a fresh "episode" dict
            # into info on every episode end — using that directly gives
            # a reward history that only ever contains THIS stage's
            # episodes, since this callback instance is recreated fresh
            # for every stage in main()'s loop.
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            # --- FIX: checkpoint-repetition bug ---
            # `episode_count % check_every_episodes == 0` stays true for
            # every timestep until the NEXT episode ends (episode_count
            # doesn't change on non-terminal steps), so evaluating it
            # after the loop unconditionally re-ran the same nominal
            # checkpoint on every single step of the following episode —
            # visible as the same checkpoint being logged 10+ times in a
            # row. That let consecutive_fails race past the `persistence`
            # threshold within what should count as ONE check, quietly
            # defeating the persistence protection above. Only flag a
            # checkpoint the one time episode_count actually CROSSES a
            # multiple of check_every_episodes.
            if self.episode_count % self.check_every_episodes == 0:
                just_reached_checkpoint = True

        elapsed_this_stage = self.model.num_timesteps - self.start_timesteps

        if self.baseline_reward is None and len(self.episode_rewards) >= self.reward_window:
            self.baseline_reward = float(np.mean(self.episode_rewards[:self.reward_window]))

        past_checkpoint = elapsed_this_stage >= self.min_fraction_before_check * self.stage_timesteps

        if past_checkpoint and just_reached_checkpoint and self.baseline_reward is not None:
            recent_reward = float(np.mean(self.episode_rewards[-self.reward_window:]))
            improvement = recent_reward - self.baseline_reward
            failing = self.docks < self.min_docks_required and improvement < self.min_improvement

            if failing:
                self.consecutive_fails += 1
            else:
                self.consecutive_fails = 0  # a good check clears the streak entirely

            print(f"[{self.stage_name}] plateau check @ {elapsed_this_stage:,}/{self.stage_timesteps:,}: "
                  f"docks={self.docks}, reward {self.baseline_reward:.1f} -> {recent_reward:.1f} "
                  f"({improvement:+.1f}), fail-streak={self.consecutive_fails}/{self.persistence}")

            if self.consecutive_fails >= self.persistence:
                self.aborted = True
                self.abort_reason = (
                    f"[{self.stage_name}] STOPPING EARLY at {elapsed_this_stage:,}/"
                    f"{self.stage_timesteps:,} timesteps INTO THIS STAGE "
                    f"({self.episode_count} episodes this stage), after "
                    f"{self.consecutive_fails} consecutive non-improving checks.\n"
                    f"  Docks so far: {self.docks} (need >= {self.min_docks_required})\n"
                    f"  Reward trend: baseline={self.baseline_reward:.1f} -> "
                    f"recent={recent_reward:.1f} (change={improvement:+.1f}, "
                    f"needed >= {self.min_improvement:+.1f})\n"
                    f"  This stage is not producing docks and reward has not "
                    f"meaningfully improved across multiple checks. Continuing "
                    f"would very likely just burn the rest of this stage's budget "
                    f"for no gain, and every later stage is harder than this one."
                )
                print("\n" + "=" * 60)
                print(self.abort_reason)
                print("=" * 60 + "\n")
                return False  # tells SB3's model.learn() to stop now

        return True

set_random_seed(42)

# --- THE FIX: A 4-Stage Curriculum ---
# We force the AI to spend over 1.5 million timesteps in the middle stages 
# where it is far enough away that it MUST use retro-thrusters to slow down 
# before hitting the target, solving the "Speed Demon" overshoot problem.
STAGES = [
    {
        "name": "stage1_fundamentals",
        "spawn_range": (-20.0, -10.0),
        "spawn_lateral": 5.0,
        "tolerance_scale": 4.0,
        "max_episode_steps": 1000,
        "timesteps": 300_000,
    },
    {
        # --- FIX: curriculum-pacing gap #1 (distance) ---
        # See stage2b/stage3b comments below for the full diagnosis. This
        # step only grows spawn distance meaningfully; tolerance stays
        # close to stage1's so only one thing gets harder at a time.
        "name": "stage1b_midrange",
        "spawn_range": (-35.0, -18.0),
        "spawn_lateral": 12.0,
        "tolerance_scale": 3.2,
        "max_episode_steps": 1400,
        "timesteps": 300_000,
    },
    {
        # --- FIX: curriculum-pacing gap #1.5 (distance THEN tolerance) ---
        # The stage1b -> stage2_braking jump still changed spawn distance
        # (-35/-18 -> -60/-30) AND tolerance (3.2x -> 2.5x) in the same
        # step, despite the stated "one thing at a time" design. A real
        # run showed hull_strike rate falling episode-over-episode while
        # mean reward kept getting WORSE and out_of_bounds rose in
        # parallel — the policy was trading one failure mode for another,
        # not converging, and the plateau guard correctly killed it at
        # 372k/600k steps with zero docks. This stage grows ONLY the
        # spawn distance, at stage1b's already-comfortable 3.2x tolerance,
        # so the model consolidates "cover more ground" before
        # stage2_braking (below) asks it to also be more precise once it
        # gets there.
        "name": "stage1c_distance",
        "spawn_range": (-60.0, -30.0),
        "spawn_lateral": 20.0,
        "tolerance_scale": 3.2,
        "max_episode_steps": 1800,
        "timesteps": 400_000,
    },
    {
        "name": "stage2_braking",
        "spawn_range": (-60.0, -30.0),
        "spawn_lateral": 20.0,
        "tolerance_scale": 2.5,
        "max_episode_steps": 1800,
        "timesteps": 600_000,
    },
    {
        # --- FIX: curriculum-pacing gap #2 (distance + tolerance) ---
        # A full curriculum run (original 4 stages) showed mean eval
        # reward getting WORSE at every single stage transition
        # (146 -> -337 -> -411 -> -492), never recovering. Every original
        # transition increased spawn distance AND tightened the
        # capture/velocity/alignment tolerance in the same step,
        # compounding two harder problems at once before the model
        # consolidated the first one. This stage (and stage3b below)
        # bisect the distance and tolerance jumps between stage2->stage3
        # and stage3->stage4 so each transition only makes one thing
        # meaningfully harder at a time.
        "name": "stage2b_extended",
        "spawn_range": (-80.0, -40.0),
        "spawn_lateral": 27.0,
        "tolerance_scale": 2.0,
        "max_episode_steps": 2800,
        "timesteps": 400_000,
    },
    {
        # --- FIX: max_episode_steps 2500 -> 4000 (250s -> 400s). ---
        # You docked this same task manually in 20-30 minutes using full
        # simultaneous multi-axis control. 250s gave the agent ~1/5 to 1/7
        # of that time using a MORE restrictive interface (one thruster
        # direction per step, no simultaneous translate+rotate). That's a
        # plausible reason for blanket timeouts here independent of how
        # good the policy itself gets.
        "name": "stage3_approach",
        "spawn_range": (-100.0, -50.0),
        "spawn_lateral": 35.0,
        "tolerance_scale": 1.5,
        "max_episode_steps": 4000,
        "timesteps": 1_000_000,
    },
    {
        # --- FIX: curriculum-pacing gap #3 (distance + tolerance) ---
        # Same reasoning as stage2b: bisects the stage3->stage4 jump
        # (100m->150m distance, 1.5x->1.0x tolerance) into two smaller
        # steps instead of one large compound jump.
        "name": "stage3b_precision",
        "spawn_range": (-125.0, -60.0),
        "spawn_lateral": 42.0,
        "tolerance_scale": 1.2,
        "max_episode_steps": 5000,
        "timesteps": 500_000,
    },
    {
        # --- FIX: max_episode_steps 3000 -> 6000 (300s -> 600s). ---
        # Same reasoning. Still less than half your manual time —
        # deliberately conservative, since longer episodes cost more
        # wall-clock training time per env step. If the agent still can't
        # solve it in 600s after this, that's a stronger signal to make
        # the action-space fix (see PROJECT_GUIDE.md) than to keep raising
        # this number indefinitely.
        "name": "stage4_full_mission",
        "spawn_range": (-150.0, -70.0),
        "spawn_lateral": 50.0,
        "tolerance_scale": 1.0,
        "max_episode_steps": 6000,
        "timesteps": 1_500_000,
    },
]

def make_env(stage):
    def _init():
        return Monitor(OrbitalDockingEnv(
            spawn_range=stage["spawn_range"],
            spawn_lateral=stage["spawn_lateral"],
            tolerance_scale=stage["tolerance_scale"],
            max_episode_steps=stage["max_episode_steps"],
        ))
    return _init

N_ENVS = 8
VECNORM_PATH = "vecnormalize_stats.pkl"

def main():
    # --- FIX: resume-from-stage support ---
    # main() had no way to skip stages whose .zip already exists on disk —
    # every run started stage1_fundamentals from scratch, meaning re-running
    # this script after an abort (or after inserting a new stage, as above)
    # re-trained everything from the top. That's hours of wasted compute
    # for a script you're re-running daily. `--resume-from <stage_name>`
    # loads that stage's model + the persisted VecNormalize stats and
    # starts training AT that stage, skipping everything earlier — those
    # earlier stages' saved .zip files are left untouched.
    #
    #   python train_agent_curriculum.py --resume-from stage2_braking
    #
    # requires ppo_orbital_docker_stage1c_distance.zip (the stage BEFORE
    # the resume point) and vecnormalize_stats.pkl to already exist.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", default=None,
                         help="Stage name to resume training at (skips all earlier stages).")
    args = parser.parse_args()

    start_idx = 0
    model = None
    results = []

    if args.resume_from:
        names = [s["name"] for s in STAGES]
        if args.resume_from not in names:
            raise SystemExit(f"Unknown stage '{args.resume_from}'. Valid stages: {names}")
        start_idx = names.index(args.resume_from)
        if start_idx == 0:
            print("Resuming from the first stage — nothing to skip.")
        else:
            prev_stage_name = STAGES[start_idx - 1]["name"]
            prev_model_path = f"ppo_orbital_docker_{prev_stage_name}.zip"
            if not os.path.exists(prev_model_path):
                raise SystemExit(
                    f"Cannot resume at '{args.resume_from}': expected the previous "
                    f"stage's model at '{prev_model_path}', which doesn't exist."
                )
            if not os.path.exists(VECNORM_PATH):
                raise SystemExit(
                    f"Cannot resume at '{args.resume_from}': '{VECNORM_PATH}' not found. "
                    f"It's required to continue reward normalization correctly."
                )
            print(f"Resuming from '{args.resume_from}' — loading '{prev_model_path}'.")
            model = PPO.load(prev_model_path)

    for i, stage in enumerate(STAGES):
        if i < start_idx:
            print(f"Skipping already-completed stage {i+1}/{len(STAGES)}: {stage['name']}")
            continue
        print(f"\n=== STAGE {i+1}/{len(STAGES)}: {stage['name']} "
              f"(spawn {stage['spawn_range']}, tolerance x{stage['tolerance_scale']}) ===")
        raw_env = DummyVecEnv([make_env(stage) for _ in range(N_ENVS)])

        # --- FIX: reward normalization ---
        # Per-step shaping rewards are O(0.1-1), terminal rewards are
        # O(500-2000), with nothing in the original script scaling either
        # one. PPO's value function has to fit both scales simultaneously,
        # which destabilizes the advantage estimates it uses to update the
        # policy. norm_obs stays False because the environment's own
        # _normalize() already puts observations in a sane, hand-verified
        # [-1,1] range (see docking_env.py) — normalizing twice would just
        # add noise. Only the reward stream is rescaled here.
        #
        # Stats are persisted across curriculum stages (loaded from the
        # previous stage, saved at the end of this one) for the same
        # reason model weights carry over: stage-to-stage continuity.
        if os.path.exists(VECNORM_PATH) and i > 0:
            env = VecNormalize.load(VECNORM_PATH, raw_env)
        else:
            env = VecNormalize(raw_env, norm_obs=False, norm_reward=True,
                                clip_reward=50.0, gamma=0.999)
        env.training = True

        checkpoint_callback = CheckpointCallback(
            save_freq=max(50000 // N_ENVS, 1),
            save_path=f"./models/{stage['name']}/",
            name_prefix="ppo_docking"
        )
        dashboard_callback = OutcomeDashboardCallback(stage_name=stage["name"], print_every=50)
        plateau_callback = PlateauGuardCallback(
            stage_name=stage["name"],
            stage_timesteps=stage["timesteps"],
        )

        if model is None:
            # ent_coef remains at 0.02 to ensure it keeps exploring rotational thrusters
            #
            # --- FIX: gamma was left at PPO's default (0.99). With dt=0.1s
            # and episodes running up to thousands of steps, gamma=0.99
            # gives an effective credit-assignment horizon of only ~100
            # steps (10 sim-seconds): 0.99**2000 ~= 2e-9. The docking/crash
            # terminal reward, landing 1000+ steps into the episode, was
            # mathematically almost invisible to the gradient for any
            # action taken more than ~200 steps before the episode ended.
            # This is very likely why training only ever produced
            # "truncated" outcomes. gamma=0.999 gives a ~1000-step (100s)
            # horizon; gae_lambda raised slightly (0.95->0.98) to reduce
            # bias in the advantage estimate over that longer horizon.
            #
            # --- FIX: n_steps raised 512->2048 (batch_size 256->512 to
            # match) so each policy update sees substantially more of an
            # episode's trajectory before bootstrapping.
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.0003,
                ent_coef=0.02,
                gamma=0.999,
                gae_lambda=0.98,
                n_steps=2048,
                batch_size=512,
                tensorboard_log="./tensorboard_logs/"
            )
        else:
            model.set_env(env)

        model.learn(total_timesteps=stage["timesteps"],
                    callback=[checkpoint_callback, dashboard_callback, plateau_callback],
                    reset_num_timesteps=False)
        dashboard_callback.finalize()

        if plateau_callback.aborted:
            abort_path = f"ppo_orbital_docker_{stage['name']}_ABORTED"
            model.save(abort_path)
            env.save(VECNORM_PATH)
            print("\n" + "=" * 60)
            print("CURRICULUM STOPPED EARLY — NOT ALL STAGES COMPLETED")
            print("=" * 60)
            print(plateau_callback.abort_reason)
            print(f"\nPartial model for this stage saved as '{abort_path}.zip' "
                  f"in case you want to inspect it.")
            if results:
                print("\nStages that DID complete normally before this one:")
                for name, mean_r, std_r in results:
                    print(f"  {name:25s} mean reward: {mean_r:>12.2f}  +/- {std_r:>10.2f}")
            print("=" * 60)
            return

        model.save(f"ppo_orbital_docker_{stage['name']}")
        env.save(VECNORM_PATH)
        print(f"Stage {i+1} training complete. Evaluating...")

        # Eval env is intentionally plain (no VecNormalize wrapper) so the
        # printed mean reward stays in the same raw units across every
        # stage and is directly comparable to pre-fix runs — only the
        # TRAINING reward stream is rescaled, not what gets reported here.
        eval_env = Monitor(OrbitalDockingEnv(
            spawn_range=stage["spawn_range"], spawn_lateral=stage["spawn_lateral"],
            tolerance_scale=stage["tolerance_scale"], max_episode_steps=stage["max_episode_steps"]))
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        results.append((stage["name"], mean_reward, std_reward))

    model.save("ppo_orbital_docker_final")

    full_env = Monitor(OrbitalDockingEnv(
        spawn_range=STAGES[-1]["spawn_range"], spawn_lateral=STAGES[-1]["spawn_lateral"],
        tolerance_scale=STAGES[-1]["tolerance_scale"], max_episode_steps=STAGES[-1]["max_episode_steps"]))
    full_mean, full_std = evaluate_policy(model, full_env, n_eval_episodes=20)
    results.append(("FULL_DIFFICULTY", full_mean, full_std))

    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    for name, mean_r, std_r in results:
        print(f"{name:25s} mean reward: {mean_r:>12.2f}  +/- {std_r:>10.2f}")
    print("=" * 60)
    print("Final model saved as 'ppo_orbital_docker_final.zip'.")

if __name__ == "__main__":
    main()