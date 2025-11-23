"""
MAIN ENTRYPOINT (prereg v1.1)

Runs:
- 7 tasks (2 MiniGrid + 5 Procgen games)
- 4 conditions:
    baseline
    adaptive_entropy
    kl_only
    trp_full
- 10 fixed seeds per task, paired across conditions

Uses pilot locks from configs/pilot_locked.json:
- baseline entropy bonus ent_coef
- dt band [ell, u]
- KL kappa

Saves per-run logs to:
results/<family>/<env>/<condition>/<seed>/logs.json
"""

import argparse, json, os, random
import numpy as np
import torch
import yaml
from pathlib import Path

from algorithms.ppo_trp import PPO_TRP_Trainer, PPOConfig
from envs.wrappers import MiniGridTRPWrapper, ProcgenTRPWrapper
import gymnasium as gym


# -------------------------
# Flatten helpers (same as pilot)
# -------------------------

class FlattenObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(np.prod(obs_shape)),), dtype=np.float32
        )

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return obs.reshape(-1)


class ProcgenSingleEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space["rgb"].shape if isinstance(env.observation_space, dict) else env.observation_space.shape
        H, W, C = shp[1], shp[2], shp[3]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(H, W, C), dtype=np.float32
        )
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}
        if isinstance(obs, dict):
            obs = obs["rgb"]
        obs = obs[0].astype(np.float32) / 255.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.array([action]))
        if isinstance(obs, dict):
            obs = obs["rgb"]
        obs = obs[0].astype(np.float32) / 255.0
        return obs, float(reward[0]), bool(terminated[0]), bool(truncated[0]), info


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------
# Env factories per prereg
# -------------------------

def make_minigrid_env(env_id):
    env = gym.make(env_id)
    # minigrid obs is usually a dict; use flatten wrapper for now.
    try:
        from minigrid.wrappers import FlatObsWrapper
        env = FlatObsWrapper(env)
    except Exception:
        # fallback: raw obs
        pass
    env = FlattenObs(env)
    return env


def make_minigrid_ladder(task_name):
    """
    Locked ladders:
    - Empty-8x8 <-> Empty-16x16
    - KeyCorridorS3R3 -> S4R3 -> S5R3
    """
    if task_name == "Empty":
        ladder = ["MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-16x16-v0"]
    elif task_name == "KeyCorridor":
        ladder = [
            "MiniGrid-KeyCorridorS3R3-v0",
            "MiniGrid-KeyCorridorS4R3-v0",
            "MiniGrid-KeyCorridorS5R3-v0",
        ]
    else:
        raise ValueError("Unknown MiniGrid ladder task.")
    return MiniGridTRPWrapper(make_minigrid_env, ladder)


def make_procgen_env(name):
    from procgen import ProcgenEnv
    env = ProcgenEnv(num_envs=1, env_name=name, distribution_mode="default", num_levels=0, start_level=0)
    env = ProcgenSingleEnv(env)
    env = FlattenObs(env)
    env = ProcgenTRPWrapper(env)  # gives harder/easier knobs
    return env


# -------------------------
# Condition run loop (unified)
# -------------------------

def run_condition(trainer, total_steps, condition, baseline_ent):
    """
    Unified training loop.
    adaptive_entropy uses a simple prereg-safe decay schedule (standard baseline class).
    """
    step_i = 0
    last_logps_t = None
    rollout_len = trainer.cfg.rollout_len

    while step_i < total_steps:
        # Adaptive entropy schedule (prereg allows standard schedule; not numerically pinned)
        if condition == "adaptive_entropy":
            frac = step_i / float(total_steps)
            # cosine decay from baseline_ent to 0.5*baseline_ent
            trainer.ent_coef = baseline_ent * (0.75 + 0.25 * np.cos(np.pi * frac))

        obses, acts, logps, ents, vals, rews, dones = trainer.rollout()

        # KL leash check for kl_only / trp_full
        if condition in ("kl_only", "trp_full") and trainer.leash is not None:
            logps_t = torch.tensor(logps, dtype=torch.float32)
            if last_logps_t is not None:
                triggered, kl = trainer.leash.check(logps_t, last_logps_t)
                trainer.logs["KL"].append(kl)
                if triggered:
                    trainer.logs["leash_fires"] += 1
                    # one-update exploration burst
                    trainer.ent_coef = min(trainer.ent_coef * 1.5, trainer.cfg.ent_coef * 2.0)
            last_logps_t = logps_t.clone()

        trainer.update(obses, acts, logps, ents, vals, rews, dones)

        step_i += rollout_len

        # TRP band control only for trp_full
        if condition == "trp_full":
            trainer._maybe_trp_control(step_i)

    return trainer.logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/main/main.yaml")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42,123,456,789,1001,2024,31415,271828,4242,666])
    args = ap.parse_args()

    with open(args.config, "r") as f:
        main_cfg = yaml.safe_load(f)

    # Load pilot locks
    with open("configs/pilot_locked.json", "r") as f:
        locks = json.load(f)

    baseline_ent = float(locks["baseline_ent_coef"])
    ell = float(locks["dt_band"]["ell"])
    u = float(locks["dt_band"]["u"])
    kappa = float(locks["kl_leash"]["kappa"])

    save_root = main_cfg.get("save_dir", "results")

    # Task lists per prereg
    minigrid_tasks = [
        ("Empty", "MiniGrid-Empty-8x8-v0"),
        ("KeyCorridor", "MiniGrid-KeyCorridorS3R3-v0")
    ]
    procgen_tasks = ["coinrun", "starpilot", "caveflyer", "jumper", "leaper"]

    conditions = main_cfg["conditions"]

    for family in main_cfg["env_families"]:

        if family == "minigrid":
            tasks = minigrid_tasks
            total_steps = int(main_cfg["steps"]["minigrid"])

            for task_name, env_id in tasks:
                for condition in conditions:
                    for seed in args.seeds:
                        set_all_seeds(seed)

                        # env with locked ladder wrapper
                        env = make_minigrid_ladder(task_name)

                        obs_dim = int(np.prod(env.observation_space.shape))
                        act_dim = env.action_space.n

                        cfg = PPOConfig(ent_coef=baseline_ent)
                        dt_band = (ell, u) if condition == "trp_full" else None
                        kk = kappa if condition in ("kl_only", "trp_full") else None

                        trainer = PPO_TRP_Trainer(
                            env=env,
                            obs_dim=obs_dim,
                            act_dim=act_dim,
                            cfg=cfg,
                            condition=("trp_full" if condition=="trp_full" else condition),
                            dt_band=dt_band,
                            kappa=kk,
                            control_interval=main_cfg["trp"]["control_interval"]
                        )

                        logs = run_condition(trainer, total_steps, condition, baseline_ent)

                        out_dir = Path(save_root) / "minigrid" / env_id / condition / str(seed)
                        out_dir.mkdir(parents=True, exist_ok=True)

                        with open(out_dir / "logs.json", "w") as f:
                            json.dump(logs, f)

                        print("Saved:", out_dir / "logs.json")

        elif family == "procgen":
            total_steps = int(main_cfg["steps"]["procgen"])

            for env_name in procgen_tasks:
                for condition in conditions:
                    for seed in args.seeds:
                        set_all_seeds(seed)

                        env = make_procgen_env(env_name)
                        obs_dim = int(np.prod(env.observation_space.shape))
                        act_dim = env.action_space.n

                        cfg = PPOConfig(ent_coef=baseline_ent)
                        dt_band = (ell, u) if condition == "trp_full" else None
                        kk = kappa if condition in ("kl_only", "trp_full") else None

                        trainer = PPO_TRP_Trainer(
                            env=env,
                            obs_dim=obs_dim,
                            act_dim=act_dim,
                            cfg=cfg,
                            condition=("trp_full" if condition=="trp_full" else condition),
                            dt_band=dt_band,
                            kappa=kk,
                            control_interval=main_cfg["trp"]["control_interval"]
                        )

                        logs = run_condition(trainer, total_steps, condition, baseline_ent)

                        out_dir = Path(save_root) / "procgen" / env_name / condition / str(seed)
                        out_dir.mkdir(parents=True, exist_ok=True)

                        with open(out_dir / "logs.json", "w") as f:
                            json.dump(logs, f)

                        print("Saved:", out_dir / "logs.json")

        else:
            raise ValueError(f"Unknown env family: {family}")

    print("Main prereg runs complete.")


if __name__ == "__main__":
    main()
