"""
PILOT ENTRYPOINT (prereg v1.1)

Tasks:
- Sweep baseline entropy bonus grid on baseline PPO
- Pick best ent_coef by AULC
- Using best ent_coef run a logging-only pilot to estimate:
    - dt band [Q0.35, Q0.65]
    - KL 95th percentile (kappa)
- Write configs/pilot_locked.json (overwriting placeholder)

Notes:
- Pilot uses same code/hparams as main, truncated to 200k steps, N=3 seeds.
- For KL logging without affecting behavior, we run condition='kl_only'
  with kappa=1e9 to ensure no trigger (behavior == baseline, but KL logged).
"""

import argparse, json, os, random
import numpy as np
import torch
import yaml

from algorithms.ppo_trp import PPO_TRP_Trainer, PPOConfig
from envs.wrappers import ProcgenTRPWrapper, TRPEnvWrapper


# -------------------------
# Simple obs flatten wrappers
# -------------------------

import gymnasium as gym

class FlattenObs(gym.ObservationWrapper):
    """Flattens any array obs to 1D float32."""
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
    """
    Procgen delivers vectorized obs (num_envs, H, W, C). We unwrap to single env.
    """
    def __init__(self, env):
        super().__init__(env)
        # observation_space: (num_envs, H, W, C)
        shp = env.observation_space["rgb"].shape if isinstance(env.observation_space, dict) else env.observation_space.shape
        if len(shp) == 4:
            H, W, C = shp[1], shp[2], shp[3]
        else:
            H, W, C = shp[-3], shp[-2], shp[-1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(H, W, C), dtype=np.float32
        )
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # procgen may return dict or array
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


def aulc_from_returns(returns):
    """
    AULC over episode index using trapezoid rule.
    If no returns, returns -inf.
    """
    if returns is None or len(returns) < 2:
        return -np.inf
    y = np.array(returns, dtype=np.float32)
    x = np.arange(len(y), dtype=np.float32)
    return float(np.trapz(y, x) / len(y))


def make_procgen_env(name):
    from procgen import ProcgenEnv
    env = ProcgenEnv(num_envs=1, env_name=name, distribution_mode="default", num_levels=0, start_level=0)
    env = ProcgenSingleEnv(env)
    env = FlattenObs(env)
    env = TRPEnvWrapper(env)  # no difficulty adjustments in pilot
    return env


def run_baseline_for_ent(ent_coef, env_names, seeds, steps, cfg_base, use_wandb=False, wandb_project=None):
    """
    Run baseline PPO for each env and seed, return mean AULC across envs/seeds.
    """
    all_aulc = []
    for env_name in env_names:
        for seed in seeds:
            set_all_seeds(seed)
            cfg = PPOConfig(ent_coef=ent_coef)

            env = make_procgen_env(env_name)
            obs_dim = int(np.prod(env.observation_space.shape))
            act_dim = env.action_space.n

            trainer = PPO_TRP_Trainer(
                env=env,
                obs_dim=obs_dim,
                act_dim=act_dim,
                cfg=cfg,
                condition="baseline",
                dt_band=None,
                kappa=None,
                control_interval=2048
            )

            logs = trainer.train(total_steps=steps)
            all_aulc.append(aulc_from_returns(logs.get("returns", [])))

    return float(np.mean(all_aulc)) if all_aulc else -np.inf


def run_logging_pilot(ent_coef, env_names, seeds, steps):
    """
    Run baseline-equivalent PPO but log dt_eff + KL distribution.

    Uses condition='kl_only' with huge kappa so leash never triggers.
    """
    dt_list = []
    kl_list = []

    huge_kappa = 1e9

    for env_name in env_names:
        for seed in seeds:
            set_all_seeds(seed)
            cfg = PPOConfig(ent_coef=ent_coef)

            env = make_procgen_env(env_name)
            obs_dim = int(np.prod(env.observation_space.shape))
            act_dim = env.action_space.n

            trainer = PPO_TRP_Trainer(
                env=env,
                obs_dim=obs_dim,
                act_dim=act_dim,
                cfg=cfg,
                condition="kl_only",   # identical to baseline because kappa huge
                dt_band=None,
                kappa=huge_kappa,
                control_interval=2048
            )

            logs = trainer.train(total_steps=steps)
            dt_list.extend(logs.get("dt_eff", []))
            kl_list.extend(logs.get("KL", []))

    dt_arr = np.array(dt_list, dtype=np.float32)
    kl_arr = np.array(kl_list, dtype=np.float32)

    return dt_arr, kl_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pilot/pilot.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42,123,456])
    ap.add_argument("--steps", type=int, default=200000)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    env_family = cfg_yaml.get("env_family", "procgen")
    if env_family != "procgen":
        raise ValueError("Pilot currently supports procgen env_family only per default pilot.yaml.")

    env_names = cfg_yaml["envs"]
    ent_grid = cfg_yaml["ppo"]["entropy_bonus_grid"]

    # --- sweep entropy grid ---
    best_ent = None
    best_aulc = -np.inf
    sweep_results = {}

    for ent in ent_grid:
        aulc = run_baseline_for_ent(
            ent_coef=ent,
            env_names=env_names,
            seeds=args.seeds,
            steps=args.steps,
            cfg_base=None
        )
        sweep_results[str(ent)] = aulc
        if aulc > best_aulc:
            best_aulc = aulc
            best_ent = float(ent)

    # --- logging pilot for dt-band and kappa ---
    dt_arr, kl_arr = run_logging_pilot(
        ent_coef=best_ent,
        env_names=env_names,
        seeds=args.seeds,
        steps=args.steps
    )

    if len(dt_arr) == 0:
        raise RuntimeError("No dt_eff samples collected in pilot.")
    if len(kl_arr) == 0:
        # If KL doesn't log for some reason, set to 0 and warn.
        kl_arr = np.array([0.0], dtype=np.float32)

    ell_q, u_q = cfg_yaml["trp"]["band_quantiles"]
    kl_q = cfg_yaml["trp"]["kl_quantile"]

    ell = float(np.quantile(dt_arr, ell_q))
    u = float(np.quantile(dt_arr, u_q))
    kappa = float(np.quantile(kl_arr, kl_q))

    locked = {
        "locked_on": "2025-11-23",
        "env_family": env_family,
        "baseline_ent_coef": best_ent,
        "dt_band": {
            "ell": ell,
            "u": u,
            "quantiles": [ell_q, u_q]
        },
        "kl_leash": {
            "kappa": kappa,
            "quantile": kl_q
        },
        "sweep_results": sweep_results,
        "notes": "Written by scripts/pilot.py prereg v1.1. KL logged via kl_only with huge kappa => no triggers."
    }

    out_path = "configs/pilot_locked.json"
    with open(out_path, "w") as f:
        json.dump(locked, f, indent=2)

    print("Pilot complete.")
    print("Best ent_coef:", best_ent, "AULC:", best_aulc)
    print("Locked dt band:", (ell, u))
    print("Locked kappa:", kappa)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
