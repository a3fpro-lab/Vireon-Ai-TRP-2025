"""
PPO + TRP Control (prereg-accurate R_t)

Implements prereg Conditions:
1) baseline
2) adaptive_entropy
3) kl_only
4) trp_full

Hooks in:
- world-model prediction error -> R_t
- policy entropy + inverse progress -> P_t
- dt_eff band controller
- KL-leash trigger bursts
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from trp.trp_time import TRPTime
from trp.kl_leash import KLLeash


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01   # baseline entropy bonus (locked from pilot)
    max_grad_norm: float = 0.5

    rollout_len: int = 256
    batch_size: int = 2048
    update_epochs: int = 4

    world_lr: float = 1e-3
    world_epochs: int = 1


class RunningEMA:
    """Simple EMA for scalar signals (for inv-progress proxy)."""
    def __init__(self, beta=0.01):
        self.beta = beta
        self.val = 0.0
        self.init = False

    def update(self, x):
        x = float(x)
        if not self.init:
            self.val = x
            self.init = True
        else:
            self.val = (1 - self.beta) * self.val + self.beta * x
        return self.val


class ActorCritic(nn.Module):
    """
    Tiny generic AC net with explicit embedding e_t.
    e_t is the output of emb_net before pi/v heads.
    """
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.hidden = hidden

        self.emb_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.pi_head = nn.Linear(hidden, act_dim)
        self.v_head  = nn.Linear(hidden, 1)

    def embed(self, obs):
        return self.emb_net(obs)

    def forward(self, obs):
        emb = self.embed(obs)
        logits = self.pi_head(emb)
        value  = self.v_head(emb).squeeze(-1)
        return logits, value, emb

    def act(self, obs):
        logits, value, emb = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy, value, emb


class WorldModel(nn.Module):
    """Online predictor for e_{t+1} from (e_t, a_t)."""
    def __init__(self, emb_dim, act_dim, hidden=128):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim)
        )

    def forward(self, emb_t, a_t):
        a_onehot = torch.nn.functional.one_hot(a_t, self.act_dim).float()
        x = torch.cat([emb_t, a_onehot], dim=-1)
        return self.net(x)


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - dones[t+1] if t < T-1 else 1.0
        nextvalue = values[t+1] if t < T-1 else 0.0
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv, adv + values


class PPO_TRP_Trainer:
    def __init__(self, env, obs_dim, act_dim, cfg: PPOConfig,
                 condition: str,
                 dt_band=None, kappa=None,
                 control_interval=2048):
        """
        condition in {'baseline','adaptive_entropy','kl_only','trp_full'}

        dt_band = (ell, u) locked from pilot
        kappa = KL threshold locked from pilot
        """
        self.env = env
        self.cfg = cfg
        self.condition = condition
        self.control_interval = control_interval

        self.model = ActorCritic(obs_dim, act_dim, hidden=128)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # TRP trackers
        self.trp = TRPTime()
        self.dt_band = dt_band
        self.ell, self.u = dt_band if dt_band else (None, None)

        self.leash = KLLeash(kappa=kappa) if kappa is not None else None

        # World model for R_t
        emb_dim = self.model.hidden
        self.world = WorldModel(emb_dim=emb_dim, act_dim=act_dim, hidden=128)
        self.world_opt = torch.optim.Adam(self.world.parameters(), lr=cfg.world_lr)

        # dynamic entropy coefficient (for TRP/adaptive_entropy conditions)
        self.ent_coef = cfg.ent_coef

        # inverse-progress proxy EMA
        self.prev_v_loss = None
        self.inv_prog_ema = RunningEMA(beta=0.01)

        # logs
        self.logs = {
            "returns": [],
            "dt_eff": [],
            "R_t": [],
            "P_t": [],
            "KL": [],
            "leash_fires": 0,
            "world_eps": [],
        }

    def _maybe_trp_control(self, step_i):
        if self.condition != "trp_full":
            return
        if self.dt_band is None:
            return
        if step_i % self.control_interval != 0:
            return

        dt = self.trp.dt_eff
        if dt < self.ell:
            self.ent_coef = min(self.ent_coef * 1.15, self.cfg.ent_coef * 2.0)
            if hasattr(self.env, "harder"):
                self.env.harder()
        elif dt > self.u:
            self.ent_coef = max(self.ent_coef * 0.90, self.cfg.ent_coef * 0.5)
            if hasattr(self.env, "easier"):
                self.env.easier()

    def rollout(self):
        obs, _ = self.env.reset()
        ep_ret = 0.0

        obses, acts, logps, ents, vals, rews, dones = [], [], [], [], [], [], []
        embes, next_embes = [], []

        for t in range(self.cfg.rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action, logp, entropy, value, emb_t = self.model.act(obs_t)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            # compute next embedding e_{t+1} with frozen trunk
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, next_emb_t = self.model.forward(next_obs_t)

            obses.append(obs)
            acts.append(action.item())
            logps.append(logp.item())
            ents.append(entropy.item())
            vals.append(value.item())
            rews.append(reward)
            dones.append(done)

            embes.append(emb_t.squeeze(0).cpu().numpy())
            next_embes.append(next_emb_t.squeeze(0).cpu().numpy())

            ep_ret += reward
            obs = next_obs

            if done:
                self.logs["returns"].append(ep_ret)
                obs, _ = self.env.reset()
                ep_ret = 0.0

        return (
            np.array(obses),
            np.array(acts),
            np.array(logps),
            np.array(ents),
            np.array(vals),
            np.array(rews),
            np.array(dones),
            np.array(embes),
            np.array(next_embes),
        )

    def update(self, obses, acts, old_logps, ents, vals, rews, dones, embes, next_embes):
        adv, rets = compute_gae(rews, vals, dones, self.cfg.gamma, self.cfg.lam)

        obses_t = torch.tensor(obses, dtype=torch.float32)
        acts_t = torch.tensor(acts, dtype=torch.int64)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        rets_t = torch.tensor(rets, dtype=torch.float32)

        # --- PPO update ---
        last_v_loss = None
        for _ in range(self.cfg.update_epochs):
            logits, value, _ = self.model(obses_t)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(acts_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logps_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_t
            pi_loss = -(torch.min(surr1, surr2)).mean()

            v_loss = ((value - rets_t) ** 2).mean()
            last_v_loss = v_loss.item()

            loss = pi_loss + self.cfg.vf_coef * v_loss - self.ent_coef * entropy

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.opt.step()

        # inv-progress proxy: EMA of value-loss improvement
        if self.prev_v_loss is None:
            inv_progress = 0.0
        else:
            inv_progress = self.prev_v_loss - last_v_loss
        self.prev_v_loss = last_v_loss
        inv_progress = self.inv_prog_ema.update(inv_progress)

        # --- World model update for R_t ---
        emb_t = torch.tensor(embes, dtype=torch.float32)
        next_emb_t = torch.tensor(next_embes, dtype=torch.float32)

        # train world model a little each update
        for _ in range(self.cfg.world_epochs):
            pred_next = self.world(emb_t, acts_t)
            eps_per = ((next_emb_t - pred_next) ** 2).mean(dim=-1)  # per-step ε_t
            world_loss = eps_per.mean()

            self.world_opt.zero_grad()
            world_loss.backward()
            self.world_opt.step()

        eps_t_scalar = float(eps_per.mean().item())  # mean ε_t across rollout
        self.logs["world_eps"].append(eps_t_scalar)

        # --- TRP updates ---
        P_t = self.trp.update_P(entropy.item(), inv_progress)
        R_t = self.trp.update_R(eps_t_scalar)
        dt = self.trp.update_dt()

        self.logs["P_t"].append(P_t)
        self.logs["R_t"].append(R_t)
        self.logs["dt_eff"].append(dt)

        return entropy.item(), last_v_loss

    def train(self, total_steps):
        step_i = 0
        last_logps_t = None

        while step_i < total_steps:
            (obses, acts, logps, ents, vals, rews, dones,
             embes, next_embes) = self.rollout()

            # KL leash check (conditions 3 & 4)
            if self.condition in ("kl_only", "trp_full") and self.leash is not None:
                logps_t = torch.tensor(logps, dtype=torch.float32)
                if last_logps_t is not None:
                    triggered, kl = self.leash.check(logps_t, last_logps_t)
                    self.logs["KL"].append(kl)
                    if triggered:
                        self.logs["leash_fires"] += 1
                        self.ent_coef = min(self.ent_coef * 1.5, self.cfg.ent_coef * 2.0)
                last_logps_t = logps_t.clone()

            self.update(obses, acts, logps, ents, vals, rews, dones, embes, next_embes)

            step_i += self.cfg.rollout_len

            # TRP band control (condition 4)
            self._maybe_trp_control(step_i)

        return self.logs
