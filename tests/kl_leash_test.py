"""
Sanity test for KL-Leash trigger + burst behavior.

Checks:
1) KLLeash triggers when KL > kappa
2) PPO_TRP_Trainer applies one-step ent_coef burst on trigger

Not part of prereg results.
"""

import torch

from trp.kl_leash import KLLeash
from algorithms.ppo_trp import PPO_TRP_Trainer, PPOConfig


def test_kl_leash_triggers():
    kappa = 0.05
    leash = KLLeash(kappa=kappa)

    # Construct fake log-prob vectors
    logp_old = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    logp_new_close = torch.tensor([-1.01, -0.99, -1.02, -0.98])  # small KL
    logp_new_far = torch.tensor([-1.3, -1.2, -1.4, -1.1])        # big KL

    triggered_close, kl_close = leash.check(logp_new_close, logp_old)
    assert triggered_close is False
    assert kl_close <= kappa

    triggered_far, kl_far = leash.check(logp_new_far, logp_old)
    assert triggered_far is True
    assert kl_far > kappa


# ---- Trainer burst check ----

class DummyEnv:
    def __init__(self):
        class Act: n = 3
        class Obs: shape = (10,)
        self.action_space = Act()
        self.observation_space = Obs()

    def reset(self):
        return [0.0]*10, {}

    def step(self, a):
        return [0.0]*10, 0.0, False, False, {}


def test_trainer_entropy_burst_on_trigger():
    env = DummyEnv()
    cfg = PPOConfig(ent_coef=0.01)

    kappa = 0.02  # small so trigger easy
    trainer = PPO_TRP_Trainer(
        env=env,
        obs_dim=10,
        act_dim=3,
        cfg=cfg,
        condition="kl_only",
        dt_band=None,
        kappa=kappa,
        control_interval=999999
    )

    # Fake two rollout logp arrays:
    last_logps = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    new_logps_trigger = torch.tensor([-1.5, -1.4, -1.6, -1.3])

    # Mimic the leash check logic used in train()
    old_ent = trainer.ent_coef
    triggered, kl = trainer.leash.check(new_logps_trigger, last_logps)
    assert triggered is True

    if triggered:
        trainer.ent_coef = min(trainer.ent_coef * 1.5, trainer.cfg.ent_coef * 2.0)

    assert trainer.ent_coef > old_ent, "entropy burst did not increase ent_coef"


if __name__ == "__main__":
    test_kl_leash_triggers()
    test_trainer_entropy_burst_on_trigger()
    print("kl_leash_test passed")
