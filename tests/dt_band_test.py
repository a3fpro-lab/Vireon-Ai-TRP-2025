"""
Sanity test for dt_eff band controller.

We simulate dt_eff values and ensure:
- below ell -> ent_coef goes up, harder() called
- above u   -> ent_coef goes down, easier() called
- inside band -> no change

Not part of prereg results.
"""

from algorithms.ppo_trp import PPO_TRP_Trainer, PPOConfig


class DummyEnv:
    def __init__(self):
        self.harder_calls = 0
        self.easier_calls = 0
        # minimal spaces
        class Act: n = 3
        class Obs: shape = (10,)
        self.action_space = Act()
        self.observation_space = Obs()

    def reset(self):
        return [0.0]*10, {}

    def step(self, a):
        return [0.0]*10, 0.0, False, False, {}

    def harder(self):
        self.harder_calls += 1

    def easier(self):
        self.easier_calls += 1


def test_dt_band_control():
    env = DummyEnv()
    cfg = PPOConfig(ent_coef=0.01)
    trainer = PPO_TRP_Trainer(
        env=env,
        obs_dim=10,
        act_dim=3,
        cfg=cfg,
        condition="trp_full",
        dt_band=( -0.5, 0.5 ),  # ell, u
        kappa=None,
        control_interval=1       # fire every step for test
    )

    # Case 1: dt below ell
    trainer.trp.dt_eff = -1.0
    old_ent = trainer.ent_coef
    trainer._maybe_trp_control(step_i=1)
    assert trainer.ent_coef > old_ent
    assert env.harder_calls == 1

    # Case 2: dt above u
    trainer.trp.dt_eff = 1.0
    old_ent = trainer.ent_coef
    trainer._maybe_trp_control(step_i=2)
    assert trainer.ent_coef < old_ent
    assert env.easier_calls == 1

    # Case 3: dt inside band
    trainer.trp.dt_eff = 0.0
    old_ent = trainer.ent_coef
    trainer._maybe_trp_control(step_i=3)
    assert trainer.ent_coef == old_ent
    assert env.harder_calls == 1 and env.easier_calls == 1


if __name__ == "__main__":
    test_dt_band_control()
    print("dt_band_test passed")
