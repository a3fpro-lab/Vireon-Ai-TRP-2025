"""
Environment wrappers for TRP prereg.

Goal:
- Provide uniform harder()/easier() hooks
- Lock MiniGrid variant ladder
- Provide Procgen difficulty/seed-pool knobs if supported

These are scaffolds; exact mechanics filled in on laptop.
"""

class TRPEnvWrapper:
    """Base wrapper giving no-op harder/easier by default."""
    def __init__(self, env):
        self.env = env

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def harder(self):
        # no-op unless overridden
        return

    def easier(self):
        # no-op unless overridden
        return

    def __getattr__(self, name):
        return getattr(self.env, name)


class MiniGridTRPWrapper(TRPEnvWrapper):
    """
    Locked ladders per prereg v1.1:
    - Empty-8x8 <-> Empty-16x16
    - KeyCorridor: S3R3 -> S4R3 -> S5R3
    """
    def __init__(self, make_env_fn, ladder_names):
        """
        make_env_fn: function(name)->env
        ladder_names: ordered list of env ids from easy->hard
        """
        self.make_env_fn = make_env_fn
        self.ladder = ladder_names
        self.idx = 0
        super().__init__(self.make_env_fn(self.ladder[self.idx]))

    def harder(self):
        if self.idx < len(self.ladder) - 1:
            self.idx += 1
            self.env = self.make_env_fn(self.ladder[self.idx])

    def easier(self):
        if self.idx > 0:
            self.idx -= 1
            self.env = self.make_env_fn(self.ladder[self.idx])


class ProcgenTRPWrapper(TRPEnvWrapper):
    """
    Procgen difficulty knobs per prereg v1.1:
    - Prefer distribution_mode easy->default->hard if env supports.
    - Else increase level_seed pool size 64->512->4096.
    """
    def __init__(self, env, difficulty_modes=("easy", "default", "hard"),
                 seed_pools=(64, 512, 4096)):
        super().__init__(env)
        self.modes = list(difficulty_modes)
        self.mode_idx = 1  # start at default
        self.seed_pools = list(seed_pools)
        self.pool_idx = 0

    def harder(self):
        # try distribution_mode if exists
        if hasattr(self.env, "distribution_mode"):
            if self.mode_idx < len(self.modes) - 1:
                self.mode_idx += 1
                try:
                    self.env.distribution_mode = self.modes[self.mode_idx]
                except Exception:
                    pass
            return

        # fallback: seed pool size if supported
        if hasattr(self.env, "num_levels"):
            if self.pool_idx < len(self.seed_pools) - 1:
                self.pool_idx += 1
                try:
                    self.env.num_levels = self.seed_pools[self.pool_idx]
                except Exception:
                    pass

    def easier(self):
        if hasattr(self.env, "distribution_mode"):
            if self.mode_idx > 0:
                self.mode_idx -= 1
                try:
                    self.env.distribution_mode = self.modes[self.mode_idx]
                except Exception:
                    pass
            return

        if hasattr(self.env, "num_levels"):
            if self.pool_idx > 0:
                self.pool_idx -= 1
                try:
                    self.env.num_levels = self.seed_pools[self.pool_idx]
                except Exception:
                    pass
