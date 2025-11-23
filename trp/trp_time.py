import numpy as np

class RunningZ:
    """Online z-scoring with EMA mean/var."""
    def __init__(self, beta=0.01, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, x):
        x = float(x)
        if not self.initialized:
            self.mean = x
            self.var = 1.0
            self.initialized = True
        else:
            dm = x - self.mean
            self.mean += self.beta * dm
            self.var = (1 - self.beta) * self.var + self.beta * dm * dm
        return (x - self.mean) / np.sqrt(self.var + self.eps)

class TRPTime:
    """Tracks R_t, P_t, dt_eff."""
    def __init__(self, beta_R=0.01, beta_P=0.01, beta_dt=0.01):
        self.zR = RunningZ(beta_R)
        self.zP = RunningZ(beta_P)
        self.zdt = RunningZ(beta_dt)
        self.R_t = 0.0
        self.P_t = 0.0
        self.dt_eff = 0.0

    def update_R(self, eps_t):
        self.R_t = self.zR.update(eps_t)
        return self.R_t

    def update_P(self, entropy_t, inv_progress_t):
        p_raw = entropy_t + inv_progress_t
        self.P_t = np.clip(self.zP.update(p_raw), -3, 3)
        return self.P_t

    def update_dt(self):
        self.dt_eff = self.zdt.update(self.P_t * self.R_t)
        return self.dt_eff
