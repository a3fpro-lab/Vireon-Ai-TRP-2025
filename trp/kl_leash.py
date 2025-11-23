import torch

class KLLeash:
    def __init__(self, kappa):
        self.kappa = kappa
        self.last_kl = 0.0
        self.triggered = False

    @torch.no_grad()
    def check(self, logp_new, logp_old):
        kl = (logp_old - logp_new).mean().item()
        self.last_kl = kl
        self.triggered = kl > self.kappa
        return self.triggered, kl
