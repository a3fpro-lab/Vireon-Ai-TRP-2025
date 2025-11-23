"""
Sanity test for prereg R_t definition.

Idea:
- Feed the world model a perfectly predictable synthetic stream.
- World-model M_phi should quickly reduce epsilon.
- This verifies eps_t = ||e_{t+1} - M_phi(e_t, a_t)||^2 behaves properly.

This is NOT part of prereg results; it's a unit sanity check.
"""

import numpy as np
import torch

from algorithms.ppo_trp import WorldModel


def make_predictable_sequence(T=512, emb_dim=32, act_dim=5, noise=0.0):
    """
    e_{t+1} = A e_t + B a_t + small noise, with fixed A,B.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(scale=0.05, size=(emb_dim, emb_dim)).astype(np.float32)
    B = rng.normal(scale=0.05, size=(act_dim, emb_dim)).astype(np.float32)

    e = rng.normal(size=(emb_dim,), scale=1.0).astype(np.float32)
    embs = []
    next_embs = []
    acts = []

    for t in range(T):
        a = rng.integers(0, act_dim)
        e_next = A @ e + B[a]
        if noise > 0:
            e_next += rng.normal(scale=noise, size=emb_dim).astype(np.float32)

        embs.append(e.copy())
        next_embs.append(e_next.copy())
        acts.append(a)

        e = e_next

    return np.array(embs), np.array(next_embs), np.array(acts)


def test_world_model_learns_predictable_map():
    emb_dim = 32
    act_dim = 5
    T = 1024

    embs, next_embs, acts = make_predictable_sequence(T=T, emb_dim=emb_dim, act_dim=act_dim, noise=0.0)

    wm = WorldModel(emb_dim=emb_dim, act_dim=act_dim, hidden=64)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)

    emb_t = torch.tensor(embs, dtype=torch.float32)
    next_emb_t = torch.tensor(next_embs, dtype=torch.float32)
    acts_t = torch.tensor(acts, dtype=torch.int64)

    eps_history = []
    for epoch in range(50):
        pred = wm(emb_t, acts_t)
        eps_per = ((next_emb_t - pred) ** 2).mean(dim=-1)
        loss = eps_per.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        eps_history.append(loss.item())

    # Should drop a lot on predictable sequences
    assert eps_history[-1] < 0.2 * eps_history[0], (
        f"epsilon didn't drop enough: start={eps_history[0]:.4f} end={eps_history[-1]:.4f}"
    )


if __name__ == "__main__":
    test_world_model_learns_predictable_map()
    print("world_model_test passed")
