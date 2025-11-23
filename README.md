# TRP Time Control (2025 Preregistered)

This repo implements and tests **TRP Time Control** in RL (primary) and DPO (secondary).

**Core control variable**
\[
dt_{\text{eff}}(t) = z(P_t)\cdot z(R_t)
\]

**Primary preregistration**
- PPO on MiniGrid + Procgen
- Pilot-then-lock dt band + KL threshold
- 4-condition ablation grid
- AULC primary outcome
- Hard falsifiers

See prereg in `prereg/PREREG_TRP_RLHF_PPO_DPO.md`.

## Repo layout
- `prereg/` locked prereg + commit hash
- `trp/` TRP variables + KL-leash
- `algorithms/` PPO + world-model
- `envs/` env wrappers & difficulty control
- `configs/` pilot + main configs
- `scripts/` run scripts
- `analysis/` plotting notebooks
- `results/` output logs

## Status
Repo scaffolded on iPhone. Core code coming next.
