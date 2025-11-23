# Vireon AI — TRP Time 2025 (Locked Prereg)

**Core law:**  
\[
dt_{\text{eff}}(t) = z(P_t)\cdot z(R_t)
\]
Subjective/effective training time is the product of:
- **R_t (Reality bandwidth):** how much structured novelty exists in the stream right now  
- **P_t (Perception gain):** how strongly the agent is extracting that structure  
Constrained by a **KL-Leash** to prevent uncontrolled belief drift.

This repo implements **TRP Time Control** for RL (primary) and DPO (secondary), with a preregistered, falsifiable evaluation.

---

## What’s in this repo

### Locked preregistration
- **Prereg doc:** `prereg/PREREG_TRP_RLHF_PPO_DPO.md`
- **Version:** v1.1-LOCKED  
- **Lock date:** 2025-11-23  
- **Violations log:** `results/violations.md` (must be updated if anything deviates)

### Core TRP system
- TRP time tracker: `trp/trp_time.py`
- KL-Leash: `trp/kl_leash.py`

### RL training code (primary)
- PPO + TRP trainer (with prereg-accurate world-model \(R_t\)):  
  `algorithms/ppo_trp.py`
- Env wrappers for prereg difficulty ladders:  
  `envs/wrappers.py`

### Scripts
- Pilot sweep & lock writer: `scripts/pilot.py`
- Main prereg grid runner: `scripts/main.py`
- Audit trail: `scripts/run_notes.md`
- Roadmap: `ROADMAP.md`

### Sanity tests (NOT part of prereg results)
- World-model ε drop check: `tests/world_model_test.py`
- dt-band trigger check: `tests/dt_band_test.py`
- KL-Leash trigger/burst check: `tests/kl_leash_test.py`

### Analysis
- Plotting stub: `analysis/plot_curves.py` (filled after main runs)

---

## Status (as of v1.0 release)
- ✅ Repo scaffold complete (iPhone build)
- ✅ Preregistration committed and locked
- ✅ PPO+TRP trainer implemented
- ✅ World-model \(R_t\), dt-band controller, KL-Leash wired per prereg
- ✅ Sanity tests added
- ⏭ **Next:** run pilots to lock `baseline_ent_coef`, `[ell,u]`, and `kappa`

Release tag: `v1.0-prereg-scaffold`

---

## Quickstart (when on laptop)

### 1) Install
```bash
pip install -r requirements.txt
