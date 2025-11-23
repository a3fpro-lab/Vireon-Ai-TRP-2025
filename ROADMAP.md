# TRP-2025 Roadmap (Prereg-Safe)

This roadmap is subordinate to the prereg in:
`prereg/PREREG_TRP_RLHF_PPO_DPO.md`

No steps here may change prereg definitions without logging a violation.

---

## Phase 0 — Scaffold (DONE)
- Repo skeleton created on iPhone.
- Prereg v1.1 locked.
- Core TRP + PPO scaffolds added.
- Pilot lock placeholder added.

---

## Phase 1 — Pilot Implementation
**Goal:** produce `configs/pilot_locked.json` with:
- baseline entropy bonus `ent_coef`
- dt band `[ell, u]`
- KL threshold `kappa`

Tasks:
1. Implement `scripts/pilot.py`
   - load pilot config `configs/pilot/pilot.yaml`
   - run baseline PPO for each ent grid value
   - compute AULC per ent value
   - pick best ent_coef (highest AULC)
   - compute dt_eff distribution → lock Q0.35/Q0.65 band
   - compute KL distribution → lock 95th percentile kappa
   - write `configs/pilot_locked.json` (overwrite placeholders)
2. Verify lock file is committed before main runs.

Output:
- `configs/pilot_locked.json`
- pilot curves in `results/pilot/`

---

## Phase 2 — Main Implementation
**Goal:** prereg main runs across 7 tasks × 4 conditions.

Tasks:
1. Implement real env creation in `scripts/main.py`
   - MiniGrid ladders per prereg
   - Procgen difficulty knobs per prereg
2. Fill in PPO rollout + update in `algorithms/ppo_trp.py`
3. Fill in world-model prediction error for true R_t
4. Wire TRP band control + KL-leash bursts exactly as prereg
5. Save per-run logs to:
   `results/<env>/<condition>/<seed>/logs.json`

Output:
- full raw logs in `results/main/`
- **no analysis yet**

---

## Phase 3 — Analysis (Post-Unblinding)
Tasks:
1. Implement `analysis/plot_curves.py` for:
   - AULC per seed
   - steps-to-threshold
   - final return
2. Run prereg stats:
   - paired t-tests
   - Holm-Bonferroni correction
3. If any prereg deviations occurred, log in:
   `results/violations.md`

Output:
- `analysis/results_summary.csv`
- figures for paper

---

## Phase 4 — Release
1. Make repo public.
2. Tag release:
   `v1.0-preregistered-results`
3. Post full curves + lock hash publicly.

---

## Notes
- Any change to envs, seeds, bands, KL thresholds, control law, or metrics
  must be logged as a prereg violation.
