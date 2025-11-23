# Run Notes / Audit Trail

**Scaffold built on iPhone (Safari)**
- Date: 2025-11-23
- Repo status: prereg + skeleton code complete, no runs executed yet.

## Prereg lock
- File: `prereg/PREREG_TRP_RLHF_PPO_DPO.md`
- Version: v1.1-LOCKED
- Lock intent: no numeric changes allowed without logging a prereg violation.

## Next steps (when laptop available)
1. Implement `scripts/pilot.py`
   - sweep entropy grid
   - save `configs/pilot_locked.json` containing:
     - baseline ent_coef
     - dt band [ell,u]
     - KL threshold kappa
2. Implement real PPO rollouts + world-model eps_t in `algorithms/ppo_trp.py`
3. Add plotting notebook in `analysis/`
4. Run pilots, then main runs, then publish curves.

## Violations
Any deviation from prereg must be recorded in:
`results/violations.md`
