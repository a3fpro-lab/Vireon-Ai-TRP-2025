# PREREG: TRP Time Control for RL (Primary) + DPO (Secondary)
**Date:** 2025-11-23  
**Version:** v1.1-LOCKED (ambiguity-hardened, no numeric changes)  
**Owner:** VIREON / TRP Time  
**Planned code release:** `github.com/vireon-lab/trp-2025` (private until results finalized; prereg commit hash will be published at lock).

---

## 0) Claim Under Test
**TRP Time Control** (online band-keeping of \(dt_{\text{eff}} = P_t \cdot R_t\) + KL-leash triggers) yields **measurable improvements** in learning speed and/or final performance over strong baselines under matched compute.

---

## 1) Primary Experiment: TRP-Augmented PPO in RL

### 1.1 Tasks / Environments
Run on **two** environment families.

**MiniGrid**
- `MiniGrid-Empty-8x8-v0`
- `MiniGrid-KeyCorridorS3R3-v0`

**Procgen**
- 5 games: `coinrun`, `starpilot`, `caveflyer`, `jumper`, `leaper`
- default difficulty + hard generalization split

### 1.2 Agents
- Algorithm: PPO (standard clipped objective)
- Backbone:
  - MiniGrid: small CNN+MLP + 2-layer GRU (hidden=128)
  - Procgen: IMPALA-style CNN trunk + MLP (hidden=256)

### 1.3 Compute Budget (Matched)
Per environment family:
- Steps per run: `10,000,000` (Procgen), `2,000,000` (MiniGrid)
- **Seeds:** `N=10` fixed seeds per condition, paired across conditions  
  Seeds = `[42, 123, 456, 789, 1001, 2024, 31415, 271828, 4242, 666]`
- Wall-clock matched via equal steps; no early stopping.

### 1.4 Conditions (Ablation Grid)
Four conditions per task:

1) **Baseline PPO**
2) **Baseline PPO + Adaptive Entropy**
3) **PPO + KL-Leash Only**
4) **Full TRP-PPO (Leash + dt Band Control)**

### 1.5 Baseline Entropy Bonus \( \lambda_H \) (Pinned)
Baseline PPO entropy bonus is selected **once** by pilot grid search:

- Grid: `{0.001, 0.003, 0.01, 0.03}`
- Search uses **pilot protocol** below (3 seeds, 200k steps)
- Choose value with highest pilot AULC, then **lock for all main runs** and all TRP conditions.
- Adaptive-entropy baseline uses the locked baseline value as its starting point.

### 1.6 Pilot Protocol (Pinned)
Pilot runs:
- use **identical codebase + hyperparameters** as main runs
- truncated to **200k steps**
- **N=3 seeds**
- produce:
  1) baseline \( \lambda_H \) selection  
  2) locked dt-band \([\ell,u]\)  
  3) locked KL threshold \(\kappa\)

**Important:** the same locked dt-band \([\ell,u]\) is used in both conditions 3 and 4.

### 1.7 TRP Observables (Definitions locked)

#### (a) \(R_t\): Reality Bandwidth proxy
World-model prediction error.

- World model \(M_\phi\): small predictor trained online to predict next observation embedding.
- Embedding \(e_t = f_{\text{emb}}(o_t)\) where \(f_{\text{emb}}\) is the policy CNN trunk frozen for the step.
- Prediction loss:
  \[
  \varepsilon_t = \| e_{t+1} - M_\phi(e_t, a_t)\|_2^2
  \]
- Running estimate:
  \[
  R_t = \text{EMA}_{\tau_R}(\varepsilon_t)
  \]
- EMA window: \(\tau_R = 0.01 \times\) episode length (fixed numeric per env; recorded).

#### (b) \(P_t\): Perception Gain proxy
Two proxies:
1. Policy entropy \(H_t\)  
2. Inverse learning progress \(L'_t\) where \(L_t\) is PPO value loss:
   \[
   L'_t = \text{EMA}_{\tau_P}(L_{t-1} - L_t)
   \]

Combine:
- Each proxy z-scored with running mean/var.
- Define:
  \[
  P_t = \text{clip}\big( z(H_t) + z(L'_t),\ -3,\ 3 \big)
  \]

#### (c) Effective time
\[
dt_{\text{eff}}(t) = z(P_t)\cdot z(R_t)
\]

### 1.8 Band Targets (Locked)
From pilot Baseline PPO curves:
\[
[\ell, u] = [Q_{0.35}, Q_{0.65}]
\]
Locked per environment family; no tuning after pilot.

### 1.9 TRP Control Law (Locked)
Control acts every `K=2048` steps.

**If \(dt_{\text{eff}} < \ell\):**
- increase exploration + novelty:
  - \( \lambda_H \leftarrow 1.15 \lambda_H \) (cap at 2× baseline)
  - **Procgen:** adjust difficulty via:
    - `distribution_mode` steps (`easy → default → hard`) **if supported**, else  
    - increase `level_seed` pool size `{64 → 512 → 4096}`
    - if neither supported, only entropy adjustment is used.
  - **MiniGrid:** switch to a harder variant with prob 0.3 from locked list below.

**If \(dt_{\text{eff}} > u\):**
- reduce overload:
  - \( \lambda_H \leftarrow 0.90 \lambda_H \) (floor at 0.5× baseline)
  - **Procgen:** difficulty notch down (`hard → default → easy`) / smaller seed pool  
  - **MiniGrid:** revert to easier variant with prob 0.3.

### 1.10 MiniGrid Variant Lists (Locked)
- For `Empty-8x8`: harder/easier cycle is  
  `Empty-8x8 ↔ Empty-16x16`  
- For `KeyCorridorS3R3`: harder chain is  
  `S3R3 → S4R3 → S5R3`  
  easier reverts back along the chain.

### 1.11 KL-Leash Trigger (Locked)
Compute KL to previous policy every update:
\[
\mathrm{KL}_t = \mathbb{E}_{s\sim\pi_t}\big[\mathrm{KL}(\pi_t(\cdot|s)\|\pi_{t-1}(\cdot|s))\big]
\]

Trigger:
\[
\mathrm{KL}_t > \kappa
\]
where \(\kappa\) is **95th percentile KL** from Baseline PPO pilot (locked per env family).

**On trigger (one update only):**
- \( \lambda_H \leftarrow 1.5\lambda_H \)
- action noise \( \sigma \leftarrow 1.25\sigma \)

### 1.12 Metrics
**Primary:** AULC of episodic return vs steps.  
**Secondary:** final return, steps-to-X% baseline final return, Procgen generalization gap.

### 1.13 Statistical Tests (Locked)
- Paired seeds across conditions.
- Primary test: paired t-test on AULC between **Full TRP-PPO** and best baseline (max of cond 1–3).
- Holm-Bonferroni correction across 7 tasks.

**Acceptance if:**  
- mean AULC gain ≥ **15%** and p < 0.05 (corrected), **or**  
- steps-to-threshold improves ≥ **20%** with no average final-return drop.

### 1.14 Falsifiers
TRP Control **not supported** if:

1) Full TRP-PPO fails to beat best baseline on AULC in ≥ 5 of 7 tasks.  
2) Speed win accompanies ≥5% worse final return on average.  
3) KL-only condition matches Full TRP gains → dt-band adds no value.

### 1.15 Logging & Release
Publish:
- per-seed curves  
- raw \(R_t, P_t, dt_{\text{eff}}\) traces  
- KL traces + trigger times  
- full config JSON + seeds  
- a single `run_all.sh` reproducer  
- all code and pilot locks in this repo.

---

## 2) Secondary Experiment: TRP-Augmented DPO for LLM Preference Tuning

### 2.1 Setup (Pinned)
- Base model: `Llama-3.1-8B` (or closest open equivalent)
- Preference dataset: `Anthropic HH-RLHF` or `Argilla/OpenHermes` prefs (locked at run time; name + hash recorded)
- Judge for evaluation: `AlpacaEval 2.0` (GPT-4o judge) + `MT-Bench`

### 2.2 Conditions
1) SFT only  
2) SFT + DPO baseline  
3) SFT + TRP-DPO (dt-band weighted routing)

### 2.3 TRP Variables (Locked)

**\(R_t\)** = pair novelty / surprise to the model:  
\[
R_t = \text{EMA}\big( -\log p_{\theta}(y_{\text{chosen}}|x)\ -\ (-\log p_{\theta}(y_{\text{rejected}}|x)) \big)
\]
(i.e., margin surprise)

**\(P_t\)** = judge reliability proxy:  
- AI judge: calibration accuracy on a fixed 1k calibrator set  
- Human judge: inter-annotator agreement rate  
Normalize to z-score.

**\(dt_{\text{eff}}=z(P_t)\cdot z(R_t)\)**

### 2.4 TRP-DPO Control (Locked)
Every epoch:
- reweight pairs so batch-average dt_eff lies in locked pilot band (Q0.35–Q0.65).  
- high dt_eff pairs are routed to stronger judges or debate rounds before acceptance.

### 2.5 Metrics
- AlpacaEval / MT-Bench win rate  
- safety refusal accuracy  
- label/judge cost per win-rate point

### 2.6 Acceptance / Falsification
TRP-DPO supported if:
- win rate improves ≥ **3 points** at same label budget, **or**
- matches baseline win rate with ≥ **20% fewer labels/judge FLOPs**.

Falsified otherwise.

---

## 3) Prereg Lock Statement
Any deviation from:
- environments  
- seeds  
- band/KL pilot definitions  
- control laws  
- metrics or acceptance criteria  
will be explicitly logged as a prereg violation in `results/violations.md`.
