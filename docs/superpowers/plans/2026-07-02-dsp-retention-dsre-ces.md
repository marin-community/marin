# DSP Retention, DS-RE, and CES Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether retention, DS-RE-style satiety, and CES-style aggregation improve the two-phase effective-exposure DSP fit and produce better Table-9 schedules.

**Architecture:** Extend the existing standalone DSP variant registry so all existing nested-CV, KL-sweep, and mixture-export diagnostics continue to work. Keep CES as a diagnostic path unless it clearly improves selection, because the prior March deck showed CES/DS-RE-CEQ can overfit badly in many-domain settings.

**Tech Stack:** Python 3.12, `uv run`, NumPy/SciPy/Pandas/Plotly, existing Marin Fieldbook and Table-9 analysis scripts.

---

### Task 1: Add Retention and DS-RE Variants

**Files:**
- Modify: `experiments/domain_phase_mix/exploratory/two_phase_many/standalone_code/dsp_exact.py`

- [ ] **Step 1: Add phase modes and variant registry entries**

Add four variants:

```python
"retained_effective_exposure"
"split_retained_exposure"
"dsre_satiety"
"retained_penalty_only"
```

Each variant must decode explicit nonlinear parameters, keep positive parameters in log coordinates, and preserve NNLS variable projection for the linear head.

- [ ] **Step 2: Implement feature builders**

Implement retained exposure:

```python
x_i = exp(-lambda * (1 - p1_i)) * e0_i + eta * e1_i
```

Implement DS-RE satiety signal:

```python
z0_i = log1p(e0_i)
z1_i = log1p(phi * e0_i + e1_i) - log1p(phi * e0_i)
s_i = exp(-lambda * (1 - p1_i)) * z0_i + eta * z1_i
```

Use the same penalty family as DSP unless the variant explicitly uses retained penalty exposure.

- [ ] **Step 3: Compile**

Run:

```bash
uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/standalone_code/dsp_exact.py
```

Expected: command exits 0.

### Task 2: Run Table-9 Diagnostics

**Files:**
- Use: `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_table9_phase_split_dsp_300m.py`

- [ ] **Step 1: Run nested-CV plus KL sweep**

Run:

```bash
uv run python experiments/domain_phase_mix/exploratory/two_phase_many/analyze_table9_phase_split_dsp_300m.py \
  --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/table9_dsp_retention_dsre_ces_20260702 \
  --variants effective_exposure,split_saturation_penalty,retained_effective_exposure,split_retained_exposure,dsre_satiety,retained_penalty_only \
  --linear-reg-values 0.0001,0.001,0.01 \
  --kl-reg-values 0.025,0.05,0.1,0.2,0.3,0.5 \
  --maxiter 20 \
  --coarse-top-k 2
```

Expected: summary CSVs and Plotly artifacts are written under the output directory.

### Task 3: CES Diagnostic

**Files:**
- Create if needed: `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_table9_ces_dsp_300m.py`

- [ ] **Step 1: Only create a separate CES diagnostic if the retained additive variants do not dominate**

Implement a fixed-weight CES diagnostic over retained per-domain signals, using simple component weights rather than high-dimensional free CES weights, so the result remains interpretable:

```python
U(w) = (sum_i a_i s_i(w)^rho_ces) ** (1 / rho_ces)
```

Compare nested OOF fit, selection regret, and optimism against additive effective exposure.

### Task 4: Record and Decide

**Files:**
- Fieldbook experiment: `exp_01kwc2ha76te8g6277h9hq520h`

- [ ] **Step 1: Add artifacts and validation notes**

Record generated CSV/HTML artifacts and a concise checkpoint summarizing whether any variant improves nested regret or produces a plausible 3e18 validation candidate.

- [ ] **Step 2: If a candidate is promising, prepare a validation mixture**

Only launch validation if the candidate is clearly better on held-out selection diagnostics and not just lower in overoptimistic predicted BPB.
