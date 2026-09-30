# Task `capability/d18.function.measure/slot-7-cwt10-gait-speed-v1`

**Compute gait speeds and pick the defensible interpretation.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d18.function.measure-7-fc81597e5398/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `capability/d18.function.measure/slot-7-cwt10-gait-speed-v1` |
| `schema_version` | `0.9` |
| `difficulty` | 4 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `0094f29b59d1c1cbaa28b5a2099907fcfe939bf4a4d3a777cb9e8eecf9524504` |

`coverage_tags`: `competency:quantitative_reasoning`, `shape:answer`, `subject:medicine.rehabilitation`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `0ce32038771d4fb0c66498f29604474d6766040e8917f894b31940a6a828149a` |
| `source.row` | `fc81597e53988e976508b471c2cc6481979b633e2f26d14d84d0ef1e8f91c5fd` |
| `source.importer_revision` | `dc6b501c8604bcd2e3c20c1e9947679845fdfef8` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d18.function.measure` — *Score and interpret functional performance* |
| subject | D18 — Nursing, Allied Health & Rehabilitation |
| proposal slot | 7 |
| task family | timed-mobility-scoring |

The capability's stated outcome: *"Map supplied performance observations to an explicit functional instrument or assistance scale, calculate scores correctly, and interpret change without exceeding the instrument's stated limits."*
Declared **excludes**: selecting a therapy intervention; assigning scores from undocumented assumptions; measuring acute physiologic instability.

---

## Environment and interface

| field | value |
| --- | --- |
| proposed environment | `reasoning` |
| `requirements.state.image` | `null` |
| `requirements.state.workdir` | `/app` |
| `requirements.state.setup_commands` | `[]` |
| `requirements.capabilities` | `[]` |
| `requirements.action_interfaces` | `[]` |
| `binding.json` | `{"environment":{"kind":"none"},"tools":[]}` |
| rendering | `{"id":"answer","instruction_surface":"original","submission":{"extractor":{"kind":"plain"},"kind":"assistant_final"},"version":"0.2"}` |
| step 0 `context_requirement` | `instruction_and_workspace` |
| step 0 `answer_requirements` | `{"kind":"json"}` |

`task.toml`:

```toml
version = "1.0"

[environment]
allow_internet = false
```

---

## The prompt

What follows is `instruction.md` in full, the complete solver-facing surface (4.9 KB, 122 lines).

---

## 1. Clinician's request

Here is Miriam Okafor's reassessment from today - the CWT-10 protocol, her trial
log, her baseline report from six weeks ago, and our unit's reference sheet.
Please work out her trial speeds and condition means for comfortable and fast
pace, classify her comfortable pace against the reference table, and tell me
which of the four drafted sentences I can defensibly put in her progress note.
I don't want to overstate her progress.

---

## 2. Protocol sheet — Corridor Walk Test-10 (CWT-10)

**Corridor Walk Test-10 (CWT-10) — administration and scoring**

- A measured corridor of 10.0 m is marked out; the patient starts standing
  still (static start) behind the start line.
- Timing begins on the "go" signal and stops when the patient crosses the
  10.0 m mark.
- Two trials are performed at a comfortable pace and two trials at a fast pace
  (as fast as safely possible), in any order on the day.
- Trial speed = 10.0 / trial time, in m/s.
- Report all trial speeds to two decimal places.
- The condition mean (comfortable, fast) is computed from the unrounded trial
  speeds and rounded to two decimal places.

*Fixture-defined instrument; values are not published psychometrics.*

---

## 3. Trial log — reassessment, 2026-09-28

**Patient:** Miriam Okafor — Age: 74 — ID: PT-4471
**Date of assessment:** 2026-09-28 — Protocol: CWT-10

Trials as recorded on the day (rows in the order logged):

| Log row | Entry |
| --- | --- |
| 1 | F1 (fast): 5.4 s |
| 2 | C1 (comfortable): 8.20 s |
| 3 | F2 (fast): 5.60 s |
| 4 | C2 (comfortable): 8.00 s |

*Fixture-defined instrument; values are not published psychometrics.*

---

## 4. Baseline report — 2026-08-17 (six weeks prior)

**Patient:** Miriam Okafor — Age at baseline: 73 — ID: PT-4471
**Date of baseline assessment:** 2026-08-17 — Protocol: CWT-10

**Condition means at baseline (same protocol):**

| Condition | Mean speed |
| --- | --- |
| Comfortable pace | 1.10 m/s |
| Fast pace | 1.80 m/s |

Raw baseline trial times are not retained in this report; only the condition
means were recorded.

*Fixture-defined instrument; values are not published psychometrics.*

---

## 5. Reference table — comfortable-pace mean gait speed by age band

**Reference table (comfortable-pace mean speeds only):**

| Age band | Impaired | Borderline | Within normal limits |
| --- | --- | --- | --- |
| 60-69 | < 0.80 m/s | 0.80 - 1.19 m/s | >= 1.20 m/s |
| 70-79 | < 0.90 m/s | 0.90 - 1.29 m/s | >= 1.30 m/s |

**Coverage note:** Reference values apply only to ages 60-79. No reference values are provided for other ages. Categories apply to comfortable-pace mean speed only.

*Fixture-defined instrument; values are not published psychometrics.*

---

## 6. Instrument page — detectable difference

**CWT-10 interpretive limits:**

Detectable difference = 0.20 m/s per pace condition. A change smaller than this value is not interpretable as a detectable change for this instrument.

*Fixture-defined instrument; values are not published psychometrics.*

---

## 7. Drafted progress-note sentences

The clinician has drafted four candidate sentences for the progress note:

- **A.** Miriam's comfortable-pace gait speed increased from a baseline mean of 1.10 m/s to a current mean of 1.23 m/s, a change of 0.13 m/s; her fast-pace mean is 1.82 m/s (change 0.02 m/s). This improvement in comfortable-pace gait speed appears clinically meaningful.
- **B.** Miriam's comfortable-pace mean gait speed is 1.23 m/s (change from baseline 0.13 m/s), with a fast-pace mean of 1.82 m/s (change 0.02 m/s); this comfortable-pace speed appears to fall within normal limits for adults over 80, so her comfortable-pace walking appears within normal limits for her age group.
- **C.** Miriam's comfortable-pace gait speed increased by 0.13 m/s from baseline (current mean 1.23 m/s), and her fast-pace mean of 1.82 m/s (change 0.02 m/s) reflects this gain, suggesting improved fast-pace walking capacity as well.
- **D.** Miriam's comfortable-pace mean gait speed is 1.23 m/s (fast-pace mean 1.82 m/s), which is borderline for her age band, and her changes from baseline (0.13 m/s comfortable pace, 0.02 m/s fast pace) are both smaller than the instrument's detectable difference of 0.20 m/s, so neither change would be interpretable as a detectable change for this instrument.

*Fixture-defined instrument; values are not published psychometrics.*

---

## 8. What to return

Return a single fenced JSON object with exactly these fields:

- `speeds_m_per_s`: `comfortable_trial_1`, `comfortable_trial_2`,
  `fast_trial_1`, `fast_trial_2` — each to two decimals
- `mean_comfortable_m_per_s`
- `mean_fast_m_per_s`
- `comfortable_change_from_baseline_m_per_s` (mean now minus baseline; faster
  is positive)
- `fast_change_from_baseline_m_per_s`
- `comfortable_pace_category`: one of `impaired`, `borderline`,
  `within normal limits`
- `interpretation_statement`: one of `A`, `B`, `C`, `D`

Return only the answer in the requested format.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `0628e44492bd0bcd…`) |
| repair budget | `{"exhausted":false,"max":4,"used":0}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 5,025 |
| `manifest.json` | 1,401 |
| `renderings.json` | 135 |
| `specification.json` | 22.2 K |
| `task.toml` | 54 |

