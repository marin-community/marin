# Task `capability/d19-surveillance-control/ewma-baseline-alert-entry`

**EWMA baseline alert on daily ED syndrome counts.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d19.surveillance-control.surveillance.signal-analysis-1-b2ed68e2d0ae/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `capability/d19-surveillance-control/ewma-baseline-alert-entry` |
| `schema_version` | `0.9` |
| `difficulty` | 3 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `ae498a7ac3c169b6270d9c4d2ac19bc2c7baf02e2070091638791674fd641195` |

`coverage_tags`: `artifact:numeric_answer`, `competency:quantitative_reasoning`, `shape:calculation`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `f18783002fc6a5e52e0445e546e342ae70029d43561f2713a9694d2c07b2aba1` |
| `source.row` | `b2ed68e2d0ae27dcd1a97b50b5a2ab7110443ef9393e05b1379fc27c6b2f222a` |
| `source.importer_revision` | `taskcompendium-dc6b501c8604bcd2e3c20c1e9947679845fdfef8` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d19.surveillance-control.surveillance.signal-analysis` — *Detect and Interpret Surveillance Signals* |
| subject | D19 — Public Health, Epidemiology & Health Systems |
| proposal slot | 1 |
| task family | ewma-baseline-alert-entry |

The capability's stated outcome: *"Distinguish actionable surveillance signals from expected variation and reporting artifacts using explicit baselines, denominators, delays, and escalation rules."*
Declared **excludes**: Long-horizon forecasting; Declaring outbreaks from raw counts alone; Source attribution.

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
| rendering | `{"id":"plain","instruction_surface":"original","submission":{"extractor":{"kind":"plain"},"kind":"assistant_final"},"version":"0.2"}` |
| step 0 `context_requirement` | `instruction_and_workspace` |
| step 0 `answer_requirements` | `{"kind":"text"}` |

`task.toml`:

```toml
version = "1.0"

[environment]
allow_internet = false
```

---

## The prompt

What follows is `instruction.md` in full, the complete solver-facing surface (4.7 KB, 82 lines).

---

# EWMA baseline alert review - ED gastrointestinal syndrome counts

## Scenario

You are the on-call epidemiologist for Memorial General Hospital's emergency department (ED) syndromic surveillance system. Each day the system records the number of ED visits meeting the gastrointestinal syndrome case definition. The table below lists 28 consecutive days of counts, from Day 1 (Mon 2025-06-02) through Day 28 (Sun 2025-06-29).

The system uses a standing EWMA-baseline alert rule, fully specified below. Your job is to apply that rule exactly as written, day by day, and report the five values requested in the answer template. Under hospital policy, if at least one evaluated day triggers an alert, you must trigger a formal outbreak review.

## Daily visit counts

| Day | Date | Gastrointestinal syndrome visits |
|----|------------|-----|
| 1 | 2025-06-02 (Mon) | 9 |
| 2 | 2025-06-03 (Tue) | 13 |
| 3 | 2025-06-04 (Wed) | 8 |
| 4 | 2025-06-05 (Thu) | 12 |
| 5 | 2025-06-06 (Fri) | 10 |
| 6 | 2025-06-07 (Sat) | 14 |
| 7 | 2025-06-08 (Sun) | 9 |
| 8 | 2025-06-09 (Mon) | 11 |
| 9 | 2025-06-10 (Tue) | 10 |
| 10 | 2025-06-11 (Wed) | 12 |
| 11 | 2025-06-12 (Thu) | 9 |
| 12 | 2025-06-13 (Fri) | 13 |
| 13 | 2025-06-14 (Sat) | 11 |
| 14 | 2025-06-15 (Sun) | 10 |
| 15 | 2025-06-16 (Mon) | 12 |
| 16 | 2025-06-17 (Tue) | 11 |
| 17 | 2025-06-18 (Wed) | 13 |
| 18 | 2025-06-19 (Thu) | 10 |
| 19 | 2025-06-20 (Fri) | 11 |
| 20 | 2025-06-21 (Sat) | 12 |
| 21 | 2025-06-22 (Sun) | 15 |
| 22 | 2025-06-23 (Mon) | 16 |
| 23 | 2025-06-24 (Tue) | 19 |
| 24 | 2025-06-25 (Wed) | 23 |
| 25 | 2025-06-26 (Thu) | 26 |
| 26 | 2025-06-27 (Fri) | 28 |
| 27 | 2025-06-28 (Sat) | 30 |
| 28 | 2025-06-29 (Sun) | 31 |

## The alert rule (apply exactly as specified; every convention is fixed)

1. **Smoothing constant:** lambda = 0.3.
2. **Control-limit multiplier:** k = 3.
3. **Warm-up period:** Days 1-7 (the first 7 days). The warm-up days are used only to initialize the rule; they are never evaluated for alerts.
4. **Baseline mean:** mu_w = arithmetic mean of the 7 warm-up counts.
5. **Baseline spread:** sigma_w = the sample standard deviation of the 7 warm-up counts with the n - 1 denominator (sum the squared deviations from mu_w, divide by 6, take the square root).
6. **Fixed sigma:** sigma_w is computed once from Days 1-7 and then held fixed. It is never re-estimated, updated, or recomputed from any later data.
7. **Initialization:** EWMA_7 = mu_w (the EWMA statistic after Day 7 equals the warm-up mean).
8. **Sequential update:** for each day t from Day 8 through Day 28, EWMA_t = lambda * x_t + (1 - lambda) * EWMA_(t-1) = 0.3 * x_t + 0.7 * EWMA_(t-1), where x_t is the Day-t count.
9. **Control limit:** for each day t from Day 8 through Day 28, UCL_t = EWMA_(t-1) + k * sqrt(lambda / (2 - lambda)) * sigma_w = EWMA_(t-1) + 3 * sqrt(0.3 / 1.7) * sigma_w. The limit is built from the previous day's EWMA statistic EWMA_(t-1), not from the same-day value EWMA_t.
10. **Alert condition:** day t alerts if and only if x_t > UCL_t, strictly greater. A count exactly equal to its control limit is not an alert.
11. **Evaluation window:** only Days 8-28 are evaluated. The first alert day, the alert-day count, and the review decision refer exclusively to Days 8-28.
12. **No day-of-week effect:** weekends and weekdays are treated identically. Do not adjust, reweight, exclude, or smooth any day because of its weekday, and do not apply holiday corrections.
13. **Full-precision intermediates:** carry every intermediate quantity (the whole EWMA chain, each control limit, each comparison) at full precision, with no rounding at intermediate steps. Round only the two final reported numbers to four decimal places.

## Worked example (Day 8 only)

This example pins the numerical conventions. It covers Day 8 only; Days 9-28 are yours to compute. Values below are displayed rounded to four decimals but are computed at full precision.

- mu_w = 10.7143 (mean of 9, 13, 8, 12, 10, 14, 9 = 75/7)
- sigma_w = 2.2887 (sample standard deviation of the same seven counts, n - 1 = 6)
- EWMA_7 = 10.7143
- UCL_8 = EWMA_7 + 3 * sqrt(0.3 / 1.7) * sigma_w = 10.7143 + 2.8843 = 13.5986
- The Day 8 count is 11. Since 11 < 13.5986, Day 8 does not alert.
- EWMA_8 = 0.3 * 11 + 0.7 * 10.7143 = 10.8000

## Answer template

Reply with exactly these five lines, each value after the colon:

```
EWMA_DAY20: <the EWMA statistic after Day 20, to four decimal places>
UCL_DAY21: <the control limit against which the Day 21 count is evaluated, to four decimal places>
FIRST_ALERT_DAY: <the first day in Days 8-28 that alerts, as a day number or its calendar date; NONE if no day alerts>
ALERT_DAY_COUNT: <the number of days in Days 8-28 that alert>
REVIEW: <YES if at least one day in Days 8-28 alerts, otherwise NO>
```

Return only the answer in the requested format.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |

Embedded resources (5):

| path | roles | lines |
| --- | --- | --- |
| `answer_key.json` | verifier | — |
| `ref_impl.py` | verifier | — |
| `verifier_config.json` | verifier | — |
| `verifier/grade.py` | verifier | — |
| `check_answer.py` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `cf13da5bfc4f3e5c…`) |
| repair budget | `{"exhausted":false,"max":4,"used":3}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 4,844 |
| `manifest.json` | 1,418 |
| `renderings.json` | 134 |
| `specification.json` | 46.8 K |
| `task.toml` | 54 |

