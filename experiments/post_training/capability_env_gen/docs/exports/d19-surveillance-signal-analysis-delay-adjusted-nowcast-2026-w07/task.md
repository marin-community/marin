# Task `synthetic/d19-surveillance-signal-analysis/delay-adjusted-nowcast-2026-w07`

**Delay-adjusted nowcast against a seasonal threshold (Riverbend District, 2026-W07).**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d19.surveillance-control.surveillance.signal-analysis-2-bf6b2c25211b/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/d19-surveillance-signal-analysis/delay-adjusted-nowcast-2026-w07` |
| `schema_version` | `0.9` |
| `difficulty` | 5 |
| `success_policy` | `all_required_steps` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `9abefda04028a6e2752aaaf3031e04fd85fee4a24019ebbab87f45cb1fcf6ed1` |

`coverage_tags`: `artifact:numeric_answer`, `competency:quantitative_reasoning`, `shape:calculation`, `subject:medicine.epidemiology`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `0ce32038771d4fb0c66498f29604474d6766040e8917f894b31940a6a828149a` |
| `source.row` | `bf6b2c25211b7f5b1bf42d7e17615e96e91f13bf20456d817a0ca4a10d3cc923` |
| `source.importer_revision` | `cap-construct-003-hc3-d1-20260928T160537Z-26099` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d19.surveillance-control.surveillance.signal-analysis` — *Detect and Interpret Surveillance Signals* |
| subject | D19 — Public Health, Epidemiology & Health Systems |
| proposal slot | 2 |
| task family | delay-adjusted-surveillance-alert-assessment |

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
| rendering | `{"id":"final","instruction_surface":"original","submission":{"extractor":{"kind":"json_path","path":"$"},"kind":"assistant_final"},"version":"0.2"}` |
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

What follows is `instruction.md` in full, the complete solver-facing surface (3.1 KB, 38 lines).

---

# Riverbend District bulletin assessment — epi week 2026-W07 (delay-adjusted) — needed before today's 14:00 bulletin

Riverbend District (population 550,000 per current census estimate; the five baseline seasons averaged 500,000).

**CURRENT WEEK, 2026-W07 (Mon 2026-02-09 to Sun 2026-02-15):** as of this morning, Wednesday 2026-02-18 — the third reporting day (Mon 16, Tue 17, Wed 18) since the week closed — the database holds **51 lab-confirmed cases** with event dates in W07.

**REPORTING-COMPLETENESS AUDIT** (average fraction of a week's eventual reports received by the end of day k after that week closes; based on our last two years of reporting history):

| day k after week close | 1 | 2 | 3 | 4 | 5 | 10 | 15 | 20 | 28+ |
|---|---|---|---|---|---|---|---|---|---|
| fraction received | 0.55 | 0.62 | 0.68 | 0.74 | 0.81 | 0.92 | 0.96 | 0.97 | 1.00 (eventual) |

The reference lab reports no unusual backlog this cycle.

**SEASONAL BASELINE for ISO week 07 (five prior seasons):** mean 9.8 cases per 100,000 population, SD 1.5 per 100,000.

**PRIOR WEEK, 2026-W06:** now final at **58 cases** (complete; no delay adjustment applicable).

**ALERT RULE:** flag a week for review when its rate is at least 2.0 standard deviations above the seasonal baseline mean (z >= 2.0). Compare like with like: adjust the current week for incomplete reporting before comparing, and put counts and rates on the current population basis.

## TASKS

1. **Assess W07:** state the completeness factor you used, the delay-adjusted eventual count, the current rate per 100,000, the z-score, and whether W07 triggers review; also state the z-score the unadjusted partial count alone would have produced.
2. **Assess W06:** z-score and decision.
3. For each week, select validation actions ONLY from this vocabulary: `recheck_next_reporting_cycle`, `verify_lab_backlog`, `cross_check_second_signal`, `inspect_reporting_unit_timeliness`, `confirm_case_definition_applied`, `request_deduplicated_linelist`, `escalate_to_outbreak_team`, `routine_monitoring_continue`.

   Mandatory minimum: for an alert week, `recheck_next_reporting_cycle` and `cross_check_second_signal`; for a non-alert week, `routine_monitoring_continue`. `escalate_to_outbreak_team` may appear only for an alert week; `routine_monitoring_continue` only for a non-alert week.

## DELIVERABLE

A single JSON object with exactly these keys — `district`, `week_under_assessment`, `completeness_factor_w07`, `adjusted_eventual_count_w07`, `current_rate_per_100k_w07`, `z_w07`, `alert_w07`, `unadjusted_z_w07`, `z_w06`, `alert_w06`, `validation_actions_w07`, `validation_actions_w06`.

Submit the JSON object as your entire response: no Markdown code fences, no text before or after it. A response that is not a single valid JSON document cannot be parsed as an answer and is recorded as an ungraded extraction error with no numeric score; a response that is valid JSON is graded on its contents.

Do not project beyond the eventual total for W07; do not attribute a source; the decision vocabulary is 'alert for review', not 'outbreak declared'.

Return valid JSON with the answer at $; do not use Markdown fences.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |

Embedded resources (2):

| path | roles | lines |
| --- | --- | --- |
| `checker.py` | verifier | — |
| `grade_entry.py` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `a70f6ba0f9a60dbf…`) |
| repair budget | `{"exhausted":false,"max":4,"used":1}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 3,177 |
| `manifest.json` | 1,500 |
| `renderings.json` | 149 |
| `specification.json` | 40.1 K |
| `task.toml` | 54 |

