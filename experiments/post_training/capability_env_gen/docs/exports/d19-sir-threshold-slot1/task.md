# Task `capability-pipeline/d19-sir-threshold-slot1`

**Herd-immunity threshold and critical transmission rate.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d19.surveillance-control.outbreaks.transmission-modeling-1-d806c1f52610/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `capability-pipeline/d19-sir-threshold-slot1` |
| `schema_version` | `0.9` |
| `difficulty` | 3 |
| `success_policy` | `all_required_steps` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `46c033873d3a38b1b972005068ab6ddcb39b60f185472b1b5642dff4cf035333` |

`coverage_tags`: `artifact:numeric_answer`, `competency:quantitative_reasoning`, `shape:calculation`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `1552ba04535f9e667fd85ece3593f5789d1470576ac18769e59c085803289e43` |
| `source.row` | `d806c1f5261052fdccec7c9667942edba23539535326b2734e40aab72625009a` |
| `source.importer_revision` | `a8ee2ab421` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d19.surveillance-control.outbreaks.transmission-modeling` — *Model Deterministic Transmission Scenarios* |
| subject | D19 — Public Health, Epidemiology & Health Systems |
| proposal slot | 1 |
| task family | analytic-sir-threshold-reasoning |

The capability's stated outcome: *"Construct and interrogate a deterministic compartment transmission model whose states, parameters, uncertainty, and intervention scenarios match the decision question."*
Declared **excludes**: Renewal, branching, network, metapopulation, and individual-based models; Named-person contact tracing; Pure statistical forecasting.

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

What follows is `instruction.md` in full, the complete solver-facing surface (3.6 KB, 58 lines).

---

# Briefing request: herd-immunity threshold and critical transmission rate

**From:** Deputy Health Officer, City Public Health Department
**To:** Analytics support
**Re:** Friday briefing on the currently circulating strain

Our city has 500,000 residents. Serosurvey data from the last two waves suggest about 20% of residents already have protective immunity. The strain now circulating has an average transmission rate of beta = 0.32 per day and an average infectious period of 6.25 days (recovery rate gamma = 0.16 per day). Assume a straightforward SIR picture for the threshold math.

I need four things for the briefing, with the formulas visible so the epi team can check them:

(a) the basic reproduction number R0 for this strain;
(b) the effective reproduction number Re given our current immunity;
(c) the critical vaccination coverage with a perfect-take vaccine that would bring Re down to exactly the threshold of 1 — tell me both the fraction (and be explicit about whether it is a fraction of currently susceptible people or of the whole population) and roughly how many people that is;
(d) the largest average transmission rate beta we could live with, at our current immunity level, without Re exceeding 1.

One more: the neighboring district has the same gamma and the same 20% immune fraction, but their beta is 0.20 per day — is their effective reproduction number strictly below one?

## Parameters

| Quantity | Value |
| --- | --- |
| City population, N | 500,000 residents |
| Residents with protective immunity from prior waves | 20% |
| Circulating strain transmission rate, beta | 0.32 per day |
| Recovery rate, gamma | 0.16 per day (average infectious period 6.25 days) |
| Neighboring district beta | 0.20 per day (same gamma, same 20% immune fraction) |

## Modeling assumptions

- Deterministic SIR model with mass-action incidence; threshold algebra only — no simulation needed.
- The 20% of residents with immunity from prior waves are fully protected and no longer susceptible.
- The vaccine, if used, has perfect take: it confers sterilizing immunity on currently susceptible residents, applied before any further transmission.
- All rates are per day.

## How to respond

- For each of (a)–(d), show the formula and the numeric substitution, not just the final number.
- Values may be given as decimals, percentages, or simple fractions (e.g., 0.375, 37.5%, 3/8).
- For (c), label which denominator you are using — currently susceptible people, or the whole population. Either framing is fine as long as you say which one.
- When you report (c) and (d), state clearly what happens at the critical value itself: where does Re sit exactly at that coverage, and exactly at that transmission rate, and how much coverage (or how much lower a rate) is needed for Re to be strictly below 1?
- Give a one-sentence plain-language reading of each result that I can read aloud in the briefing.
- End your reply with the completed answer block, filled in, using exactly the structure below — the same labels, one line per field — so the numbers can be lifted cleanly into the briefing deck.

## Answer block template

```
S_current: <number of currently susceptible residents>
R0: <basic reproduction number>
Re_current: <effective reproduction number at current immunity>
p_c: <critical vaccination coverage fraction> (denominator: <susceptible or total>)
p_c_people: <number of people that coverage represents>
beta_max: <largest average transmission rate without Re exceeding 1, per day>
district_Re: <neighboring district's effective reproduction number>
district_below_one: <yes or no — is the district's Re strictly below 1?>
```

Return only the answer in the requested format.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |

Embedded resources (3):

| path | roles | lines |
| --- | --- | --- |
| `grade.py` | verifier | — |
| `evaluator.py` | verifier | — |
| `patterns.json` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `94db8d5f93adbc58…`) |
| repair budget | `{"exhausted":false,"max":4,"used":2}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 3,729 |
| `manifest.json` | 1,386 |
| `renderings.json` | 134 |
| `specification.json` | 33.1 K |
| `task.toml` | 54 |

