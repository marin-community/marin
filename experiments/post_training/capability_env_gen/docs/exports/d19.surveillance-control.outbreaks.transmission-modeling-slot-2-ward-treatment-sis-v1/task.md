# Task `capability/d19.surveillance-control.outbreaks.transmission-modeling/slot-2-ward-treatment-sis-v1`

**Endemic equilibrium and critical treatment rate in a treatment-SIS model.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d19.surveillance-control.outbreaks.transmission-modeling-2-5504908c63dd/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `capability/d19.surveillance-control.outbreaks.transmission-modeling/slot-2-ward-treatment-sis-v1` |
| `schema_version` | `0.9` |
| `difficulty` | 4 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `e87880121d59292b6b03923c2ba9ba9bc58ed019596f7c65e957be4800c8468e` |

`coverage_tags`: `competency:quantitative_reasoning`, `shape:calculation`, `subject:medicine.public_health`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `18da81f78d8eb60bcc6a4c25e1cc45ccde26823e09a3598982f0db2d3d7abb48` |
| `source.row` | `5504908c63dde39cbaa3d014568d5a765711098068faafb1133896d11f308c28` |
| `source.importer_revision` | `dc6b501c8604bcd2e3c20c1e9947679845fdfef8` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d19.surveillance-control.outbreaks.transmission-modeling` — *Model Deterministic Transmission Scenarios* |
| subject | D19 — Public Health, Epidemiology & Health Systems |
| proposal slot | 2 |
| task family | deterministic-compartment-equilibrium-and-threshold-algebra |

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
| rendering | `{"id":"answers","instruction_surface":"original","submission":{"extractor":{"kind":"plain"},"kind":"assistant_final"},"version":"0.2"}` |
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

What follows is `instruction.md` in full, the complete solver-facing surface (4.6 KB, 111 lines).

---

# Ward treatment program: model numbers for Thursday's budget meeting

You are the quantitative support for a district health ward with a population
of **N = 10,000** people. The ward has had a treatable infection at a
stubbornly stable level for months. The ward medical officer needs the model
numbers before Thursday's budget meeting — closed-form results worked out from
the model, not a simulation.

## The model

The ward manages the infection with a deterministic treatment-SIS compartment
model. The infectious population I(t) evolves as:

  dI/dt = β·S·I/N − (γ + τ)·I

where S = N − I is the susceptible population. Individuals who clear the
infection naturally (rate γ) and individuals who are successfully treated
(rate τ) **return to the susceptible pool** — this is an SIS-style model, not
SIR: nobody acquires lasting immunity, and people leave the susceptible pool
only while they are infectious.

### Parameters

| Symbol | Meaning | Value |
| --- | --- | --- |
| N | ward population | 10,000 persons |
| β | transmission rate | 0.40 per day |
| γ | natural clearance rate | 0.10 per day |
| τ | current effective treatment rate | 0.05 per day |

## Context memo from a colleague

> Been thinking ahead to the budget meeting. Our treatment program is what's
> keeping this thing from exploding, and the effect should be roughly
> proportional: doubling the treatment rate should roughly halve the number of
> infected people at steady state. I'd put that on the slides.

The officer wants actual model numbers rather than this rule of thumb. Work
from the stated equations; your results will implicitly confirm or contradict
the memo either way.

## What elimination means here

"Elimination" in this task means the **disease-free equilibrium** (I = 0,
S = N) **becomes locally stable**: the initial effective reproduction number —
the expected number of secondary infections caused by one infectious
individual when everyone is still susceptible (S = N) — drops below one.

## What the officer needs

Answer from the stated equations (closed-form algebra suffices; no
simulation, external data, or code execution is required):

1. **Current endemic equilibrium prevalence** — the prevalence the current
   situation settles at (I*).
2. **Effective reproduction number at that endemic equilibrium** — the
   expected number of secondary infections caused by one infectious
   individual at the equilibrium state.
3. **Critical treatment rate τ_c** — the treatment rate that would achieve
   elimination, i.e. at which the disease-free state just becomes stable
   (bring it below threshold).
4. **Equilibrium prevalence if the treatment program doubles τ** (i.e. at
   τ = 0.10 per day).
5. **Local sensitivity of equilibrium prevalence to the treatment rate**:
   dI*/dτ at the current τ — how many persons the equilibrium prevalence
   moves per unit of τ.

Also state the **equilibrium condition you used**: the balance between
inflow into and outflow out of the infectious compartment (what equals what
at equilibrium) — not merely a restatement of the differential equation.

## Answer format

Return a single JSON object (inline in your final message) with exactly these
fields. Give numeric values to at least 3 significant digits, or exactly.
Units are fixed by the field names:

| Field | Meaning | Unit |
| --- | --- | --- |
| `I_star_persons` | endemic equilibrium prevalence at the current τ | persons |
| `I_star_percent` | same prevalence, as a percent of N | percent |
| `Re_equilibrium` | effective reproduction number at the endemic equilibrium | dimensionless |
| `tau_c_per_day` | critical treatment rate for elimination | per day |
| `I_star_tau_doubled_persons` | equilibrium prevalence at doubled τ (τ = 0.10/day) | persons |
| `dI_star_dtau_persons_per_unit_tau` | dI*/dτ at the current τ | persons per unit τ |
| `equilibrium_condition` | the inflow–outflow balance used | string |

An optional string field `notes` for formulas or a short derivation is
allowed (it is not part of the requested numbers).

Shape example (placeholder values only, not answers):

```json
{
  "I_star_persons": 0,
  "I_star_percent": 0.0,
  "Re_equilibrium": 0.0,
  "tau_c_per_day": 0.00,
  "I_star_tau_doubled_persons": 0,
  "dI_star_dtau_persons_per_unit_tau": 0,
  "equilibrium_condition": "...",
  "notes": "..."
}
```

The field table above fully defines the required JSON shape: those seven
fields (plus the optional `notes`) and no others. Every numeric field must be
a finite number; `equilibrium_condition` must be a non-empty string.

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
| quality review | `accept` (artifact sha256 `804b9fe8a0b3b3aa…`) |
| repair budget | `{"exhausted":false,"max":4,"used":1}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 4,661 |
| `manifest.json` | 1,406 |
| `renderings.json` | 136 |
| `specification.json` | 44.1 K |
| `task.toml` | 54 |

