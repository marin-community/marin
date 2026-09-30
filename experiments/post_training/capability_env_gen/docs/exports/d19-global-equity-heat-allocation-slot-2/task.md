# Task `synthetic/d19-global-equity/heat-allocation-slot-2`

**Tiered heat-response allocation under a fixed budget.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d19.global-equity.intervention-2-d1903fe68baa/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/d19-global-equity/heat-allocation-slot-2` |
| `schema_version` | `0.9` |
| `difficulty` | 8 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `43a75e652ea5ac67e3d50ea75fef67015c59d323a085c4e789d34519fc812b06` |

`coverage_tags`: `artifact:json`, `competency:quantitative_reasoning`, `context:provided_documents`, `shape:constrained_design`, `subject:government`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `483a09f7780b8cc7bdb19e9f14756f2ad768d60454fd75165e69c2d93a70721a` |
| `source.row` | `d1903fe68baa00f59315204a0c0cb6ebfb0cbc82a3705c2d23f9c085d5b902d3` |
| `source.importer_revision` | `cap-construct-003-hc3-d1-20260928T160537Z-26099` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d19.global-equity.intervention` — *Design Equity-Centered Population Interventions* |
| subject | D19 — Public Health, Epidemiology & Health Systems |
| proposal slot | 2 |
| task family | proportionately-universal-allocation-design |

The capability's stated outcome: *"Design population interventions explicitly changing inequity-producing pathways and distributing participation, benefits, burdens, and accountability fairly."*
Declared **excludes**: Generic intervention design; Post hoc disparity measurement only; Token participation.

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
| rendering | `{"id":"response","instruction_surface":"original","submission":{"extractor":{"kind":"json_path","path":"$"},"kind":"assistant_final"},"version":"0.2"}` |
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

What follows is `instruction.md` in full, the complete solver-facing surface (7.5 KB, 108 lines).

---

# Heat-Response Supplemental Allocation - Marlow Bend

## Your role and the season you are planning for

You are the heat-response program officer for the synthetic City of Marlow Bend (population aged 65 and over: 43,200). Last heat season the universal cooling-center network reached 50.7% of high-income-quintile (Q5) residents but only 19.9% of lowest-quintile (Q1) residents citywide, and reach was worst in the highest-need neighborhoods.
The citywide baseline Q5-Q1 reach gap is 30.8 percentage points.

Council has approved a one-season supplemental budget of $120,000 and adopted a proportionate-universalism policy: the universal cooling-center offer stays citywide, every neighborhood must receive at least one incremental component, and allocated intensity must not be inversely related to need.
The Office of Heat Resilience requires two hard outcomes: a post-plan citywide Q5-Q1 reach gap of no more than 14.0 percentage points, and Q1 reach of at least 35.0% in every neighborhood.

Produce `allocation_plan.json` as described at the end of this packet. Every number you state will be recomputed from the yield table; inconsistent claims fail.

## Baseline summary (stated so you need not re-derive it)

Citywide population-weighted reach last season:

| Measure | Value |
| --- | --- |
| Q1 reach (citywide, pop-weighted) | 19.94% |
| Q5 reach (citywide, pop-weighted) | 50.70% |
| Q5-Q1 gap | 30.76 pp |

## Neighborhood table

Each income quintile is 20% of each neighborhood's 65+ population. The need index comes from the city heat vulnerability assessment (components: AC prevalence, 65+ density, heat-mortality risk).

| Neighborhood | Pop 65+ | AC prevalence | Need index | Cooling-center sites | Q1/Q2/Q3/Q4/Q5 baseline reach (%) |
| --- | --- | --- | --- | --- | --- |
| N01 | 4,200 | 0.91 | 2 | 2 | 38 / 44 / 50 / 55 / 60 |
| N02 | 3,600 | 0.88 | 3 | 2 | 34 / 40 / 47 / 53 / 58 |
| N03 | 5,100 | 0.79 | 5 | 1 | 27 / 34 / 41 / 48 / 55 |
| N04 | 4,800 | 0.74 | 6 | 1 | 23 / 30 / 38 / 46 / 53 |
| N05 | 6,200 | 0.66 | 7 | 2 | 19 / 26 / 34 / 43 / 51 |
| N06 | 5,400 | 0.58 | 8 | 1 | 15 / 22 / 30 / 39 / 48 |
| N07 | 7,100 | 0.47 | 9 | 1 | 11 / 18 / 26 / 36 / 46 |
| N08 | 6,800 | 0.39 | 10 | 1 | 8 / 14 / 22 / 32 / 43 |

## Component menu

All costs are city planning estimates for this synthetic season. Per-unit base reach yields are in percentage points by quintile, before the effectiveness multiplier below. Yields are additive across units and across components, subject to per-neighborhood caps.

| Component | Cost (USD) | Unit | Cap | Q1/Q2/Q3/Q4/Q5 base yield (pp/unit) |
| --- | --- | --- | --- | --- |
| extended_hours | $9,000 | per site-season | max 1 per site (see site count above) | 6 / 5 / 4 / 3 / 2 |
| transport_vouchers | $2,500 | per block of 100 vouchers | max 3 per neighborhood | 6 / 5 / 3 / 1 / 0 |
| ac_repair_grants | $7,000 | per batch of 20 grants | max 2 per neighborhood | 4 / 3 / 2 / 1 / 0 |
| outreach_workers | $5,200 | per worker-season | max 4 per neighborhood | 3 / 3 / 2 / 2 / 1 |

## Yield model

Effective yield per unit = base yield x f, where:

f = 0.5 + 0.06 x need_index

Rationale (supplied planning assumption): a larger share of residents in high-need neighborhoods lack home air conditioning and vehicles and are therefore more responsive to the offer, so the same unit buys more additional reach where need is higher.

Projected post-plan reach for a cell = baseline reach + sum over all allocated units of (base yield x f), capped at 92% in every neighborhood-quintile cell.

### Worked example (one neighborhood, not a recommendation)

Neighborhood N03 has need index 5, so f = 0.5 + 0.06 x 5 = 0.80.
One transport-voucher block there adds 4.80 pp of Q1 reach (base yield 6 x f), so projected Q1 reach = 27 + 4.80 = 31.80%, and projected Q5 reach = 55 + 0.00 = 55.00%.
Additional Q1 residents reached = (31.80 - 27)% x 1,020 = 48.96 persons.

## Budget, required targets, and policy constraints

Total spend must not exceed $120,000. Unit counts are non-negative integers within the per-neighborhood caps in the component menu (extended_hours are capped per site).

Required targets and policy:

- R1: post-plan citywide Q5-Q1 reach gap <= 14.0 pp (population-weighted across all neighborhoods).
- R2: projected Q1 reach >= 35.0% in every neighborhood.
- R4: every neighborhood receives at least 1 component unit.
- Spend per neighborhood must be non-decreasing in need index (ties allowed); a plan that concentrates resources in low-need neighborhoods while high-need neighborhoods stay untreated is invalid.
- Your own stated gap-reduction target must be no looser than the required gap target and must be achieved by your recomputed projections.

## Arithmetic consistency requirement

Stated projected reach for all neighborhood-quintile cells must match recomputation from this yield table within 0.5 pp, and stated persons reached must be within max(2%, 5 persons) of recomputation. Rounding to one decimal place is fine.

## Deliverable: `allocation_plan.json`

Submit one JSON document with exactly these top-level fields:

- `neighborhood_allocations`: eight entries keyed N01, N02, N03, N04, N05, N06, N07, N08, each with:
  - `units`: object mapping component name to integer unit count (components: extended_hours, transport_vouchers, ac_repair_grants, outreach_workers)
  - `spend_usd`: integer dollars for that neighborhood
  - `projected_reach_pct`: projected reach percentage for Q1, Q2, Q3, Q4, Q5
  - `projected_additional_persons_reached`: persons by quintile and `total`
- `stated_gap_target_pp`: your own numeric gap-reduction target in percentage points
- `benefit_accounting`: short summary of who gains and by how much (persons, by subgroup where useful)
- `burden_and_harm`: array of at least 3 entries, each with `type` (one of: travel_burden, heat_exposure_during_travel, stigma_of_targeting, worker_burden, displacement_from_familiar_sites, digital_access_barrier), `affected_subgroups`, `description`, and a non-empty `mitigation`
- `governance`: `schedule_authority` and `site_authority`, each with `holder`, `community_role` (one of: holds, shared, advisory_only, none), `named_community_entity`, `decision_review_cadence`
- `reallocation_trigger`: `metric` (one of: citywide_q5_q1_gap_pp, q1_reach_worst_neighborhood_pct, q1_reach_neighborhood_pct), numeric `threshold_pp`, `review_point` (one of: mid_season, season_end, post_season_review), `action_type` (one of: shift_intensity, shift_budget, add_workers, change_sites, escalate), `action`, `escalation`

Burden and harm entries must cover at least one of travel_burden or heat_exposure_during_travel, stigma_of_targeting, and at least one other type. Governance must name a specific community or neighborhood entity, and at least one of schedule or site authority must be held or shared with the community rather than advisory-only. The reallocation trigger must reference a gap or subgroup metric with a numeric threshold in pp, a season review point, and a shift/escalate action.

## Data provenance

All figures in this packet are synthetic city planning estimates for the fictional City of Marlow Bend, designed for this planning exercise; magnitudes were chosen to be plausible for a mid-sized city heat season and are not measured program data.


## Final answer

Return the complete `allocation_plan.json` document as your final answer: one valid JSON object with exactly the top-level fields listed above. Return valid JSON only; do not use Markdown fences and do not add any commentary before or after the document.

Return valid JSON with the answer at $; do not use Markdown fences.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |

Embedded resources (4):

| path | roles | lines |
| --- | --- | --- |
| `grade_entry.py` | verifier | — |
| `validator.py` | verifier | — |
| `data_module.py` | verifier | — |
| `allocation_schema.json` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `1d7696b7dcb5e9d1…`) |
| repair budget | `{"exhausted":false,"max":4,"used":1}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 7,655 |
| `manifest.json` | 1,492 |
| `renderings.json` | 152 |
| `specification.json` | 62.4 K |
| `task.toml` | 54 |

