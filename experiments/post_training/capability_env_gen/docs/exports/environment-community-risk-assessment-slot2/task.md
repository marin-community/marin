# Task `synthetic/environment-community-risk-assessment/slot2`

**Multi-pathway arsenic risk for a rural well community.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d19.environment-community.risk-assessment-2-62164482e41e/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/environment-community-risk-assessment/slot2` |
| `schema_version` | `0.9` |
| `difficulty` | 7 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `15dbc044c2c6cae320f88087bc2b5ccf468fd837143981a1df6b4189bd60181d` |

`coverage_tags`: `artifact:numeric_answer`, `competency:quantitative_reasoning`, `context:provided_documents`, `shape:calculation`, `subject:medicine.environmental_health`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `0ce32038771d4fb0c66498f29604474d6766040e8917f894b31940a6a828149a` |
| `source.row` | `62164482e41ee2c82074554403fb8cf361f10846ebd27f856b96b67f8c772616` |
| `source.importer_revision` | `cap-construct-003-hc3-d1-20260928T160537Z-26099` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d19.environment-community.risk-assessment` — *Characterize Chemical and Radiological Health Risk* |
| subject | D19 — Public Health, Epidemiology & Health Systems |
| proposal slot | 2 |
| task family | multi-pathway-chemical-risk-characterization |

The capability's stated outcome: *"Integrate chemical or radiological hazard, dose-response, and exposure evidence to estimate population risk and communicate uncertainty, assumptions, and susceptible groups."*
Declared **excludes**: Microbial dose-response, injury-risk, and mixture frameworks; Control selection; Clinical prognosis.

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
| step 0 `answer_requirements` | `{"kind":"json"}` |

`task.toml`:

```toml
version = "1.0"

[environment]
allow_internet = false
```

---

## The prompt

What follows is `instruction.md` in full, the complete solver-facing surface (7.5 KB, 104 lines).

---

# Screening-Level Risk Assessment: Marsh Family Property (Former Orchard)

You are supporting a county environmental health office. A screening-level human health risk assessment is needed for the Marsh family — two adults and one child (age 2–6) — who live on a former orchard property and drink from a private well. Recent sampling found arsenic in the well water and in the yard soil. Using the exposure parameters and toxicity values below, compute chronic daily doses, hazard quotients, and excess lifetime cancer risks for the adult and the child, separately for the water-ingestion pathway and the soil-ingestion pathway. Identify which pathway dominates for each age group. The family is considering a treatment system expected to cut well-water arsenic by 50% (from 60 micrograms/L to 30 micrograms/L) while soil is unchanged; recompute everything for that scenario and state which group(s) still exceed a hazard quotient of 1 and which pathway now drives each group's exposure.

Your deliverables are (1) the JSON code block described below, containing all computed results, and (2) a short narrative. Both are required parts of the response; the JSON block is the machine-read record of your results, and values stated only in the narrative are not counted as results.

Report per-pathway values only; do not sum hazard quotients or risks across pathways, age groups, or endpoints.

## Site and chemical information

- Chemical of concern: inorganic arsenic
- Measured well-water arsenic concentration: 60 micrograms/L (one confirmed sample round; assumed representative of chronic conditions)
- Measured yard-soil arsenic concentration: 400 mg/kg (dry weight, 0-15 cm composite)
- Scenario: well-water arsenic reduced by 50% to 30 micrograms/L; soil concentration unchanged

## Exposure parameters

| Parameter | Adult | Child (age 2-6) |
| --- | --- | --- |
| Water intake rate | 2.5 L/day | 1.0 L/day |
| Incidental soil ingestion rate | 50 mg/day | 100 mg/day |
| Body weight | 70 kg | 15 kg |
| Exposure frequency | 350 days/year | 350 days/year |
| Exposure duration | 30 years | 6 years |

## Toxicity values (oral, inorganic arsenic)

- Oral reference dose (noncancer): 0.0003 mg/kg-day
- Oral cancer slope factor: 1.5 (mg/kg-day)^-1

Use only these provided toxicity values; do not substitute values from other sources.

## Conventions

- Chronic daily intake: CDI = C x IR x EF x ED / (BW x AT), where C is the medium concentration, IR the intake rate, EF the exposure frequency, ED the exposure duration, BW the body weight, and AT the averaging time. The same formula applies to the water and soil pathways, with the unit conversions below, and to both the noncancer and cancer endpoints; the endpoints differ only in the averaging time AT.
- Noncancer endpoint: averaging time equals the exposure duration, expressed in days (AT = ED x 365 days).
- Cancer endpoint: averaging time is 70 years = 25,550 days, applied to each age group's exposure period separately (do not merge the child and adult periods into one lifetime profile). Compute the child's cancer dose and risk from the child's own exposure parameters with the 70-year averaging time, and the adult's from the adult's parameters with the 70-year averaging time.
- Hazard quotient: HQ = CDI(noncancer) / oral reference dose. Excess lifetime cancer risk: ELCR = CDI(cancer) x oral cancer slope factor.
- Relative absorption is 100% (no absorption factor is applied).
- Unit conversions: water 60 micrograms/L = 0.060 mg/L (1 microgram = 0.001 mg); soil 1 kg = 10^6 mg, so a soil concentration in mg/kg times an ingestion rate in mg/day is converted with 10^-6 kg/mg.
- Each age group is evaluated with its own parameters; do not swap or blend body weights, intake rates, or exposure durations between adult and child.

## Required JSON output

Report your results in a single fenced JSON code block (a block that opens with three backticks followed by json and closes with three backticks) with exactly this structure (all dose fields in mg/kg-day; hazard quotients and risks are unitless):

```json
{
  "baseline": {
    "adult": {
      "water": {"cd_noncancer": 0, "hq": 0, "cd_cancer": 0, "elcr": 0},
      "soil":  {"cd_noncancer": 0, "hq": 0, "cd_cancer": 0, "elcr": 0}
    },
    "child": {
      "water": {"cd_noncancer": 0, "hq": 0, "cd_cancer": 0, "elcr": 0},
      "soil":  {"cd_noncancer": 0, "hq": 0, "cd_cancer": 0, "elcr": 0}
    }
  },
  "scenario": {
    "adult": {
      "water": {"cd_noncancer": 0, "hq": 0, "cd_cancer": 0, "elcr": 0},
      "soil":  {"cd_noncancer": 0, "hq": 0, "cd_cancer": 0, "elcr": 0}
    },
    "child": {
      "water": {"cd_noncancer": 0, "hq": 0, "cd_cancer": 0, "elcr": 0},
      "soil":  {"cd_noncancer": 0, "hq": 0, "cd_cancer": 0, "elcr": 0}
    }
  },
  "dominant_pathway": {
    "baseline": {"adult": "", "child": ""},
    "scenario": {"adult": "", "child": ""}
  },
  "scenario_hq_exceeds_1": {
    "adult": {"water": null, "soil": null},
    "child": {"water": null, "soil": null}
  },
  "any_pathway_hq_exceeds_1": {"adult": null, "child": null},
  "assumptions": ["...", "..."]
}
```

Field descriptions:

- `baseline` / `scenario`: for each age group and each pathway, `cd_noncancer` is the chronic daily dose for the noncancer endpoint (mg/kg-day), `hq` is the hazard quotient (unitless), `cd_cancer` is the chronic daily dose for the cancer endpoint (mg/kg-day), and `elcr` is the excess lifetime cancer risk (unitless). `baseline` uses the measured concentrations (60 micrograms/L water, 400 mg/kg soil); `scenario` uses 30 micrograms/L water with the same soil concentration.
- `dominant_pathway`: for each case (`baseline` or `scenario`) and each age group, the single pathway ("water" or "soil") with the larger hazard quotient for that group in that case.
- `scenario_hq_exceeds_1`: for each age group and pathway in the 50%-reduction scenario, whether that pathway's hazard quotient exceeds 1 (true/false).
- `any_pathway_hq_exceeds_1`: for each age group in the 50%-reduction scenario, whether any single pathway's hazard quotient exceeds 1 (evaluate each pathway on its own; do not sum pathways).
- `assumptions`: at least two distinct assumptions you made, each at least a sentence long (for example: 100% relative absorption; concentrations assumed constant over the exposure duration; the single sampling round is representative of chronic conditions).

All values shown in the example block above are placeholders illustrating the required structure only, not answers; replace every one with your computed result (empty strings, nulls, and zeros must all be replaced). Report numeric values to at least 2 significant figures (scientific notation such as 2.1e-3 is acceptable). Do not add extra keys to the JSON structure; every field shown must be filled in.

## Narrative

Include a short narrative (200-400 words) summarizing the dominant contributors at baseline, the effect of the water reduction, and which group(s) remain above a hazard quotient of 1 in the scenario.

## Computation rules

- Derive all numeric results from the provided parameters only.
- Apply the noncancer averaging time (exposure duration) and the cancer averaging time (70 years) exactly as stated above.
- Recompute scenario values from first principles; soil-pathway values are identical between baseline and scenario because soil is unchanged.
- Do not compute or report a summed hazard index across pathways, a summed ELCR across pathways, an adult-plus-child combined metric, or any combined noncancer-plus-cancer metric. Report each age group, pathway, and endpoint separately, exactly as the JSON structure requires.

Return only the answer in the requested format.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |

Embedded resources (4):

| path | roles | lines |
| --- | --- | --- |
| `composed_verifier.py` | verifier | — |
| `evaluator.py` | verifier | — |
| `params.json` | verifier | — |
| `verifier_config.json` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `d67ebca8319d6301…`) |
| repair budget | `{"exhausted":false,"max":4,"used":0}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 7,713 |
| `manifest.json` | 1,483 |
| `renderings.json` | 134 |
| `specification.json` | 66.1 K |
| `task.toml` | 54 |

