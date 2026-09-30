# Task `synthetic/biostatistics-evidence-appraisal/abstract-only-domain-judgment-1`

**Abstract-Only Trial: Domain Judgment Under Missing Information.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d19.biostatistics-evidence.synthesis.appraisal-1-faf0910b367a/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/biostatistics-evidence-appraisal/abstract-only-domain-judgment-1` |
| `schema_version` | `0.9` |
| `difficulty` | 3 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `07d88be377aaed08571a9826ab6c8dba68b6ea632488639fa32eb041da1bd2bf` |

`coverage_tags`: `competency:evidence_synthesis`, `context:provided_documents`, `shape:multiple_choice`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `0ce32038771d4fb0c66498f29604474d6766040e8917f894b31940a6a828149a` |
| `source.row` | `faf0910b367ab131f33e8be6bb9870cbf0609ffa6749fb5bc38661da2b13581f` |
| `source.importer_revision` | `cap-construct-003-hc1-d1-20260928T160515Z-25428` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d19.biostatistics-evidence.synthesis.appraisal` — *Appraise Study Credibility and Applicability* |
| subject | D19 — Public Health, Epidemiology & Health Systems |
| proposal slot | 1 |
| task family | trial-report-appraisal-constrained-choice |

The capability's stated outcome: *"Appraise a study's internal validity and applicability using design-specific evidence rather than result direction or prestige."*
Declared **excludes**: Eligibility decisions; Pooling estimates; Judging credibility from result direction.

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
| step 0 `answer_requirements` | `{"kind":"text"}` |

`task.toml`:

```toml
version = "1.0"

[environment]
allow_internet = false
```

---

## The prompt

What follows is `instruction.md` in full, the complete solver-facing surface (4.5 KB, 63 lines).

---

# A 6-Month Text-Message Physical Activity Program for Sedentary Office Workers: A Randomized Controlled Trial

12th Northgate Congress on Preventive Medicine, June 2024 — Oral Abstract O-142

**Background:** Desk-based office workers accumulate little moderate-to-vigorous physical activity (MVPA). We tested whether a tailored text-message support program increases MVPA in sedentary employees.

**Methods:** We conducted a two-arm, parallel, individually randomized controlled trial. Sedentary adults aged 18–65 were recruited from four municipal employers and randomized 1:1 to a 6-month tailored text-message program promoting home-based MVPA (n=206) or to a healthy-activity leaflet (n=206). The program sent weekly personalized messages that set each participant's weekly MVPA goal and tracked progress toward it. Delivery was open-label: participants and program staff were aware of assignment. The primary outcome was self-reported MVPA (minutes/week, IPAQ-short) at 6 months, collected through an online questionnaire. Secondary outcomes were self-reported sitting time and program satisfaction. Follow-up was 89% overall, with attrition balanced across arms. No trial protocol or registration number is available for this trial.

**Results:** At 6 months, mean self-reported MVPA was 168 min/week in the intervention arm versus 126 min/week in the leaflet arm (adjusted mean difference +41 min/week; p=0.003). Program satisfaction was higher in the intervention arm.

**Conclusions:** A 6-month tailored text-message program significantly increases physical activity in sedentary office workers.

---

# Conference Abstract Appraisal — Three Questions

You are appraising the randomized controlled trial reported in the conference abstract below. No trial protocol, registry entry, or full report exists for this trial; the abstract is the only available documentation. Use only the information contained in the abstract. Judge from design information present in or absent from the abstract — not from the trial's results, conclusions, or plausibility.

The three questions ask you to (1) identify the risk-of-bias domain most directly implicated by how the primary outcome was measured, (2) select the domain-level judgment best supported by the abstract alone, and (3) identify the single additional datum that would most change that domain's judgment.

For reference, the five standard risk-of-bias domains for an individually randomized trial are: bias arising from the randomization process; bias due to deviations from the intended interventions; bias due to missing outcome data; bias in measurement of the outcome; and bias in selection of the reported result.

---

## Question 1

Considering only how the primary outcome was measured, which risk-of-bias domain is most directly implicated by this trial's outcome-measurement arrangement?

- **A.** Bias arising from the randomization process
- **B.** Bias due to deviations from the intended interventions
- **C.** Bias in measurement of the outcome
- **D.** Bias due to missing outcome data
- **E.** Bias in selection of the reported result

## Question 2

Based only on this abstract, which judgment for the domain you identified in Question 1 is best supported by the reported design information?

- **A.** Low risk, because the abstract reports no measurement problems
- **B.** Low risk, because the primary outcome difference was statistically significant
- **C.** Some concerns, because attrition was balanced and modest at 11%
- **D.** High risk, because participants were aware of their assignment and the primary outcome was self-reported on a subjective measure plausibly influenced by that awareness
- **E.** No judgment is possible, because the abstract does not state whether outcome assessors were blinded

## Question 3

Which single additional datum, if obtained, would most change your judgment for the domain identified in Question 1?

- **A.** Whether the 6-month questionnaire was administered by staff blinded to allocation
- **B.** Whether attrition differed by participants' baseline MVPA
- **C.** That the primary outcome was measured objectively (hip-worn accelerometer) rather than by self-report
- **D.** The exact 95% confidence interval for the between-group difference
- **E.** Whether the allocation sequence was concealed until assignment
- **F.** Whether both arms received equal frequency of contact with study staff

---

## Answer format

Output exactly three lines: `Q1: <letter>`, `Q2: <letter>`, `Q3: <letter>`, where each letter is the option you select for that question.

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
| `grade_answer.py` | verifier | — |
| `simple_verifier.py` | verifier | — |
| `verifier_config.json` | verifier | — |
| `answer_key.json` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `d25e72897bdb8b2f…`) |
| repair budget | `{"exhausted":false,"max":4,"used":0}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 4,641 |
| `manifest.json` | 1,423 |
| `renderings.json` | 136 |
| `specification.json` | 27.2 K |
| `task.toml` | 54 |

