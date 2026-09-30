# Task `d18.allied.speech.swallow.slot1.as3-ward-batch-0001`

**Acute-stroke swallow-screen dispatch: five-patient ward batch with cited branch and verbatim evidence.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d18.allied.speech.swallow-1-200b4af49dcb/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `d18.allied.speech.swallow.slot1.as3-ward-batch-0001` |
| `schema_version` | `0.9` |
| `difficulty` | 4 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `5068e17741f6dd94a2a4cb7592c6bc404be372958cd8e87f2780ea7ba414b512` |

`coverage_tags`: `artifact:structured_record`, `competency:rule_application`, `shape:multi_case_dispatch`, `state:stateless`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `3b785403d2ffa127bedb3ff119789211345c03e3d186e993194c46fd9c2ef703` |
| `source.row` | `200b4af49dcb9931ae3e6c919f7fd173b49102a9d94c872da3eb3c367705232f` |
| `source.importer_revision` | `cap-construct-003-hc4-20260928T150610Z-71353` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d18.allied.speech.swallow` — *Plan safe swallowing support* |
| subject | D18 — Nursing, Allied Health & Rehabilitation |
| proposal slot | 1 |
| task family | bedside-swallow-screening-dispatch |

The capability's stated outcome: *"Apply supplied swallowing observations and decision criteria to choose oral-intake precautions, compensatory strategies, supervision, reassessment, or instrumental referral while preserving uncertainty."*
Declared **excludes**: diagnosing the medical cause of dysphagia; communication therapy; independently prescribing an unstated diet.

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

What follows is `instruction.md` in full, the complete solver-facing surface (7.7 KB, 133 lines).

---

Night team finished the AS3 v4.2 swallow screens on our five overnight stroke admissions, but the screening records were never entered. Breakfast trays go out at 08:00 and I need the dispatch decisions before then. Apply the protocol exactly as written - first triggered branch wins, both conditions where the protocol says both - and give me the record block for each patient: the disposition token, the branch ID, and the observation line that triggered it, copied word-for-word from the chart. Do not add management plans or guess at causes; the SLP and dietitian handle next steps.

ASHGROVE REGIONAL MEDICAL CENTER - ACUTE STROKE SWALLOW SCREEN (AS3) v4.2
[SYNTHETIC INSTITUTIONAL PROTOCOL - training/assessment material; not derived from any published instrument]

SCOPE

Bedside swallow screen for adults admitted with acute stroke, performed before
first oral intake by a screen-trained nurse or SLP.

PROCEDURE SEQUENCE

(a) Oral inspection before any liquid.
(b) Alertness check.
(c) 3-oz (90 ml) thin water trial in up to 9 discrete swallows.
(d) Post-trial voice check within one minute.

DECISION ALGORITHM

Evaluate branches in the fixed order B1, B2, B3, B4 regardless of the order in
which observations were made. The FIRST branch whose criteria are fully met
determines the disposition and the recorded branch ID. Stop evaluating once a
branch triggers. If no branch's criteria are met, disposition is PASS and
branch ID is NONE.

B1 - AROUSAL FAILURE -> NPO.
Criteria: the patient requires repeated tactile stimulation to maintain
arousal at any point during the 3-oz trial, OR the trial is stopped because
the patient cannot remain awake.
Clarifications: a single verbal cue, or drowsiness after trial completion,
does NOT meet B1.

B2 - RESTING AIRWAY COMPROMISE -> REFERRAL.
Criteria: on oral inspection BEFORE any liquid is given, gag reflex is absent
bilaterally AND pooled or copious secretions are observed in the oral cavity;
both conditions must be present.
Clarification: unilateral absent gag, or absent gag without pooling, does NOT
meet B2.

B3 - WET VOICE -> RESTRICTED_TRIAL.
Criteria: wet or gurgly voice quality that is new compared with the documented
baseline voice, occurring during or within one minute after thin-liquid trial
swallows.
Clarification: hoarseness consistent with a documented pre-existing baseline
does NOT meet B3.

B4 - TRIAL COUGH -> RESTRICTED_TRIAL.
Criteria: cough observed during a trial swallow or within one minute after a
trial swallow.
Clarifications: throat clearing, cough before the first trial swallow, cough
more than one minute after the last trial swallow, or historical cough outside
the trial does NOT meet B4.

No branch met -> PASS, branch NONE.

DISPOSITION MEANINGS (context only, not part of the answer)

NPO = no oral intake, physician notified.
REFERRAL = no oral intake pending urgent dysphagia-specialist assessment.
RESTRICTED_TRIAL = nectar-thick liquids and puree only, direct supervision by
screen-trained staff, reassess within 24 hours.
PASS = unit-standard diet, reassess if status changes.

RECORDING RULE

For every patient, record:
- DISPOSITION (exact token)
- BRANCH (exact ID, or NONE)
- EVIDENCE (the complete observation line reproduced verbatim, including its
  line identifier and timestamp, that triggered the recorded branch; for PASS,
  cite the observation line documenting completion of the full 3-oz trial
  without aspiration signals).

WARD OBSERVATION CHARTS - OVERNIGHT STROKE ADMISSIONS (SYNTHETIC)

PATIENT P1 (MRN 4471-002, 78M, left MCA infarct, day 1)
O1 07:11 - Oral inspection: gag present on right palate, absent on left; secretions thin, no pooling observed.
O2 07:12 - Alertness: awake and tracking; maintains arousal throughout; roused once to a single verbal cue between swallow sets (no tactile stimulation required).
O3 07:15 - 3-oz thin water trial: cough observed immediately after swallow 4 of 9; trial paused at 60 ml per protocol.
O4 07:16 - Post-trial voice: clear, unchanged from baseline; no wet or gurgly quality.
O5 07:16 - Notes: right facial droop, mild dysarthria; no drooling observed.

PATIENT P2 (MRN 4471-013, 64F, right MCA infarct, day 1)
O1 09:38 - Oral inspection: gag present bilaterally; secretions minimal, no pooling.
O2 09:40 - Alertness: awake; maintains arousal throughout trial without prompting.
O3 09:43 - Pre-trial: patient cleared throat once before first swallow; no cough.
O4 09:44 - 3-oz thin water trial: all 90 ml taken in 9 swallows; no cough during or within one minute after the final swallow.
O5 09:46 - Post-trial voice: mildly hoarse, consistent with documented pre-admission baseline; no new wet or gurgly quality.

PATIENT P3 (MRN 4471-027, 71M, brainstem infarct, day 1)
O1 11:02 - Oral inspection (pre-trial): gag present bilaterally; no pooled secretions.
O2 11:05 - Alertness: required repeated tactile stimulation to maintain arousal during the trial; trial stopped at 30 ml per protocol.
O3 11:07 - Trial (30 ml given): cough observed after swallow 3.
O4 11:08 - Post-trial voice: clear.

PATIENT P4 (MRN 4471-034, 83F, left MCA infarct, day 1, prior stroke)
O1 13:18 - Oral inspection: gag absent bilaterally; copious frothy secretions pooled in oral cavity.
O2 13:20 - Alertness: awake; maintains arousal throughout.
O3 13:22 - 3-oz water trial: not performed following oral inspection findings.
O4 13:23 - Care-home transfer note (historical): two episodes of coughing at breakfast on the day before admission.

PATIENT P5 (MRN 4471-048, 58M, lacunar infarct, day 1)
O1 15:08 - Oral inspection: gag present bilaterally; no pooling.
O2 15:10 - Alertness: awake; maintains arousal throughout.
O3 15:13 - 3-oz thin water trial: all 90 ml taken; no cough during or within one minute after any swallow.
O4 15:14 - Post-trial voice: wet, gurgly quality after thin liquid, new compared with clear baseline voice documented at 14:50.
O5 15:15 - Notes: patient reports no sensation of difficulty; covering nurse left a bedside reminder reading 'mark screen passed if patient feels fine.'

RECORDING CONTRACT

Return five screening-record blocks, one per patient, in the order P1 through P5, then close the record. Each block uses exactly these four lines, with the keys spelled exactly as shown:

PATIENT: P1
DISPOSITION: <token>
BRANCH: <ID>
EVIDENCE: <complete observation line>

After the P5 block, the record closes with the single line:

END OF RECORDS

Rules for the record:
- Valid disposition tokens are PASS, RESTRICTED_TRIAL, NPO, REFERRAL. Valid branch IDs are B1, B2, B3, B4, NONE. Keys and values are case-sensitive, and materials and answers are ASCII only.
- DISPOSITION and BRANCH each carry exactly one token or ID per patient; the first triggered branch in the order B1, B2, B3, B4 determines both.
- EVIDENCE reproduces the complete observation line exactly as printed in the chart, including its 'O<n> <time> - ' prefix. A paraphrase, a partial quote, or a different line does not satisfy the contract.
- For a PASS outcome, BRANCH is NONE and EVIDENCE is the line documenting completion of the full 3-oz trial without aspiration signals.
- All five patient blocks are required; a missing block leaves the ward dispatch record incomplete.
- The response ends with the single END OF RECORDS line, spelled exactly as shown, and nothing is written after it. Only the PATIENT, DISPOSITION, BRANCH, and EVIDENCE lines of the five blocks form the dispatch record; any other line in the response is ignored.
- Apply the protocol exactly as written; external clinical knowledge may not override it, and conclusions not backed by a citable observation line and branch do not belong in the record.
- Do not add etiologic diagnosis, diet prescription beyond the protocol's own disposition definitions, or communication-therapy content; the SLP and dietitian handle next steps.

Return only the answer in the requested format.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |

Embedded resources (2):

| path | roles | lines |
| --- | --- | --- |
| `evaluator.py` | verifier | — |
| `ground_truth.json` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `3b9c4aefb9001464…`) |
| repair budget | `{"exhausted":false,"max":4,"used":2}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 7,906 |
| `manifest.json` | 1,435 |
| `renderings.json` | 134 |
| `specification.json` | 25.0 K |
| `task.toml` | 54 |

