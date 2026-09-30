# Task `synthetic/d18-allied-respiratory/oxygen-titration-ed-0001`

**Next permitted oxygen setting under a stepwise emergency-department protocol.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d18.allied.respiratory-1-fca5becb195f/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/d18-allied-respiratory/oxygen-titration-ed-0001` |
| `schema_version` | `0.9` |
| `difficulty` | 4 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `79ac54f636699b211c53ff8eb295a02907d87e6c6d412e0cae151a2966c39df0` |

`coverage_tags`: `competency:rule_application`, `context:provided_documents`, `shape:constrained_generation`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `42226dbf757507649e43183d7c8d4688b6c88de749eb704331da8e8a1beec66d` |
| `source.row` | `fca5becb195fc4f11300abcbca577a9ef1e49181356de0d907435b8501d37500` |
| `source.importer_revision` | `taskcompendium-dc6b501c8604bcd2e3c20c1e9947679845fdfef8` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d18.allied.respiratory` — *Plan and monitor respiratory therapy interventions* |
| subject | D18 — Nursing, Allied Health & Rehabilitation |
| proposal slot | 1 |
| task family | oxygen-titration-under-protocol |

The capability's stated outcome: *"Apply supplied respiratory measurements, orders, device data, and protocols to select or adjust a respiratory intervention and define delivery, monitoring, failure, and escalation criteria."*
Declared **excludes**: diagnosing cardiopulmonary disease; changing an order outside the supplied protocol; general nursing assessment without a respiratory intervention.

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

What follows is `instruction.md` in full, the complete solver-facing surface (6.0 KB, 82 lines).

---

# ED oxygen titration: next permitted setting

**Synthetic-content label.** Everything in this task — the patient, the order, the nursing record, the device data, the protocol, and the worked example — is authored synthetic content created for this exercise. It is not sourced clinical guidance, and no outside clinical knowledge is needed or credited.

You are the respiratory therapist covering the ED. Your patient (synthetic: 68-year-old, admitted with pneumonia — the diagnosis is already made and is not part of your task) is on nasal cannula oxygen under the order below. The unit's ED Oxygen Titration Protocol and the nursing record for the last 40 minutes are supplied. Using only the supplied protocol, order, trend, and device limits, state the next permitted action.

## Order sheet (synthetic)

"Oxygen via nasal cannula, titrate to SpO2 94–98%. Do not change device or order; select settings within it."

## ED Oxygen Titration Protocol v4

(Authored for this exercise: synthetic, self-contained rules. Follow them exactly as written; they are not sourced clinical guidance, and no outside clinical knowledge is needed or credited.)

**How to read the nursing record.** Each entry shows the time, the SpO2 reading, and any action taken. The first entry records the start of oxygen therapy: the flow is set first, and that entry's reading is taken on the started flow. Starting therapy is the initial setting, not a flow change. In every later entry, the reading is taken first, and any flow change listed in that entry (a flow increase or a device switch) was made after that reading. A *flow change* is any increase or adjustment of an already-running flow, including a device switch; a flow change is *between* two readings when it was made after the earlier of the two readings and before the later one.

**Terms.** A *step increase* is a single move to the next step on the device ladder — a flow increase on the current device or a switch to the next device (for example, nasal cannula 6 L/min → simple face mask 8 L/min). Starting oxygen therapy at the ordered initial setting is not a step increase.

**Rules.**

1. Titrate only to the ordered target range; never change the order. The order fixes the target range and the starting device; the only permitted changes to the patient's oxygen are the titration steps defined in rules 2–5. A move up the device ladder under rule 2 is a titration step within the order — not a change of device or order.
2. Device ladder: nasal cannula 1–6 L/min in 1 L/min steps; simple face mask 6–10 L/min in 2 L/min steps, started at 8 L/min when stepping up from a nasal cannula already at 6 L/min; nonrebreather mask at 15 L/min only.
3. Increase trigger: if SpO2 is below the ordered target range on two consecutive readings that are at least 10 minutes apart and were taken with no flow change between them, increase by one step on the device ladder (rule 2).
4. Never increase more than one step per reassessment, and never exceed the current device's permitted range.
5. After every flow change, reassess SpO2 15 minutes after the change.
6. Escalation threshold: notify the ED provider immediately if SpO2 is 88% or less at any reading, or if SpO2 is still below the target range at the first reassessment after two step increases have been given.
7. If SpO2 is within the target range, make no change and recheck within 30 minutes.
8. If any escalation criterion in rule 6 is met, notify the ED provider before making any further change; escalation takes precedence over every titration step, including a device switch.

## Nursing/therapy record, last 40 minutes (synthetic)

- 13:30 — Nasal cannula started at 3 L/min; SpO2 92%.
- 13:45 — SpO2 92%; flow increased to 4 L/min per protocol (one step).
- 14:00 — SpO2 92% (reassessment after the change).
- 14:10 — SpO2 92% (current reading; time now).

Current device and flow: nasal cannula at 4 L/min.

**Device data.** Current device: nasal cannula. Permitted range: 1–6 L/min. Protocol step size on this device: 1 L/min.

## Answer template (exact output contract)

Respond with EXACTLY three lines and nothing else, in this template:

```
NEXT SETTING: <device> at <flow> L/min
REASSESS IN: <n> minutes
ESCALATE IF: SpO2 88% or less at any reading, or SpO2 below <target lower bound>% after two step increases
```

Do not add preamble, a fourth line, or any units other than L/min and %. Each of the three lines must copy its label (`NEXT SETTING:`, `REASSESS IN:`, `ESCALATE IF:`) exactly as shown, followed by the requested content. Plain text only: no markdown formatting, no bold, no bullet points, no code fences, no explanation.

## Worked example (a DIFFERENT patient — study the method; do not copy its numbers)

**Example order (synthetic).** "Oxygen via nasal cannula, titrate to SpO2 92–96%. Do not change device or order; select settings within it."

**Example nursing record (synthetic).**

- 09:00 — Nasal cannula started at 2 L/min; SpO2 90%.
- 09:10 — SpO2 90% (current reading; time now).

**Example device data.** Current device: nasal cannula. Permitted range: 1–6 L/min. Protocol step size on this device: 1 L/min.

**How this example is solved, using only the protocol:**

1. Escalation check (rules 6 and 8): no reading is 88% or less, and no step increase has been given, so no escalation criterion is met.
2. The current reading (90%) is below this patient's target range (92–96%), so rule 7 (in range → no change) does not apply.
3. Increase trigger (rule 3): 09:00 and 09:10 are two consecutive readings, exactly 10 minutes apart, with no flow change between them, and both are below 92% → the trigger is met, so increase by one step.
4. One step (rule 4): 2 → 3 L/min on the nasal cannula; never two steps at once.
5. Reassessment (rule 5): 15 minutes after the change.
6. Escalation line (rule 6): substitute THIS patient's target lower bound, 92.

**Example answer:**

```
NEXT SETTING: nasal cannula at 3 L/min
REASSESS IN: 15 minutes
ESCALATE IF: SpO2 88% or less at any reading, or SpO2 below 92% after two step increases
```

Return only the answer in the requested format.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `tasktrove` |
| `verifier.mode` | `exact` |
| implementation revision | `b76d03131cd88bd9fc711dba206659027edba3a8` |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `60008b815779f095…`) |
| repair budget | `{"exhausted":false,"max":4,"used":0}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 6,173 |
| `manifest.json` | 1,112 |
| `renderings.json` | 134 |
| `specification.json` | 7,465 |
| `task.toml` | 54 |

