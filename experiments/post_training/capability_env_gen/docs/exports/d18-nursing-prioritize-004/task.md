# Task `synthetic/d18-nursing-prioritize/004`

**Triage a messy monitor export.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d18.nursing.prioritize-4-f33dd13f3363/` (9 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/d18-nursing-prioritize/004` |
| `schema_version` | `0.9` |
| `difficulty` | 6 |
| `success_policy` | `final` |
| `metadata.task_shape` | `structured_extraction` |
| steps | 1 |
| `specification_sha256` | `c4c9c90c98c9275f0fa9fbc5573951fc39376ab03c29a9ddd149d845297cc7ef` |

`coverage_tags`: `artifact:workspace_state`, `competency:rule_application`, `context:provided_documents`, `context:text/csv`, `interaction:terminal`, `shape:structured_extraction`, `state:workspace`, `subject:medicine.nursing`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `0a972219b5dad6e037c7d98f72a72680a1a6fbdf09b78808ad05424985cdb388` |
| `source.row` | `f33dd13f336300234e49feff383f9b6107b8dda8b1693cc709f8add3238b90c0` |
| `source.importer_revision` | `cap-construct-003-hc3-d1-20260928T160537Z-26099` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d18.nursing.prioritize` — *Recognize deterioration and escalate* |
| subject | D18 — Nursing, Allied Health & Rehabilitation |
| proposal slot | 4 |
| task family | single-patient monitor-export deterioration triage |

The capability's stated outcome: *"Compare one patient's timestamped observations with supplied deterioration criteria to identify the first actionable change, select the immediate protocol-supported response, and communicate the evidence and escalation requirement."*
Declared **excludes**: multi-patient ranking or shift scheduling; creating a complete care plan; escalating from unstated criteria.

---

## Environment and interface

| field | value |
| --- | --- |
| proposed environment | `shellsim` |
| `requirements.state.image` | `null` |
| `requirements.state.workdir` | `/home/nurse` |
| `requirements.state.setup_commands` | `[]` |
| `requirements.capabilities` | `["filesystem","shell"]` |
| `requirements.action_interfaces` | `[]` |
| `binding.json` | `{"environment":{"additional_directories":[],"kind":"shellsim","max_output_bytes":4194304,"max_steps":1000000,"setup_commands":[],"workdir":"/home/nurse"},"tools":[{"backend":"shellsim","kind":"shell","name":"shell"}]}` |
| rendering | `{"id":"workspace","instruction_surface":"original","submission":{"excluded_paths":[],"kind":"final_state","paths":["decision.txt"]},"version":"0.2"}` |
| step 0 `context_requirement` | `instruction_and_workspace` |
| step 0 `answer_requirements` | `{"kind":"final_state"}` |

`task.toml`:

```toml
version = "1.0"

[environment]
allow_internet = false
workdir = "/home/nurse"
```

Solver-visible input files shipped with the export:

| file | bytes |
| --- | --- |
| `environment/inputs/audit_format.txt` | 1,638 |
| `environment/inputs/monitor_export.csv` | 568 |
| `environment/inputs/ward_protocol.txt` | 2,754 |

---

## The prompt

What follows is `instruction.md` in full, the complete solver-facing surface (0.7 KB, 1 lines).

---

Before handover, triage the overnight monitor export for bed 7 (postoperative day 1, abdominal surgery, single patient). The export is at /home/nurse/monitor_export.csv. Apply the ward protocol at /home/nurse/ward_protocol.txt — including its monitor-export handling rules — and write the audit decision record to /home/nurse/decision.txt in the exact format specified in /home/nurse/audit_format.txt. The record must state the first time any escalation criterion was met, every criterion met at that time, the parsed observation set at that moment, the notification deadline implied by the protocol, and the required actions. Exports from this monitor are known to be messy; follow the documented handling rules rather than assuming the file is clean.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `tasktrove` |
| `verifier.mode` | `script` |
| `parameters.path` | `"grade_decision.py"` |
| `parameters.timeout` | `60.0` |
| `parameters.args` | `[]` |
| runtime | container `python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`, workspace `{"kind":"empty"}`, timeout 120.0 s |
| implementation revision | `b76d03131cd88bd9fc711dba206659027edba3a8` |

Embedded resources (3):

| path | roles | lines |
| --- | --- | --- |
| `audit_format.txt` | agent | — |
| `monitor_export.csv` | agent | — |
| `ward_protocol.txt` | agent | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `7abc7cfe964b848b…`) |
| repair budget | `{"exhausted":false,"max":4,"used":0}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 217 |
| `environment/inputs/audit_format.txt` | 1,638 |
| `environment/inputs/monitor_export.csv` | 568 |
| `environment/inputs/ward_protocol.txt` | 2,754 |
| `instruction.md` | 756 |
| `manifest.json` | 1,821 |
| `renderings.json` | 150 |
| `specification.json` | 51.1 K |
| `task.toml` | 78 |

