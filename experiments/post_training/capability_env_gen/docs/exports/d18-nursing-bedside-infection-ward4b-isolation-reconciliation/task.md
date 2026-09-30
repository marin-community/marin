# Task `synthetic/d18-nursing-bedside-infection/ward4b-isolation-reconciliation`

**Correct stale isolation flags in a simulated EHR export.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d18.nursing.bedside.infection-5-5af9f75c6578/` (13 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/d18-nursing-bedside-infection/ward4b-isolation-reconciliation` |
| `schema_version` | `0.9` |
| `difficulty` | 6 |
| `success_policy` | `final` |
| `metadata.task_shape` | `shell_workflow` |
| steps | 1 |
| `specification_sha256` | `c02e01f31c0dafcf28b3e8ecb787f2674ae5334d4e65ac57f17bfe78cd5a7aff` |

`coverage_tags`: `artifact:text/plain`, `competency:information_extraction`, `competency:rule_application`, `context:provided_documents`, `interaction:multi_action_workflow`, `shape:shell_workflow`, `state:workspace`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `24409315697a1ac83c3836c25bc36232a3ffc86ef9dd21cd488b7450c06b455d` |
| `source.row` | `5af9f75c6578df94f8675009af0eec3ca098f0635a9b0729dbdd7d925dc0ab63` |
| `source.importer_revision` | `dc6b501c8604bcd2e3c20c1e9947679845fdfef8` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d18.nursing.bedside.infection` — *Apply infection-prevention precautions* |
| subject | D18 — Nursing, Allied Health & Rehabilitation |
| proposal slot | 5 |
| task family | infection-prevention EHR reconciliation on an in-memory ward filesystem |

The capability's stated outcome: *"Use supplied exposure facts, organism or syndrome information, task details, and infection-control rules to select and sequence precautions, identify breaches, and limit transmission."*
Declared **excludes**: treating an established infection; generic workplace cleaning; noninfectious fall or pressure-injury prevention.

---

## Environment and interface

| field | value |
| --- | --- |
| proposed environment | `shellsim` |
| `requirements.state.image` | `null` |
| `requirements.state.workdir` | `/ward` |
| `requirements.state.setup_commands` | `[]` |
| `requirements.capabilities` | `["filesystem","shell"]` |
| `requirements.action_interfaces` | `[]` |
| `binding.json` | `{"environment":{"additional_directories":[],"kind":"shellsim","max_output_bytes":4194304,"max_steps":1000000,"setup_commands":[],"workdir":"/ward"},"tools":[{"backend":"shellsim","kind":"shell","name":"shell"}]}` |
| rendering | `{"id":"ward","instruction_surface":"original","submission":{"excluded_paths":[],"kind":"final_state","paths":["."]},"version":"0.2"}` |
| step 0 `context_requirement` | `instruction_and_workspace` |
| step 0 `answer_requirements` | `{"kind":"final_state"}` |

`task.toml`:

```toml
version = "1.0"

[environment]
allow_internet = false
workdir = "/ward"
```

Solver-visible input files shipped with the export:

| file | bytes |
| --- | --- |
| `environment/inputs/README.txt` | 6,685 |
| `environment/inputs/census.txt` | 489 |
| `environment/inputs/isolation_orders.txt` | 244 |
| `environment/inputs/isolation_policy.txt` | 1,279 |
| `environment/inputs/lab_results.txt` | 964 |
| `environment/inputs/nursing_notes.txt` | 282 |
| `environment/inputs/rooms.txt` | 159 |

---

## The prompt

What follows is `instruction.md` in full, the complete solver-facing surface (1.5 KB, 3 lines).

---

Overnight infection-prevention review, Ward 4B. Our EHR export has drifted from this morning's results: some isolation orders are missing required upgrades and some are carrying stale precautions that no longer apply. Cross-reference the census, isolation orders, room assignments, lab results, and nursing notes in /ward/ against the unit isolation policy (/ward/isolation_policy.txt). Update /ward/isolation_orders.txt so every precaution flag and every required room type matches current policy, and update /ward/rooms.txt so every bed placement is consistent with the corrected orders (airborne patients in negative pressure, contact and contact-enteric patients in private rooms, no double occupancy). Leave /ward/changes.log documenting each change you made in the exact format specified in /ward/README.txt, citing the policy rule that justifies it. Do not modify the source-of-truth files (census, labs, notes, policy); tampering with evidence is a reportable offense and voids the review.

You are working in a simulated POSIX shell with a persistent in-memory filesystem. The available utilities are cat, grep, sed, awk, sort, cut, head, tail, wc, echo, printf, tee, ls, mv, cp, touch, rm, and mkdir; there is no network access and no scripting runtime beyond the shell. /ward/README.txt documents the exact file schemas, the changes.log line format, the case-sensitive token contract, the immutable-file list, and shell behaviors confirmed on this system - read it before editing.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `tasktrove` |
| `verifier.mode` | `script` |
| `parameters.path` | `"verify.py"` |
| `parameters.timeout` | `60.0` |
| `parameters.args` | `[]` |
| runtime | container `docker.io/library/python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`, workspace `{"kind":"empty"}`, timeout 120.0 s |
| implementation revision | `b76d03131cd88bd9fc711dba206659027edba3a8` |

Embedded resources (19):

| path | roles | lines |
| --- | --- | --- |
| `README.txt` | agent | — |
| `census.txt` | agent | — |
| `isolation_orders.txt` | agent | — |
| `rooms.txt` | agent | — |
| `lab_results.txt` | agent | — |
| `nursing_notes.txt` | agent | — |
| `isolation_policy.txt` | agent | — |
| `verify.py` | verifier | — |
| `composed_verifier.py` | verifier | — |
| `check_ward.py` | verifier | — |
| `extension_config.json` | verifier | — |
| `ground_truth.json` | verifier | — |
| `fixtures/ward_initial/README.txt` | verifier | — |
| `fixtures/ward_initial/census.txt` | verifier | — |
| `fixtures/ward_initial/isolation_orders.txt` | verifier | — |
| `fixtures/ward_initial/rooms.txt` | verifier | — |
| `fixtures/ward_initial/lab_results.txt` | verifier | — |
| `fixtures/ward_initial/nursing_notes.txt` | verifier | — |
| `fixtures/ward_initial/isolation_policy.txt` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `6846aa92bb234803…`) |
| repair budget | `{"exhausted":false,"max":4,"used":3}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 211 |
| `environment/inputs/README.txt` | 6,685 |
| `environment/inputs/census.txt` | 489 |
| `environment/inputs/isolation_orders.txt` | 244 |
| `environment/inputs/isolation_policy.txt` | 1,279 |
| `environment/inputs/lab_results.txt` | 964 |
| `environment/inputs/nursing_notes.txt` | 282 |
| `environment/inputs/rooms.txt` | 159 |
| `instruction.md` | 1,491 |
| `manifest.json` | 1,820 |
| `renderings.json` | 134 |
| `specification.json` | 104.1 K |
| `task.toml` | 72 |

