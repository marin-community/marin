# Task `capability/d19.environment-community.preparedness/slot-4-bexley-protective-action-package-v1`

**Chemical Release Protective-Action Decisions.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d19.environment-community.preparedness-4-a249aa47e3df/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `capability/d19.environment-community.preparedness/slot-4-bexley-protective-action-package-v1` |
| `schema_version` | `0.9` |
| `difficulty` | 6 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `101721f3c8b05ec55bf6fdf38907f8d78e2c9e6083b67263d1bf20d60815e8f8` |

`coverage_tags`: `artifact:application/json`, `competency:rule_application`, `context:provided_documents`, `shape:constrained_generation`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `9233810c1e8e1d80052a94b634eebddad46d9cd019cdfa43ab9657edb45d9896` |
| `source.row` | `a249aa47e3dff7a33c3c5e8aeec9415c227d604ea316a02b44603c0793273ed3` |
| `source.importer_revision` | `taskcompendium-dc6b501c8604bcd2e3c20c1e9947679845fdfef8` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d19.environment-community.preparedness` — *Plan for Public-Health Emergencies* |
| subject | D19 — Public Health, Epidemiology & Health Systems |
| proposal slot | 4 |
| task family | public-health-emergency-protective-actions |

The capability's stated outcome: *"Build and test population-health preparedness plans connecting hazards and vulnerabilities to triggers, roles, resources, continuity, and recovery actions."*
Declared **excludes**: Routine disease control; Hospital clinical incident command; Long-term policy without an emergency.

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
| rendering | `{"id":"capability/d19.environment-community.preparedness/slot-4-bexley-protective-action-package-v1/assistant-final","instruction_surface":"original","submission":{"extractor":{"kind":"plain"},"kind":"assistant_final"},"version":"0.2"}` |
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

What follows is `instruction.md` in full, the complete solver-facing surface (5.7 KB, 64 lines).

---

# Chemical Release Protective-Action Decision Package — Bexley County

## Situation

You are the on-call public-health preparedness officer for Bexley County. At 06:10 today a freight tanker car derails at the Bexley County rail yard and begins releasing a synthetic irritant cloud. The county's regional hazard model (RHM-9) has produced its plume characterization, and the county facility registry is current as of this morning. County doctrine requires your protective-action recommendation to the county emergency manager within 30 minutes.

This is a fully synthetic exercise. Every zone, time, population, and facility below is a fictional fixture authored for this drill, and the model output is authoritative: use ONLY the county decision rules quoted verbatim below. Do not substitute outside protective-action doctrine, regulation, or any real agency's or real standard's guidance; none is named or implied by this scenario.

## RHM-9 plume table

| Zone | Downwind corridor | In plume footprint | Plume passage time (min) | General evacuation route time (min) | Resident population |
|------|-------------------|--------------------|--------------------------|------------------------------------|---------------------|
| A    | 0–2 km            | yes                | 30                       | 20                                 | 5,200               |
| B    | 2–5 km            | yes                | 70                       | 25                                 | 9,800               |
| C    | 5–9 km            | yes                | 50                       | 95                                 | 4,100               |
| D    | crosswind/upwind corridor | no         | n/a                      | n/a                                | 6,300               |

Plume passage time = modeled minutes for the cloud to pass over the zone. General evacuation route time = modeled minutes for the zone's general population to clear the zone via its assigned evacuation routes.

## Facility registry

| Facility | Zone | On-site population | Ambulatory status | Facility evacuation/mobilization time (min) |
|----------|------|--------------------|-------------------|----------------------------------------------|
| Hillview Care Center (skilled nursing) | A | 112 | non-ambulatory | 120 |
| Elmsworth Elementary School | B | 460 | ambulatory | 45 |
| Larkfield Dialysis Clinic | B | 38 | ambulatory | 70 |
| Ashgrove Distribution Warehouse | C | 85 | ambulatory | 40 |

No facilities are listed in zone D. Facility evacuation/mobilization time = modeled minutes for that facility's full on-site population to mobilize for either evacuation or sheltering, as applicable under the rules below. The parenthetical "(skilled nursing)" after Hillview Care Center is a care-level descriptor, not part of the facility name: use "Hillview Care Center" as the facility name in your output.

## County decision rules (verbatim; apply exactly as written)

R1 (zone): if the zone is outside the plume footprint, decision is monitor with reason code MONITOR_UPWIND; if the zone is in the footprint and the general evacuation route time is strictly less than the plume passage time, decision is evacuate with reason code EVAC_DOWNWIND_FAST; otherwise decision is shelter_in_place with reason code SHELTER_ROUTE_TOO_SLOW.

R2 (facility): begin with the zone decision. If the zone decision is evacuate: first check the ambulatory flag - if the facility is flagged non-ambulatory, override to shelter_in_place with reason code FAC_OVERRIDE_NON_AMBULATORY (this check takes precedence over the time check when both apply); otherwise, if facility mobilization time is greater than or equal to the zone's plume passage time (equality means shelter), override to shelter_in_place with reason code FAC_OVERRIDE_MOBILIZATION_TIME; otherwise the facility evacuates with reason code FAC_FOLLOWS_ZONE. If the zone decision is shelter_in_place or monitor, the facility follows the zone decision with reason code FAC_FOLLOWS_ZONE; facilities never override upward from a sheltering or monitoring zone to evacuate.

## Required output

Return one JSON object (not an array) with exactly two top-level keys, both case-sensitive: `zone_decisions` and `facility_decisions`.

`zone_decisions` is an array of exactly four objects, one per zone A, B, C, and D, each zone appearing exactly once, each object with these keys:

- `zone`: the zone letter, "A", "B", "C", or "D"
- `decision`: one of `evacuate`, `shelter_in_place`, `monitor` (lowercase, case-sensitive)
- `reason_code`: for zone entries, one of `MONITOR_UPWIND`, `EVAC_DOWNWIND_FAST`, `SHELTER_ROUTE_TOO_SLOW` (uppercase, case-sensitive)
- `population_affected`: the zone's resident population from the plume table, as a plain integer (no thousands separators, no strings)

`facility_decisions` is an array of exactly four objects, one per facility listed in the registry, each facility named exactly as in the registry and appearing exactly once, each object with these keys:

- `facility`: the exact facility name from the registry
- `zone`: the facility's zone letter
- `decision`: one of `evacuate`, `shelter_in_place`, `monitor` (lowercase, case-sensitive)
- `reason_code`: for facility entries, one of `FAC_OVERRIDE_NON_AMBULATORY`, `FAC_OVERRIDE_MOBILIZATION_TIME`, `FAC_FOLLOWS_ZONE` (uppercase, case-sensitive)

No extra, renamed, or missing entries are permitted. Key names and enum values are case-sensitive. Shape skeleton (placeholder dots, not values):

```json
{"zone_decisions": [{"zone": "...", "decision": "...", "reason_code": "...", "population_affected": 0}], "facility_decisions": [{"facility": "...", "zone": "...", "decision": "...", "reason_code": "..."}]}
```

Produce the decision package now using only the rules and tables above.

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
| `composition_config.json` | verifier | — |
| `composition_prepass.py` | verifier | — |
| `pad_checker.py` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `feb3d242b33f5bad…`) |
| repair budget | `{"exhausted":false,"max":4,"used":0}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 5,853 |
| `manifest.json` | 1,580 |
| `renderings.json` | 237 |
| `specification.json` | 46.7 K |
| `task.toml` | 54 |

