# Task `synthetic/function/accessibility/openplan-hearing-light-conflict-slot5`

**Resolving a hearing-versus-light-sensitivity accommodation conflict in an open-plan office.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d18.function.accessibility-5-c679a4053244/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/function/accessibility/openplan-hearing-light-conflict-slot5` |
| `schema_version` | `0.9` |
| `difficulty` | 3 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `cd9701c8a78fd9413db5695a34c649038eb7f16d7765cd674a46d73b89b49260` |

`coverage_tags`: `competency:constraint_reasoning`, `context:provided_documents`, `shape:answer`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `0ce32038771d4fb0c66498f29604474d6766040e8917f894b31940a6a828149a` |
| `source.row` | `c679a40532444cd04a729b590c6bd02129b91a22c2cc79c54a241c200a3ddf27` |
| `source.importer_revision` | `cap-construct-003-hc3-r2-20260928T152631Z-87785` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d18.function.accessibility` — *Design person-centered accessibility accommodations* |
| subject | D18 — Nursing, Allied Health & Rehabilitation |
| proposal slot | 5 |
| task family | forced-choice accommodation conflict resolution (workplace sensory access) |

The capability's stated outcome: *"Analyze a supplied person's functional access needs against task, communication, service, and environmental demands, then propose feasible accommodations and a method for verifying effective access."*
Declared **excludes**: clinical treatment planning; device fitting as the central task; generic building design without a defined user and activity.

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

What follows is `instruction.md` in full, the complete solver-facing surface (11.0 KB, 123 lines).

---

# Workplace accommodation selection — Harbrook Insurance Services

You are the workplace accommodation coordinator at Harbrook Insurance Services. Jordan Osei, a claims-processing specialist on a 40-desk open-plan floor, has submitted an accommodation request. Jordan has bilateral moderate-severe hearing loss and wears hearing aids; phone and face-to-face communication are currently managed well with the hearing aids, but Jordan cannot reliably hear the fire-alarm horn over the open-plan ambient noise. Jordan also has migraine with aura, physician-documented as triggered by flashing light, including the building's existing wall strobes and the visibly flickering fluorescent troffers near Jordan's desk. The building runs monthly alarm-system tests and quarterly evacuation drills during which all notification appliances, including strobes, activate. A vendor has quoted five fixed accommodation bundles (no substitutions, partial selections, or mixing). Using the documents and constraints below, select the single bundle that resolves Jordan's fire-alarm alerting need and light-sensitivity restrictions without violating the fire-code constraint or the $2,400 all-in budget, and state that bundle's total cost.

---

## 1. Accommodation request memo

> **From:** Jordan Osei, Claims-Processing Specialist, desk 14 (employed since 2019)
> **To:** Workplace Accommodation Coordinator
> **Date:** June 2
>
> I'm submitting this request about my workstation on the open floor. My hearing aids handle phone calls and face-to-face conversation well, so I'm not asking about those. What I can't manage is the fire alarm: I cannot reliably hear the alarm horn over the open-plan ambient noise, and I need to know I'll be alerted during tests, drills, and a real event. Separately, the fluorescent troffers right above and next to my desk flicker visibly, and the wall strobes are a problem for me during the monthly tests and quarterly drills. Please review the vendor's quoted bundles and tell me which one to order.

## 2. Physician letter summary (on file)

- Diagnosis: bilateral moderate-severe sensorineural hearing loss; hearing-aid user.
- Diagnosis: migraine with aura.
- Triggers include strobe and flashing light, including peripheral exposure.
- Routine exposure to flashing light must be eliminated; rare emergency exposure is accepted.
- Jordan must not depend on strobes for alarm alerting.
- Visible fluorescent flicker at the workstation must be eliminated.

## 3. Facilities memo

> **From:** Building Facilities
> **Re:** Open-plan floor, desk 14
>
> - The floor holds 40 desks in one open plan.
> - The existing fire-alarm system has an audible horn plus 12 code-required wall strobes. The floor is divided into three alarm zones; desk 14 sits in Zone 2, and the four Zone 2 wall strobes are the only strobes visible from desk 14.
> - Monthly alarm-system tests and quarterly evacuation drills activate all notification appliances, including the strobes.
> - Eight fluorescent troffers near desk 14 have aging ballasts with visible flicker.
> - A quiet room is available for booking on the adjacent floor.

## 4. Constraints

1. **Budget:** $2,400 all-in for the selected bundle, labor included.
2. **Fire code** (summary from the building's licensed fire-safety contractor): existing alarm notification appliances may not be removed, disconnected, or disabled. Any change to the alarm system must use listed equipment installed by the licensed contractor, with fire-marshal sign-off.
3. **Medical restrictions:** as stated in the physician letter summary above.
4. **Fixed packages:** the vendor's bundles are fixed packages. No substitutions, no partial selections, and no mixing of items across bundles are offered.

## 5. Vendor quotes

The vendor has quoted five fixed bundles. Each bundle is itemized below; the vendor does not print bundle totals. Every price includes the labor described in its line item.

### Bundle A — Enhanced visual notification

This package strengthens visual alarm coverage at and around desk 14. The existing horn and all existing strobes stay in service; this package adds synchronized visual coverage on top of them.

- 3 additional listed synchronized wall strobes covering desk 14 and the evacuation route, placed on the wall sections nearest the desk, supplied, installed, and commissioned by the licensed contractor with fire-marshal sign-off, at $290 each.
- 8 LED retrofit fixtures for the flickering fluorescent troffers above and adjacent to desk 14, matched to the existing ceiling grid, at $80 per troffer.
- 8 removals of the old fixtures, including disposal and fitting of a new diffuser for each retrofit, at $15 per troffer.
- 1 adjustable task lamp with dimmable output and a flicker-free driver for desk 14, for use during detailed claim review, at $120.
- 1 acceptance documentation package at no charge, recording the commissioning checks and fire-marshal sign-off with facilities.
- 1 re-aim visit at no charge if a later coverage check shows a shadowed strobe, booked within one test cycle of the finding.

Pricing includes all labor, materials, and scheduling around the monthly test calendar. Equipment carries the manufacturer warranty.

### Bundle B — De-clutter and substitute

This package trims the alarm hardware around desk 14 and substitutes low-frequency sounders for the removed strobe coverage. The horn and the remaining strobes elsewhere on the floor stay in service.

- 4 listed low-frequency 520 Hz audible notification appliances, supplied, installed, and programmed by the licensed contractor, with panel functional testing of each appliance after installation, at $185 each.
- 8 LED retrofit fixtures for the flickering fluorescent troffers, matched to the existing ceiling grid, at $80 per troffer.
- 8 removals of the old fixtures, including disposal and fitting of a new diffuser for each retrofit, at $15 per troffer.
- 1 electrician line item: remove and decommission the 4 existing Zone 2 strobe heads, which clears the wall and ceiling area above desk 14, at $180 for the group.
- 1 panel programming verification record at no charge, entered in the alarm panel log after the functional testing.
- 1 patching and painting pass at no charge for the cleared wall sections above desk 14, matched to the existing paint.

Pricing includes all labor, materials, and scheduling around the monthly test calendar. Sounder equipment carries the manufacturer warranty.

### Bundle C — Layered non-visual alerting

This package builds alerting and environment controls that do not rely on sight. Alerting, lighting, and scheduling changes arrive as one package.

- 4 listed low-frequency 520 Hz audible notification appliances, supplied, installed, and programmed by the licensed contractor, with panel functional testing of each appliance after installation, at $185 each.
- 1 personal vibrotactile wrist receiver with a listed interface to the fire-alarm panel, at $310.
- LED retrofit of the 8 flickering fluorescent troffers, including removal and disposal of each old fixture, with new diffusers matched to the existing ceiling grid, at $95 per troffer.
- 1 adjustable task lamp with dimmable output and a flicker-free driver, at $120.
- 1 procedural item at no charge: 48-hour advance notice of every scheduled test and drill so Jordan can work from the quiet room during strobe activation.
- 1 acceptance test at no charge: the contractor activates the sounders alone in panel test mode, Jordan confirms audibility with hearing aids, and the pager is verified from the panel transmitter with fire-marshal sign-off.

Pricing includes all labor, materials, and scheduling around the monthly test calendar. The receiver carries the manufacturer warranty, and the vendor holds a spare on site.

### Bundle D — Private office relocation

This package moves Jordan's workstation into a newly built private office on the same floor. The office is positioned near the claims team, and the floor's desk count is unchanged.

- 4 listed low-frequency 520 Hz audible notification appliances, supplied, installed, and programmed by the licensed contractor, with panel functional testing of each appliance after installation, at $185 each.
- 1 personal vibrotactile wrist receiver with a listed interface to the fire-alarm panel, at $310.
- LED retrofit of the 8 flickering fluorescent troffers, including removal and disposal of each old fixture, with new diffusers matched to the existing ceiling grid, at $95 per troffer.
- 1 construction line item: build a single-occupancy office with a door and interior glazing positioned so strobe light from the open area does not reach the workspace, at $3200 for the build.
- 1 post-build inspection at no charge, covering light levels, door operation, alarm-appliance placement, and fire-marshal sign-off.
- 1 move-coordination meeting at no charge, arranged with facilities around the claims-team schedule, covering furniture, files, and IT handover.

Pricing includes all labor, materials, and scheduling around the monthly test calendar; the office build is quoted all-in. Equipment carries the manufacturer warranty.

### Bundle E — Lighting and communications refresh

This package upgrades the workstation lighting and Jordan's communication equipment. The work is scheduled in evening blocks so desk 14 stays usable during the day.

- 8 LED retrofit fixtures for the flickering fluorescent troffers above and adjacent to desk 14, matched to the existing ceiling grid, at $80 per troffer.
- 8 removals of the old fixtures, including disposal and fitting of a new diffuser for each retrofit, at $15 per troffer.
- 1 adjustable task lamp with dimmable output and a flicker-free driver for desk 14, for use during detailed claim review, at $120.
- 1 amplified office headset with t-coil compatibility for phone and dictation work, programmed to work with Jordan's hearing-aid t-coil setting, at $140.
- 1 headset fitting session at no charge, with the vendor's on-site audiologist, including a follow-up adjustment visit within 30 days.
- 1 haul-away and recycling visit at no charge for the old fixtures, packaging, and any leftover materials.

Pricing includes all labor, materials, and scheduling around the monthly test calendar. Communications equipment carries the manufacturer warranty, and the audiologist records each fitting adjustment in the vendor's file.

## 6. Output format

Return your answer in exactly this format, copying the labels exactly as shown, uppercase, with nothing else in your response:

```
ANSWER: X
COST: 1234
```

Line 1 is `ANSWER:` followed by the single uppercase letter of the bundle you select (A, B, C, D, or E). Line 2 is `COST:` followed by that bundle's total cost in dollars — digits only, no $ sign, no commas. The example above uses a dummy letter and a dummy amount to show the format only; `X` is not one of the offered bundles and 1234 is not a stated price. The two labels are case-sensitive: `answer:` or a lowercase bundle letter is not a valid response. Compute the bundle total yourself by summing its itemized line items; the vendor does not print totals. Do not include any other text, explanation, or blank lines beyond the two answer lines.

Return only the answer in the requested format.

---

## How it is graded

The verifier is private evaluator material and never reaches the solver.

| field | value |
| --- | --- |
| `verifier.kind` | `code_answer` |

Embedded resources (1):

| path | roles | lines |
| --- | --- | --- |
| `check_answer.py` | verifier | — |

---

## Pipeline verdicts

| check | result |
| --- | --- |
| item state | `quality_accepted` |
| runtime validated | `True` |
| repeated diagnostics | `ready` |
| quality review | `accept` (artifact sha256 `3e8218e8a960db2b…`) |
| repair budget | `{"exhausted":false,"max":4,"used":0}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 11.0 K |
| `manifest.json` | 1,424 |
| `renderings.json` | 134 |
| `specification.json` | 16.3 K |
| `task.toml` | 54 |

