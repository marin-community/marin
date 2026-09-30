# Task `synthetic/capability/d18-nursing-medication-verify-administer-1-cf693ef9ad43`

**Hold-parameter gate on an oral liquid with an erroneous pharmacy-verified volume.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d18.nursing.medication.verify_administer-1-cf693ef9ad43/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/capability/d18-nursing-medication-verify-administer-1-cf693ef9ad43` |
| `schema_version` | `0.9` |
| `difficulty` | 4 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `e76d5708d7cf0a82c50cd11d6c7b1ed250cabdf49348232e9a2532d03806e055` |

`coverage_tags`: `artifact:json`, `competency:rule_application`, `context:provided_documents`, `shape:constrained_generation`, `subject:medicine.nursing`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `7292f2f7dbd17b76cac8f16fb524651dd12522d8e5b485a22c945e3c75af4662` |
| `source.row` | `cf693ef9ad43ec37b757faf2b280060583c2529974ca42bfa5fa1e3c6d3c2534` |
| `source.importer_revision` | `cap-construct-003-hc2-d1-20260928T160526Z-25754` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d18.nursing.medication.verify_administer` — *Verify and plan safe medication administration* |
| subject | D18 — Nursing, Allied Health & Rehabilitation |
| proposal slot | 1 |
| task family | d18.verify_administer.single-dose-packet |

The capability's stated outcome: *"Compare a supplied order, patient record, product label, current parameters, and administration protocol to determine whether and how a medication may be administered, held, or clarified."*
Declared **excludes**: choosing a medication without an order; calculating an absent dose or concentration; monitoring effects after administration.

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

What follows is `instruction.md` in full, the complete solver-facing surface (10.1 KB, 182 lines).

---

You are the nurse caring for Marta Delgado. Before giving the 08:00 dose of her scheduled oral medication, review the complete packet below — the medication order, her record and current vital signs, the pharmacy preparation slip, the product label, the co-administration record, and the facility administration policy POL-MM-104. Decide whether this dose should be administered, held, or clarified. Return a single JSON object (schema below) containing your disposition, every failed or unresolved check with its category and the specific evidence, the administration volume implied by the product label, and the follow-up actions the policy requires. Do not administer any dose that fails a policy check, do not accept any supplied quantity you have not verified against the label, and do not select or substitute a medication without a prescriber order.

# Pre-Administration Medication Packet — Marta Delgado, 08:00 dose

## Section 1 — Medication order

```
RIVERBEND REGIONAL — MEDICATION ORDER
Order #: 77341
Date written: yesterday (day before the current dose)
Prescriber: Dr. A. Okafor

Patient: Marta Delgado        DOB: 1958-03-12        MRN: 441-207-883

Medication:   metoprolol tartrate oral solution
Dose:         25 mg
Route:        PO (oral)
Frequency:    every 12 hours
Scheduled times: 08:00 and 20:00

Hold parameters (per POL-MM-104 §4):
  Hold if HR < 55 bpm or SBP < 100 mmHg.
  Obtain vital signs within 30 minutes before each dose.

Status: active, scheduled for the next dose at 08:00 today.
```

## Section 2 — Patient record

```
RIVERBEND REGIONAL — PATIENT RECORD (Ward 4B)
Patient: Marta Delgado        DOB: 1958-03-12        MRN: 441-207-883
Wristband: name and DOB on wristband match the record and the order above.

Allergies:
  Sulfonamide antibiotics — rash

Status this morning: alert, oriented, swallowing intact.

Vital signs (taken 07:40 today):
  HR 52 (apical)
  BP 118/74
  RR 16
  T 36.8 C
  SpO2 97% on room air
```

## Section 3 — Pharmacy preparation slip

```
RIVERBEND REGIONAL — PHARMACY PREPARATION SLIP
metoprolol tartrate 25 mg PO — administer 5 mL
Verified: RPh J. Whitfield
Batch: RB-4471
Prepared: 06:15 today
```

## Section 4 — Product label

```
RIVERBEND REGIONAL PRODUCT LABEL
Metoprolol tartrate oral solution
50 mg per 5 mL (10 mg/mL)
120 mL bottle
For oral use.
Lot: RB-4471
Expires: 2027-06-30 (valid)
```

## Section 5 — Co-administration record

```
RIVERBEND REGIONAL — CO-ADMINISTRATION RECORD (08:00 due time)
lisinopril 10 mg PO — due 08:00 today.
No interaction listed between lisinopril and metoprolol tartrate
in POL-MM-104 Appendix A.
```

## Section 6 — Facility policy POL-MM-104

# POL-MM-104 — Pre-Administration Verification Policy (Oral Medications)

**Issuing body:** Riverbend Regional Medication Safety Committee (synthetic facility policy, authored for this training packet)
**Applies to:** every scheduled oral medication dose administered at Riverbend Regional
**Effective:** current revision, supersedes all prior versions

## 1. Purpose and scope

1.1. Before any scheduled dose is administered, the administering nurse must complete all seven pre-administration checks defined in §2, using only the documents in the medication packet: the medication order, the patient record, the pharmacy preparation slip, the product label, the co-administration record, and this policy.

1.2. A dose may be administered only when **all seven checks pass**. If any check fails, the dose must be **held** (§7). If the packet cannot be completed or contains an unresolvable ambiguity, the disposition is **clarify** (§7).

1.3. No nurse may resolve a conflict in the packet by selecting or substituting a different medication, dose, or route without a prescriber order.

1.4. Every quantity supplied in the packet — including a quantity verified by a pharmacist — must be checked against the product label before administration (§2.7). A pharmacist verification does not replace the administering nurse's label check.

## 2. The seven pre-administration checks

2.1. **patient_order_product_match** — Passes when (a) the patient is identified on the order by at least two identifiers and the same two identifiers appear on the patient record/wristband, and (b) the medication named on the product label and on the pharmacy preparation slip is the same medication, in the same form, as the ordered medication. Fails if any identifier or medication mismatch exists.

2.2. **allergy** — Passes when the ordered medication has no documented cross-reactivity with any allergy recorded in the patient record, according to the allergy cross-reference table in §5. Fails if the ordered medication has a documented cross-reactivity with a recorded allergy, or if the allergy status is unknown.

2.3. **parameter** — Passes when every hold parameter attached to the order is met, using the most recent valid vital signs (§3.2) and the hold-parameter table in §4. Fails if any measured value is outside the ordered or policy hold range. A dose whose required pre-dose vital signs are missing or not valid under §3.2 fails this check as unresolved.

2.4. **timing** — Passes when (a) the planned administration time falls inside the administration window of §3.1, and (b) the vital signs required by the order are valid under §3.2 at the planned administration time. Fails if either condition is not met.

2.5. **route** — Passes when the route on the order, the route stated on the pharmacy preparation slip, and the route the patient can safely receive are all the same route. Fails on any route mismatch or on a patient who cannot safely receive the medication by the ordered route.

2.6. **compatibility** — Passes when every medication scheduled to be given at the same time as this dose has no documented interaction with it in Appendix A. Fails if any same-time medication pair is listed in Appendix A or if the same-time medication list is unknown.

2.7. **quantity** — Passes when the volume to be administered equals the **label-implied volume**: the volume that delivers exactly the ordered dose at the concentration stated on the product label. The label-implied volume is derived only from the ordered dose and the label concentration; any other supplied volume (including the volume on a pharmacy-verified preparation slip) must agree with it. Fails if the supplied volume differs from the label-implied volume. A failed quantity check is a preparation discrepancy and must be reported under §6.1.

## 3. Timing rules

3.1. **Administration window.** A scheduled dose may be administered from **30 minutes before** to **60 minutes after** its scheduled time. Outside this window the timing check fails and the dose is held pending rescheduling.

3.2. **Valid pre-dose vital signs.** When an order requires vital signs before a dose, the vital signs are valid only if obtained **within the 30 minutes immediately before** the planned administration time. Vital signs older than 30 minutes at the planned administration time are not valid; obtain fresh vital signs before proceeding.

## 4. Hold-parameter table

The hold parameters for a dose are those stated on the order; where the order refers to this policy, the table below applies. A dose is held when any measured value falls below the stated minimum or above the stated maximum.

| Medication (ordered form) | Parameter | Hold if |
| --- | --- | --- |
| Metoprolol tartrate oral solution | Heart rate (apical) | HR < 55 bpm |
| Metoprolol tartrate oral solution | Systolic blood pressure | SBP < 100 mmHg |

Hold parameters attached to the order supersede this table when they are stricter.

## 5. Allergy cross-reference table

This table lists, for each allergy recorded in a patient record, the cross-reactivities recognized at Riverbend Regional. A recorded allergy with no listed cross-reactivity against the ordered medication passes the allergy check.

| Recorded allergy | Recognized cross-reactivity |
| --- | --- |
| Sulfonamide antibiotics | No cross-reactivity with metoprolol tartrate; no cross-reactivity with lisinopril |
| Sulfonamide diuretics | Cross-reactivity with sulfonamide antibiotics |

## 6. Required follow-up actions

6.1. **Pharmacy notification.** Notify the pharmacy of any preparation discrepancy — any supplied volume that disagrees with the label-implied volume (§2.7), or any other mismatch between the preparation slip and the product label.

6.2. **Hold documentation.** Whenever a dose is held, document the hold reason together with the parameter value that triggered it (e.g., the measured heart rate and the threshold it fell below).

6.3. **Vital-sign recheck.** When a dose is held for a hold parameter, recheck vital signs before the next scheduled dose of the same medication.

## 7. Dispositions

- **administer** — all seven checks pass and the dose is given as ordered.
- **hold** — at least one check fails; the dose is not given, and the follow-up actions of §6 apply as relevant to the failed checks.
- **clarify** — the packet is incomplete or contains an unresolvable ambiguity; the dose is not given until the ambiguity is resolved with the prescriber or pharmacy.

## Appendix A — Documented same-time interactions

Only the pairs listed below have documented interactions at Riverbend Regional. Pairs not listed here have no documented interaction.

| Medication A | Medication B |
| --- | --- |
| Sulfonamide diuretics | Sulfonamide antibiotics |

## Appendix B — Required decision output

Return a single JSON object with exactly these four required keys (unknown top-level keys are ignored):

```
{
  "disposition": "administer" | "hold" | "clarify",
  "failed_checks": [
    { "category": "patient_order_product_match" | "allergy" | "parameter" | "timing" | "route" | "compatibility" | "quantity",
      "finding": "<free text citing the specific packet evidence>" }
  ],
  "correct_volume_ml": <number>,
  "followup": ["<free-text actions>"]
}
```

- `disposition` is matched case-insensitively against the three tokens above.
- `failed_checks[].category` is matched case-insensitively against the seven tokens above; a token outside this list counts as no category.
- `correct_volume_ml` is the label-implied volume (§2.7) for the ordered dose, in mL.

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
| quality review | `accept` (artifact sha256 `7f16dc3edda76555…`) |
| repair budget | `{"exhausted":false,"max":4,"used":0}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 10.1 K |
| `manifest.json` | 1,465 |
| `renderings.json` | 134 |
| `specification.json` | 44.3 K |
| `task.toml` | 54 |

