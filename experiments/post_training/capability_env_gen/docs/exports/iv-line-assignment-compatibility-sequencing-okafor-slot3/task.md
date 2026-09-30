# Task `synthetic/iv-line-assignment-compatibility-sequencing/okafor-slot3`

**Intravenous line assignment and compatibility sequencing with a pump-limit rate violation.**  The exported, validated deliverable of run `/muchanem/cap-construct-003-hc* (pilot 003)`.

This document is a rendering of the exported TaskSpec.  The authoritative bytes are `validated/d18.nursing.medication.verify_administer-3-288d8440ed8b/` (6 files).

---

## Identity

| field | value |
| --- | --- |
| `id` | `synthetic/iv-line-assignment-compatibility-sequencing/okafor-slot3` |
| `schema_version` | `0.9` |
| `difficulty` | 6 |
| `success_policy` | `final` |
| `metadata.task_shape` | `answer` |
| steps | 1 |
| `specification_sha256` | `a0d9cd8e35fd790da5357f733bb161011fba394f117d2b698c0c2036f86a517d` |

`coverage_tags`: `artifact:application/json`, `competency:constraint_satisfaction`, `context:provided_documents`, `shape:constrained_generation`, `subject:medicine`

### Provenance

| field | value |
| --- | --- |
| `source.dataset` | `synthetic-capability-proposals` |
| `source.revision` | `582247f5e8359a4c6b47f8e6b6210f9add2c4ca7f2a5117cda7631953aec121b` |
| `source.row` | `288d8440ed8b1d2d4201319afadf162b0da9fc0504ee079a569cce99119a1d5b` |
| `source.importer_revision` | `cap-construct-003-hc2-d1-20260928T160526Z-25754` |
| catalog | `new_catalog.json` @ `2026.09.20-cross-domain-v3` |
| capability | `d18.nursing.medication.verify_administer` — *Verify and plan safe medication administration* |
| subject | D18 — Nursing, Allied Health & Rehabilitation |
| proposal slot | 3 |
| task family | iv-line-assignment-compatibility-sequencing |

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
| step 0 `answer_requirements` | `{"kind":"json"}` |

`task.toml`:

```toml
version = "1.0"

[environment]
allow_internet = false
```

---

## The prompt

What follows is `instruction.md` in full, the complete solver-facing surface (20.3 KB, 484 lines).

---

IV THERAPY DECISION PACKET - DANA R. OKAFOR
===========================================

You are the nurse caring for Dana R. Okafor on a medical unit at Riverside
Regional Medical Center (synthetic facility). The time now is 13:40.

Three new IV orders have just arrived while her single-lumen PICC is running
a potassium-containing maintenance fluid. Using ONLY the supplied documents
below, decide for each order whether it may be administered now, held, or
clarified; assign every product to a line and time; and produce the exact
ordered administration sequence, including any flushes and carrier
stops/resumes required by facility Policy IV-7.2.

Before anything is programmed, review the pharmacy pump rate sheet against
the orders and Table P-3 of Policy IV-7.2.

The documents, in order:

  1. patient_record.txt      - access report, allergy record, vitals, hold
                               parameters
  2. orders.txt              - the running carrier order and the three new
                               orders
  3. product_labels.txt      - the four supplied product labels
  4. pharmacy_rate_sheet.txt - the pharmacy-verified pump rate sheet
  5. policy_IV-7.2.txt       - facility policy (Tables Y-4, P-3, A-1, H-2;
                               rules D-1, C-1, F-1, T-1, DOC-1)
  6. output_contract.txt     - your required output, including the JSON
                               schema and the answer rules

The facility tables in policy_IV-7.2.txt are authoritative for this task;
outside references and real-world compatibility knowledge are not. Do not
compute any dose, concentration, volume, or rate: every quantity you need
is supplied. Cite the policy rule or table for every failed or unresolved
check in your reasons.

RIVERSIDE REGIONAL MEDICAL CENTER - MEDICAL UNIT 4WEST
PATIENT RECORD (SYNTHETIC TRAINING PACKET - NOT A REAL PATIENT)

Patient:        Dana R. Okafor
MRN:            88-4471
DOB:            1971-03-14
Attending:      Dr. M. Ellison

Record current as of 13:40 today (day 6 of admission).

ALLERGIES
---------
Sulfa (sulfonamide antibiotics) - reaction: hives. See Policy IV-7.2
Table A-1 for the facility cross-reactivity reference for every product
supplied in this packet.

VITALS (13:30 today)
--------------------
HR 78 (regular)   BP 118/74   Temp 37.0 C   RR 16   SpO2 96% RA
Patient alert and oriented, no acute distress.

ACCESS DEVICE REPORT
--------------------
Device:          PICC, single lumen, 4 Fr
Site:            right upper arm
Inserted:        5 days ago
Tip position:    cavoatrial (confirmed on insertion chest x-ray)
Status:          patent; blood return present; dressing intact and dry;
                no redness, swelling, or tenderness at site
Lumens:          one (1) distal lumen only

NOTE: No peripheral access available; two PIV attempts failed this
admission. Do not plan peripheral administration without new access.

CURRENT INFUSIONS (as of 13:40)
-------------------------------
ORD-091  0.9% Sodium Chloride with 20 mEq Potassium Chloride per liter
         ("KCl carrier"), continuous IV, 80 mL/hr, running in the PICC
         distal lumen since 09:00 today.

RIVERSIDE REGIONAL MEDICAL CENTER - ACTIVE IV MEDICATION ORDERS
Patient: Dana R. Okafor   MRN 88-4471   DOB 1971-03-14
Orders current as of 13:40 today.

ORD-091  0.9% Sodium Chloride with 20 mEq Potassium Chloride per liter
         IV, continuous infusion, 80 mL/hr, via PICC. Running since
         09:00 today. Continue.

ORD-102  Pantoprazole 40 mg in 100 mL 0.9% Sodium Chloride
         IV piggyback (IVPB), infuse over 30 minutes, once every
         24 hours (q24h). Next dose due 14:00 today.

ORD-103  Cefepime 2 g in 100 mL 0.9% Sodium Chloride
         IV piggyback (IVPB), infuse over 30 minutes, every 12 hours
         (q12h). Next dose due 14:00 today.

ORD-104  Amiodarone 900 mg in 250 mL D5W
         IV, continuous infusion, ordered rate 35 mL/hr.
         Comment: "Start now."

All scheduled doses above are non-time-critical per the facility
time-critical list (Policy IV-7.2, Section T-1). Administration windows
are governed by Policy IV-7.2, Section T-1.

PRODUCT LABELS SUPPLIED TO THE UNIT (SYNTHETIC)

LBL-091 (running bag, ORD-091)
  0.9% Sodium Chloride Injection with 20 mEq Potassium Chloride
  per liter. 1000 mL. Single bag. Administer by IV infusion.
  Concentration: KCl 20 mEq/L.

LBL-102 (ORD-102)
  Pantoprazole 40 mg / 100 mL 0.9% Sodium Chloride Injection.
  Single-dose IVPB container. Total volume 100 mL.
  Infuse over 30 minutes per order ORD-102.

LBL-103 (ORD-103)
  Cefepime 2 g / 100 mL 0.9% Sodium Chloride Injection.
  Single-dose IVPB container. Total volume 100 mL.
  Infuse over 30 minutes per order ORD-103.

LBL-104 (ORD-104)
  Amiodarone 900 mg / 250 mL D5W Injection.
  Single-dose continuous-infusion container. Total volume 250 mL.
  Infuse continuously; rate per order ORD-104.

All four products are also listed in the facility Y-site compatibility
matrix (Policy IV-7.2, Table Y-4) and the pump maximum-rate table
(Policy IV-7.2, Table P-3).

RIVERSIDE REGIONAL MEDICAL CENTER - PHARMACY PUMP RATE SHEET
Patient: Dana R. Okafor   MRN 88-4471
Pharmacy-verified pump settings for today's new IV orders.
Verification date/time: 13:25 today. Verified by: J. Ruiz, RPh.

  Order    Product        Pharmacy-Verified Rate   Comment
  ------   ------------   ---------------------   ---------------------------
  ORD-091  KCl carrier    80 mL/hr                continue current setting
  ORD-102  Pantoprazole   200 mL/hr                IVPB over 30 minutes
  ORD-103  Cefepime       200 mL/hr                IVPB over 30 minutes
  ORD-104  Amiodarone     125 mL/hr                continuous infusion

Settings above were verified by pharmacy against the orders.
Program pump as listed unless bedside verification finds a
discrepancy; bedside verification against the order and Policy
IV-7.2 Table P-3 is still required before programming.

RIVERSIDE REGIONAL MEDICAL CENTER
POLICY IV-7.2: INTRAVENOUS MEDICATION ADMINISTRATION - COMPATIBILITY,
LINE ASSIGNMENT, AND PUMP RATE VERIFICATION
(Synthetic facility policy; internal training document)

SECTION 1 - AUTHORITY AND SCOPE
--------------------------------
1.1 This policy governs the assignment of IV medications to lines,
compatibility of products sharing a lumen or Y-site, verification of
pump rates, and the required administration sequence when products
that cannot share a lumen must use the same lumen.
1.2 The tables and lists in this policy (Tables Y-4, P-3, A-1, H-2
and Sections D-1, C-1, F-1, T-1, DOC-1) are the AUTHORITATIVE
compatibility, rate, and timing references for this facility. Outside
references, memory, and real-world compatibility charts are NOT
authoritative and must not override these tables.
1.3 No dose, concentration, volume, or rate may be recalculated at the
bedside for this purpose; the nurse compares the supplied order,
label, pharmacy rate sheet, and the tables below. If any supplied
quantity disagrees with the tables or the order, the discrepancy must
be resolved before administration.

TABLE Y-4 - Y-SITE AND SAME-LUMEN COMPATIBILITY MATRIX
------------------------------------------------------
C = compatible (may share a lumen at a Y-site, or run sequentially
    in the same lumen with no intervening flush)
I = incompatible (may NOT infuse concurrently or sequentially in the
    same lumen without the full Section F-1 sequence between them)

                       | KCl carrier | Pantoprazole | Cefepime | Amiodarone
  KCl carrier          |     -      |      C       |    I     |     I
  Pantoprazole         |     C      |      -       |    C     |     I
  Cefepime             |     I      |      C       |    -     |     I
  Amiodarone           |     I      |      I       |    I     |     -

TABLE P-3 - PUMP MAXIMUM RATES
-------------------------------
  Product                    | Maximum pump rate
  ---------------------------+------------------
  KCl carrier (this packet)  | 150 mL/hr
  Pantoprazole (IVPB)        | 300 mL/hr
  Cefepime (IVPB)            | 200 mL/hr
  Amiodarone (infusion)      |  60 mL/hr

2.1 A programmed rate MUST NOT EXCEED the Table P-3 maximum for that
product. A rate EQUAL TO the listed maximum is acceptable; only a
rate GREATER THAN the maximum is a violation. Example: cefepime
programmed at exactly 200 mL/hr is acceptable because 200 mL/hr is
the Table P-3 maximum for cefepime; 201 mL/hr or any higher rate is a
violation.
2.2 Before any infusion is programmed, the administering nurse MUST
verify the rate against BOTH the prescriber's order AND Table P-3.
2.3 A pharmacy-verified rate sheet does not replace this bedside
verification. If a pharmacy-verified rate is greater than the ordered
rate OR greater than the Table P-3 maximum for that product, the
nurse MUST NOT program that rate; the entry is unacceptable, and the
nurse must refuse the setting and clarify it with pharmacy before
administration proceeds on that setting.

TABLE A-1 - ALLERGY CROSS-REFERENCE (SULFA)
---------------------------------------------
  Product        | Cross-reactivity with sulfa allergy
  ---------------+-------------------------------------
  0.9% NaCl      | none
  KCl            | none
  Pantoprazole   | none
  Cefepime       | none
  Amiodarone     | none
  D5W            | none

3.1 No product in this packet cross-reacts with a documented sulfa
allergy. A sulfa allergy is therefore not a reason to hold or clarify
any product supplied in this packet.

TABLE H-2 - HOLD PARAMETERS (AMIODARONE)
-----------------------------------------
  Parameter                              | Hold amiodarone and notify
                                         | the prescriber if:
  ---------------------------------------+---------------------------
  Heart rate                             | HR < 50 beats/min
  Systolic blood pressure                | SBP < 90 mmHg

4.1 Hold parameters are pre-administration gates for new amiodarone
infusions. Both parameters must be within the limits above (HR 50 or
greater, and SBP 90 or greater) before amiodarone is started.

SECTION D-1 - DEDICATED-LUMEN LIST
-----------------------------------
The following products may NOT share a lumen with ANY other product,
including via a Y-site, regardless of Table Y-4:
  - Amiodarone: requires a dedicated lumen with no other product
    infusing or connected to that lumen.

SECTION C-1 - CENTRAL-ACCESS-ONLY LIST
----------------------------------------
The following products must be administered through a central line
(tunneled, PICC, or port) and may NOT be administered through a
peripheral IV:
  - Amiodarone (continuous infusion).

SECTION F-1 - FLUSH PROTOCOL FOR INCOMPATIBLE PRODUCTS SHARING A LUMEN
-----------------------------------------------------------------------
5.1 When two products rated I in Table Y-4 (or any product on the
Section D-1 dedicated-lumen list) must use a lumen that another
product is using or has used, the ONLY permitted sequence on that
lumen is:
    a. STOP all other infusions on that lumen;
    b. FLUSH the lumen with at least 10 mL 0.9% Sodium Chloride;
    c. ADMINISTER the product;
    d. FLUSH the lumen with at least 10 mL 0.9% Sodium Chloride;
    e. RESUME the stopped infusions on that lumen.
5.2 A flush of exactly 10 mL satisfies the flush requirement. Flushes
larger than 10 mL also satisfy it. A flush smaller than 10 mL does
NOT satisfy it.
5.3 Products rated C in Table Y-4 may share a lumen at a Y-site with
no intervening flush, and may run sequentially in the same lumen with
no intervening flush.
5.4 Extra flushes are permitted but not required; performing the
Section F-1 sequence when it is not required does not violate this
policy.
5.5 While a Section F-1 sequence is in progress on a lumen, no other
product may infuse on that lumen between step (a) and step (e).

SECTION T-1 - ADMINISTRATION TIMING
-------------------------------------
6.1 Non-time-critical scheduled doses MUST start within plus or minus
60 minutes of the scheduled due time. The window is INCLUSIVE on both
ends: a dose due at 14:00 may start as early as 13:00 and as late as
15:00. A start at exactly 13:00 is within the window; a start at
exactly 15:00 is within the window; a start before 13:00 or after
15:00 (for example 15:01) is outside the window.
6.2 The facility time-critical medication list is: IV regular
insulin, IV heparin, and all chemotherapy agents. Time-critical
medications are outside this policy's window rule. All other
scheduled medications, including pantoprazole and cefepime, are
non-time-critical and follow rule 6.1.

SECTION DOC-1 - DOCUMENTATION REQUIREMENTS
--------------------------------------------
7.1 For every administered product, document: product, dose, route,
rate, start and stop times, and the line used.
7.2 For every held order, document: the order, the hold reason, and
prescriber notification.
7.3 For every clarification request, document: the role contacted
(for example prescriber or pharmacy), the topic, and the time.
7.4 All entries are made in the electronic medication administration
record (MAR) at the time of the event.

SECTION 8 - PATIENT IDENTIFICATION
------------------------------------
8.1 Before administering any medication, confirm the patient using
two identifiers (full name plus MRN or DOB) against the patient
record.

OUTPUT CONTRACT - IV LINE ASSIGNMENT AND COMPATIBILITY SEQUENCING
(Read together with patient_record.txt, orders.txt, product_labels.txt,
pharmacy_rate_sheet.txt, and policy_IV-7.2.txt.)

YOUR TASK
---------
You are the nurse caring for Dana R. Okafor at 13:40. Using ONLY the
supplied packet, decide for each order whether it is administered now,
held, or clarified; assign every product to a line and mode; produce the
exact ordered administration sequence including any flushes and carrier
stops/resumes required by Policy IV-7.2; review the pharmacy pump rate
sheet against the orders and Table P-3 before anything is programmed;
request any clarifications; and list documentation entries.

RULES FOR YOUR ANSWER
---------------------
1. Return ONE valid JSON object matching the schema below. JSON keys
   are matched case-sensitively; exactly these key names are required.
2. Disposition, action, mode, contact-role, and entry-type TOKENS are
   matched case-insensitively; any case of a listed token is accepted.
   Unlisted tokens are wrong answers, not format errors.
3. Do NOT compute any dose, concentration, volume, or rate. Every
   quantity you need is supplied in the packet; compare and sequence.
4. The facility tables in policy_IV-7.2.txt are authoritative. Outside
   references and real-world compatibility knowledge are NOT
   authoritative for this task.
5. Cite the policy rule or table for every failed or unresolved check
   in your reasons.
6. Extra flushes and different valid orderings are permitted as long
   as every policy constraint holds.

REQUIRED JSON SHAPE (schema)
-----------------------------
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "IV plan for Dana R. Okafor",
  "type": "object",
  "required": ["patient_verification", "dispositions", "line_assignments",
               "administration_sequence", "rate_review", "clarifications",
               "documentation"],
  "additionalProperties": true,
  "properties": {
    "patient_verification": {
      "type": "object",
      "required": ["patient_name", "mrn", "dob", "two_identifiers_confirmed"],
      "additionalProperties": true,
      "properties": {
        "patient_name": {"type": "string"},
        "mrn": {"type": "string"},
        "dob": {"type": "string"},
        "two_identifiers_confirmed": {"type": "boolean"}
      }
    },
    "dispositions": {
      "type": "array",
      "minItems": 4,
      "maxItems": 4,
      "items": {
        "type": "object",
        "required": ["order_id", "disposition", "reasons"],
        "additionalProperties": true,
        "properties": {
          "order_id": {"type": "string",
            "enum": ["ORD-091", "ORD-102", "ORD-103", "ORD-104"]},
          "disposition": {"type": "string",
            "description": "token: continue | administer | hold | clarify"},
          "reasons": {"type": "array", "minItems": 1,
            "items": {"type": "string"}}
        }
      }
    },
    "line_assignments": {
      "type": "array",
      "minItems": 4,
      "maxItems": 4,
      "items": {
        "type": "object",
        "required": ["order_id", "product", "line", "mode"],
        "additionalProperties": true,
        "properties": {
          "order_id": {"type": "string",
            "enum": ["ORD-091", "ORD-102", "ORD-103", "ORD-104"]},
          "product": {"type": "string"},
          "line": {"type": "string",
            "description": "e.g. picc_single_lumen, or unassigned"},
          "mode": {"type": "string",
            "description": "token: y_site | sequential | continuous | not_assigned"}
        }
      }
    },
    "administration_sequence": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["time", "action", "order_id", "product",
                     "rate_mL_hr", "volume_mL", "detail"],
        "additionalProperties": true,
        "properties": {
          "time": {"type": "string", "pattern": "^[0-2][0-9]:[0-5][0-9]$"},
          "action": {"type": "string",
            "description": "token: ysite_start | carrier_stop | flush | administer | carrier_resume | hold | clarify"},
          "order_id": {"type": "string"},
          "product": {"type": "string"},
          "rate_mL_hr": {"type": ["number", "null"]},
          "volume_mL": {"type": ["number", "null"]},
          "detail": {"type": "string"}
        }
      }
    },
    "rate_review": {
      "type": "object",
      "required": ["order_id", "product", "sheet_rate_mL_hr",
                   "pump_max_mL_hr", "ordered_rate_mL_hr", "acceptable",
                   "action"],
      "additionalProperties": true,
      "properties": {
        "order_id": {"type": "string"},
        "product": {"type": "string"},
        "sheet_rate_mL_hr": {"type": "number"},
        "pump_max_mL_hr": {"type": "number"},
        "ordered_rate_mL_hr": {"type": "number"},
        "acceptable": {"type": "boolean"},
        "action": {"type": "string",
          "description": "what you did about the sheet entry"}
      }
    },
    "clarifications": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["contact_role", "topic", "detail"],
        "additionalProperties": true,
        "properties": {
          "contact_role": {"type": "string",
            "description": "token: prescriber | pharmacy"},
          "topic": {"type": "string"},
          "detail": {"type": "string"}
        }
      }
    },
    "documentation": {
      "type": "array",
      "minItems": 3,
      "items": {
        "type": "object",
        "required": ["order_id", "entry_type", "detail"],
        "additionalProperties": true,
        "properties": {
          "order_id": {"type": "string"},
          "entry_type": {"type": "string",
            "description": "token: administer | hold | clarify"},
          "detail": {"type": "string"}
        }
      }
    }
  }
}

FIELD NOTES
-----------
- dispositions: exactly one entry for each order ORD-091, ORD-102,
  ORD-103, ORD-104.
- line_assignments: exactly one entry for each order; a product that
  must not be given on any currently available line is listed with
  line "unassigned" and mode "not_assigned".
- administration_sequence: time-ordered events covering the interval
  from now (13:40) through the last planned event.
    * ysite_start: start a piggyback concurrently with a running
      compatible infusion on the same lumen via a Y-site.
    * carrier_stop: stop another infusion on that lumen.
    * flush: volume_mL is the flush volume; product is 0.9% NaCl.
    * administer: start a product that is not concurrent (rate_mL_hr
      and volume_mL carry the programmed rate and dose volume).
    * carrier_resume: restart a previously stopped infusion.
    * hold / clarify: record a hold or clarification decision for an
      order.
- rate_review: review of the pharmacy pump rate sheet entry that
  fails bedside verification against the order and Table P-3, with the
  sheet rate, the Table P-3 maximum, the ordered rate, whether the
  sheet rate is acceptable, and your action.
- clarifications: contact_role must name the correct role for the
  topic (prescriber for access/order decisions, pharmacy for pump
  rate sheet errors).
- documentation: entries for every administration, hold, and
  clarification performed.

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
| quality review | `accept` (artifact sha256 `f38ea1d4c9216fe0…`) |
| repair budget | `{"exhausted":false,"max":4,"used":2}` |

---

## Files in the export

| file | bytes |
| --- | --- |
| `binding.json` | 42 |
| `instruction.md` | 20.3 K |
| `manifest.json` | 1,458 |
| `renderings.json` | 134 |
| `specification.json` | 88.0 K |
| `task.toml` | 54 |

