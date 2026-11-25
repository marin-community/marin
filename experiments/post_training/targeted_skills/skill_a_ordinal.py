# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generator and deterministic verifier for Skill A: Nth-of-many-similar-passages retrieval (8K-64K, 4-16 needles).

Strictly disjoint from Probe 1 (12 domains, crews, regions, statuses) and MRCR literary genres.
Uses dedicated training seed namespace ``mochi-train-skillA-ordinal-v1``.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from difflib import SequenceMatcher
from typing import Any

from .common import EOT_ID, render_shifted_row

SKILL_A_SEED_NAMESPACE = "mochi-train-skillA-ordinal-v1"

# Explicit build-time blocklist of Probe 1 domains, vocabularies, and MRCR literary genres.
BLOCKED_PROBE1_DOMAINS = frozenset(
    {
        "substation fault brief",
        "greenhouse irrigation log",
        "harbor berth dispatch",
        "zoning variance memo",
        "observatory exposure entry",
        "coldchain batch audit",
        "waterworks valve ticket",
        "geothermal well bulletin",
        "rail switch advisory",
        "aquifer piezometer note",
        "cleanroom filter record",
        "bridge bearing inspection",
    }
)
BLOCKED_PROBE1_VOCAB = frozenset(
    {
        "alpha-2",
        "bravo-4",
        "cedar-7",
        "delta-3",
        "echo-9",
        "flint-5",
        "granite-1",
        "helios-6",
        "indigo-8",
        "kestrel-4",
    }
)
BLOCKED_MRCR_GENRES = frozenset({"poem", "riddle", "short story", "essay", "haiku", "blog post"})

# 20 fresh operational domains (zero overlap with Probe 1 domains or MRCR literary genres).
SKILL_A_DOMAINS: list[tuple[str, str, list[str]]] = [
    (
        "aviation maintenance workcard",
        "Hangar",
        [
            "flap actuator backlash measure",
            "pitot heater resistance sweep",
            "landing gear oleo strut pressure",
            "avionics rack cooling airflow",
            "bleed-air duct clamp torque",
            "rudder trim tab travel verify",
            "fuel boost pump seal inspection",
            "cabin outflow valve seat check",
        ],
    ),
    (
        "library acquisition slip",
        "Archive",
        [
            "folio binding spine consolidation",
            "microfilm reel silver-halide assay",
            "incunabula provenance collation",
            "acid-free clamshell box sizing",
            "map drawer linen backing pass",
            "serials barcode registry entry",
            "parchment humidity chamber cycle",
            "rare monograph deacidification",
        ],
    ),
    (
        "freight customs declaration",
        "Depot",
        [
            "tariff heading tariff-code cross-check",
            "bonded pallet tamper-seal audit",
            "phytosanitary pallet stamp check",
            "bill-of-lading tare weight verify",
            "hazmat placard class reconciliation",
            "origin certificate consular stamp",
            "container manifest tally review",
            "transit bond release endorsement",
        ],
    ),
    (
        "museum conservation note",
        "Gallery",
        [
            "varnish UV fluorescence mapping",
            "pigment flake consolidation pass",
            "bronze patina chloride stabilization",
            "silica-gel vitrine buffer refresh",
            "canvas stretcher key tensioning",
            "stone relief salt poultice lift",
            "textile silk crepeline backing",
            "display lux-hour dosimeter read",
        ],
    ),
    (
        "pharmacy compounding sheet",
        "Dispensary",
        [
            "laminar hood differential check",
            "excipient lot gravimetric tally",
            "sterile vial stopper crimping",
            "pH buffer electrode two-point cal",
            "endotoxin limulus reagent run",
            "suspension viscosity spindle read",
            "amber bottle tamper-band seal",
            "cold-pack shipping lane packout",
        ],
    ),
    (
        "hydrology gauge bulletin",
        "Weir",
        [
            "stilling-well float encoder sync",
            "acoustic doppler velocity transect",
            "crest-stage cork-line survey",
            "turbidity nephelometer zero check",
            "staff-plate datum offset verify",
            "sediment trap bottle retrieval",
            "dissolved-oxygen membrane swap",
            "telemetry radio antenna SWR sweep",
        ],
    ),
    (
        "textile dye-lot card",
        "Mill",
        [
            "spectrophotometer delta-E check",
            "vat liquor alkali titration pass",
            "crockmeter wet-rub fastness test",
            "scouring bath surfactant dosing",
            "stenter frame selvage tensioning",
            "yarn skein wash-shrinkage check",
            "mordant fixation dwell timing",
            "steam ager condensate trap drain",
        ],
    ),
    (
        "acoustics chamber log",
        "Chamber",
        [
            "anechoic wedge absorption sweep",
            "condenser mic pistonphone cal",
            "reverberation RT60 decay curve",
            "floor isolation pad deflection",
            "dodecahedron speaker phase verify",
            "background noise NC-15 floor scan",
            "impedance tube standing-wave run",
            "baffle door perimeter seal check",
        ],
    ),
    (
        "foundry melt ticket",
        "Furnace",
        [
            "induction coil cooling flow check",
            "optical spectrometer alloy assay",
            "slag skimmer refractory coat",
            "thermocouple immersion dip read",
            "sand mold permeability gauge",
            "ladle preheat burner soak time",
            "degassing argon lance purge",
            "chill-block fracture grain check",
        ],
    ),
    (
        "vineyard harvest record",
        "Block",
        [
            "refractometer brix sugar reading",
            "titratable acidity malic assay",
            "cluster stem lignification score",
            "canopy leaf-pull sunburn tally",
            "picking bin tare weight balance",
            "press-pan free-run juice turbidity",
            "yeast assimilable nitrogen test",
            "cold-soak tank jacket setpoint",
        ],
    ),
    (
        "satellite pass bulletin",
        "Dish",
        [
            "S-band downlink carrier lock",
            "ephemeris two-line-element sync",
            "azimuth servo backlash profile",
            "cryogenic LNA noise-temp read",
            "doppler shift tracking residual",
            "solid-state recorder frame dump",
            "beacon polarization skew trim",
            "radome blower static pressure",
        ],
    ),
    (
        "municipal fleet workorder",
        "Garage",
        [
            "hydraulic lift cylinder bleed",
            "air-brake slack adjuster stroke",
            "DPF regeneration soot load check",
            "alternator ripple voltage test",
            "wheel-end hub oil seal inspect",
            "coolant refractometer freeze-pt",
            "steering kingpin dial-indicator",
            "PTO driveline u-joint greasing",
        ],
    ),
    (
        "botanical herbarium voucher",
        "Cabinet",
        [
            "pressed specimen linen mounting",
            "freezer beetle-quarantine cycle",
            "taxonomic determinavit slip tag",
            "seed packet silica desiccant pack",
            "type-sheet high-res flatbed scan",
            "locality datum WGS84 georef",
            "anther pollen slide glycerine mount",
            "accession ledger cross-indexing",
        ],
    ),
    (
        "quarry blast schedule",
        "Bench",
        [
            "borehole burden spacing survey",
            "seismograph tri-axial geophone set",
            "stemming crushed-stone depth check",
            "detonator millisecond delay plan",
            "overpressure airblast mic station",
            "highwall crest crack-gauge read",
            "emulsion truck density cup test",
            "perimeter clearance horn sweep",
        ],
    ),
    (
        "dairy pasteurization ledger",
        "Plant",
        [
            "plate heat-exchanger hold timer",
            "flow-diversion valve seat verify",
            "phosphatase enzyme negative test",
            "homogenizer second-stage pressure",
            "CIP caustic conductivity titration",
            "silo agitator baffle weld inspect",
            "butterfat infrared analyzer check",
            "chart recorder ink-stylus align",
        ],
    ),
    (
        "geodesy benchmark card",
        "Monument",
        [
            "GNSS static carrier-phase session",
            "digital invar-rod leveling loop",
            "brass disk witness-post offset",
            "tribrach optical-plummet verify",
            "tropospheric zenith delay log",
            "gravity meter spring drift check",
            "EDM prism constant verification",
            "azimuth mark backsight closure",
        ],
    ),
    (
        "timber kiln schedule",
        "Kiln",
        [
            "wet-bulb wick reservoir refill",
            "pin-meter core moisture sample",
            "case-hardening prong stress cut",
            "baffle curtain air-velocity scan",
            "fan reversal cycle timer check",
            "steam spray conditioning soak",
            "sticker alignment stack check",
            "vent damper actuator stroke test",
        ],
    ),
    (
        "metrology calibration sheet",
        "Lab",
        [
            "gauge-block wringing flatness",
            "deadweight tester piston spin",
            "platinum PRT triple-point check",
            "coordinate-arm sphere probing",
            "torque transducer clockwise ramp",
            "optical flat fringe count verify",
            "mass-comparator buoyancy correct",
            "barometer mercury meniscus zero",
        ],
    ),
    (
        "pressroom makeready sheet",
        "Press",
        [
            "blanket cylinder packing gauge",
            "fountain solution conductivity",
            "registration louver vernier trim",
            "densitometer solid-ink trap read",
            "plate-bender punch pin alignment",
            "feeder vacuum-sucker dwell check",
            "IR dryer web-temperature pyrometer",
            "folder buckle-chute stop setting",
        ],
    ),
    (
        "apiary hive inspection",
        "Yard",
        [
            "brood pattern capped-cell ratio",
            "varroa alcohol-wash mite count",
            "queen-excluder frame spacing",
            "entrance reducer airflow check",
            "pollen-patty protein weight log",
            "honey-super refractometer check",
            "bottom-board debris tray audit",
            "propolis vent-screen clearance",
        ],
    ),
]

ZONES = [
    "Sector-A",
    "Sector-B",
    "Sector-C",
    "Sector-D",
    "Zone-11",
    "Zone-22",
    "Zone-33",
    "Zone-44",
    "Wing-E",
    "Wing-W",
    "Annex-N",
    "Annex-S",
]
OUTCOMES = ["PASSED", "HOLD", "RECHECK", "SIGNED", "ARCHIVED", "FLAGGED"]
OPERATORS = [
    "L. Alvarez",
    "M. Chen",
    "R. Patel",
    "S. Okafor",
    "K. Novak",
    "T. Lindqvist",
    "D. Sato",
    "N. Rossi",
    "H. Haddad",
    "P. Kowalski",
    "J. Morales",
    "A. Mensah",
]
STYLES = ("prose", "bullets", "kv_memo", "table", "qa_log", "minutes")

# Stratified needle counts and weights from design: 4(15%), 6(15%), 8(25%), 12(25%), 16(20%)
# Interleaved across the 20-item cycle so small validation/test splits also cover all 5 needle counts.
NEEDLE_SCHEDULE = [8, 12, 16, 4, 6, 8, 12, 16, 8, 12, 4, 6, 16, 8, 12, 4, 6, 12, 16, 8]
BAND_TARGETS = [
    ("8k", 9_400),
    ("16k", 17_000),
    ("32k", 32_000),
    ("56k", 53_000),
]

ORDINAL_WORDS = {
    1: "1st",
    2: "2nd",
    3: "3rd",
    4: "4th",
    5: "5th",
    6: "6th",
    7: "7th",
    8: "8th",
    9: "9th",
    10: "10th",
    11: "11th",
    12: "12th",
    13: "13th",
    14: "14th",
    15: "15th",
    16: "16th",
}


def validate_skill_a_catalog() -> None:
    """Assert build-time domain and vocabulary disjointness from Probe 1 and MRCR."""
    seen = set()
    for label, fac, events in SKILL_A_DOMAINS:
        low = label.lower()
        assert low not in BLOCKED_PROBE1_DOMAINS, f"Domain collides with Probe 1: {label}"
        assert low not in seen, f"Duplicate Skill A domain: {label}"
        seen.add(low)
        for genre in BLOCKED_MRCR_GENRES:
            assert genre not in low, f"Domain collides with MRCR genre {genre}: {label}"
        assert fac.lower() not in {
            "substation",
            "conservatory",
            "terminal",
            "parcel",
            "dome",
            "vault",
            "main",
            "borehole",
            "turnout",
            "station",
            "bay",
            "span",
        }
        assert len(events) >= 8
    for op in OPERATORS:
        assert op.lower() not in BLOCKED_PROBE1_VOCAB


def _rng_for(tag: str) -> random.Random:
    h = hashlib.sha256(f"{SKILL_A_SEED_NAMESPACE}:{tag}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _make_styled_needle(
    r: random.Random,
    dom_idx: int,
    entity_name: str,
    occ_idx: int,
    style: str,
    used_codes: set[str],
) -> tuple[str, str, dict[str, str]]:
    """Create one needle user/assistant turn pair with unique ID and chosen structural style."""
    label, fac_prefix, events = SKILL_A_DOMAINS[dom_idx]
    ev1, ev2 = r.sample(events, 2)
    while True:
        num = r.randint(10000, 99999)
        c1, c2 = r.choice(string.ascii_uppercase), r.choice(string.ascii_uppercase)
        code = f"{fac_prefix[:3].upper()}-{num}-{c1}{c2}"
        if code not in used_codes:
            used_codes.add(code)
            break
    val1 = round(r.uniform(12.1, 489.9), 2)
    val2 = round(r.uniform(0.05, 19.75), 3)
    op = r.choice(OPERATORS)
    outcome = r.choice(OUTCOMES)
    month = r.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = r.randint(1, 28)
    hh = r.randint(0, 23)
    mm = r.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    recheck_h = r.randint(6, 72)
    stamp = f"{month} {day:02d} {hh:02d}:{mm:02d} UTC"
    remarks = [
        f"bench tag {r.choice(string.ascii_uppercase)}{r.randint(10, 99)} cleared without rework",
        f"ambient temp {r.uniform(14.0, 31.5):.1f}C logged on handheld meter #{r.randint(10, 99)}",
        f"calibration sticker valid through Q{r.randint(1, 4)}-{r.randint(2026, 2028)}",
        f"shift supervisor counter-signed checklist rev-{r.randint(2, 9)}",
        f"spare seal kit #{r.randint(200, 999)} staged at intake desk",
        f"archived photo set IMG-{r.randint(1000, 9999)} attached to dossier",
        f"zero fault flags on telemetry channel ch-{r.randint(1, 32)}",
        f"manual verification matched sensor bus #{r.randint(1, 16)} within tolerance",
    ]
    remark = r.choice(remarks)

    u = f"Create a {label} for {entity_name} (entry #{occ_idx}, format: {style})."
    if style == "prose":
        a = (
            f"Ticket {code} for {entity_name} logged on {stamp} by {op} closed as {outcome}. "
            f"Primary procedure covered {ev1} at {val1} units (drift {val2}), followed by {ev2} "
            f"with follow-up interval set to {recheck_h} hours; {remark}."
        )
    elif style == "bullets":
        a = (
            f"{label.title()} — {code} ({entity_name})\n"
            f"- Timestamp & Owner: {stamp} | {op} | Outcome: {outcome}\n"
            f"- Primary Check: {ev1} measured {val1} (offset {val2})\n"
            f"- Secondary Check: {ev2} verified; next inspection in {recheck_h}h ({remark})"
        )
    elif style == "kv_memo":
        a = (
            f"ID: {code}\n"
            f"Unit: {entity_name}\n"
            f"Recorded: {stamp} by {op}\n"
            f"Task_A: {ev1} = {val1} (delta {val2})\n"
            f"Task_B: {ev2} -> {outcome} (recheck {recheck_h}h)\n"
            f"Note: {remark}"
        )
    elif style == "table":
        a = (
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Code | {code} |\n"
            f"| Unit | {entity_name} ({stamp}) |\n"
            f"| Primary | {ev1}: {val1} (delta {val2}) |\n"
            f"| Secondary | {ev2} [{op} / {outcome}, +{recheck_h}h] |\n"
            f"| Remark | {remark} |"
        )
    elif style == "qa_log":
        a = (
            f"Q: Which ticket covers {entity_name} at {stamp}?\n"
            f"A: {code} handled by {op} ({outcome}).\n"
            f"Q: What readings and notes were recorded?\n"
            f"A: {ev1} at {val1} (offset {val2}); confirmed {ev2}; revisit in {recheck_h} hours; {remark}."
        )
    else:  # minutes
        a = (
            f"Shift note {code} [{entity_name}, {stamp}]: {op} reported {ev1} reading {val1} "
            f"(variance {val2}) and signed off on {ev2}. Disposition: {outcome}; revisit window {recheck_h}h; {remark}."
        )

    anchor = f"{code} ({stamp}, {outcome})"
    return u, a, {"code": code, "anchor": anchor, "style": style}


def _make_filler_turn(r: random.Random, exclude_dom: int, batch_idx: int, approx_words: int) -> tuple[str, str]:
    d_idx = r.choice([i for i in range(len(SKILL_A_DOMAINS)) if i != exclude_dom])
    label, fac_prefix, events = SKILL_A_DOMAINS[d_idx]
    unit = f"{fac_prefix} {r.choice(ZONES)}-{r.randint(100, 999)}"
    u = f"Compile background archive notes ({label}) for {unit}, dossier #{batch_idx}."
    lines = [f"Dossier #{batch_idx} — {unit} ({label}):"]
    wc = len(lines[0].split())
    while wc < approx_words:
        ev = r.choice(events)
        ref = f"DOS-{r.randint(100000, 999999)}"
        line = (
            f"* Ref {ref}: {r.choice(OPERATORS)} completed {ev} at {r.uniform(5.0, 390.0):.2f} "
            f"(dev {r.uniform(0.02, 8.40):.2f}), disposition {r.choice(OUTCOMES)}."
        )
        lines.append(line)
        wc += len(line.split())
    return u, "\n".join(lines)


def generate_skill_a_example(
    tokenizer: Any,
    *,
    split: str,
    index: int,
    training_template: str | None = None,
) -> dict[str, Any]:
    """Generate and deterministically verify one Skill A example."""
    validate_skill_a_catalog()
    item_id = f"skillA-{split}-{index:05d}"
    r = _rng_for(f"{split}:{index}")

    # Deterministic balanced stratification across bands, needle counts, modes, prefix, and styles.
    band_name, target_prompt_tok = BAND_TARGETS[index % len(BAND_TARGETS)]
    n_needles = NEEDLE_SCHEDULE[(index // len(BAND_TARGETS)) % len(NEEDLE_SCHEDULE)]
    enable_thinking = (index % 2) == 1  # exact 50% /nothink, 50% /think
    use_prefix = ((index // 2) % 2) == 0  # exact 50% prepend token, 50% direct copy
    target_k = ((index * 7 + r.randint(0, n_needles - 1)) % n_needles) + 1  # uniform over 1..N

    dom_idx = (index + r.randint(0, len(SKILL_A_DOMAINS) - 1)) % len(SKILL_A_DOMAINS)
    label, fac_prefix, _ = SKILL_A_DOMAINS[dom_idx]
    entity_name = f"{fac_prefix} {r.choice(ZONES)}-{r.randint(100, 999)}"

    used_codes: set[str] = set()
    needles: list[tuple[str, str, dict[str, str]]] = []
    for occ in range(1, n_needles + 1):
        for attempt in range(25):
            style = STYLES[(index + occ + attempt) % len(STYLES)]
            cand = _make_styled_needle(r, dom_idx, entity_name, occ, style, used_codes)
            if all(
                SequenceMatcher(None, prev[1], cand[1]).ratio() < 0.82
                and SequenceMatcher(None, cand[1], prev[1]).ratio() < 0.82
                for prev in needles
            ):
                needles.append(cand)
                break
        else:
            raise RuntimeError(f"{item_id}: could not sample distinct needle #{occ}")

    # Calibrate filler size so total rendered prompt falls cleanly inside [8_000, 64_000]
    # and near target_prompt_tok. Empirical ratio ~1.55 tokens/word on structured logs.
    needle_words = sum(len(u.split()) + len(a.split()) for u, a, _ in needles)
    est_needle_toks = int(needle_words * 1.6) + n_needles * 24 + 250
    remaining_toks = max(5_000, target_prompt_tok - est_needle_toks)
    # Place n_needles + 1 filler slots interspersed across early, middle, and late positions.
    n_slots = n_needles + 1
    words_per_slot = max(80, int((remaining_toks / 1.65) / n_slots))

    def build_messages(slot_words: int) -> tuple[list[dict[str, Any]], str, str, str | None]:
        rr = _rng_for(f"{split}:{index}:fill:{slot_words}")
        msgs: list[dict[str, Any]] = []
        batch_id = 1
        for slot in range(n_slots):
            # Vary filler lengths moderately across early/middle/late turns
            w = max(50, int(slot_words * rr.uniform(0.75, 1.25)))
            fu, fa = _make_filler_turn(rr, dom_idx, batch_id, w)
            batch_id += 1
            msgs.append({"role": "user", "content": fu})
            msgs.append({"role": "assistant", "content": fa})
            if slot < n_needles:
                nu, na, _ = needles[slot]
                msgs.append({"role": "user", "content": nu})
                msgs.append({"role": "assistant", "content": na})

        prefix_str = ""
        if use_prefix:
            chars = string.ascii_letters + string.digits
            plen = rr.randint(8, 12)
            prefix_str = "".join(rr.choice(chars) for _ in range(plen))

        ord_txt = ORDINAL_WORDS[target_k]
        if use_prefix:
            final_q = (
                f"Prepend {prefix_str} to the {ord_txt} {label} for {entity_name} from our conversation above. "
                f"Output the exact verbatim text immediately after the prefix with no extra commentary."
            )
            expected_answer = prefix_str + needles[target_k - 1][1]
        else:
            final_q = (
                f"Copy the {ord_txt} {label} for {entity_name} from our conversation above verbatim. "
                f"Output only the exact passage with no extra commentary."
            )
            expected_answer = needles[target_k - 1][1]

        msgs.append({"role": "user", "content": final_q})

        reasoning: str | None = None
        if enable_thinking:
            anchor_lines = [f"{i + 1}. {meta['anchor']}" for i, (_, _, meta) in enumerate(needles)]
            reasoning = (
                f"Listing all {n_needles} {label} entries for {entity_name} in order:\n"
                + "\n".join(anchor_lines)
                + f"\nTarget is #{target_k} ({needles[target_k - 1][2]['code']}); returning exact text."
            )
            msgs.append({"role": "assistant", "reasoning_content": reasoning, "content": expected_answer})
        else:
            msgs.append({"role": "assistant", "content": expected_answer})
        return msgs, expected_answer, prefix_str, reasoning

    # Up to 3 calibration passes to guarantee rendered prompt tokens are strictly in [8_000, 64_000]
    # and near the band target.
    slot_words = words_per_slot
    encoded: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = []
    expected_answer = ""
    prefix_str = ""
    reasoning: str | None = None

    for _attempt in range(4):
        messages, expected_answer, prefix_str, reasoning = build_messages(slot_words)
        encoded = render_shifted_row(
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
            select_turns="last",
            training_template=training_template,
        )
        prompt_tokens = encoded["selected_spans"][0][0]
        total_tokens = encoded["sequence_tokens"]
        lo_band = max(8_000, int(target_prompt_tok * 0.82))
        hi_band = min(64_000, int(target_prompt_tok * 1.15))
        if lo_band <= prompt_tokens <= hi_band and total_tokens <= 65_536:
            break
        scale = target_prompt_tok / max(prompt_tokens, 1)
        slot_words = max(35, int(slot_words * scale))

    assert encoded is not None
    example = {
        "id": item_id,
        "skill": "skill_a_ordinal_retrieval",
        "split": split,
        "seed_namespace": SKILL_A_SEED_NAMESPACE,
        "domain": label,
        "entity": entity_name,
        "band": band_name,
        "n_needles": n_needles,
        "target_k": target_k,
        "mode": "think" if enable_thinking else "nothink",
        "has_prefix": use_prefix,
        "prefix": prefix_str,
        "needle_codes": [m["code"] for _, _, m in needles],
        "needle_texts": [a for _, a, _ in needles],
        "expected_answer": expected_answer,
        "reasoning_content": reasoning or "",
        "messages": messages,
        "prompt_tokens": encoded["selected_spans"][0][0],
        "sequence_tokens": encoded["sequence_tokens"],
        "target_tokens": encoded["target_tokens"],
        "selected_spans": encoded["selected_spans"],
        "input_ids": encoded["input_ids"],
        "loss_weights": encoded["loss_weights"],
        "rendered_text": encoded["rendered_text"],
        "decoded_targets": encoded["decoded_targets"],
    }
    verify_skill_a_example(example, tokenizer)
    return example


def verify_skill_a_example(example: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Deterministic 100% gate verifier for Skill A examples (zero LLM judge)."""
    dom = example["domain"].lower()
    if dom in BLOCKED_PROBE1_DOMAINS:
        raise ValueError(f"Domain collides with Probe 1: {dom}")
    for genre in BLOCKED_MRCR_GENRES:
        if genre in dom:
            raise ValueError(f"Domain collides with MRCR genre {genre}: {dom}")

    prompt_toks = example["prompt_tokens"]
    seq_toks = example["sequence_tokens"]
    if not (8_000 <= prompt_toks <= 64_000):
        raise ValueError(f"{example['id']}: prompt_tokens={prompt_toks} outside [8000, 64000]")
    if seq_toks > 65_536:
        raise ValueError(f"{example['id']}: sequence_tokens={seq_toks} > 65536")

    n_needles = example["n_needles"]
    k = example["target_k"]
    if n_needles not in (4, 6, 8, 12, 16) or not (1 <= k <= n_needles):
        raise ValueError(f"{example['id']}: invalid N={n_needles}, k={k}")

    codes = example["needle_codes"]
    texts = example["needle_texts"]
    if len(codes) != n_needles or len(set(codes)) != n_needles:
        raise ValueError(f"{example['id']}: needle codes not pairwise unique")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
            if ratio >= 0.85:
                raise ValueError(f"{example['id']}: needles {i} and {j} too similar (ratio={ratio:.3f})")

    # Verify shifted mask decodes exact terminal assistant turn + EOT.
    ids = example["input_ids"]
    weights = example["loss_weights"]
    spans = example["selected_spans"]
    if len(spans) != 1:
        raise ValueError(f"{example['id']}: Skill A must supervise exactly 1 (terminal) turn, got {len(spans)}")
    if any(weights[: spans[0][0] - 1]):
        raise ValueError(f"{example['id']}: non-terminal turn has non-zero loss weight")

    decoded = tokenizer.decode(
        [ids[i + 1] for i in range(len(ids) - 1) if weights[i] > 0.0],
        skip_special_tokens=False,
    )
    if not decoded.endswith("<|eot_id|>"):
        raise ValueError(f"{example['id']}: decoded target missing <|eot_id|>")
    body = decoded[: -len("<|eot_id|>")]

    mode = example["mode"]
    if mode == "think":
        if "Reasoning: /think" not in example["rendered_text"][:200]:
            raise ValueError(f"{example['id']}: /think item missing Reasoning: /think header")
        if body.count("<|start_think|>") != 1 or body.count("<|end_think|>") != 1:
            raise ValueError(f"{example['id']}: /think target must have exactly one think span")
        m = re.match(r"^<\|start_think\|>(.*?)<\|end_think\|>(.*)$", body, re.S)
        if not m:
            raise ValueError(f"{example['id']}: malformed think span structure")
        think_text, final_ans = m.group(1), m.group(2)
        think_toks = len(tokenizer.encode(think_text, add_special_tokens=False))
        if think_toks > 512:
            raise ValueError(f"{example['id']}: think span {think_toks} tokens exceeds 512 cap")
        enum_lines = re.findall(r"(?m)^\d+\.\s+", think_text)
        if len(enum_lines) != n_needles:
            raise ValueError(f"{example['id']}: enumerated count {len(enum_lines)} != N={n_needles}")
    elif mode == "nothink":
        if "Reasoning: /nothink" not in example["rendered_text"][:200]:
            raise ValueError(f"{example['id']}: /nothink item missing Reasoning: /nothink header")
        if "<|start_think|>" in body or "<|end_think|>" in body:
            raise ValueError(f"{example['id']}: /nothink target contains think markers")
        final_ans = body
    else:
        raise ValueError(f"{example['id']}: unknown mode {mode!r}")

    expected = example["expected_answer"]
    if final_ans != expected:
        raise ValueError(f"{example['id']}: final answer mismatch against exact k-th passage")
    if example["has_prefix"] and not final_ans.startswith(example["prefix"]):
        raise ValueError(f"{example['id']}: missing exact prefix")

    if ids[-1] != 128001 or weights[-1] != 0.0:
        raise ValueError(f"{example['id']}: missing EOS=128001 boundary at weight 0")
    if ids[spans[0][1] - 1] != EOT_ID:
        raise ValueError(f"{example['id']}: terminal span does not end with EOT={EOT_ID}")

    return {"passed": True, "id": example["id"], "prompt_tokens": prompt_toks, "target_tokens": example["target_tokens"]}


def to_parquet_row(example: dict[str, Any]) -> dict[str, Any]:
    """Strip transient list fields into Parquet-friendly provenance + int32/float32 tensors."""
    return {
        "id": example["id"],
        "skill": example["skill"],
        "split": example["split"],
        "seed_namespace": example["seed_namespace"],
        "domain": example["domain"],
        "entity": example["entity"],
        "band": example["band"],
        "n_needles": int(example["n_needles"]),
        "target_k": int(example["target_k"]),
        "mode": example["mode"],
        "has_prefix": bool(example["has_prefix"]),
        "prompt_tokens": int(example["prompt_tokens"]),
        "sequence_tokens": int(example["sequence_tokens"]),
        "target_tokens": int(example["target_tokens"]),
        "messages_json": json.dumps(example["messages"], ensure_ascii=False),
        "input_ids": example["input_ids"],
        "loss_weights": example["loss_weights"],
    }
