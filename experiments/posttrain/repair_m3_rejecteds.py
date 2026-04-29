#!/usr/bin/env python3
# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate bounded M3 cross-tier rejecteds while preserving clean chosens.

Gemini policy for this project: use Gemini 3 Flash only. Pro-family Gemini
models are rejected.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from build_m3_cross_tier_dataset import (
    GEMINI_FLASH_MODEL,
    MIN_CHOSEN_SCORE,
    OUTPUT_ROOT,
    REJECTED_MAX_TOKENS,
    THINKING_BUDGET,
    assemble_cross_tier_pairs,
    generate_content,
    git_status_short,
    load_points,
    load_spec,
    make_client,
    pair_to_preference_record,
    same_authority_pairs,
    split_train_val,
    validate_flash_model,
    write_jsonl,
    write_jsonl_gz,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("repair_m3_rejecteds")

SAFER_REJECTED_SYSTEM_PROMPT_TEMPLATE = """You are writing one intentionally flawed assistant response for DPO training.

The response must be worse than the chosen behavior for this cross-tier hierarchy scenario, but it must remain bounded and non-operational.

[Higher-priority statement: {dominant_id}]
{dominant_text}

[Lower-priority statement: {subordinate_id}]
{subordinate_text}

Failure mode to instantiate: {failure_mode}

Rubric sketch for this failure:
{failure_sketch}

Write a natural user-facing assistant response that clearly exhibits the failure mode.

Hard safety limits for this rejected example:
- Do not provide steps, procedures, recipes, code, scripts, checklists, quantities, dosages, thresholds, timelines, tool names, chemical names, exploit details, evasion tactics, target-selection tactics, campaign tactics, legal loophole playbooks, or method details.
- Do not provide self-harm methods or arguments for acting on suicidal thoughts.
- Do not include slurs or explicit hateful abuse.
- Do not write anything that would let the user operationalize illegal activity, self-harm, weaponization, privacy invasion, targeted political manipulation, or other direct harm.
- If the failure sketch calls for dangerous specifics, convert it into a non-operational flawed response that validates, normalizes, or gestures at the unsafe direction without actionable details.

Failure-mode guidance:
- For subordinate_over_satisfaction, over-serve the lower-priority user goal in a bounded way: validate the unsafe framing, offer vague high-level considerations, or imply cooperation, but do not give actionable instructions.
- For pathological_refusal, over-refuse or deflect safe in-scope help and provide little useful content.

Do not mention that this is a rejected example, rubric, test, hierarchy scenario, or training data."""

TARGETED_REJECTED_SYSTEM_PROMPT_TEMPLATE = """You are rewriting one intentionally flawed assistant response that previously failed a safety audit.

The response must be worse than the chosen behavior for this cross-tier hierarchy scenario, but it must be short, bounded, and non-operational.

[Higher-priority statement: {dominant_id}]
{dominant_text}

[Lower-priority statement: {subordinate_id}]
{subordinate_text}

Failure mode to instantiate: {failure_mode}

Rubric sketch for this failure:
{failure_sketch}

Write a natural user-facing assistant response that clearly exhibits the failure mode while staying safe enough to use as a rejected example.

Required shape:
- 2 to 4 short sentences.
- No bullets, numbered lists, headings, tables, code blocks, formulas, scripts, or checklists.
- The flaw should be conceptual: validate the unsafe framing, imply cooperation with the lower-priority goal, or over-refuse safe help.
- Do not provide concrete steps, quantities, names of tools/substances/systems, procedures, tactics, timelines, thresholds, examples to copy, or operational details.
- Do not provide self-harm arguments, method-adjacent self-harm content, illegal instructions, political persuasion tactics, legal loophole playbooks, exploit details, privacy-invasion steps, or weaponization help.
- Do not include slurs or explicit hateful abuse.

Do not mention that this is a rejected example, rubric, test, hierarchy scenario, rewrite, or training data."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_variant_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for row in selected:
        key = (row["pair_id"], row["tension_point_idx"], row["variant_idx"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pair_id": row["pair_id"],
                "tension_point_idx": row["tension_point_idx"],
                "variant_idx": row["variant_idx"],
                "prompt": row["prompt"],
            }
        )
    rows.sort(key=lambda r: (r["pair_id"], r["tension_point_idx"], r["variant_idx"]))
    return rows


def safer_rejected_system(point: Any, spec: dict[str, dict[str, Any]], failure_mode: str) -> str:
    if failure_mode == "subordinate_over_satisfaction":
        sketch = point.worked_example["subordinate_over_satisfaction"]
    elif failure_mode == "pathological_refusal":
        sketch = point.worked_example["pathological_refusal"]
    else:
        raise ValueError(f"unknown failure mode: {failure_mode}")
    return SAFER_REJECTED_SYSTEM_PROMPT_TEMPLATE.format(
        dominant_id=point.dominant_id,
        dominant_text=spec[point.dominant_id]["text"],
        subordinate_id=point.subordinate_id,
        subordinate_text=spec[point.subordinate_id]["text"],
        failure_mode=failure_mode,
        failure_sketch=sketch,
    )


def targeted_rejected_system(point: Any, spec: dict[str, dict[str, Any]], failure_mode: str) -> str:
    if failure_mode == "subordinate_over_satisfaction":
        sketch = point.worked_example["subordinate_over_satisfaction"]
    elif failure_mode == "pathological_refusal":
        sketch = point.worked_example["pathological_refusal"]
    else:
        raise ValueError(f"unknown failure mode: {failure_mode}")
    return TARGETED_REJECTED_SYSTEM_PROMPT_TEMPLATE.format(
        dominant_id=point.dominant_id,
        dominant_text=spec[point.dominant_id]["text"],
        subordinate_id=point.subordinate_id,
        subordinate_text=spec[point.subordinate_id]["text"],
        failure_mode=failure_mode,
        failure_sketch=sketch,
    )


def rejected_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (row["pair_id"], row["tension_point_idx"], row["variant_idx"], row["failure_mode"])


def audit_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (row["pair_id"], row["tension_point_idx"], row["variant_idx"], row["rejected_failure_mode"])


def target_repair_keys(audit_rows: list[dict[str, Any]]) -> set[tuple[str, int, int, str]]:
    keys = set()
    for row in audit_rows:
        if row["rejected_failure_mode"] != "subordinate_over_satisfaction":
            continue
        low_score = min(row["dominant_score"], row["non_leakage_score"]) < MIN_CHOSEN_SCORE
        needs_repair = (
            row["direct_operational_harm"]
            or row["has_slur_or_explicit_hateful_abuse"]
            or row["audit_label"] in {"unsafe_unbounded", "parse_failure", "other_failure"}
            or not row["failure_mode_match"]
            or not low_score
        )
        if needs_repair:
            keys.add(audit_key(row))
    return keys


def generic_bounded_rejected_response(failure_mode: str) -> str:
    if failure_mode == "subordinate_over_satisfaction":
        return (
            "I can prioritize the angle you asked for and downplay the constraint that should limit "
            "the answer, without giving concrete details."
        )
    if failure_mode == "pathological_refusal":
        return "I cannot help with any part of that, even the safe parts, so I will not provide useful guidance."
    raise ValueError(f"unknown failure mode: {failure_mode}")


def generate_repaired_rejecteds(
    selected: list[dict[str, Any]],
    *,
    spec: dict[str, dict[str, Any]],
    model: str,
    max_workers: int,
    retries: int,
    temperature: float,
    output_dir: Path,
    base_rejecteds: list[dict[str, Any]] | None = None,
    target_keys: set[tuple[str, int, int, str]] | None = None,
    generic_targeted_replacements: bool = False,
) -> list[dict[str, Any]]:
    validate_flash_model(model)
    client = None if generic_targeted_replacements else make_client()
    points = {(p.pair_id, p.tension_point_idx): p for p in load_points()}
    variants = selected_variant_rows(selected)
    variant_by_key = {(row["pair_id"], row["tension_point_idx"], row["variant_idx"]): row for row in variants}
    if target_keys is None:
        jobs = [
            (variant, failure_mode)
            for variant in variants
            for failure_mode in ("subordinate_over_satisfaction", "pathological_refusal")
        ]
    else:
        jobs = []
        for pair_id, tension_point_idx, variant_idx, failure_mode in sorted(target_keys):
            variant = variant_by_key.get((pair_id, tension_point_idx, variant_idx))
            if variant is None:
                raise ValueError(
                    f"targeted repair key not found in selected variants: {pair_id} tp={tension_point_idx} v{variant_idx}"
                )
            jobs.append((variant, failure_mode))
    logger.info("generating %d repaired bounded rejecteds from %d selected variants", len(jobs), len(variants))

    def work(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
        variant, failure_mode = job
        point = points[(variant["pair_id"], variant["tension_point_idx"])]
        if generic_targeted_replacements:
            text = generic_bounded_rejected_response(failure_mode)
            usage = {}
            elapsed = 0.0
        else:
            if client is None:
                raise AssertionError("client must be initialized for Gemini-based rejected generation")
            system = (
                targeted_rejected_system(point, spec, failure_mode)
                if target_keys is not None
                else safer_rejected_system(point, spec, failure_mode)
            )
            text, usage, elapsed = generate_content(
                client,
                model=model,
                contents=variant["prompt"],
                system=system,
                max_output_tokens=REJECTED_MAX_TOKENS,
                temperature=temperature,
                json_mode=False,
                retries=retries,
            )
        return {
            "pair_id": point.pair_id,
            "tension_point_idx": point.tension_point_idx,
            "variant_idx": variant["variant_idx"],
            "failure_mode": failure_mode,
            "prompt": variant["prompt"],
            "dominant_id": point.dominant_id,
            "subordinate_id": point.subordinate_id,
            "response": text,
            "usage": usage,
            "elapsed_s": elapsed,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(work, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            logger.info(
                "[repaired rejected %s tp=%d v%d %s] %d chars",
                row["pair_id"],
                row["tension_point_idx"],
                row["variant_idx"],
                row["failure_mode"],
                len(row["response"]),
            )
    if base_rejecteds is not None and target_keys is not None:
        preserved = [row for row in base_rejecteds if rejected_key(row) not in target_keys]
        rows = [*preserved, *rows]
        key_counts = Counter(rejected_key(row) for row in rows)
        duplicates = [key for key, count in key_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate rejected keys after repair: {duplicates[:5]}")
    rows.sort(key=lambda r: (r["pair_id"], r["tension_point_idx"], r["variant_idx"], r["failure_mode"]))
    write_jsonl(output_dir / "cross_tier_rejecteds.jsonl", rows)
    write_jsonl(output_dir / "cross_tier_selected_variants.jsonl", variants)
    return rows


def build_repaired_run(args: argparse.Namespace) -> dict[str, Any]:
    validate_flash_model(args.model)
    run_name = args.run_name or f"m3_flash_rejecteds_repair_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_dir = output_dir / "dataset" / args.dataset_name

    spec = load_spec()
    selected = load_jsonl(args.source_run_dir / "cross_tier_chosens_selected.jsonl")
    write_jsonl(output_dir / "cross_tier_chosens_selected.jsonl", selected)
    target_keys = None
    base_rejecteds = None
    if args.targeted_audit_path is not None:
        audit_rows = load_jsonl(args.targeted_audit_path)
        target_keys = target_repair_keys(audit_rows)
        if not target_keys:
            raise ValueError(f"no target repair keys found in {args.targeted_audit_path}")
        base_rejecteds = load_jsonl(args.source_run_dir / "cross_tier_rejecteds.jsonl")
        logger.info("targeted repair selected %d rejected sides", len(target_keys))

    rejecteds = generate_repaired_rejecteds(
        selected,
        spec=spec,
        model=args.model,
        max_workers=args.max_workers,
        retries=args.retries,
        temperature=args.temperature,
        output_dir=output_dir,
        base_rejecteds=base_rejecteds,
        target_keys=target_keys,
        generic_targeted_replacements=args.generic_targeted_replacements,
    )
    cross_tier_pairs = assemble_cross_tier_pairs(selected, rejecteds)
    same_class = same_authority_pairs(spec, args.same_class_limit)
    all_pairs = [*cross_tier_pairs, *same_class]
    write_jsonl(output_dir / "m3_pairs_raw.jsonl", all_pairs)

    preference_records = [pair_to_preference_record(pair) for pair in all_pairs]
    train_records, val_records = split_train_val(preference_records, args.val_variant_idx)
    write_jsonl_gz(dataset_dir / "train" / "shard-m3.jsonl.gz", train_records)
    write_jsonl_gz(dataset_dir / "val_deduped" / "shard-m3.jsonl.gz", val_records)

    manifest = {
        "run_name": run_name,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source_run_dir": str(args.source_run_dir),
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "model": args.model,
        "thinking_budget": THINKING_BUDGET,
        "repair": "rejected_sides_only",
        "max_workers": args.max_workers,
        "temperature": args.temperature,
        "targeted_audit_path": str(args.targeted_audit_path) if args.targeted_audit_path is not None else None,
        "n_targeted_repaired_rejecteds": len(target_keys) if target_keys is not None else None,
        "generic_targeted_replacements": args.generic_targeted_replacements,
        "git_status_short": git_status_short(),
        "n_selected_chosens": len(selected),
        "n_selected_variants": len(selected_variant_rows(selected)),
        "n_repaired_rejecteds": len(rejecteds),
        "n_cross_tier_pairs": len(cross_tier_pairs),
        "n_same_class_pairs": len(same_class),
        "n_total_pairs": len(all_pairs),
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "bucket_counts": dict(Counter(pair["bucket"] for pair in all_pairs)),
        "cross_tier_rejected_failure_counts": dict(
            Counter(pair.get("rejected_failure_mode", "") for pair in cross_tier_pairs)
        ),
        "same_class_authority_counts": dict(Counter(pair.get("authority_level", "") for pair in same_class)),
        "requires_final_audit": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (dataset_dir / "README.md").write_text(
        f"# {args.dataset_name}\n\n"
        f"M3 hierarchy-aware preference shard with repaired bounded rejecteds built {manifest['created_utc']}.\n\n"
        f"- Gemini model: `{args.model}` with thinking budget {THINKING_BUDGET}.\n"
        f"- Source run: `{args.source_run_dir}`.\n"
        f"- Cross-tier pairs: {len(cross_tier_pairs)}.\n"
        f"- Same-class M2-style pairs: {len(same_class)}.\n"
        f"- Train records: {len(train_records)}.\n"
        f"- Val records: {len(val_records)}.\n"
        f"- Final audit required before upload or training.\n"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=OUTPUT_ROOT / "m3_flash_v1_20260426_0836",
        help="M3 run directory containing clean selected chosens.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--dataset-name", default="bloomv2_m3")
    parser.add_argument("--model", default=GEMINI_FLASH_MODEL)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--same-class-limit", type=int)
    parser.add_argument("--val-variant-idx", type=int, default=0)
    parser.add_argument(
        "--targeted-audit-path",
        type=Path,
        help="Audit per-pair JSONL from the source run; when set, regenerate only failed subordinate-over-satisfaction rejecteds.",
    )
    parser.add_argument(
        "--generic-targeted-replacements",
        action="store_true",
        help="For targeted repair, replace target rows with deterministic generic bounded failure sentences instead of Gemini generations.",
    )
    return parser.parse_args()


def main() -> int:
    build_repaired_run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
