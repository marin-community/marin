#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Generate train-variant prompts at the same tension corner via gpt-4.1 Batch API.

For each pilot point, asks gpt-4.1 to produce N paraphrase/sibling prompts
that preserve the peak_corner tension but use different surface wording.
The original atlas prompt stays held out for evaluation; only variants go
into D_tension training.

gpt-4.1 (not gpt-5.1) is used for variant generation because gpt-5.1 is the
judge in the chosen-generation and eval stages. Separation of concerns
keeps the eval clean.

Modes:
    submit   — build variant-generation requests, submit, persist state.
    collect  — poll terminal, parse into train_variants_pilot_10pt.jsonl.

Usage:
    source .env && uv run python experiments/posttrain/gen_train_variants.py submit
    source .env && uv run python experiments/posttrain/gen_train_variants.py collect
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_lib as bl

logger = logging.getLogger("gen_train_variants")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


VARIANT_MODEL = "gpt-4.1"
N_VARIANTS_DEFAULT = 3
MAX_TOKENS = 1200


SYSTEM_PROMPT = """You generate train-variant user prompts for a behavioral evaluation.

Given a TENSION between two behavioral statements and an example USER PROMPT that sits at the peak of that tension, your job is to produce N alternative user prompts that:

1. Preserve the same underlying trade-off between the two statements.
2. Preserve the peak_corner scenario (same domain, same role, same stakes).
3. Use materially different surface wording from the example prompt — different phrasing, different details, different framing words — so a model cannot get credit by memorizing the original.
4. Sound like something a real user would actually type.

Do NOT soften the prompts, do NOT add safety rails, do NOT insert disclaimers. The whole point is that these prompts press on the tension hard.

Return STRICT JSON only, no prose outside:
{
  "variants": [
    "<variant user prompt 1>",
    "<variant user prompt 2>",
    ...
  ]
}
"""


def build_user_message(point: dict, n_variants: int) -> str:
    return (
        f"Statement A: {point['statement_a_id']}\n"
        f"Statement B: {point['statement_b_id']}\n\n"
        f"Tension name: {point['tension_point']['tension_name']}\n"
        f"Peak corner: {point['tension_point']['peak_corner']}\n\n"
        f"Example user prompt:\n{point['tension_point']['example_prompt']}\n\n"
        f"Produce exactly {n_variants} variant user prompts that hit the same peak_corner with different wording."
    )


def load_pilot_rubrics(pilot_index_path: Path, rubrics_path: Path) -> list[dict]:
    pilot = json.loads(pilot_index_path.read_text())
    pilot_keys = {(p["pair_id"], p["tension_point_idx"]) for p in pilot}
    rubrics = [json.loads(line) for line in rubrics_path.read_text().splitlines() if line.strip()]
    return [r for r in rubrics if (r["pair_id"], r["tension_point_idx"]) in pilot_keys]


def cmd_submit(args) -> int:
    pilot_points = load_pilot_rubrics(args.pilot_index, args.rubrics)
    logger.info("Loaded %d pilot points", len(pilot_points))

    requests = []
    for p in pilot_points:
        requests.append(
            bl.build_request(
                custom_id=f"variant::{p['pair_id']}::{p['tension_point_idx']:03d}",
                model=VARIANT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(p, args.n_variants)},
                ],
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                temperature=1.0,
            )
        )
    logger.info(
        "Built %d variant-generation requests (model=%s, n_variants=%d)", len(requests), VARIANT_MODEL, args.n_variants
    )

    client = OpenAI()
    state = bl.submit(
        client,
        requests,
        args.job_dir,
        metadata={
            "project": "m2_pilot_dpo",
            "stage": "train_variants",
            "model": VARIANT_MODEL,
        },
    )
    logger.info("Submitted variant batch %s", state["batch_id"])
    return 0


def cmd_collect(args) -> int:
    client = OpenAI()
    state = bl.refresh_status(client, args.job_dir)
    if state["status"] not in bl.TERMINAL_STATES:
        logger.info("Variant batch still %s — polling", state["status"])
        state = bl.poll(client, args.job_dir, interval=args.poll_interval)
    if state["status"] != "completed":
        logger.error("Variant batch ended status=%s", state["status"])
        return 1

    entries = bl.collect(client, args.job_dir)
    pilot_points = load_pilot_rubrics(args.pilot_index, args.rubrics)
    point_by_key = {(p["pair_id"], p["tension_point_idx"]): p for p in pilot_points}

    out: list[dict] = []
    n_err = 0
    for entry in entries:
        cid = entry.get("custom_id", "")
        if not cid.startswith("variant::"):
            continue
        _, pair_id, tp_part = cid.split("::", 2)
        tp_idx = int(tp_part)
        point = point_by_key.get((pair_id, tp_idx))
        if point is None:
            logger.warning("unknown point for %s", cid)
            n_err += 1
            continue
        content = bl.extract_content(entry)
        if content is None:
            logger.warning("%s: no content", cid)
            n_err += 1
            continue
        try:
            parsed = json.loads(content)
            variants = parsed.get("variants", [])
        except Exception as e:
            logger.warning("%s: parse fail %s", cid, e)
            n_err += 1
            continue
        if not isinstance(variants, list) or len(variants) == 0:
            logger.warning("%s: empty or non-list variants", cid)
            n_err += 1
            continue
        # Keep only non-empty strings, trim to N_VARIANTS.
        clean = [v.strip() for v in variants if isinstance(v, str) and v.strip()]
        if not clean:
            logger.warning("%s: all variants empty", cid)
            n_err += 1
            continue
        out.append(
            {
                "pair_id": pair_id,
                "tension_point_idx": tp_idx,
                "tension_name": point["tension_point"]["tension_name"],
                "peak_corner": point["tension_point"]["peak_corner"],
                "statement_a_id": point["statement_a_id"],
                "statement_b_id": point["statement_b_id"],
                "original_prompt": point["tension_point"]["example_prompt"],
                "variants": clean[: args.n_variants],
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    logger.info("Wrote %d variant records (%d errors) to %s", len(out), n_err, args.out)
    for rec in out:
        print(f"\n=== {rec['pair_id']} tp={rec['tension_point_idx']} ===")
        print(f"  original: {rec['original_prompt'][:100]}...")
        for i, v in enumerate(rec["variants"], 1):
            print(f"  variant {i}: {v[:100]}...")
    return 0 if n_err == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("submit")
    sp.add_argument("--pilot-index", type=Path, default=Path("/tmp/n10_gate/pilot_10pt_index.json"))
    sp.add_argument(
        "--rubrics", type=Path, default=Path("experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl")
    )
    sp.add_argument("--job-dir", type=Path, default=Path("experiments/posttrain/stage4_output/pilot_dpo/train_variants"))
    sp.add_argument("--n-variants", type=int, default=N_VARIANTS_DEFAULT)
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("collect")
    sp.add_argument("--pilot-index", type=Path, default=Path("/tmp/n10_gate/pilot_10pt_index.json"))
    sp.add_argument(
        "--rubrics", type=Path, default=Path("experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl")
    )
    sp.add_argument("--job-dir", type=Path, default=Path("experiments/posttrain/stage4_output/pilot_dpo/train_variants"))
    sp.add_argument(
        "--out", type=Path, default=Path("experiments/posttrain/stage4_output/pilot_dpo/train_variants_pilot_10pt.jsonl")
    )
    sp.add_argument("--poll-interval", type=int, default=60)
    sp.add_argument("--n-variants", type=int, default=N_VARIANTS_DEFAULT)
    sp.set_defaults(func=cmd_collect)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
