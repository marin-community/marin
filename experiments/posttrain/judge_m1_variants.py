#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge M1-on-variants generations on paired A/B rubrics via gpt-5.1 batch.

Custom_id schema includes variant_idx so each variant's 10 samples are tracked
separately:
    m1var_score::<pair_id>::<tp:03d>::v<vi:02d>::s<si:02d>::<side>

Subcommands: submit, collect.
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

from stage4_bcg_eval import SCORE_SYSTEM_PROMPT

logger = logging.getLogger("judge_m1_variants")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


JUDGE_MODEL = "gpt-5.1"
MAX_TOKENS = 1000


def custom_id(pair_id: str, tp: int, vi: int, si: int, side: str) -> str:
    return f"m1var_score::{pair_id}::{tp:03d}::v{vi:02d}::s{si:02d}::{side}"


def parse_id(cid: str) -> tuple[str, int, int, int, str]:
    _, pair_id, tp_part, v_part, s_part, side = cid.split("::", 5)
    return pair_id, int(tp_part), int(v_part[1:]), int(s_part[1:]), side


def build_user_content(stmt_id: str, rubric: dict, peak_corner: str, prompt: str, response: str) -> str:
    return (
        f"Statement to score against: {stmt_id}\n\n"
        f"Rubric:\n"
        f"GOOD: {rubric['GOOD']}\n"
        f"BAD: {rubric['BAD']}\n"
        f"KEY_TENSION: {rubric['KEY_TENSION']}\n\n"
        f"Scenario (peak_corner): {peak_corner}\n\n"
        f"User prompt: {prompt}\n\n"
        f"Assistant response to score:\n{response}\n\n"
        f"Score the response 0-10 against this rubric only."
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def cmd_submit(args) -> int:
    gens = load_jsonl(args.generations)
    rubrics = load_jsonl(args.rubrics)
    variants = load_jsonl(args.variants)

    rubric_by = {(r["pair_id"], r["tension_point_idx"]): r for r in rubrics}
    variant_prompts = {}
    for v in variants:
        for vi, p in enumerate(v["variants"]):
            variant_prompts[(v["pair_id"], v["tension_point_idx"], vi)] = p

    requests = []
    skipped = 0
    for g in gens:
        key = (g["pair_id"], g["tension_point_idx"])
        rubric_rec = rubric_by.get(key)
        if rubric_rec is None:
            skipped += 1
            continue
        tp = rubric_rec["tension_point"]
        vi = g["variant_idx"]
        si = g["sample_idx"]
        prompt = variant_prompts.get((*key, vi), g.get("prompt", ""))
        for side, rubric_key in (("A", "A_rubric"), ("B", "B_rubric")):
            rubric = rubric_rec[rubric_key]
            stmt_id = rubric_rec[f"statement_{side.lower()}_id"]
            user_content = build_user_content(stmt_id, rubric, tp["peak_corner"], prompt, g["response"])
            requests.append(
                bl.build_request(
                    custom_id=custom_id(g["pair_id"], g["tension_point_idx"], vi, si, side),
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": SCORE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                    reasoning_effort="none" if bl.is_next_gen(JUDGE_MODEL) else None,
                )
            )
    logger.info("Built %d judge requests (skipped %d missing-rubric)", len(requests), skipped)
    client = OpenAI()
    state = bl.submit(
        client,
        requests,
        args.job_dir,
        metadata={"project": "m2_pilot_dpo", "stage": "m1_variants_judge", "judge_model": JUDGE_MODEL},
    )
    logger.info("Submitted M1-variants judge batch %s", state["batch_id"])
    return 0


def cmd_collect(args) -> int:
    client = OpenAI()
    state = bl.refresh_status(client, args.job_dir)
    if state["status"] not in bl.TERMINAL_STATES:
        state = bl.poll(client, args.job_dir, interval=args.poll_interval)
    if state["status"] != "completed":
        logger.error("Batch ended %s", state["status"])
        return 1
    entries = bl.collect(client, args.job_dir)
    out = []
    n_err = 0
    for entry in entries:
        cid = entry.get("custom_id", "")
        if not cid.startswith("m1var_score::"):
            continue
        try:
            pid, tp, vi, si, side = parse_id(cid)
        except Exception as e:
            logger.warning("bad custom_id %s: %s", cid, e)
            n_err += 1
            continue
        content = bl.extract_content(entry)
        if content is None:
            n_err += 1
            continue
        try:
            parsed = json.loads(content)
            score = int(parsed.get("score", -1))
            explanation = parsed.get("explanation", "")
        except Exception as e:
            logger.warning("%s: parse fail %s", cid, e)
            n_err += 1
            continue
        out.append(
            {
                "pair_id": pid,
                "tension_point_idx": tp,
                "variant_idx": vi,
                "sample_idx": si,
                "side": side,
                "score": score,
                "explanation": explanation,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d scores (%d errors) to %s", len(out), n_err, args.out)
    return 0 if n_err == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    defaults = {
        "generations": Path("experiments/posttrain/stage4_output/tier_b/m1_variants/generations.jsonl"),
        "rubrics": Path("experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl"),
        "variants": Path("experiments/posttrain/stage4_output/tier_b/variants/train_variants_tier_b.jsonl"),
        "job_dir": Path("experiments/posttrain/stage4_output/tier_b/m1_variants/judge"),
    }
    for name, fn in [("submit", cmd_submit), ("collect", cmd_collect)]:
        sp = sub.add_parser(name)
        sp.add_argument("--generations", type=Path, default=defaults["generations"])
        sp.add_argument("--rubrics", type=Path, default=defaults["rubrics"])
        sp.add_argument("--variants", type=Path, default=defaults["variants"])
        sp.add_argument("--job-dir", type=Path, default=defaults["job_dir"])
        sp.add_argument("--poll-interval", type=int, default=60)
        sp.add_argument(
            "--out", type=Path, default=Path("experiments/posttrain/stage4_output/tier_b/m1_variants/scores.jsonl")
        )
        sp.set_defaults(func=fn)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
