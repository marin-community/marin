#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate + judge oracle chosens for each (pilot point, variant) via gpt-5.1 Batch API.

Two-stage pipeline:

    Stage 1 — gen: for each (pair, tp, variant): draw K chosens from gpt-5.1
        (reasoning_effort="none", temperature=1.0) against that variant prompt.
        Output: chosens_gen.jsonl with {pair_id, tp_idx, variant_idx, draw_idx, response}.

    Stage 2 — judge: score each chosen on BOTH the A_rubric and the B_rubric via
        gpt-5.1 using the same SCORE_SYSTEM_PROMPT/user-template as stage4_bcg_eval.py.
        Then select argmax(min(A,B)) conditional on min(A,B) >= MIN_CHOSEN_SCORE.

    Final output: chosens_pilot_10pt.jsonl with one selected chosen per (pair, tp, variant),
    plus a diagnostic file listing drops (variants with no qualifying chosen).

Subcommands:
    gen-submit, gen-collect, judge-submit, judge-collect, select
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

# Reuse the scoring system prompt + user template from stage4_bcg_eval so chosens
# are judged with exactly the same rubric machinery as the eval pipeline.
from stage4_bcg_eval import SCORE_SYSTEM_PROMPT

logger = logging.getLogger("gen_chosens")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


CHOSEN_MODEL = "gpt-5.1"
JUDGE_MODEL = "gpt-5.1"
K_DRAWS_PER_VARIANT = 5
MIN_CHOSEN_SCORE = 7  # min(A, B) must meet or exceed this
CHOSEN_MAX_TOKENS = 1500
JUDGE_MAX_TOKENS = 1000


CHOSEN_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Respond to the user's message directly "
    "and helpfully. Be concise where appropriate, warm where appropriate, "
    "and honest throughout. Answer in natural prose — no JSON, no preamble "
    "like 'here is my response'. Just the response itself."
)


def gen_custom_id(pair_id: str, tp: int, variant_idx: int, draw_idx: int) -> str:
    return f"chosen_gen::{pair_id}::{tp:03d}::v{variant_idx:02d}::d{draw_idx:02d}"


def judge_custom_id(pair_id: str, tp: int, variant_idx: int, draw_idx: int, side: str) -> str:
    return f"chosen_judge::{pair_id}::{tp:03d}::v{variant_idx:02d}::d{draw_idx:02d}::{side}"


def parse_gen_id(cid: str) -> tuple[str, int, int, int]:
    _, pair_id, tp_part, v_part, d_part = cid.split("::", 4)
    return pair_id, int(tp_part), int(v_part[1:]), int(d_part[1:])


def parse_judge_id(cid: str) -> tuple[str, int, int, int, str]:
    _, pair_id, tp_part, v_part, d_part, side = cid.split("::", 5)
    return pair_id, int(tp_part), int(v_part[1:]), int(d_part[1:]), side


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_variants_and_rubrics(variants_path: Path, rubrics_path: Path) -> list[dict]:
    """Returns one record per (point, variant_idx) with rubrics attached."""
    variants = load_jsonl(variants_path)
    rubrics = load_jsonl(rubrics_path)
    rubric_by = {(r["pair_id"], r["tension_point_idx"]): r for r in rubrics}
    expanded = []
    for v in variants:
        rubric = rubric_by.get((v["pair_id"], v["tension_point_idx"]))
        if rubric is None:
            logger.warning("no rubric for %s tp=%d", v["pair_id"], v["tension_point_idx"])
            continue
        for vi, variant_prompt in enumerate(v["variants"]):
            expanded.append(
                {
                    "pair_id": v["pair_id"],
                    "tension_point_idx": v["tension_point_idx"],
                    "variant_idx": vi,
                    "variant_prompt": variant_prompt,
                    "peak_corner": v["peak_corner"],
                    "tension_name": v["tension_name"],
                    "statement_a_id": v["statement_a_id"],
                    "statement_b_id": v["statement_b_id"],
                    "A_rubric": rubric["A_rubric"],
                    "B_rubric": rubric["B_rubric"],
                }
            )
    return expanded


# ---- Stage 1: chosen generation ----


def cmd_gen_submit(args) -> int:
    rows = load_variants_and_rubrics(args.variants, args.rubrics)
    logger.info("Expanded %d (point, variant) rows", len(rows))
    requests = []
    for row in rows:
        for d in range(args.k_draws):
            requests.append(
                bl.build_request(
                    custom_id=gen_custom_id(row["pair_id"], row["tension_point_idx"], row["variant_idx"], d),
                    model=CHOSEN_MODEL,
                    messages=[
                        {"role": "system", "content": CHOSEN_SYSTEM_PROMPT},
                        {"role": "user", "content": row["variant_prompt"]},
                    ],
                    max_tokens=CHOSEN_MAX_TOKENS,
                    reasoning_effort="none" if bl.is_next_gen(CHOSEN_MODEL) else None,
                )
            )
    logger.info(
        "Built %d chosen-generation requests (model=%s, k=%d per variant)", len(requests), CHOSEN_MODEL, args.k_draws
    )

    client = OpenAI()
    state = bl.submit(
        client,
        requests,
        args.job_dir / "gen",
        metadata={"project": "m2_pilot_dpo", "stage": "chosen_gen", "model": CHOSEN_MODEL, "k_draws": str(args.k_draws)},
    )
    logger.info("Submitted chosen-gen batch %s", state["batch_id"])
    return 0


def cmd_gen_collect(args) -> int:
    client = OpenAI()
    jd = args.job_dir / "gen"
    state = bl.refresh_status(client, jd)
    if state["status"] not in bl.TERMINAL_STATES:
        state = bl.poll(client, jd, interval=args.poll_interval)
    if state["status"] != "completed":
        logger.error("Gen batch ended %s", state["status"])
        return 1

    entries = bl.collect(client, jd)
    rows = load_variants_and_rubrics(args.variants, args.rubrics)
    row_by = {(r["pair_id"], r["tension_point_idx"], r["variant_idx"]): r for r in rows}
    out = []
    n_err = 0
    for entry in entries:
        cid = entry.get("custom_id", "")
        if not cid.startswith("chosen_gen::"):
            continue
        try:
            pid, tp, vi, di = parse_gen_id(cid)
        except Exception as e:
            logger.warning("bad custom_id %s: %s", cid, e)
            n_err += 1
            continue
        row = row_by.get((pid, tp, vi))
        if row is None:
            logger.warning("no row for %s", cid)
            n_err += 1
            continue
        content = bl.extract_content(entry)
        if content is None:
            logger.warning("%s: no content", cid)
            n_err += 1
            continue
        out.append(
            {
                "pair_id": pid,
                "tension_point_idx": tp,
                "variant_idx": vi,
                "draw_idx": di,
                "variant_prompt": row["variant_prompt"],
                "response": content,
            }
        )

    gen_path = args.job_dir / "chosens_gen.jsonl"
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    with gen_path.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d chosen generations (%d errors) to %s", len(out), n_err, gen_path)
    return 0 if n_err == 0 else 1


# ---- Stage 2: chosen judging ----


def build_judge_user_content(statement_id: str, rubric: dict, peak_corner: str, prompt: str, response: str) -> str:
    return (
        f"Statement to score against: {statement_id}\n\n"
        f"Rubric:\n"
        f"GOOD: {rubric['GOOD']}\n"
        f"BAD: {rubric['BAD']}\n"
        f"KEY_TENSION: {rubric['KEY_TENSION']}\n\n"
        f"Scenario (peak_corner): {peak_corner}\n\n"
        f"User prompt: {prompt}\n\n"
        f"Assistant response to score:\n{response}\n\n"
        f"Score the response 0-10 against this rubric only."
    )


def cmd_judge_submit(args) -> int:
    gen_path = args.job_dir / "chosens_gen.jsonl"
    gens = load_jsonl(gen_path)
    rows = load_variants_and_rubrics(args.variants, args.rubrics)
    row_by = {(r["pair_id"], r["tension_point_idx"], r["variant_idx"]): r for r in rows}

    requests = []
    for g in gens:
        row = row_by[(g["pair_id"], g["tension_point_idx"], g["variant_idx"])]
        for side, rubric_key in (("A", "A_rubric"), ("B", "B_rubric")):
            stmt_id = row[f"statement_{side.lower()}_id"]
            rubric = row[rubric_key]
            user_content = build_judge_user_content(
                stmt_id, rubric, row["peak_corner"], g["variant_prompt"], g["response"]
            )
            requests.append(
                bl.build_request(
                    custom_id=judge_custom_id(
                        g["pair_id"], g["tension_point_idx"], g["variant_idx"], g["draw_idx"], side
                    ),
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": SCORE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=JUDGE_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    reasoning_effort="none" if bl.is_next_gen(JUDGE_MODEL) else None,
                )
            )
    logger.info("Built %d chosen-judge requests (judge=%s)", len(requests), JUDGE_MODEL)
    client = OpenAI()
    state = bl.submit(
        client,
        requests,
        args.job_dir / "judge",
        metadata={
            "project": "m2_pilot_dpo",
            "stage": "chosen_judge",
            "judge_model": JUDGE_MODEL,
            "num_generations": str(len(gens)),
        },
    )
    logger.info("Submitted chosen-judge batch %s", state["batch_id"])
    return 0


def cmd_judge_collect(args) -> int:
    client = OpenAI()
    jd = args.job_dir / "judge"
    state = bl.refresh_status(client, jd)
    if state["status"] not in bl.TERMINAL_STATES:
        state = bl.poll(client, jd, interval=args.poll_interval)
    if state["status"] != "completed":
        logger.error("Judge batch ended %s", state["status"])
        return 1

    entries = bl.collect(client, jd)
    out = []
    n_err = 0
    for entry in entries:
        cid = entry.get("custom_id", "")
        if not cid.startswith("chosen_judge::"):
            continue
        try:
            pid, tp, vi, di, side = parse_judge_id(cid)
        except Exception as e:
            logger.warning("bad custom_id %s: %s", cid, e)
            n_err += 1
            continue
        content = bl.extract_content(entry)
        if content is None:
            logger.warning("%s: no content", cid)
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
                "draw_idx": di,
                "side": side,
                "score": score,
                "explanation": explanation,
            }
        )
    scores_path = args.job_dir / "chosens_scores.jsonl"
    with scores_path.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d chosen scores (%d errors) to %s", len(out), n_err, scores_path)
    return 0 if n_err == 0 else 1


# ---- Stage 3: selection ----


def cmd_select(args) -> int:
    gens = load_jsonl(args.job_dir / "chosens_gen.jsonl")
    scores = load_jsonl(args.job_dir / "chosens_scores.jsonl")

    score_by = {
        (s["pair_id"], s["tension_point_idx"], s["variant_idx"], s["draw_idx"], s["side"]): s["score"] for s in scores
    }

    # Group gens by (pair, tp, variant_idx), attach scores.
    by_variant: dict[tuple, list[dict]] = {}
    for g in gens:
        key = (g["pair_id"], g["tension_point_idx"], g["variant_idx"])
        sa = score_by.get((*key, g["draw_idx"], "A"))
        sb = score_by.get((*key, g["draw_idx"], "B"))
        if sa is None or sb is None:
            continue
        enriched = {**g, "score_A": sa, "score_B": sb, "min_side": min(sa, sb)}
        by_variant.setdefault(key, []).append(enriched)

    selected = []
    dropped = []
    for key, draws in sorted(by_variant.items()):
        qualifying = [d for d in draws if d["min_side"] >= args.min_score]
        if not qualifying:
            dropped.append(
                {
                    "pair_id": key[0],
                    "tension_point_idx": key[1],
                    "variant_idx": key[2],
                    "reason": "no_draw_above_min_score",
                    "draws_scored": len(draws),
                    "best_min_side": max(d["min_side"] for d in draws) if draws else None,
                }
            )
            continue
        # Prefer highest min(A,B); break ties by higher sum, then lower draw_idx.
        qualifying.sort(key=lambda d: (-d["min_side"], -(d["score_A"] + d["score_B"]), d["draw_idx"]))
        # Take top-K per variant.
        for best in qualifying[: args.top_k]:
            selected.append(
                {
                    "pair_id": best["pair_id"],
                    "tension_point_idx": best["tension_point_idx"],
                    "variant_idx": best["variant_idx"],
                    "draw_idx": best["draw_idx"],
                    "variant_prompt": best["variant_prompt"],
                    "chosen_response": best["response"],
                    "chosen_score_A": best["score_A"],
                    "chosen_score_B": best["score_B"],
                    "chosen_min_side": best["min_side"],
                    "n_drawn": len(draws),
                    "n_qualifying": len(qualifying),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in selected:
            f.write(json.dumps(rec) + "\n")
    drops_path = args.out.parent / (args.out.stem + "_drops.json")
    drops_path.write_text(json.dumps(dropped, indent=2))

    logger.info("Selected %d chosens; dropped %d variants (no qualifying draw)", len(selected), len(dropped))
    print(f"\nSelected {len(selected)} chosens; dropped {len(dropped)} variants.")
    for s in selected:
        print(
            f"  {s['pair_id']} tp={s['tension_point_idx']} v={s['variant_idx']} "
            f"A={s['chosen_score_A']} B={s['chosen_score_B']} "
            f"min={s['chosen_min_side']} ({s['n_qualifying']}/{s['n_drawn']} qualified)"
        )
    if dropped:
        print("\nDROPS:")
        for d in dropped:
            print(
                f"  {d['pair_id']} tp={d['tension_point_idx']} v={d['variant_idx']}: "
                f"best_min_side={d.get('best_min_side')}"
            )
    return 0


# ---- CLI ----


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    defaults = {
        "variants": Path("experiments/posttrain/stage4_output/pilot_dpo/train_variants_pilot_10pt.jsonl"),
        "rubrics": Path("experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl"),
        "job_dir": Path("experiments/posttrain/stage4_output/pilot_dpo/chosens"),
    }

    for name, fn in [
        ("gen-submit", cmd_gen_submit),
        ("gen-collect", cmd_gen_collect),
        ("judge-submit", cmd_judge_submit),
        ("judge-collect", cmd_judge_collect),
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("--variants", type=Path, default=defaults["variants"])
        sp.add_argument("--rubrics", type=Path, default=defaults["rubrics"])
        sp.add_argument("--job-dir", type=Path, default=defaults["job_dir"])
        sp.add_argument("--k-draws", type=int, default=K_DRAWS_PER_VARIANT)
        sp.add_argument("--poll-interval", type=int, default=60)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("select")
    sp.add_argument("--job-dir", type=Path, default=defaults["job_dir"])
    sp.add_argument(
        "--out", type=Path, default=Path("experiments/posttrain/stage4_output/pilot_dpo/chosens_pilot_10pt.jsonl")
    )
    sp.add_argument("--min-score", type=int, default=MIN_CHOSEN_SCORE)
    sp.add_argument("--top-k", type=int, default=1, help="How many chosens to keep per variant (default 1).")
    sp.set_defaults(func=cmd_select)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
