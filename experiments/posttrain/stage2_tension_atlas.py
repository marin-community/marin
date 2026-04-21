# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M2 Stage 2 — Tension Atlas: cross-axis tension analysis per statement pair.

For each statement pair in an input working-set, prompt a judge LLM with:
- Both statements' text + examples + understanding (behavior_understanding,
  scientific_motivation, variation axes)
- Variation axes: behavior-specific axes only; demographic axes
  (user_cultural_context, user_demographic_identity) are filtered out.

The judge returns up to 10 tension points per pair, each naming the axes
(from A, B, or both) whose values drive a trade-off, the peak-tension corner,
a concrete example user prompt at that corner, and reasoning. If no real
trade-off exists, returns an empty list.

Usage:
    source .env && uv run python experiments/posttrain/stage2_tension_atlas.py \\
        --spec experiments/posttrain/specs/openai_model_spec.jsonl \\
        --understanding-dir experiments/posttrain/stage1_inputs/understandings_gpt41 \\
        --pairs experiments/posttrain/stage1_output/pairs_gpt41_rich_3lvl.jsonl \\
        --also-pairs experiments/posttrain/stage1_output/pairs_gpt41_full_gpt41u_3lvl.jsonl \\
        --also-pairs experiments/posttrain/stage1_output/pairs_gpt51_rich_3lvl.jsonl \\
        --also-pairs experiments/posttrain/stage1_output/pairs_gpt51_full_gpt41u_3lvl.jsonl \\
        --require-label possible_tension \\
        --output experiments/posttrain/stage2_output/tension_atlas_gpt41.jsonl \\
        --judge-model gpt-4.1 \\
        --workers 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger("stage2_tension_atlas")


DEMOGRAPHIC_AXES = {"user_cultural_context", "user_demographic_identity"}

SYSTEM_PROMPT = """You are analyzing two behavioral statements from a model spec (call them A and B).

Each statement has:
- Text: the rule itself.
- Examples: good_response / bad_response / user_query pairs from the spec.
- Understanding: what the behavior is and why evaluating it matters.
- Behavior-specific variation axes: dimensions along which the behavior manifests differently. Each axis has a name, description, and an ordered spectrum of values.

Your task: identify the specific axis combinations where a trade-off between A and B sharpens — the axis values (from A's axes, from B's axes, or spanning both) that pull a response in opposite directions on realistic prompts.

For each tension point, specify:

- tension_name: a short label (<= 8 words) for this tension corner.
- axes: a list of objects, each describing ONE axis contributing to this tension:
    {
      "axis": "<axis name>",
      "from_statement": "A" or "B",
      "toward_A": "<value on this axis that pulls the response toward satisfying A>",
      "toward_B": "<value on this axis that pulls the response toward satisfying B>"
    }
  Include 1–3 axes per tension point. If a single axis alone produces the tension, that is fine.
- peak_corner: one sentence describing the specific combination of axis values where the trade-off is sharpest.
- example_prompt: a concrete, realistic user message (1–2 sentences) that instantiates the peak corner. A user would plausibly send this.
- reasoning: 1–3 sentences explaining why this axis combination drives tension between A and B.

Return AT MOST 10 tension points, ordered by sharpness and realism. Most pairs will have fewer (0–4).

IMPORTANT: If after careful analysis you believe these two statements do NOT produce any substantial trade-off on realistic prompts, return an empty list. Do not invent tensions.

Return strict JSON:
{
  "tension_points": [
    {
      "tension_name": "...",
      "axes": [{"axis": "...", "from_statement": "A", "toward_A": "...", "toward_B": "..."}],
      "peak_corner": "...",
      "example_prompt": "...",
      "reasoning": "..."
    }
  ]
}

No prose outside the JSON object."""


@dataclass(frozen=True)
class Example:
    description: str
    user_query: str
    good_response: str
    bad_response: str


@dataclass(frozen=True)
class VariationAxis:
    axis: str
    description: str
    spectrum: tuple[str, ...]
    why_it_matters: str


@dataclass(frozen=True)
class Understanding:
    behavior_understanding: str
    scientific_motivation: str
    variation_axes: tuple[VariationAxis, ...]


@dataclass(frozen=True)
class Statement:
    id: str
    text: str
    authority_level: str
    type: str
    section: str
    subsection: str
    related_statements: tuple[str, ...]
    examples: tuple[Example, ...]
    understanding: Understanding


def load_spec(path: Path) -> dict[str, Statement]:
    out: dict[str, Statement] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples_raw = row.get("metadata", {}).get("examples", []) or []
            examples = tuple(
                Example(
                    description=e.get("description", ""),
                    user_query=e.get("user_query", ""),
                    good_response=e.get("good_response", ""),
                    bad_response=e.get("bad_response", ""),
                )
                for e in examples_raw
            )
            out[row["id"]] = Statement(
                id=row["id"],
                text=row["text"],
                authority_level=row.get("authority_level", ""),
                type=row.get("type", ""),
                section=row.get("section", ""),
                subsection=row.get("subsection", ""),
                related_statements=tuple(row.get("related_statements", []) or []),
                examples=examples,
                understanding=Understanding(
                    behavior_understanding="", scientific_motivation="", variation_axes=()
                ),  # filled in later
            )
    return out


def load_understandings(statements: dict[str, Statement], understanding_dir: Path) -> dict[str, Statement]:
    updated: dict[str, Statement] = {}
    for sid, stmt in statements.items():
        u_path = understanding_dir / sid / "understanding.json"
        if not u_path.exists():
            raise FileNotFoundError(f"Missing understanding for {sid}: {u_path}")
        with open(u_path) as f:
            u = json.load(f)
        axes = tuple(
            VariationAxis(
                axis=a.get("axis", ""),
                description=a.get("description", ""),
                spectrum=tuple(a.get("spectrum", []) or []),
                why_it_matters=a.get("why_it_matters", ""),
            )
            for a in (u.get("variation_axes") or [])
            if a.get("axis", "") not in DEMOGRAPHIC_AXES  # FILTER demographic axes
        )
        updated[sid] = Statement(
            id=stmt.id,
            text=stmt.text,
            authority_level=stmt.authority_level,
            type=stmt.type,
            section=stmt.section,
            subsection=stmt.subsection,
            related_statements=stmt.related_statements,
            examples=stmt.examples,
            understanding=Understanding(
                behavior_understanding=u.get("understanding", ""),
                scientific_motivation=u.get("scientific_motivation", ""),
                variation_axes=axes,
            ),
        )
    return updated


def load_unanimous_possible_tension_pairs(
    primary_path: Path,
    also_paths: list[Path],
    required_label: str,
) -> list[str]:
    def load_pair_labels(p: Path) -> dict[str, str]:
        out = {}
        with open(p) as f:
            for line in f:
                row = json.loads(line.strip())
                out[row["pair_id"]] = row["edge_type"]
        return out

    primary = load_pair_labels(primary_path)
    others = [load_pair_labels(p) for p in also_paths]
    unanimous: list[str] = []
    for pid, lbl in primary.items():
        if lbl != required_label:
            continue
        if not all(o.get(pid) == required_label for o in others):
            continue
        unanimous.append(pid)
    return sorted(unanimous)


def render_statement(s: Statement, label: str) -> str:
    lines = [
        f"Statement {label}",
        f"  id: {s.id}",
        f"  authority_level: {s.authority_level}",
        f"  type: {s.type}",
        f"  section: {s.section}",
    ]
    if s.subsection:
        lines.append(f"  subsection: {s.subsection}")
    if s.related_statements:
        lines.append(f"  related_statements: {', '.join(s.related_statements)}")
    lines.append("  text: |")
    for t in s.text.splitlines() or [s.text]:
        lines.append(f"    {t}")
    if s.examples:
        lines.append(f"  examples ({len(s.examples)}):")
        for i, ex in enumerate(s.examples, 1):
            lines.append(f"    - example {i}:")
            if ex.description:
                lines.append(f"        description: {ex.description}")
            lines.append(f"        user_query: {ex.user_query}")
            lines.append(f"        good_response: {ex.good_response}")
            lines.append(f"        bad_response: {ex.bad_response}")
    lines.append("  understanding:")
    lines.append("    behavior_understanding: |")
    for t in s.understanding.behavior_understanding.splitlines() or [s.understanding.behavior_understanding]:
        lines.append(f"      {t}")
    lines.append("    scientific_motivation: |")
    for t in s.understanding.scientific_motivation.splitlines() or [s.understanding.scientific_motivation]:
        lines.append(f"      {t}")
    lines.append(f"    behavior_specific_variation_axes ({len(s.understanding.variation_axes)}):")
    for ax in s.understanding.variation_axes:
        lines.append(f"      - axis: {ax.axis}")
        if ax.description:
            lines.append(f"        description: {ax.description}")
        if ax.spectrum:
            lines.append(f"        spectrum: {list(ax.spectrum)}")
        if ax.why_it_matters:
            lines.append(f"        why_it_matters: {ax.why_it_matters}")
    return "\n".join(lines)


def classify_pair(
    client: OpenAI,
    judge_model: str,
    a: Statement,
    b: Statement,
    max_retries: int = 3,
) -> dict:
    user = (
        render_statement(a, "A")
        + "\n\n"
        + render_statement(b, "B")
        + "\n\nAnalyze the (A, B) pair for tension-driving axis combinations."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    is_next_gen = judge_model.startswith(("gpt-5", "o1", "o3", "o4"))
    kwargs: dict = {
        "model": judge_model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if is_next_gen:
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["max_tokens"] = 3000
        kwargs["temperature"] = 0.0

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            parsed = json.loads(content)
            if "tension_points" not in parsed:
                raise ValueError("Missing 'tension_points' key")
            if not isinstance(parsed["tension_points"], list):
                raise ValueError("'tension_points' must be a list")
            if len(parsed["tension_points"]) > 10:
                # Trim to 10; don't fail
                parsed["tension_points"] = parsed["tension_points"][:10]
            # Basic schema validation per point
            for i, tp in enumerate(parsed["tension_points"]):
                for fld in ("tension_name", "axes", "peak_corner", "example_prompt", "reasoning"):
                    if fld not in tp:
                        raise ValueError(f"tension_points[{i}] missing {fld!r}")
                if not isinstance(tp["axes"], list) or not tp["axes"]:
                    raise ValueError(f"tension_points[{i}].axes must be non-empty list")
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"classify_pair failed after {max_retries} attempts: {last_err}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--understanding-dir", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path, help="Primary Stage-1 output JSONL defining candidate pairs.")
    ap.add_argument(
        "--also-pairs",
        type=Path,
        action="append",
        default=[],
        help="Additional Stage-1 JSONL files; a pair is kept only if it has the required label in ALL of {pairs, also-pairs}.",
    )
    ap.add_argument(
        "--require-label",
        default="possible_tension",
        help="Label that the pair must have in all Stage-1 files. Default: possible_tension.",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--judge-model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not set. Run with `source .env && ...`.")
        return 2

    statements = load_spec(args.spec)
    statements = load_understandings(statements, args.understanding_dir)
    logger.info("Loaded %d statements (understandings + examples, demographic axes filtered).", len(statements))

    pairs = load_unanimous_possible_tension_pairs(args.pairs, args.also_pairs, args.require_label)
    logger.info("Unanimous-%s pairs across %d file(s): %d", args.require_label, 1 + len(args.also_pairs), len(pairs))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["pair_id"])
                except Exception:
                    continue
        logger.info("Resume: %d pair_ids already present", len(done_ids))

    todo: list[str] = [p for p in pairs if p not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit]
    logger.info("Pairs to process: %d (judge=%s, workers=%d)", len(todo), args.judge_model, args.workers)

    client = OpenAI()
    t_start = time.time()
    n_ok, n_err = 0, 0
    zero_tension_count = 0
    tension_points_total = 0

    with open(args.output, "a") as out_f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_pair = {}
        for pid in todo:
            a_id, b_id = pid.split("__", 1)
            a, b = statements.get(a_id), statements.get(b_id)
            if a is None or b is None:
                logger.warning("Skipping %s: missing statement(s)", pid)
                n_err += 1
                continue
            future_to_pair[pool.submit(classify_pair, client, args.judge_model, a, b)] = pid

        for fut in as_completed(future_to_pair):
            pid = future_to_pair[fut]
            a_id, b_id = pid.split("__", 1)
            try:
                parsed = fut.result()
                tp = parsed["tension_points"]
                if not tp:
                    zero_tension_count += 1
                tension_points_total += len(tp)
                record = {
                    "pair_id": pid,
                    "statement_a_id": a_id,
                    "statement_b_id": b_id,
                    "judge_model": args.judge_model,
                    "demographic_axes_filtered": True,
                    "n_tension_points": len(tp),
                    "tension_points": tp,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_ok += 1
            except Exception as e:
                logger.warning("pair %s failed: %s", pid, e)
                n_err += 1
            total = n_ok + n_err
            if total % 25 == 0 or total == len(future_to_pair):
                elapsed = time.time() - t_start
                rate = total / elapsed if elapsed > 0 else 0.0
                eta = (len(future_to_pair) - total) / rate if rate > 0 else float("inf")
                logger.info(
                    "progress: %d/%d ok=%d err=%d zero=%d mean_points=%.2f rate=%.2f/s eta=%.0fs",
                    total, len(future_to_pair), n_ok, n_err,
                    zero_tension_count,
                    tension_points_total / max(n_ok, 1),
                    rate, eta,
                )

    elapsed = time.time() - t_start
    logger.info(
        "done: ok=%d err=%d zero=%d mean_points=%.2f elapsed=%.1fs output=%s",
        n_ok, n_err, zero_tension_count,
        tension_points_total / max(n_ok, 1),
        elapsed, args.output,
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
