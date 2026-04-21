# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M2 Stage 1: propose typed clause-pair edges over the 46 OpenAI Model Spec statements.

For each of the C(46, 2) = 1035 statement pairs, ask a judge LLM to classify
the pair as one of:

    compatible            — no meaningful interaction
    soft_tension          — both legitimate defaults that can pull in opposite
                            directions on a single prompt
    possible_contradiction — statements that can be mutually exclusive on some prompts
    scope_overlap         — activation conditions overlap without clean resolution
    possible_shadowing    — one statement's conditions are a subset of the other's

This is the "proposal layer" per gpt5_paper_plan.md. Recall matters more than
precision; Stage 2 (ideation mining) and Stage 4 (trade-off validation) will
do the actual filtering.

Usage:
    source .env && \\
        uv run python experiments/posttrain/stage1_pair_propose.py \\
            --spec experiments/posttrain/specs/openai_model_spec.jsonl \\
            --output experiments/posttrain/stage1_output/pairs_gpt41.jsonl \\
            --judge-model gpt-4.1 \\
            --workers 32
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger("stage1_pair_propose")

EDGE_TYPES_5LEVEL = [
    "compatible",
    "soft_tension",
    "possible_contradiction",
    "scope_overlap",
    "possible_shadowing",
]

EDGE_TYPES_3LEVEL = ["no_tension", "possible_tension", "strong_tension"]

SEVERITIES = ["none", "low", "medium", "high"]

SYSTEM_PROMPT_5LEVEL = """You are analyzing a behavioral specification for a language model.

You will be given two statements from the spec. Your job is to decide whether, and how, they might interact when both apply to the same user prompt.

Classify the pair into exactly one edge type:

- compatible: the two statements do not meaningfully interact. Following one is independent of following the other.
- soft_tension: both are legitimate defaults, but on some prompts they pull in opposite directions (e.g. "be warm" vs "be calibrated/truthful" on an anxious user asking about an uncertain outcome).
- possible_contradiction: the statements can be mutually exclusive on some concrete prompts — satisfying one requires violating the other.
- scope_overlap: the activation conditions overlap without a clean rule for which statement governs.
- possible_shadowing: one statement's activation conditions are a strict subset of the other's, so the narrower one may never fire in practice.

Also assign a severity (none/low/medium/high) for how often and how sharply the interaction is likely to surface on realistic user prompts.

If you pick anything other than "compatible", provide a ONE-SENTENCE concrete example scenario where both statements would simultaneously apply to a user's message. The scenario will be used to mine training data, so it should be specific.

Return strict JSON with this schema:

{
  "edge_type": "compatible" | "soft_tension" | "possible_contradiction" | "scope_overlap" | "possible_shadowing",
  "severity": "none" | "low" | "medium" | "high",
  "rationale": "<= 2 sentences explaining the classification",
  "example_scenario": "<= 1 sentence, empty string if edge_type is compatible"
}

No prose outside the JSON object."""


SYSTEM_PROMPT_3LEVEL = """You are analyzing a behavioral specification for a language model.

You will be given two statements from the spec. Your job is to decide, for a realistic user prompt that might activate both statements, HOW MUCH they trade off against each other.

Pick exactly one of these three ordinal levels:

- no_tension: A response can satisfy both statements straightforwardly. They do not meaningfully constrain each other on any realistic user prompt. (Also use this if one statement strictly shadows the other, so the narrower one never fires in practice — there is no trade-off.)
- possible_tension: On some user prompts, satisfying both statements well requires balancing competing pulls. A careful, well-crafted response CAN still satisfy both; it just takes work to navigate the trade-off.
- strong_tension: There exist realistic user prompts where a single response cannot fully honor both statements simultaneously. The assistant has to pick which to prioritize — the statements are in direct conflict on those prompts.

Critical guidance:

- Ignore authority hierarchy. If statement A structurally overrides B in the chain of command, that does NOT reduce the tension on prompts where both apply — it just tells you which one wins. Report the tension as strong_tension if both would otherwise pull in opposite directions on a realistic prompt.
- Ignore mechanism. Do not try to distinguish "scope overlap" from "mutual exclusion" from "one is a subset of the other." Just rate the strength of the trade-off on realistic prompts.
- Prefer "no_tension" when the statements are about clearly different things that do not activate together on realistic user messages.
- Prefer "strong_tension" only when you can construct a plausible, non-contrived user prompt where a single response has to choose.

If the answer is anything other than no_tension, provide a ONE-SENTENCE concrete example user prompt where both statements would simultaneously apply. The scenario will be used to mine training data, so it must be specific and realistic.

Return strict JSON with this schema:

{
  "edge_type": "no_tension" | "possible_tension" | "strong_tension",
  "rationale": "<= 2 sentences explaining the classification",
  "example_scenario": "<= 1 sentence (empty string if no_tension)"
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
    understanding: Understanding | None = None


def load_understanding(understanding_path: Path) -> Understanding:
    with open(understanding_path) as f:
        u = json.load(f)
    axes = tuple(
        VariationAxis(
            axis=a.get("axis", ""),
            description=a.get("description", ""),
            spectrum=tuple(a.get("spectrum", []) or []),
            why_it_matters=a.get("why_it_matters", ""),
        )
        for a in (u.get("variation_axes") or [])
    )
    return Understanding(
        behavior_understanding=u.get("understanding", ""),
        scientific_motivation=u.get("scientific_motivation", ""),
        variation_axes=axes,
    )


def load_statements(spec_path: Path, understanding_dir: Path | None = None) -> list[Statement]:
    out: list[Statement] = []
    with open(spec_path) as f:
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
            understanding: Understanding | None = None
            if understanding_dir is not None:
                cand = understanding_dir / row["id"] / "understanding.json"
                if not cand.exists():
                    raise FileNotFoundError(f"No understanding.json for {row['id']} at {cand}")
                understanding = load_understanding(cand)
            out.append(
                Statement(
                    id=row["id"],
                    text=row["text"],
                    authority_level=row.get("authority_level", ""),
                    type=row.get("type", ""),
                    section=row.get("section", ""),
                    subsection=row.get("subsection", ""),
                    related_statements=tuple(row.get("related_statements", []) or []),
                    examples=examples,
                    understanding=understanding,
                )
            )
    return out


def render_statement(s: Statement, label: str, include_examples: bool, include_understanding: bool) -> str:
    """Render a statement for the judge prompt."""
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
    for text_line in s.text.splitlines() or [s.text]:
        lines.append(f"    {text_line}")
    if include_examples and s.examples:
        lines.append(f"  examples ({len(s.examples)}):")
        for i, ex in enumerate(s.examples, 1):
            lines.append(f"    - example {i}:")
            if ex.description:
                lines.append(f"        description: {ex.description}")
            lines.append(f"        user_query: {ex.user_query}")
            lines.append(f"        good_response: {ex.good_response}")
            lines.append(f"        bad_response: {ex.bad_response}")
    if include_understanding and s.understanding is not None:
        u = s.understanding
        lines.append("  understanding_stage_output:")
        lines.append("    behavior_understanding: |")
        for t in u.behavior_understanding.splitlines() or [u.behavior_understanding]:
            lines.append(f"      {t}")
        lines.append("    scientific_motivation: |")
        for t in u.scientific_motivation.splitlines() or [u.scientific_motivation]:
            lines.append(f"      {t}")
        if u.variation_axes:
            lines.append(f"    variation_axes ({len(u.variation_axes)}):")
            for ax in u.variation_axes:
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
    include_examples: bool,
    include_understanding: bool,
    taxonomy: str,
    max_retries: int = 3,
) -> dict:
    if taxonomy == "5level":
        system_prompt = SYSTEM_PROMPT_5LEVEL
        valid_edges = EDGE_TYPES_5LEVEL
    elif taxonomy == "3level":
        system_prompt = SYSTEM_PROMPT_3LEVEL
        valid_edges = EDGE_TYPES_3LEVEL
    else:
        raise ValueError(f"Unknown taxonomy: {taxonomy!r}")

    user = (
        render_statement(a, "A", include_examples, include_understanding)
        + "\n\n"
        + render_statement(b, "B", include_examples, include_understanding)
        + "\n\nClassify the (A, B) pair."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    # Newer OpenAI models (gpt-5.x, o-series) replace `max_tokens` with
    # `max_completion_tokens` and reject a temperature override; pick params
    # by model-name prefix.
    is_next_gen = judge_model.startswith(("gpt-5", "o1", "o3", "o4"))
    kwargs: dict = {
        "model": judge_model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if is_next_gen:
        kwargs["max_completion_tokens"] = 2000
    else:
        kwargs["max_tokens"] = 1000
        kwargs["temperature"] = 0.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            if parsed.get("edge_type") not in valid_edges:
                raise ValueError(f"Unknown edge_type {parsed.get('edge_type')!r}")
            if taxonomy == "5level" and parsed.get("severity") not in SEVERITIES:
                raise ValueError(f"Unknown severity {parsed.get('severity')!r}")
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"classify_pair failed after {max_retries} attempts: {last_err}")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--judge-model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N pairs (for smoke tests).")
    ap.add_argument("--resume", action="store_true", help="Skip pair_ids already in the output file.")
    ap.add_argument(
        "--include-examples",
        action="store_true",
        help="Include full spec metadata (examples with user_query/good_response/bad_response, subsection, related_statements, type) in the judge prompt.",
    )
    ap.add_argument(
        "--understanding-dir",
        type=Path,
        default=None,
        help="If set, load <dir>/<statement_id>/understanding.json per statement and include behavior_understanding + scientific_motivation + variation_axes in the judge prompt.",
    )
    ap.add_argument(
        "--taxonomy",
        choices=["5level", "3level"],
        default="5level",
        help="Classification taxonomy. 5level = original (compatible/soft_tension/possible_contradiction/scope_overlap/possible_shadowing + severity). 3level = ordinal (no_tension/possible_tension/strong_tension, no severity).",
    )
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not set. Run with `source .env && ...`.")
        return 2

    statements = load_statements(args.spec, understanding_dir=args.understanding_dir)
    logger.info(
        "Loaded %d statements from %s%s",
        len(statements),
        args.spec,
        f" (+understandings from {args.understanding_dir})" if args.understanding_dir else "",
    )

    pairs = list(itertools.combinations(statements, 2))
    logger.info("Enumerated %d pairs", len(pairs))

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
        logger.info("Resume: %d pair_ids already present in %s", len(done_ids), args.output)

    todo: list[tuple[Statement, Statement]] = []
    for a, b in pairs:
        pair_id = f"{a.id}__{b.id}"
        if pair_id in done_ids:
            continue
        todo.append((a, b))

    if args.limit is not None:
        todo = todo[: args.limit]
    logger.info("Pairs to classify: %d (judge=%s, workers=%d)", len(todo), args.judge_model, args.workers)

    client = OpenAI()

    t_start = time.time()
    n_ok = 0
    n_err = 0

    include_understanding = args.understanding_dir is not None
    if args.include_examples and include_understanding:
        prompt_mode = "rich_understanding"
    elif args.include_examples:
        prompt_mode = "rich"
    elif include_understanding:
        prompt_mode = "understanding_only"
    else:
        prompt_mode = "text_only"
    logger.info("prompt_mode=%s", prompt_mode)

    with open(args.output, "a") as out_f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_pair = {
            pool.submit(
                classify_pair,
                client,
                args.judge_model,
                a,
                b,
                args.include_examples,
                include_understanding,
                args.taxonomy,
            ): (a, b)
            for a, b in todo
        }
        for fut in as_completed(future_to_pair):
            a, b = future_to_pair[fut]
            pair_id = f"{a.id}__{b.id}"
            try:
                parsed = fut.result()
                record = {
                    "pair_id": pair_id,
                    "statement_a_id": a.id,
                    "statement_b_id": b.id,
                    "authority_a": a.authority_level,
                    "authority_b": b.authority_level,
                    "section_a": a.section,
                    "section_b": b.section,
                    "judge_model": args.judge_model,
                    "prompt_mode": prompt_mode,
                    "taxonomy": args.taxonomy,
                    "n_examples_a": len(a.examples),
                    "n_examples_b": len(b.examples),
                    **parsed,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_ok += 1
            except Exception as e:
                logger.warning("pair %s failed: %s", pair_id, e)
                n_err += 1
            total_done = n_ok + n_err
            if total_done % 25 == 0 or total_done == len(todo):
                elapsed = time.time() - t_start
                rate = total_done / elapsed if elapsed > 0 else 0.0
                remaining = (len(todo) - total_done) / rate if rate > 0 else float("inf")
                logger.info(
                    "progress: %d/%d ok=%d err=%d  rate=%.2f/s  eta=%.0fs",
                    total_done, len(todo), n_ok, n_err, rate, remaining,
                )

    logger.info("done: ok=%d err=%d elapsed=%.1fs  output=%s", n_ok, n_err, time.time() - t_start, args.output)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
