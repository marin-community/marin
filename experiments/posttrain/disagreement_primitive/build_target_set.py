# ruff: noqa: B007
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 target-set selector.

Picks 70 stratified target pairs from the Phase 1B `pair_candidate_*.jsonl`
pool per the Codex plan ("Phase 2 - Build a zero-shot target set"):

- 20 dominance-like
- 20 bidirectional_tradeoff
- 10 modifier
- 10 ambiguous
- 10 no_tension controls

Selection rule per bucket: prefer pairs where ≥2 compilers agree on the
relation; within agreement, sort by max confidence; break ties by
canonical pair key. This biases the panel toward high-quality cross-
compiler-agreed cases (cleanest input for the disagreement primitive)
while keeping enough ambiguous/control cases for false-positive
measurement.

The 22 atlas seed pairs (cross-tier + same-class) are *also* emitted as
a separate validation slice (`atlas_validation_set.jsonl`) for the
scenario-bound track, since H2 showed they are scenario-bound rather
than pair-intrinsic.

Output: `target_set.jsonl` (70 PairCandidate-shaped rows with a
`bucket` field) + `atlas_validation_set.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
SEED_PAIRS_PATH = WORKTREE / "experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl"
DEFAULT_INPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"

BUCKETS_DEFAULT = {
    "dominance": 20,
    "bidirectional_tradeoff": 20,
    "modifier": 10,
    "ambiguous": 10,
    "no_tension": 10,
}


def load_spec(path: Path = SPEC_PATH) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in (json.loads(line) for line in path.open() if line.strip())}


def canonical_pair_key(a_id: str, b_id: str) -> tuple[str, str]:
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_seed_pairs(spec: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return atlas seed pairs annotated with cross_tier flag."""
    rows = load_jsonl(SEED_PAIRS_PATH)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        a, b = row["statement_a_id"], row["statement_b_id"]
        if a not in spec or b not in spec:
            continue
        ka = canonical_pair_key(a, b)
        if ka in seen:
            continue
        seen.add(ka)
        a_plat = spec[a]["authority_level"] == "PLATFORM"
        b_plat = spec[b]["authority_level"] == "PLATFORM"
        out.append(
            {
                "statement_a_id": ka[0],
                "statement_b_id": ka[1],
                "cross_tier": a_plat ^ b_plat,
                "tension_point_idx": row.get("tension_point_idx"),
                "tension_name": row.get("tension_point", {}).get("tension_name", ""),
                "example_prompt": row.get("tension_point", {}).get("example_prompt", ""),
            }
        )
    return out


def best_classification_per_compiler(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        ka = canonical_pair_key(r["statement_a_id"], r["statement_b_id"])
        by_pair[ka].append(r)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for ka, rs in by_pair.items():
        non_nt = [r for r in rs if r["predicted_relation"] != "no_tension"]
        winner = (
            max(non_nt, key=lambda r: r.get("confidence", 0.0))
            if non_nt
            else max(rs, key=lambda r: r.get("confidence", 0.0))
        )
        out[ka] = winner
    return out


def merge_compilers(
    runs: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Merge per-compiler best-classifications into a single per-pair view.

    Per pair, attach: agreed_relation (only if all compilers agree),
    relation_votes (compiler->relation map), max_confidence, prefer_record
    (the compiler-record with the highest confidence among non-no_tension).
    """
    all_pairs: set[tuple[str, str]] = set()
    for c in runs.values():
        all_pairs.update(c.keys())
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    compilers = list(runs.keys())
    for ka in all_pairs:
        votes: dict[str, str] = {}
        confs: dict[str, float] = {}
        records: dict[str, dict[str, Any]] = {}
        for c in compilers:
            r = runs[c].get(ka)
            if r is None:
                continue
            votes[c] = r["predicted_relation"]
            confs[c] = float(r.get("confidence", 0.0))
            records[c] = r
        relations = set(votes.values())
        agreed = next(iter(relations)) if len(relations) == 1 and len(votes) == len(compilers) else None
        # Prefer the record from the highest-confidence non-no_tension call.
        preferred_compiler = None
        prefer_record: dict[str, Any] | None = None
        for c in sorted(records, key=lambda c: -confs.get(c, 0.0)):
            if records[c]["predicted_relation"] != "no_tension":
                preferred_compiler = c
                prefer_record = records[c]
                break
        if prefer_record is None:
            preferred_compiler = max(records, key=lambda c: confs.get(c, 0.0)) if records else None
            prefer_record = records.get(preferred_compiler) if preferred_compiler else None
        if prefer_record is None:
            continue
        merged[ka] = {
            "pair": ka,
            "agreed_relation": agreed,
            "votes": votes,
            "confidences": confs,
            "max_confidence": max(confs.values()) if confs else 0.0,
            "preferred_compiler": preferred_compiler,
            "prefer_record": prefer_record,
        }
    return merged


def stratify(
    merged: dict[tuple[str, str], dict[str, Any]],
    buckets: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """For each bucket, return up to N pairs sorted by (agreed, max_conf)."""
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ka, m in merged.items():
        # Use agreed_relation if all compilers agree; else fall back to the
        # preferred record's relation.
        relation = m["agreed_relation"] or m["prefer_record"]["predicted_relation"]
        by_relation[relation].append(m)
    out: dict[str, list[dict[str, Any]]] = {}
    for relation, target_n in buckets.items():
        candidates = by_relation.get(relation, [])
        # Prefer cross-compiler agreement, then high confidence.
        candidates.sort(key=lambda m: (-(1 if m["agreed_relation"] else 0), -m["max_confidence"], m["pair"]))
        out[relation] = candidates[:target_n]
    return out


def to_target_record(m: dict[str, Any], bucket: str) -> dict[str, Any]:
    """Materialize a stratified target row for `target_set.jsonl`."""
    pr = m["prefer_record"]
    return {
        "statement_a_id": m["pair"][0],
        "statement_b_id": m["pair"][1],
        "bucket": bucket,
        "agreed_relation": m["agreed_relation"],
        "compiler_votes": m["votes"],
        "compiler_confidences": m["confidences"],
        "max_confidence": m["max_confidence"],
        "preferred_compiler": m["preferred_compiler"],
        "predicted_relation": pr["predicted_relation"],
        "predicted_controller": pr.get("predicted_controller"),
        "candidate_source": pr.get("candidate_source"),
        "why_pair_matters": pr.get("why_pair_matters", ""),
        "expected_failure_mode": pr.get("expected_failure_mode", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT_DIR / "target_set.jsonl")
    parser.add_argument("--atlas-output", type=Path, default=DEFAULT_INPUT_DIR / "atlas_validation_set.jsonl")
    parser.add_argument("--dominance", type=int, default=20)
    parser.add_argument("--bidirectional", type=int, default=20)
    parser.add_argument("--modifier", type=int, default=10)
    parser.add_argument("--ambiguous", type=int, default=10)
    parser.add_argument("--no-tension", type=int, default=10)
    args = parser.parse_args()

    spec = load_spec()

    runs: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for f in sorted(args.input_dir.glob("pair_candidate_*.jsonl")):
        if "allpairs" in f.stem:
            continue
        rows = load_jsonl(f)
        if not rows:
            continue
        compiler = rows[0].get("classifier_model", f.stem.replace("pair_candidate_", ""))
        runs[compiler] = best_classification_per_compiler(rows)

    if not runs:
        raise SystemExit(f"No pair_candidate_*.jsonl found in {args.input_dir}")

    merged = merge_compilers(runs)
    buckets = {
        "dominance": args.dominance,
        "bidirectional_tradeoff": args.bidirectional,
        "modifier": args.modifier,
        "ambiguous": args.ambiguous,
        "no_tension": args.no_tension,
    }
    selection = stratify(merged, buckets)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for bucket, picks in selection.items():
            for m in picks:
                fh.write(json.dumps(to_target_record(m, bucket), ensure_ascii=False))
                fh.write("\n")
                n_written += 1
    print(f"wrote {args.output} ({n_written} target pairs)")
    for bucket, picks in selection.items():
        target = buckets[bucket]
        agreed = sum(1 for m in picks if m["agreed_relation"])
        print(f"  {bucket}: {len(picks)}/{target} ({agreed} agreed across compilers)")

    seeds = load_seed_pairs(spec)
    args.atlas_output.parent.mkdir(parents=True, exist_ok=True)
    with args.atlas_output.open("w", encoding="utf-8") as fh:
        for s in seeds:
            fh.write(json.dumps(s, ensure_ascii=False))
            fh.write("\n")
    print(f"wrote {args.atlas_output} ({len(seeds)} atlas seed pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
