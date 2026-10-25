# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the harness's all-position log-probs with vLLM's (local, CPU only).

Inputs: ``harness.py positions`` output (``positions.pt``) and ``vllm_reference.py`` output (JSON), and
optionally the task #301 scorer's ``contrastive.jsonl`` for the same checkpoint.

Per group (check, contrastive, long, and the long sequence's positions past the 2,048-token window):

- top-1 agreement over positions. A disagreement where vLLM's top two are within ``NEAR_TIE`` nats is
  counted separately as a near tie.
- abs log-prob difference over each position's vLLM top-20 tokens with p > ``MIN_PROB``: max, p99, p50.
  ``missing`` counts such tokens outside the harness's top 50.

Contrastive candidate scores (the sum of the candidate tokens' log-probs) are compared harness vs vLLM and,
when given, harness vs the scorer, with each side's win count per condition. Per row-condition it also counts
where harness and vLLM agree on whether a correct candidate wins (``rows_won_agree``) and on the top-scoring
candidate (``rows_top_candidate_agree``), and lists every row where either differs.
"""

import argparse
import io
import json
import math
from collections import defaultdict

import torch
from rigging.filesystem.buckets import filesystem_for

from experiments.interp.snowball_identity.prompts import SNOWBALL_V01, load_tokenizer
from experiments.interp.snowball_identity.sequences import LONG_TOKENS, Sequence, build_sequences

MIN_PROB = 1e-3
NEAR_TIE = 0.1
WINDOW = 2048


def read(path: str) -> bytes:
    fs, local = filesystem_for(path)
    return fs.cat_file(local)


def stored_sequences(positions_path: str) -> list[Sequence]:
    """The sequences of a stored ``positions.pt``, with its exact token ids, in stored order.

    The group is the id's prefix (``check``, ``contrastive``, ``long``). Use these, not a fresh ``build_sequences``,
    as model inputs when comparing against the pinned #298 reference."""
    stored = torch.load(io.BytesIO(read(positions_path)), map_location="cpu")
    return [
        Sequence(sequence_id, sequence_id.split("|")[0], tuple(record["tokens"]))
        for sequence_id, record in stored.items()
    ]


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def position_stats(ours: dict, theirs: dict, groups: dict[str, list[tuple[str, int]]]) -> dict:
    stats = {}
    for group, positions in groups.items():
        agree = near_tie = disagree = missing = 0
        diffs: list[float] = []
        for sequence_id, i in positions:
            top = {int(token): value for token, value in theirs[sequence_id]["top"][i].items()}
            ids = ours[sequence_id]["top_ids"][i].tolist()
            values = dict(zip(ids, ours[sequence_id]["top_lp"][i].tolist(), strict=True))
            ranked = sorted(top.values(), reverse=True)
            if ids[0] == max(top, key=top.get):
                agree += 1
            elif len(ranked) > 1 and ranked[0] - ranked[1] < NEAR_TIE:
                near_tie += 1
            else:
                disagree += 1
            for token, value in top.items():
                if value < math.log(MIN_PROB):
                    continue
                if token in values:
                    diffs.append(abs(values[token] - value))
                else:
                    missing += 1
        stats[group] = {
            "positions": len(positions),
            "top1_agree": agree,
            "top1_near_tie": near_tie,
            "top1_disagree": disagree,
            "tokens_compared": len(diffs),
            "max_abs_logprob_diff": max(diffs, default=float("nan")),
            "p99_abs_logprob_diff": quantile(diffs, 0.99),
            "p50_abs_logprob_diff": quantile(diffs, 0.50),
            "missing_from_harness_top50": missing,
        }
    return stats


def contrastive_stats(ours: dict, theirs: dict, scorer_rows: list[dict] | None) -> dict:
    """Candidate scores per (row, condition) from both backends, plus the scorer's when given."""
    rows: dict[tuple[str, str], dict] = defaultdict(dict)
    for sequence_id, record in ours.items():
        if not sequence_id.startswith("contrastive|"):
            continue
        _, row_id, condition, index = sequence_id.split("|")
        (start, end), correct = record["score"]
        rows[(row_id, condition)]["correct"] = set(correct)
        rows[(row_id, condition)][int(index)] = (
            float(record["actual"][start:end].sum()),
            sum(theirs[sequence_id]["actual"][start:end]),
        )
    scorer = {(row["row_id"], row["condition"]): row for row in scorer_rows or []}
    vllm_diffs, scorer_diffs = [], []
    wins: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    won_agree, top_agree, disagreements = 0, 0, []
    for (row_id, condition), candidates in sorted(rows.items()):
        correct = candidates.pop("correct")
        harness_scores = [candidates[i][0] for i in sorted(candidates)]
        vllm_scores = [candidates[i][1] for i in sorted(candidates)]
        vllm_diffs.extend(abs(a - b) for a, b in zip(harness_scores, vllm_scores, strict=True))
        reference = scorer.get((row_id, condition))
        if reference is not None:
            scorer_diffs.extend(abs(a - b) for a, b in zip(harness_scores, reference["logprobs"], strict=True))
            wins[condition]["scorer"] += reference["won"]
        won = {}
        for name, scores in (("harness", harness_scores), ("vllm", vllm_scores)):
            best_wrong = max(score for i, score in enumerate(scores) if i not in correct)
            won[name] = max(scores[i] for i in correct) > best_wrong
            wins[condition][name] += won[name]
        wins[condition]["rows"] += 1
        top = {
            name: max(range(len(scores)), key=scores.__getitem__)
            for name, scores in (("harness", harness_scores), ("vllm", vllm_scores))
        }
        if won["harness"] != won["vllm"] or top["harness"] != top["vllm"]:
            disagreements.append({"row_id": row_id, "condition": condition, "won": won, "top_candidate": top})
        won_agree += won["harness"] == won["vllm"]
        top_agree += top["harness"] == top["vllm"]
    return {
        "row_conditions": len(rows),
        "max_abs_score_diff_vs_vllm": max(vllm_diffs, default=float("nan")),
        "p99_abs_score_diff_vs_vllm": quantile(vllm_diffs, 0.99),
        "max_abs_score_diff_vs_scorer": max(scorer_diffs, default=float("nan")),
        "p99_abs_score_diff_vs_scorer": quantile(scorer_diffs, 0.99),
        "wins": {condition: dict(counts) for condition, counts in wins.items()},
        # Per row-condition, harness vs vLLM: whether a correct candidate wins, and which candidate scores highest.
        "rows_won_agree": won_agree,
        "rows_top_candidate_agree": top_agree,
        "row_disagreements": disagreements,
    }


def compare(positions_path: str, vllm_path: str, scorer_path: str | None, identity_scores: dict) -> dict:
    """``identity_scores`` maps a contrastive sequence id to ((start, end), correct candidate indices)."""
    ours = torch.load(io.BytesIO(read(positions_path)), map_location="cpu")
    theirs = json.loads(read(vllm_path))
    if set(ours) != set(theirs):
        raise ValueError("The harness and vLLM scored different sequence sets")
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for sequence_id, record in ours.items():
        if record["tokens"] != theirs[sequence_id]["tokens"]:
            raise ValueError(f"{sequence_id}: different token IDs")
        record["score"] = identity_scores.get(sequence_id)
        group = sequence_id.split("|")[0]
        for i in range(1, len(record["tokens"])):
            groups[group].append((sequence_id, i))
            if group == "long" and i >= WINDOW:
                groups["long_past_window"].append((sequence_id, i))
    if "long" in groups and len(groups["long"]) != LONG_TOKENS - 1:
        raise ValueError(f"long sequence has {len(groups['long']) + 1} tokens, expected {LONG_TOKENS}")
    scorer_rows = None
    if scorer_path:
        scorer_rows = [json.loads(line) for line in read(scorer_path).decode().splitlines() if line.strip()]
    return {
        "positions": position_stats(ours, theirs, groups),
        "contrastive": contrastive_stats(ours, theirs, scorer_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", required=True, help="harness positions.pt")
    parser.add_argument("--vllm", required=True, help="vllm_reference.py JSON")
    parser.add_argument("--identity", required=True, help="identity.jsonl used to build the sequences")
    parser.add_argument("--scorer", help="task #301 contrastive.jsonl for the same checkpoint")
    args = parser.parse_args()

    spans = {
        s.id: (s.score, s.meta["correct"])
        for s in build_sequences(load_tokenizer(SNOWBALL_V01), args.identity)
        if s.score
    }
    print(json.dumps(compare(args.positions, args.vllm, args.scorer, spans), indent=1))


if __name__ == "__main__":
    main()
