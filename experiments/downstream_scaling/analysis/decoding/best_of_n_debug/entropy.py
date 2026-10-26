# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute next-token entropy from saved hard best-of-N debug traces, without inference.

Run from the repository root with a cached Llama-3 tokenizer:
    uv run --no-sync python -m experiments.downstream_scaling.analysis.decoding.best_of_n_debug.entropy \
        --tokenizer LOCAL_TOKENIZER_DIR --output-dir OUTPUT_DIR RUN_DIR [RUN_DIR ...]

Entropy includes the selector's random tie-breaking. Average within each
completion, then equally over completions, including terminal decisions.
"""

import argparse
import gzip
import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from experiments.downstream_scaling.evals.algorithms import xtok_selection


@lru_cache(maxsize=None)
def subset_weights(size):
    counts = np.array([0], dtype=np.int8)
    for _ in range(size):
        counts = np.concatenate((counts, counts + 1))
    coefficients = np.array([1 / (size * math.comb(size - 1, k)) for k in range(size)] + [0.0])
    return coefficients[counts]


def output_probabilities(proposals, ranking_scores, n):
    """Marginalize uniformly random orderings inside tied ranking-score groups."""
    if n == 1:
        return proposals.copy()
    order = np.argsort(ranking_scores)
    groups = np.split(order, np.flatnonzero(np.diff(ranking_scores[order])) + 1)
    probabilities = np.zeros_like(proposals)
    lower_mass = 0.0
    for group in groups:
        masses = proposals[group]
        group_mass = math.fsum(masses)
        winning_mass = (lower_mass + group_mass) ** n - lower_mass**n
        if len(group) == 1 or np.all(masses == masses[0]):
            probabilities[group] = winning_mass / len(group)
        elif n == 2:
            probabilities[group] = masses * (2 * lower_mass + group_mass)
        elif winning_mass > 0:
            sums = np.array([0.0])
            for mass in masses:
                sums = np.concatenate((sums, sums + mass))
            values = (lower_mass + sums) ** n
            weights = subset_weights(len(group))
            for bit, index in enumerate(group):
                paired = values.reshape(-1, 2, 1 << bit)
                # A particular subset of k other tied tokens precedes this one
                # with probability 1 / (m * choose(m - 1, k)).
                probabilities[index] = np.sum(
                    (paired[:, 1, :] - paired[:, 0, :]) * weights.reshape(-1, 2, 1 << bit)[:, 0, :]
                )
        lower_mass += group_mass
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError(f"Invalid output probabilities: {probabilities}")
    return probabilities / probabilities.sum()


def entropy_bits(probabilities):
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log2(positive)))


def summarize_run(run_dir, ns, student_vocab, teacher_vocab):
    inputs = json.loads((run_dir / "inputs.json").read_text())
    sampling = inputs["sampling"]
    if sampling["selection_rule"] != "best_of_n":
        raise ValueError(f"{run_dir}: only hard best_of_n is supported")
    # The initial student-only control predates the proposer field.
    proposer = inputs.get("proposer", "student")
    temperature = sampling["temperature"]
    grades = json.loads((run_dir / "graded.json").read_text())
    records = []
    for n in sampling["advisor_weights"]:
        if n not in ns:
            continue
        rollouts = {}
        with gzip.open(run_dir / f"tokens_n{n:g}.jsonl.gz", "rt") as stream:
            for line in stream:
                row = json.loads(line)
                student = xtok_selection.candidates(student_vocab, row["student_topk"])
                teacher = xtok_selection.candidates(teacher_vocab, row["teacher_topk"])
                proposer_scores = xtok_selection.avg_bytes_union_scores(student, teacher, float(proposer == "teacher"))
                ranker_scores = xtok_selection.avg_bytes_union_scores(student, teacher, float(proposer == "student"))
                proposal_dict = xtok_selection._softmax({key: value / temperature for key, value in proposer_scores.items()})
                keys = list(proposal_dict)
                proposals = np.array([proposal_dict[key] for key in keys])
                # Check reconstruction against a statistic recorded during generation.
                if not np.isclose(proposals @ proposals, row["two_draw_duplicate_probability"], rtol=0, atol=1e-12):
                    raise ValueError(f"{run_dir}: proposal distribution differs at {row['id']}, step {row['step']}")
                if row["selected_eos"]:
                    if row["selected_student_ids"] != [student_vocab.eos_id] or row["selected_teacher_ids"] != [teacher_vocab.eos_id]:
                        raise ValueError(f"{run_dir}: EOS mapping differs from the recorded selection")
                scores = np.array([ranker_scores[key] for key in keys])
                entropy = entropy_bits(output_probabilities(proposals, scores, n))
                key = (row["id"], row["rollout"])
                values = rollouts.setdefault(key, [])
                if row["step"] != len(values) or row["n"] != n:
                    raise ValueError(f"{run_dir}: unexpected decision index at {key}")
                values.append(entropy)
        for grade in (grade for grade in grades if grade["n"] == n):
            values = rollouts.pop((grade["id"], grade["rollout"]))
            if len(values) != grade["n_steps"]:
                raise ValueError(f"{run_dir}: decision count differs from grading record")
            records.append({
                "run_dir": str(run_dir), "slug": inputs["slug"], "proposer": proposer,
                "temperature": temperature, "n": n, "id": grade["id"], "rollout": grade["rollout"],
                "n_steps": len(values), "mean_entropy_bits": float(np.mean(values)),
                "first_step_entropy_bits": values[0], "accuracy": grade["grade"]["score"],
            })
        if rollouts:
            raise ValueError(f"{run_dir}: decisions without grades")
        print(f"Finished {inputs['slug']} {proposer}-proposed T={temperature:g} N={n:g}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ns", type=float, nargs="+", default=[1, 2, 8, 32, 128])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    student_vocab = xtok_selection.load_vocab(tokenizer)
    teacher_vocab = replace(student_vocab, eos_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    records = []
    for run_dir in args.run_dirs:
        records.extend(summarize_run(run_dir, args.ns, student_vocab, teacher_vocab))
    per_rollout = pd.DataFrame(records)
    summary = per_rollout.groupby(["slug", "proposer", "temperature", "n"], as_index=False).agg(
        mean_entropy_bits=("mean_entropy_bits", "mean"),
        first_step_entropy_bits=("first_step_entropy_bits", "mean"),
        accuracy=("accuracy", "mean"), n_rollouts=("rollout", "size"), n_steps=("n_steps", "sum"),
    )
    per_rollout.to_csv(args.output_dir / "per_rollout.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
