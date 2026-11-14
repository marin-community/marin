# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize paired accuracy, truncation and unclosed thinking for a fixed grid."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.weight_merging.playground.evaluation import item_bootstrap, sign_test_pvalue, unclosed_thinking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    order = [candidate["name"] for candidate in json.loads(args.candidates.read_text())]
    selected = set(order)
    reference = {}
    observations = []
    counts = defaultdict(
        lambda: {"n": 0, "correct": 0, "truncated": 0, "unclosed_think": 0, "wins": 0, "losses": 0, "output_tokens": 0}
    )
    for line in (args.run / "outputs.jsonl").open():
        row = json.loads(line)
        if row["candidate"] not in selected:
            continue
        if row.get("pending_execution"):
            raise ValueError("Code execution is pending; run rescore on an isolated Iris CPU worker first")
        key = (row["candidate"], row["dataset"], row["mode"], row["seed"])
        c = counts[key]
        c["n"] += 1
        c["output_tokens"] += len(row["tokens"])
        c["correct"] += row["correct"]
        c["truncated"] += row["finish_reason"] == "length"
        c["unclosed_think"] += unclosed_thinking(row["completion"])
        pair = row["item_id"], row["mode"], row["seed"]
        observations.append((key, pair, row["correct"]))
        if row["candidate"] == "instruct":
            reference[pair] = row["correct"]
    item_differences = defaultdict(lambda: defaultdict(list))
    for key, pair, correct in observations:
        if pair in reference:
            item_differences[key[:2]][pair[0]].append(int(correct) - int(reference[pair]))
            counts[key]["wins"] += correct and not reference[pair]
            counts[key]["losses"] += reference[pair] and not correct
    pooled = {}
    for (candidate, dataset, _, _), values in counts.items():
        group = pooled.setdefault((candidate, dataset, "pooled", None), {field: 0 for field in values})
        for field, value in values.items():
            group[field] += value
    counts.update(pooled)
    summaries = []
    for (candidate, dataset, mode, seed), counts_row in counts.items():
        summaries.append(
            {
                "candidate": candidate,
                "dataset": dataset,
                "mode": mode,
                "seed": seed,
                **counts_row,
                **(item_bootstrap(list(item_differences[candidate, dataset].values())) if mode == "pooled" else {}),
                "mean_output_tokens": counts_row["output_tokens"] / counts_row["n"],
                "sign_test_p_two_sided": sign_test_pvalue(counts_row["wins"], counts_row["losses"]),
            }
        )
    cap = json.loads((args.run / "run-metadata.json").read_text())["max_new_tokens"]
    lines = [
        (
            f"| Candidate (max_tokens={cap}) | GSM greedy | Strict MBPP greedy | Unclosed GSM / MBPP | "
            "Truncated GSM / MBPP | Mean tokens GSM / MBPP | GSM W/L (p) | MBPP W/L (p) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def rate(counts_row, field):
        return f"{100 * counts_row[field] / counts_row['n']:.2f}%"

    for candidate in order:
        if (candidate, "gsm8k", "greedy", 0) not in counts:
            continue
        gsm, mbpp = (counts[candidate, dataset, "greedy", 0] for dataset in ("gsm8k", "mbpp"))
        values = [
            candidate,
            rate(gsm, "correct"),
            rate(mbpp, "correct"),
            f"{rate(gsm, 'unclosed_think')} / {rate(mbpp, 'unclosed_think')}",
            f"{rate(gsm, 'truncated')} / {rate(mbpp, 'truncated')}",
            f"{gsm['output_tokens'] / gsm['n']:.1f} / {mbpp['output_tokens'] / mbpp['n']:.1f}",
        ]
        for row in (gsm, mbpp):
            values.append(f"{row['wins']}/{row['losses']} ({sign_test_pvalue(row['wins'], row['losses']):.3g})")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            (
                "Exact two-sided sign tests compare paired items with Instruct. "
                "These exploratory p-values do not correct for recipe selection. "
                "Mean output length counts stored generated token IDs, including truncated outputs."
            ),
            "",
            "| Candidate | GSM sampled seed17 / seed29 | MBPP sampled seed17 / seed29 |",
            "| --- | ---: | ---: |",
        ]
    )
    for candidate in order:
        if (candidate, "gsm8k", "sample", 17) not in counts:
            continue
        values = [candidate]
        for dataset in ("gsm8k", "mbpp"):
            values.append(" / ".join(rate(counts[candidate, dataset, "sample", seed], "correct") for seed in (17, 29)))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "| Candidate | Pooled GSM W/L (p) | Pooled MBPP W/L (p) | Item-mean GSM gain [95% CI] | "
            "Item-mean MBPP gain [95% CI] |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in order:
        if (candidate, "gsm8k", "pooled", None) not in counts:
            continue
        values = [candidate]
        for dataset in ("gsm8k", "mbpp"):
            row = counts[candidate, dataset, "pooled", None]
            values.append(f"{row['wins']}/{row['losses']} ({sign_test_pvalue(row['wins'], row['losses']):.3g})")
        for dataset in ("gsm8k", "mbpp"):
            summary = next(
                r for r in summaries if r["candidate"] == candidate and r["dataset"] == dataset and r["mode"] == "pooled"
            )
            low, high = summary["item_bootstrap_ci95"]
            values.append(f"{100 * summary['mean_item_gain']:.2f} [{100 * low:.2f}, {100 * high:.2f}] pp")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Item confidence intervals use 5,000 seeded bootstrap resamples of per-item mean paired differences. "
            "Each item averages its three decodes before resampling.",
            "Pooled rows combine greedy and sampled seeds 17/29. Repeated decodes of the same item are correlated; "
            "pooled sign-test p-values are descriptive and do not establish independent-item significance.",
        ]
    )
    text = "\n".join(lines) + "\n"
    args.output.write_text(text)
    args.output.with_suffix(".json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
