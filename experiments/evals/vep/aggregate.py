# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute variant scores from a complete scoring run without model calls."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.evals.vep.targets import Target, targets


def target_key(item: dict) -> tuple:
    return tuple(item[key] for key in ("variant", "method", "strand", "mapping", "candidate"))


def expected_key(target: Target) -> tuple:
    return (target.variant, target.method, target.strand, target.mapping, target.candidate)


def aggregate(panels_path: Path, scores_path: Path) -> list[dict]:
    panels_bytes = panels_path.read_bytes()
    panels = json.loads(panels_bytes)
    records = [json.loads(line) for line in scores_path.read_text().splitlines()]
    assert records[0]["type"] == "start" and records[-1]["type"] == "complete", "Run is incomplete"
    header = records[0]
    assert header["panels_sha256"] == hashlib.sha256(panels_bytes).hexdigest()
    rows = panels["panels"][header["panel"]][: header["variant_count"]]
    expected = {
        expected_key(target): target
        for target in targets(rows, panels["panels"]["demonstrations"], set(header["format_check_ids"]))
    }
    scores = {}
    for record in records[1:-1]:
        assert record["type"] == "score"
        key = target_key(record)
        assert key in expected and key not in scores
        target = expected[key]
        assert record["input_sha256"] == hashlib.sha256((target.prefix + target.text).encode()).hexdigest()
        assert math.isfinite(record["logprob"])
        scores[key] = record["logprob"]
    assert set(scores) == set(expected), "Missing scoring targets"
    output = []
    for row in rows:
        item = {key: row[key] for key in ("id", "label", "subset", "match_group", "chrom")}
        methods = ["null", "healthy", "zero-shot", "two-shot"]
        if row["id"] in header["format_check_ids"]:
            methods.append("explicit-pair")
        for method in methods:
            strands = {}
            for strand in ("+", "-"):
                if method in {"null", "healthy"}:
                    ref = scores[(row["id"], method, strand, "none", "reference")]
                    alt = scores[(row["id"], method, strand, "none", "alternate")]
                    strands[strand] = {"primary": ref - alt}
                else:
                    primary = (
                        scores[(row["id"], method, strand, "primary", " A")]
                        - scores[(row["id"], method, strand, "primary", " B")]
                    )
                    reversed_score = (
                        scores[(row["id"], method, strand, "reversed", " B")]
                        - scores[(row["id"], method, strand, "reversed", " A")]
                    )
                    strands[strand] = {
                        "primary": primary,
                        "reversed": reversed_score,
                        "mapping-average": (primary + reversed_score) / 2,
                    }
            for mapping in strands["+"]:
                name = method if mapping == "primary" else f"{method}/{mapping}"
                forward, reverse = strands["+"][mapping], strands["-"][mapping]
                item[name] = (forward + reverse) / 2
                item[f"{name}/strand-difference"] = forward - reverse
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.panels, args.scores)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
