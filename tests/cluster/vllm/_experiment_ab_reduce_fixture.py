#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distill the Experiment A/B raw job logs into a reduced fixture.

The fixture is data, not repo content: it is written to the input directory (or an
explicit path), for hosting in a marin bucket, not committing. Provenance: raw
client-stream logs of the three measurement jobs (Iris log queries truncate long
lines, so these came from the submitting client's stream):

  snowball-experiment-b-vllm-bd7f6354      launch 1, chunked + unchunked ladders
  snowball-experiment-b-vllm-de845a00      launch 2, same code and flags
  snowball-experiment-b-levanter-9f283c4b  Levanter arm, one load

Keeps per (job, config, length, rank): greedy token, top-1 token/logprob, and the
logprobs of tokens 423/426 (golden top-2 at full length). The summarized tables the
report and plan cite are small enough to live inline in those documents.

    uv run python tests/cluster/vllm/_experiment_ab_reduce_fixture.py <raw_dir> [<out.json>]
"""

import json
import pathlib
import sys

JOBS = {
    "b8nh8jrmi.output": "snowball-experiment-b-vllm-bd7f6354",
    "b0jce5cr9.output": "snowball-experiment-b-vllm-de845a00",
    "blhdsolf8.output": "snowball-experiment-b-levanter-9f283c4b",
}
BEGIN, END = "EXPERIMENT_B_JSON_BEGIN", "EXPERIMENT_B_JSON_END"


def extract_blocks(text: str) -> list[dict]:
    blocks = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if BEGIN in line and index + 1 < len(lines):
            payload = lines[index + 1]
            blocks.append(json.loads(payload[payload.index("{") :]))
    return blocks


def reduce_observation(observation: dict) -> dict:
    logprobs = {int(token): value for token, value in observation["logprobs"].items()}
    top_token = max(logprobs, key=lambda token: logprobs[token])
    return {
        "rank": observation["rank"],
        "greedy": observation["greedy_token_id"],
        "top_token": top_token,
        "top_logprob": logprobs[top_token],
        "lp423": logprobs.get(423),
        "lp426": logprobs.get(426),
    }


def main() -> int:
    raw_dir = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else raw_dir / "rank_variance_experiment_ab_reduced.json"
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for filename, job in JOBS.items():
        for block in extract_blocks((raw_dir / filename).read_text(errors="replace")):
            for observation in block["observations"]:
                key = (job, observation["config"], observation["length"])
                groups.setdefault(key, []).append(reduce_observation(observation))
    fixture = {
        "description": "Reduced Experiment A/B measurements; see _experiment_ab_reduce_fixture.py",
        "sentinel_case": "knowledge-longbench-02",
        "golden_lp423_full_length": None,  # golden top-1 p=0.557848; kept in the goldens resource
        "rows": [
            {
                "job": job,
                "config": config,
                "length": length,
                "observations": sorted(observations, key=lambda o: o["rank"]),
            }
            for (job, config, length), observations in sorted(groups.items())
        ],
    }
    out_path.write_text(json.dumps(fixture, indent=1) + "\n")
    row_count = sum(len(row["observations"]) for row in fixture["rows"])
    print(f"wrote {out_path} ({len(fixture['rows'])} groups, {row_count} observations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
