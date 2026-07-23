#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distil the Experiment C job logs into a table-regenerating fixture.

The fixture is data, not repo content: the full raw blocks live in finelog (per-task
stdout, shipped to GCS), retrievable with ``iris job logs /romain/<job>``, and the
reduced copy this produces is meant to be uploaded to a marin bucket and referenced
by URL (the pattern ``tests/cluster/vllm/snowball.py`` uses for the prompt fixture) —
not committed. It regenerates every table in the findings issue. Dropped relative to
raw: the per-boundary tensor statistics and the large per-rank environment dumps;
kept: reduced observations (golden-token logprobs, enough for the span and
golden-error tables), the microreproducer result rows, and per-call boundary
checksums (enough for the divergence table).

Input is the extracted JSONL per job (one EXPERIMENT_C_JSON payload per line); the
output path defaults to the input directory so nothing lands in the repo tree.

    uv run python tests/cluster/vllm/_experiment_c_reduce_fixture.py <dir-of-jsonl> [<out.json>]
"""

import json
import pathlib
import sys

GOLDEN_PATH = pathlib.Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_42150_representative_eval_golden.json"
SENTINEL_CASE_ID = "knowledge-longbench-02"
BOUNDARIES = ("b1_predispatch_hidden", "b2_gathered_hidden", "b4_precombine_partials", "b5_postcombine")


def golden_tokens() -> list[int]:
    payload = json.loads(GOLDEN_PATH.read_bytes())
    (case,) = [case for case in payload["cases"] if case["id"] == SENTINEL_CASE_ID]
    return [score["token_id"] for score in case["top_logprobs"]]


def reduce_observation(observation: dict, keep_tokens: set[str]) -> dict:
    logprobs = observation["logprobs"]
    return {
        "rank": observation["rank"],
        "greedy": observation["greedy_token_id"],
        "logprobs": {token: logprobs[token] for token in keep_tokens if token in logprobs},
    }


def main() -> int:
    raw_dir = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else raw_dir / "rank_variance_experiment_c_reduced.json"
    keep_tokens = {str(token) for token in golden_tokens()} | {"423", "426"}
    fixture: dict = {
        "description": (
            "Reduced Experiment C measurements; regenerate with _experiment_c_reduce_fixture.py. "
            "Full raw blocks: finelog, `iris job logs /romain/<job>`."
        ),
        "jobs": {},
        "observations": [],
        "micro": [],
        "trace": [],
    }
    for path in sorted(raw_dir.glob("*.jsonl")):
        job = path.stem
        blocks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        fixture["jobs"][job] = {"blocks": len(blocks)}
        trace_calls: dict[tuple, dict] = {}
        for block in blocks:
            kind = block.get("kind")
            if kind == "observations":
                fixture["observations"].append(
                    {
                        "job": job,
                        "session": block["session"],
                        "mode": block["mode"],
                        "length": block["length"],
                        "round": block.get("round", 0),
                        "observations": [reduce_observation(o, keep_tokens) for o in block["observations"]],
                    }
                )
            elif kind == "probe_micro":
                data = block["data"]
                for result in data["results"]:
                    fixture["micro"].append(
                        {
                            "job": job,
                            "session": block["session"],
                            "rank": data["dp_rank"],
                            "mode": result["mode"],
                            "scale": result["scale"],
                            "checksum": result["output_checksum"],
                            "iter_stable": result["bitwise_stable_across_iters"],
                            "max_abs_vs_ref": result["vs_reference_max_abs"]["rank_order"],
                            "mismatch_elems": result["vs_reference_mismatch_elements"]["rank_order"],
                        }
                    )
            elif kind == "trace" and "r" in block["tag"].split("len")[-1]:
                length_text, serving_text = block["tag"].removeprefix("len").split("r")
                length, serving_rank = int(length_text), int(serving_text)
                file_rank = int(block["file"].split("rank")[1].split("_")[0])
                if file_rank != serving_rank:
                    continue
                for entry in block["data"]:
                    # Keep the serving rank's own rows. At a single-chunk length shape[0]
                    # equals the length; at multi-chunk lengths it is the chunk size (<=512),
                    # so filter on "not a lone dummy token" rather than on the length.
                    if entry.get("shape", [0])[0] <= 1:
                        continue
                    key = (block["session"], length, serving_rank, entry["call"])
                    row = trace_calls.setdefault(key, {"checks": {}})
                    if entry["tag"] == "b1_topk_ids":
                        row["experts"] = entry.get("last_token_expert_ids")
                    elif entry["tag"] in BOUNDARIES:
                        row["checks"][entry["tag"]] = entry["checksum"]
        for (session, length, serving_rank, call), row in sorted(trace_calls.items()):
            fixture["trace"].append(
                {
                    "job": job,
                    "session": session,
                    "length": length,
                    "serving_rank": serving_rank,
                    "call": call,
                    "checksums": row["checks"],
                    "experts": row.get("experts"),
                }
            )
    out_path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(
        f"wrote {out_path}: {len(fixture['observations'])} obs, {len(fixture['micro'])} micro, "
        f"{len(fixture['trace'])} trace rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
