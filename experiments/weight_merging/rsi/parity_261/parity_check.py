"""Recover the archived Step12 result and compare independent binomial SEMs."""

import argparse
import ast
import json
import math
import re
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--h100-items", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    document = json.loads(json.loads((root / "eval-results.json").read_text())[0]["document"])
    outcome = document["task_outcomes"]["MATH500"]
    assert outcome["status"] == "succeeded"
    assert outcome["expected_count"] == outcome["generated_count"] == outcome["scored_count"] == 500
    samples = pq.read_table(root / "samples.parquet").to_pylist()
    assert len(samples) == len({int(row["doc_id"]) for row in samples}) == 500
    correct = sum(row["correct"] for row in samples)
    assert correct == document["results"]["MATH500"]["num_solved"] == 467
    score = correct / 500
    comparisons = {}
    for name, reference in [("H100", 0.944), ("fixed_singlehost_TPU", 0.940)]:
        sem = math.sqrt(score * (1 - score) / 500 + reference * (1 - reference) / 500)
        z = (score - reference) / sem
        comparisons[name] = dict(reference=reference, delta=score - reference, combined_sem=sem, z=z, parity=abs(z) <= 2)
    h100 = {int(row["id"]): row for row in json.loads(args.h100_items.read_text())}
    items = [
        dict(
            id=int(row["doc_id"]),
            correct=row["correct"],
            h100_correct=h100[int(row["doc_id"])]["correct"],
            extracted=row["extracted"],
            target=row["target_text"],
        )
        for row in samples
    ]
    (root / "per-item.json").write_text(json.dumps(items, indent=2))
    paired = dict(
        gained=sum(x["correct"] and not x["h100_correct"] for x in items),
        lost=sum(not x["correct"] and x["h100_correct"] for x in items),
        agreement=sum(x["correct"] == x["h100_correct"] for x in items),
    )

    def counter(filename, metric):
        rows = (root / filename).read_text().splitlines()
        return sum(float(line.rsplit(" ", 1)[1]) for line in rows if line.startswith(metric + "{"))

    generated = counter("metrics-after.txt", "vllm:generation_tokens_total") - counter(
        "metrics-before.txt", "vllm:generation_tokens_total"
    )
    finishes = {}
    for reason in ("stop", "length", "abort", "error", "repetition"):

        def count(filename):
            return sum(
                float(line.rsplit(" ", 1)[1])
                for line in (root / filename).read_text().splitlines()
                if line.startswith("vllm:request_success_total{") and f'finished_reason="{reason}"' in line
            )

        finishes[reason] = int(count("metrics-after.txt") - count("metrics-before.txt"))
    assert sum(finishes.values()) == 500 and finishes["stop"] == 500
    hbm = {}
    logs = re.sub(r"\x1b\[[0-9;]*m", "", (root / "server.log").read_text())
    for line in logs.splitlines():
        if "TASK261_POST_LOAD_HBM " in line:
            for device, stats in ast.literal_eval(line.split("TASK261_POST_LOAD_HBM ", 1)[1]):
                hbm[device] = stats
    assert len(hbm) == 16
    wall = json.loads((root / "host-0/wall.json").read_text())
    gate = json.loads((root / "gate.json").read_text())
    eval_seconds = float(document["total_evaluation_time_seconds"])
    result = dict(
        score=score,
        correct=correct,
        scored=500,
        comparisons=comparisons,
        versus_H100_items=paired,
        generated_tokens=int(generated),
        finish_reasons=finishes,
        eval_seconds=eval_seconds,
        eval_tokens_per_second=generated / eval_seconds,
        items_per_minute=500 / eval_seconds * 60,
        total_worker_wall_seconds=wall["elapsed_seconds"],
        ready_seconds=gate["ready_seconds"],
        post_load_HBM=hbm,
        post_load_HBM_bytes_min=min(s["bytes_in_use"] for s in hbm.values()),
        post_load_HBM_bytes_max=max(s["bytes_in_use"] for s in hbm.values()),
        measured_device_limit_bytes=sorted({s["bytes_limit"] for s in hbm.values()}),
        notes=[
            "Generation tokens and finish reasons are Prometheus after-before deltas, excluding the17 warmup/microbenchmark requests.",
            "Effective eval throughput includes harness/scoring overhead; warm short-output microbenchmark is separate.",
            "HBM snapshot is immediately post-weight-load, not peak runtime memory.",
            "Server context73728 differs from reference32768; evaluator effective max_length32704 retained.",
        ],
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "post_load_HBM"}, indent=2))


if __name__ == "__main__":
    main()
