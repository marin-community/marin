# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize the visual-token-budget sweep into the per-budget planning table.

Reads every ``results/ocr-budget-*.jsonl`` produced by ``launch_budget_sweep.sh``
and prints, per budget: the max-throughput point (arm, concurrency, latency), the
render cost, and the CPU:GPU ratio — render cores needed to keep one GPU fed at
that max throughput. Failed points (collapsed runs) are listed separately so the
in-flight boundary they mark stays visible.
"""

import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_rows() -> tuple[list[dict], dict[int, float]]:
    bench_rows: list[dict] = []
    render_rate: dict[int, float] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "ocr-budget-*.jsonl"))):
        arm = os.path.basename(path).removeprefix("ocr-budget-").removesuffix(".jsonl")
        with open(path) as file:
            for line in file:
                row = json.loads(line)
                if row["mode"] == "cpu_only":
                    render_rate[row["config"]["max_visual_tokens"]] = row["cpu"]["cpu_pages_per_core_second"]
                else:
                    row["arm"] = arm
                    bench_rows.append(row)
    return bench_rows, render_rate


def main() -> None:
    bench_rows, render_rate = load_rows()
    budgets = sorted({r["config"]["max_visual_tokens"] for r in bench_rows})

    header = f"{'budget':>7} {'MP':>5} | {'best p/s/gpu':>12} {'arm':>8} {'conc':>5} {'p50 s':>6}"
    print(header + f" | {'render p/c/s':>12} {'cores:GPU':>9}")
    for budget in budgets:
        points = [r for r in bench_rows if r["config"]["max_visual_tokens"] == budget]
        clean = [r for r in points if r["requests_failed"] == 0]
        best = max(clean, key=lambda r: r["pages_per_second_per_gpu"])
        rate = render_rate[budget]
        ratio = best["pages_per_second_per_gpu"] / rate
        print(
            f"{budget:>7} {best['cpu']['mean_megapixels']:>5} | {best['pages_per_second_per_gpu']:>12} "
            f"{best['arm']:>8} {best['config']['concurrency']:>5} {best['latency_p50']:>6} | "
            f"{rate:>12} {ratio:>9.2f}"
        )

    failed = [r for r in bench_rows if r["requests_failed"]]
    if failed:
        print("\ncollapsed points (in-flight boundary):")
        for r in failed:
            c = r["config"]
            print(
                f"  {r['arm']:>8} budget={c['max_visual_tokens']} conc={c['concurrency']} "
                f"api={c['api_server_count']} ram={c['gpu_worker_ram_gb']}g: "
                f"{r['requests_failed']}/{c['num_requests']} failed"
            )


if __name__ == "__main__":
    main()
