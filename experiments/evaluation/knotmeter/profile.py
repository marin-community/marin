# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen Snowball 64K AgentX benchmark against a running server."""

import argparse
from pathlib import Path

from experiments.evaluation.knotmeter.artifacts import pack_server_metrics, read_record
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import (
    SERIES_PATH,
    SERVER_PORT,
    benchmark_command,
    load_series,
    workload_environment,
)
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, help="Receipt from the CPU workload preparation")
    parser.add_argument("--url", default=f"http://127.0.0.1:{SERVER_PORT}")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    series = load_series()
    bundle = args.output / "bundle"
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = read_record(args.workload)
    restore_workload(receipt, bundle, series=series)
    check_native_cache(bundle, SERIES_PATH, args.output / "workload-check.log")
    run_logged(
        benchmark_command(series, args.concurrency, args.url, args.output / "aiperf_artifacts"),
        workload_environment(bundle, series),
        args.output / "agentx.log",
        timeout=series["warmup_grace_period"] + series["duration"] + 300,
        sample=args.output / "client-resource-samples.jsonl",
    )
    pack_server_metrics(args.output)


if __name__ == "__main__":
    main()
