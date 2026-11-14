# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed two-replica comparison after the single-replica arm drains."""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--prerequisite", type=Path, required=True)
    args = parser.parse_args()
    if len(args.base_url) != 2:
        raise ValueError("This comparison requires exactly two endpoints")
    root = args.root / args.name
    root.mkdir(exist_ok=False)
    deadline = time.monotonic() + 7200
    while not args.prerequisite.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Prerequisite benchmark has not completed")
        time.sleep(15)
    previous = json.loads(args.prerequisite.read_text())
    if not previous["minimum_duration_and_rows_met"]:
        raise ValueError("Prerequisite benchmark is too small for the comparison")
    servers = [url.removesuffix("/v1") for url in args.base_url]
    for server in servers:
        while True:
            try:
                with urllib.request.urlopen(server + "/health", timeout=10) as response:
                    assert response.status == 200
                break
            except urllib.error.URLError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(15)
        idle_samples = []
        while len(idle_samples) < 2:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Endpoint has not drained: {server}")
            with urllib.request.urlopen(server + "/metrics", timeout=10) as response:
                metrics = response.read().decode()
            counts = {}
            for metric in ("num_requests_running", "num_requests_waiting"):
                values = re.findall(r"^vllm:" + metric + r"\{[^\n]*\} ([\d.]+)$", metrics, re.M)
                if not values:
                    raise ValueError(f"Missing request accounting: {server} {metric}")
                counts[metric] = sum(float(value) for value in values)
            if any(counts.values()):
                idle_samples.clear()
            else:
                idle_samples.append({"at_unix": time.time(), "counts": counts})
            if len(idle_samples) < 2:
                time.sleep(5)
        (root / f"drain-{servers.index(server) + 1}.json").write_text(
            json.dumps({"server": server, "idle_samples": idle_samples}, indent=2)
        )
    (root / "prerequisite-result.json").write_text(json.dumps(previous, indent=2))
    commands = []
    run_roots = []
    for index, base_url in enumerate(args.base_url, 1):
        arm = args.root / f"{args.name}-r{index}"
        run_roots.append(arm / "run")
        commands.append(
            [
                sys.executable,
                "-m",
                "experiments.glm_tool_serving.benchmark",
                "--temperature",
                "1.0",
                "--top-p",
                "0.95",
                "--thinking-token-budget",
                "8192",
                "--template",
                str(args.root / "template-output32k.json"),
                "--manifest",
                str(args.root / f"pool2-manifest-r{index}.jsonl"),
                "--base-url",
                base_url,
                "--api-key-file",
                str(args.root / "local-api-key.txt"),
                "--workers",
                "16",
                "--root",
                str(arm),
            ]
        )
    (root / "launch.json").write_text(json.dumps({"commands": commands, "at_unix": time.time()}, indent=2))
    metric_names = (
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
        "num_preemptions_total",
        "prompt_tokens_total",
        "generation_tokens_total",
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
    )
    with ExitStack() as stack:
        logs = [stack.enter_context((root / f"replica-{i}.log").open("w")) for i in (1, 2)]
        samples = stack.enter_context((root / "metrics.jsonl").open("w"))
        processes = [
            subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            for command, log in zip(commands, logs, strict=True)
        ]
        while True:
            for index, server in enumerate(servers, 1):
                try:
                    with urllib.request.urlopen(server + "/metrics", timeout=10) as response:
                        text = response.read().decode()
                except (urllib.error.URLError, TimeoutError) as error:
                    samples.write(json.dumps({"replica": index, "at_unix": time.time(), "error": str(error)}) + "\n")
                    samples.flush()
                    continue
                lines = [
                    line
                    for line in text.splitlines()
                    if any(line.startswith("vllm:" + key + "{") for key in metric_names)
                ]
                samples.write(json.dumps({"replica": index, "at_unix": time.time(), "metrics": lines}) + "\n")
                samples.flush()
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(15)
        if any(process.returncode for process in processes):
            raise RuntimeError("A replica benchmark exited unsuccessfully; inspect its log")
    command = [
        sys.executable,
        "-m",
        "experiments.glm_tool_serving.summarize_pool",
        "--output",
        str(root / "summary.json"),
    ]
    for run_root in run_roots:
        command.extend(["--run-root", str(run_root)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
