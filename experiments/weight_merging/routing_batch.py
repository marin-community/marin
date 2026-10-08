# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure expert selections on saved benchmark token trajectories."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.serve import local_inference
from rigging.filesystem.buckets import filesystem_for

from experiments.weight_merging.routing_replay import TokenTrajectory, replay_request, summarize_replay


def measure_trajectory(row: dict, base_url: str, timeout: float, output_prefix: str) -> dict:
    trajectory = TokenTrajectory(row["request_id"], row["model"], row["prompt"], row["generated"], row["finish_reason"])
    response = requests.post(f"{base_url.rstrip('/')}/completions", json=replay_request(trajectory), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    summary = summarize_replay(payload, trajectory, num_layers=26, num_experts=256, top_k=4)
    summary.update(task=row["task"], source_response_sha256=row["response_sha256"])
    # Hash the request ID so arbitrary provider IDs cannot create nested output paths.
    name = hashlib.sha256(trajectory.request_id.encode()).hexdigest()
    fs, path = filesystem_for(f"{output_prefix}/{name}.json")
    with fs.open(path, "w") as output:
        json.dump(summary, output)
    return {"request_id": trajectory.request_id, "task": row["task"], "summary": f"{name}.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    source_fs, source_path = filesystem_for(args.inputs)
    with source_fs.open(source_path, "rb") as source:
        raw = source.read()
    if hashlib.sha256(raw).hexdigest() != args.input_sha256:
        raise ValueError("Replay input hash does not match the verified input artifact")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("Replay inputs must contain unique request identities")
    model = load_model_config(args.model_config)
    if any(row["model"] != model.name for row in rows):
        raise ValueError("Captured model identity does not match the replay server")
    output_fs, output_path = filesystem_for(args.output)
    if output_fs.exists(output_path):
        raise FileExistsError(args.output)
    config = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        env_vars={},
        priority=0,
        api_model=model.name,
    )
    os.environ.update(config.iris.worker_environment.env_vars)
    provenance = {
        "model_config": args.model_config.read_text(),
        "inputs": args.inputs,
        "input_sha256": args.input_sha256,
        "requests": len(rows),
        "tasks": len({row["task"] for row in rows}),
        "code_revision": args.code_revision,
        "vllm_revision": VLLM_GPU_RELEASE.source_commit,
        "measurement": "Approximate routing from teacher-forced replay; not a benchmark score",
        "concurrency": args.concurrency,
    }
    with output_fs.open(f"{output_path}/provenance.json", "w") as output:
        json.dump(provenance, output, indent=2)
    with local_inference(config.model, config.engine, num_chips=8) as server:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(measure_trajectory, row, server.model.endpoint.base_url, args.timeout, args.output)
                for row in rows
            ]
            results = [future.result() for future in futures]
    with output_fs.open(f"{output_path}/complete.json", "w") as output:
        json.dump({**provenance, "results": results}, output, indent=2)


if __name__ == "__main__":
    main()
