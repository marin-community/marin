# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate routed-expert responses using the campaign's serving configuration."""

import argparse
import base64
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import requests
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import VllmEngineConfig
from marin.inference.serve import local_inference

from experiments.weight_merging.routing_capture import summarize_routing
from experiments.weight_merging.routing_replay import (
    TokenTrajectory,
    replay_request,
    streaming_trajectory,
    summarize_replay,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--weights", required=True, help="Existing regional S3 checkpoint copy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = load_model_config(args.model_config)
    model = replace(
        original,
        location=args.weights,
        tokenizer=original.tokenizer or original.location,
        tokenizer_revision=original.effective_tokenizer_revision,
    )
    config = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        env_vars={},
        priority=0,
        api_model=model.name,
    )
    assert isinstance(config.engine, VllmEngineConfig)
    engine = replace(
        config.engine,
        extra_args=(
            *config.engine.extra_args,
            "--enable-return-routed-experts",
            "--middleware",
            "experiments.weight_merging.record_requests.RecordRequests",
        ),
    )
    os.environ.update(config.iris.worker_environment.env_vars)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "model_config": args.model_config.read_text(),
                "weights": args.weights,
                "vllm_source_commit": VLLM_GPU_RELEASE.source_commit,
                "purpose": "routing capture smoke test; not a benchmark evaluation",
            },
            indent=2,
        )
        + "\n"
    )
    with local_inference(config.model, engine, num_chips=8) as session:
        endpoint = session.model.endpoint
        for index, prompt in enumerate(("What is 17 plus 24?", "Explain in one sentence why the sky looks blue.")):
            request = {
                "model": model.name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "seed": 20222943,
                "max_tokens": 32,
                "stream": False,
                "return_token_ids": True,
                "routed_experts_prompt_start": 0,
            }
            response = requests.post(f"{endpoint.base_url}/chat/completions", json=request, timeout=300)
            response.raise_for_status()
            payload = response.json()
            (args.output / f"request-{index}.json").write_text(json.dumps(request, indent=2) + "\n")
            (args.output / f"response-{index}.json").write_text(json.dumps(payload) + "\n")
            summary = summarize_routing(payload, num_layers=26, num_experts=256, top_k=4)
            (args.output / f"summary-{index}.json").write_text(json.dumps(summary, indent=2) + "\n")
            trajectory = TokenTrajectory(
                payload["id"],
                model.name,
                payload["prompt_token_ids"],
                payload["choices"][0]["token_ids"],
                payload["choices"][0]["finish_reason"],
            )
            replay_body = replay_request(trajectory)
            replay_response = requests.post(f"{endpoint.base_url}/completions", json=replay_body, timeout=300)
            replay_response.raise_for_status()
            replay_payload = replay_response.json()
            replay_summary = summarize_replay(replay_payload, trajectory, num_layers=26, num_experts=256, top_k=4)
            (args.output / f"replay-request-{index}.json").write_text(json.dumps(replay_body) + "\n")
            (args.output / f"replay-response-{index}.json").write_text(json.dumps(replay_payload) + "\n")
            (args.output / f"replay-summary-{index}.json").write_text(json.dumps(replay_summary, indent=2) + "\n")
            direct_routes, replay_routes = [
                np.load(io.BytesIO(base64.b64decode(item["choices"][0]["routed_experts"])), allow_pickle=False)
                for item in (payload, replay_payload)
            ]
            matches = np.all(np.sort(direct_routes, axis=-1) == np.sort(replay_routes, axis=-1), axis=-1)
            agreement = {}
            for phase, values in (
                ("prompt", matches[: len(trajectory.prompt)]),
                ("generated", matches[len(trajectory.prompt) :]),
            ):
                agreement[phase] = {"token_layer_pairs": values.size, "identical_expert_sets": int(values.sum())}
            (args.output / f"replay-agreement-{index}.json").write_text(json.dumps(agreement, indent=2) + "\n")
        streaming_request = {**request, "stream": True}
        (args.output / "stream-request.json").write_text(json.dumps(streaming_request, indent=2) + "\n")
        with requests.post(
            f"{endpoint.base_url}/chat/completions", json=streaming_request, timeout=300, stream=True
        ) as response:
            response.raise_for_status()
            with (args.output / "stream-response.bin").open("wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    output.write(chunk)
        trajectory = streaming_trajectory((args.output / "stream-response.bin").read_bytes())
        replay_body = replay_request(trajectory)
        response = requests.post(f"{endpoint.base_url}/completions", json=replay_body, timeout=300)
        response.raise_for_status()
        payload = response.json()
        summary = summarize_replay(payload, trajectory, num_layers=26, num_experts=256, top_k=4)
        (args.output / "stream-replay-request.json").write_text(json.dumps(replay_body) + "\n")
        (args.output / "stream-replay-response.json").write_text(json.dumps(payload) + "\n")
        (args.output / "stream-replay-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "complete.json").write_text(
        json.dumps({"validated_requests": 2, "streamed_requests": 1, "validated_replays": 3}) + "\n"
    )


if __name__ == "__main__":
    main()
