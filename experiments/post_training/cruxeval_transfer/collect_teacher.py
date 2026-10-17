# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect native E completions without executing generated code."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import requests
from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.cruxeval_transfer.constants import GENERATION_TOKENS


def validated_prompts(rows: list[dict]) -> list[dict]:
    identities = [row["custom_id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate prompt IDs")
    groups = {direction: [row for row in rows if row["task"] == direction] for direction in ("input", "output")}
    if any(len(group) != 800 for group in groups.values()) or len(rows) != 1600:
        raise ValueError("Expected 800 prompts per direction")
    sources = {direction: {row["source_id"] for row in group} for direction, group in groups.items()}
    if len(sources["input"]) != 800 or sources["input"] != sources["output"]:
        raise ValueError("Expected matching unique source problems in both directions")
    return rows


def request_body(prompt: dict, sample: int, model: str) -> dict:
    identity = f"{prompt['custom_id']}::{sample}"
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
    return {
        "model": model,
        "messages": prompt["b_request"]["messages"],
        "max_tokens": GENERATION_TOKENS,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": seed,
        "n": 1,
        "chat_template_kwargs": {"enable_thinking": True},
        "skip_special_tokens": False,
        "return_token_ids": True,
    }


def collect_one(endpoint: OpenAIEndpoint, destination: StoragePath, item: tuple[dict, int]) -> None:
    prompt, sample = item
    identity = f"{prompt['custom_id']}::{sample}"
    filename = hashlib.sha256(identity.encode()).hexdigest() + ".json"
    target = destination / "responses" / filename
    body = request_body(prompt, sample, endpoint.model)
    if target.exists():
        previous = json.loads(target.read_text())
        if previous["custom_id"] != identity or previous["request"] != body:
            raise ValueError(f"Stored request differs: {identity}")
        if not 200 <= previous["status_code"] < 300:
            raise ValueError(f"Stored failed response requires explicit retry decision: {identity}")
        if not previous["response"].get("choices"):
            raise ValueError(f"Stored response has no completion: {identity}")
        return
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    started = time.time()
    response = requests.post(endpoint.url("chat/completions"), json=body, headers=headers, timeout=1800)
    record = {
        "custom_id": identity,
        "prompt_id": prompt["custom_id"],
        "source_id": prompt["source_id"],
        "task": prompt["task"],
        "sample": sample,
        "request": body,
        "status_code": response.status_code,
        "started_at": started,
        "elapsed_seconds": time.time() - started,
        "response_text": response.text,
    }
    target.write_text(json.dumps(record) + "\n")
    response.raise_for_status()
    record["response"] = response.json()
    target.write_text(json.dumps(record) + "\n")
    if not record["response"].get("choices"):
        raise ValueError(f"Response has no completion: {identity}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.concurrency < 1:
        raise ValueError("Samples and concurrency must be positive")
    source = StoragePath(args.prompts).read_text()
    prompts = validated_prompts([json.loads(line) for line in source.splitlines() if line])
    model = load_model_config(args.model_config)
    model = replace(model, serve=replace(model.serve, reasoning_parser=None))
    manifest = {
        "teacher": model.location,
        "tokenizer": model.tokenizer,
        "tokenizer_revision": model.tokenizer_revision,
        "prompt_source": args.prompts,
        "prompt_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "prompt_ids": [row["custom_id"] for row in prompts],
        "samples_per_prompt": args.samples,
        "requests": len(prompts) * args.samples,
        "sampling": request_body(prompts[0], 0, model.name),
        "test_exposed": True,
    }
    print(json.dumps(manifest), flush=True)
    if not args.run:
        return
    destination = StoragePath(args.output)
    destination.mkdirs()
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output prefix belongs to a different collection")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "responses").mkdirs()
    inference = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    info = get_job_info()
    assert info is not None and info.controller_address is not None
    with IrisClient.in_cluster(info.controller_address, workspace=Path.cwd()) as client:
        context = IrisContext.from_job_info(info, client=client)
        with iris_ctx_scope(context), set_current_client(FrayIrisClient.from_iris_client(client)):
            with remote_inference(inference) as session:
                with ThreadPoolExecutor(args.concurrency) as pool:
                    items = [(prompt, sample) for prompt in prompts for sample in range(args.samples)]
                    list(pool.map(lambda item: collect_one(session.model.endpoint, destination, item), items))
    (destination / "complete.json").write_text(json.dumps({"requests": manifest["requests"]}) + "\n")


if __name__ == "__main__":
    main()
