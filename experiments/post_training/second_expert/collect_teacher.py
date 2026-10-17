# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect native E_2 demonstrations on the fixed MMLU-Pro subset."""

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
from experiments.post_training.second_expert.constants import (
    CANONICAL,
    CANONICAL_SHA256,
    PROMPT_COUNT,
    SAMPLES_PER_PROMPT,
    SUBSET_SHA256,
)
from experiments.post_training.second_expert.grading import grade_response

PROMPTS_SHA256 = "a22cb1dcd6310215efc01d62e1b6572cbc611e8e47b5ea0bdb552e3c334af2c5"


def selected_prompts(rows: list[dict], canonical: list[dict], custom_ids: list[str]) -> list[dict]:
    """Select captured documents by immutable ID, never by question text or source ID."""
    requests_by_id = {row["custom_id"]: row for row in rows}
    labels_by_id = {row["custom_id"]: row for row in canonical}
    if len(requests_by_id) != len(rows) or len(labels_by_id) != len(canonical):
        raise ValueError("Duplicate custom_id")
    if set(requests_by_id) != set(labels_by_id):
        raise ValueError("Canonical and captured identities differ")
    if len(set(custom_ids)) != len(custom_ids):
        raise ValueError("Subset repeats a document")
    return [
        requests_by_id[identity]
        | {
            "source_id": str(labels_by_id[identity]["question_id"]),
            "gold": labels_by_id[identity]["canonical_gold"],
            "group": labels_by_id[identity]["group"],
            "category": labels_by_id[identity]["category"],
            "task": "MMLUPro",
        }
        for identity in sorted(custom_ids)
    ]


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
        (destination / "grades" / filename).write_text(json.dumps(grade_response(previous, prompt["gold"])) + "\n")
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
    (destination / "grades" / filename).write_text(json.dumps(grade_response(record, prompt["gold"])) + "\n")


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
    canonical_bytes = (StoragePath(CANONICAL) / "canonical.jsonl").read_bytes()
    subset_bytes = (StoragePath(CANONICAL) / "subset-groups-1600-seed20261009.json").read_bytes()
    if hashlib.sha256(source.encode()).hexdigest() != PROMPTS_SHA256:
        raise ValueError("Prompt capture hash mismatch")
    if hashlib.sha256(canonical_bytes).hexdigest() != CANONICAL_SHA256:
        raise ValueError("Canonical hash mismatch")
    if hashlib.sha256(subset_bytes).hexdigest() != SUBSET_SHA256:
        raise ValueError("Subset hash mismatch")
    subset = json.loads(subset_bytes)
    prompts = selected_prompts(
        [json.loads(line) for line in source.splitlines() if line],
        [json.loads(line) for line in canonical_bytes.splitlines() if line],
        subset["custom_ids"],
    )
    if len(prompts) != PROMPT_COUNT or args.samples != SAMPLES_PER_PROMPT:
        raise ValueError("Frozen collection requires 1606 documents and eight samples each")
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
        "canonical_sha256": CANONICAL_SHA256,
        "subset_sha256": SUBSET_SHA256,
        "teacher_selection": "step254 passed gate; provisional pending earlier checkpoint comparisons",
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
    (destination / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prompts))
    (destination / "responses").mkdirs()
    (destination / "grades").mkdirs()
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
