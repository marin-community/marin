# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise thinking rendering and both EOS stops on the actual SFT export."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import httpx
import yaml
from huggingface_hub import hf_hub_download
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.inference.iris import remote_inference
from rigging.config_discovery import find_project_root
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.post_training.long_loop.preflight import EndpointEvidence, check_endpoint
from experiments.post_training.long_loop.serving import check_context_limit, checked_serving_release
from experiments.post_training.long_loop.sft import (
    HANDOFF_DIR,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    read_json,
    validate_handoff,
)

PROBE_MODEL_ALIAS = "long-loop-sft-probe"


def prompt_tokens(tokenizer: PreTrainedTokenizerBase, messages: list[dict]) -> list[int]:
    """Match the token-ID list returned by the serving tokenize API."""
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True
    )


def probe(
    *,
    output: str,
    model_root: str,
    tokenizer_revision: str,
    settings: dict | None = None,
    relative_path: str = HANDOFF_DIR,
) -> None:
    """Serve the HF export at ``model_root/relative_path``; ``""`` selects the root itself."""
    validate_handoff(model_root, relative_path)
    hf = prefix_join(model_root, relative_path) if relative_path else model_root
    root = find_project_root()
    assert root is not None, "Run from the Marin worktree"
    if settings is None:
        settings = {
            "model_config": (
                "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/"
                "laion-snowball-67b-a2b-relay-sft-acont-step999.yaml"
            ),
            "window": 24576,
            "tokenizer_repo": TOKENIZER_REPO,
            "gpu_count": 8,
            "cluster_config": "lib/iris/config/marin.yaml",
            "priority": "production",
        }
    window = settings["window"]
    tokenizer_repo = settings["tokenizer_repo"]
    check_context_limit("SFT endpoint", read_json(prefix_join(hf, "config.json"))["max_position_embeddings"], window)
    model = load_model_config(root / settings["model_config"])
    model = replace(
        model,
        name=PROBE_MODEL_ALIAS,
        location=hf,
        revision=None,
        tokenizer=tokenizer_repo,
        tokenizer_revision=tokenizer_revision,
        serve=replace(model.serve, max_model_len=window),
    )
    origin = yaml.safe_load((root / settings["cluster_config"]).read_text())["dashboard_url"]
    inference = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=settings["gpu_count"], target_cluster=None),
        env_vars={},
        priority=priority_band_value(settings["priority"]),
        capability_origin=origin,
        api_model=PROBE_MODEL_ALIAS,
    )
    inference = replace(inference, engine=replace(inference.engine, gpu_release=checked_serving_release()))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, revision=tokenizer_revision)
    messages = [{"role": "user", "content": "Reply with the number 2."}]
    with remote_inference(inference) as served:
        endpoint = served.model.endpoint
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
        base = endpoint.base_url.rstrip("/")
        with httpx.Client(headers=headers, timeout=180) as client:
            models = client.get(base + "/models")
            models.raise_for_status()
            request = {
                "model": PROBE_MODEL_ALIAS,
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": True},
                "add_generation_prompt": True,
            }
            tokenized = client.post(base.removesuffix("/v1") + "/tokenize", json=request)
            tokenized.raise_for_status()
            served_tokens = tokenized.json()["tokens"]
            expected_tokens = prompt_tokens(tokenizer, messages)
            fs, path = filesystem_for(prefix_join(output, "tokenization.json"))
            fs.pipe_file(
                path,
                json.dumps({"served": served_tokens, "expected": expected_tokens, "request": request}).encode(),
            )
            if served_tokens != expected_tokens:
                raise ValueError("Served prompt tokens differ from the pinned tokenizer/template")
            rendered = tokenizer.decode(served_tokens, skip_special_tokens=False)
            stops = {}
            receipts = {}
            for token in (128001, 128009):
                response = client.post(
                    base + "/chat/completions",
                    json={
                        "model": PROBE_MODEL_ALIAS,
                        "messages": messages,
                        "max_tokens": 4,
                        "temperature": 0,
                        "logit_bias": {str(token): 100},
                        "skip_special_tokens": False,
                        "stop_token_ids": [128001, 128009],
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                )
                response.raise_for_status()
                receipts[str(token)] = response.json()
                stops[token] = response.json()["choices"][0]["finish_reason"]
            fs, path = filesystem_for(prefix_join(hf, "tokenizer.json"))
            tokenizer_hash = hashlib.sha256(fs.cat_file(path)).hexdigest()
            check_endpoint(
                EndpointEvidence(
                    PROBE_MODEL_ALIAS,
                    models.json()["data"][0]["id"],
                    hashlib.sha256(
                        Path(hf_hub_download(tokenizer_repo, "tokenizer.json", revision=tokenizer_revision)).read_bytes()
                    ).hexdigest(),
                    tokenizer_hash,
                    rendered,
                    "/think",
                    stops,
                )
            )
            record = {
                "configured_checkpoint": model.location,
                "served_models": models.json(),
                "tokenizer_revision": tokenizer_revision,
                "export_tokenizer_sha256": tokenizer_hash,
                "rendered_prompt": rendered,
                "tokenization": tokenized.json(),
                "forced_eos": receipts,
                "weight_identity_basis": "verified export plus pinned server launch, not models endpoint alone",
            }
    fs, path = filesystem_for(prefix_join(output, "endpoint.json"))
    fs.pipe_file(path, json.dumps(record, indent=2).encode())


def endpoint_step(version: str, model: ArtifactStep) -> ArtifactStep:
    # Both plumbing packing and the base bound cover this probe; runtime checks the actual export.
    check_context_limit("SFT plumbing endpoint", 32768, 24576)
    return apply(
        "long-loop/endpoint-probe",
        probe,
        version=version,
        output=OUT,
        model_root=model,
        tokenizer_revision=TOKENIZER_REVISION,
    )
