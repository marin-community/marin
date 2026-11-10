# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a versioned co-hosted judge to an evaluation-only SkyRL launch."""

import json
import os
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import yaml
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from marin.rl.skyrl import SkyRLRun, SkyRLRunConfig, run_skyrl
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.launch_validation import validate_before_allocation, validate_selected_context
from experiments.post_training.long_loop.timing import EventLog


@dataclass(frozen=True)
class JudgedEvaluation:
    skyrl: SkyRLRunConfig
    judge: ModelConfig
    capability_origin: str
    judge_gpu_count: int = 8


def run_with_existing_judge(config: SkyRLRunConfig) -> SkyRLRun:
    """Bind the original Tau3 counterpart credential before submitting its task."""
    previous = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.environ["TOGETHER_API_KEY"]
    try:
        return run_skyrl(config)
    finally:
        if previous is None:
            del os.environ["OPENAI_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = previous


def run_judged(config: JudgedEvaluation) -> SkyRLRun:
    validation = validate_before_allocation(config.skyrl.launch_config_yaml)
    context_started = time.monotonic()
    validation["selected_context"] = validate_selected_context(config.skyrl.launch_config_yaml)
    validation["selected_context_seconds"] = time.monotonic() - context_started
    launch = yaml.safe_load(config.skyrl.launch_config_yaml)
    validation_fs, validation_key = filesystem_for(
        str(StoragePath(launch["artifacts"]["terminal_manifest_uri"]).parent / "preallocation-validation.json")
    )
    validation_fs.pipe_file(validation_key, json.dumps(validation, indent=2).encode())
    inference = inference_config_for_model(
        config.judge,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=config.judge_gpu_count, target_cluster=None),
        env_vars={},
        priority=priority_band_value(launch["iris"]["priority"]),
        capability_origin=config.capability_origin,
        api_model=config.judge.location,
    )
    attempt = uuid4().hex
    destination = str(StoragePath(launch["artifacts"]["terminal_manifest_uri"]).parent)
    fs, key = filesystem_for(prefix_join(destination, f"timing/judge-{attempt}.jsonl"))
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        log = EventLog(Path(directory) / "events.jsonl", launch["run"]["id"], "eval", attempt)
        try:
            with log.span("judge_startup_including_queue", gpus=config.judge_gpu_count):
                judge = stack.enter_context(remote_inference(inference))
            fs.pipe_file(key, log.path.read_bytes())
            base_url = judge.model.endpoint.base_url
            launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["base_url"] = base_url
            launch["skyrl"]["extra_env"]["JUDGE_BASE_URL"] = base_url
            launch["runtime"]["task_env"]["JUDGE_BASE_URL"] = base_url
            # Resolve credentials only in the submitting process; never serialize a key into YAML.
            previous = os.environ.get("JUDGE_API_KEY")
            os.environ["JUDGE_API_KEY"] = judge.model.endpoint.api_key or "EMPTY"
            try:
                with log.span("eval_execution_including_queue"):
                    return run_with_existing_judge(replace(config.skyrl, launch_config_yaml=yaml.safe_dump(launch)))
            finally:
                if previous is None:
                    del os.environ["JUDGE_API_KEY"]
                else:
                    os.environ["JUDGE_API_KEY"] = previous
        finally:
            fs.pipe_file(key, log.path.read_bytes())
