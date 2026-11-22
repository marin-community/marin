# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a Together serverless judge to an evaluation-only SkyRL launch."""

import json
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import yaml
from marin.rl.skyrl import SkyRLRun, SkyRLRunConfig, run_skyrl
from rigging.event_log import EventLog
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import resolve_secret_spec

from experiments.evaluation.judge_config import JudgeConfig
from experiments.post_training.long_loop.launch_validation import validate_before_allocation, validate_selected_context


@dataclass(frozen=True)
class JudgedEvaluation:
    skyrl: SkyRLRunConfig
    judge: JudgeConfig


def run_with_existing_judge(config: SkyRLRunConfig, *, openai_api_key: str, judge_api_key: str) -> SkyRLRun:
    """Bind the counterpart and verifier credentials to this launch only."""
    return run_skyrl(
        config,
        env_vars={"OPENAI_API_KEY": openai_api_key, "JUDGE_API_KEY": judge_api_key},
    )


def run_judged(config: JudgedEvaluation) -> SkyRLRun:
    api_key = resolve_secret_spec((config.judge.credential,)).value
    validation = validate_before_allocation(config.skyrl.launch_config_yaml, secrets={"JUDGE_API_KEY": api_key})
    context_started = time.monotonic()
    validation["selected_context"] = validate_selected_context(config.skyrl.launch_config_yaml)
    validation["selected_context_seconds"] = time.monotonic() - context_started
    launch = yaml.safe_load(config.skyrl.launch_config_yaml)
    validation_fs, validation_key = filesystem_for(
        str(StoragePath(launch["artifacts"]["terminal_manifest_uri"]).parent / "preallocation-validation.json")
    )
    validation_fs.pipe_file(validation_key, json.dumps(validation, indent=2).encode())
    attempt = uuid4().hex
    destination = str(StoragePath(launch["artifacts"]["terminal_manifest_uri"]).parent)
    fs, key = filesystem_for(prefix_join(destination, f"timing/judge-{attempt}.jsonl"))
    with tempfile.TemporaryDirectory() as directory:
        log = EventLog(Path(directory) / "events.jsonl", launch["run"]["id"], "eval", attempt)
        try:
            base_url = config.judge.base_url
            launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["base_url"] = base_url
            launch["skyrl"]["extra_env"]["JUDGE_BASE_URL"] = base_url
            launch["runtime"]["task_env"]["JUDGE_BASE_URL"] = base_url
            # Pass credentials only to this subprocess; never serialize them into YAML.
            with log.span("eval_execution_including_queue"):
                return run_with_existing_judge(
                    replace(config.skyrl, launch_config_yaml=yaml.safe_dump(launch)),
                    openai_api_key=api_key,
                    judge_api_key=api_key,
                )
        finally:
            fs.pipe_file(key, log.path.read_bytes())
