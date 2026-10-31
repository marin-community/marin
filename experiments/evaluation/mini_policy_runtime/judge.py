# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a versioned co-hosted judge to an evaluation-only SkyRL launch."""

import os
from dataclasses import dataclass, replace

import yaml
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from marin.rl.skyrl import SkyRLRun, SkyRLRunConfig, run_skyrl


@dataclass(frozen=True)
class JudgedEvaluation:
    skyrl: SkyRLRunConfig
    judge: ModelConfig
    capability_origin: str


def run_judged(config: JudgedEvaluation) -> SkyRLRun:
    inference = inference_config_for_model(
        config.judge,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster="cw-rno2a"),
        env_vars={},
        priority=priority_band_value("batch"),
        capability_origin=config.capability_origin,
        api_model=config.judge.location,
    )
    with remote_inference(inference) as judge:
        launch = yaml.safe_load(config.skyrl.launch_config_yaml)
        base_url = judge.model.endpoint.base_url
        launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["base_url"] = base_url
        launch["skyrl"]["extra_env"]["JUDGE_BASE_URL"] = base_url
        launch["runtime"]["task_env"]["JUDGE_BASE_URL"] = base_url
        # Resolve credentials only in the submitting process; never serialize a key into YAML.
        previous = os.environ.get("JUDGE_API_KEY")
        os.environ["JUDGE_API_KEY"] = judge.model.endpoint.api_key or "EMPTY"
        try:
            return run_skyrl(replace(config.skyrl, launch_config_yaml=yaml.safe_dump(launch)))
        finally:
            if previous is None:
                del os.environ["JUDGE_API_KEY"]
            else:
                os.environ["JUDGE_API_KEY"] = previous
