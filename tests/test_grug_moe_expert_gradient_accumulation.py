# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig

from experiments.grug.moe import launch_cw_jaxpp_may_d2560
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugJaxPPConfig, GrugRunConfig, GrugTrainerConfig

_RUN_SCRIPT = Path(__file__).parents[1] / "experiments/grug/moe/run_cw_jaxpp_may_d2560.sh"


def _pipeline_config(**overrides) -> GrugJaxPPConfig:
    values: dict[str, Any] = {
        "stages": 4,
        "microbatches": 4,
        "schedule": "std_1f1b",
        "implementation": "explicit_mpmd",
        "expert_gradient_accumulation": "fused_fp32_data_local",
        **overrides,
    }
    return GrugJaxPPConfig(**values)


def _model_config(moe_implementation="ring") -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=64,
        num_layers=4,
        num_heads=2,
        num_kv_heads=2,
        num_experts=8,
        num_experts_per_token=2,
        moe_implementation=moe_implementation,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"implementation": "auto"}, "explicit_mpmd"),
        ({"schedule": "gpipe"}, "std_1f1b"),
        ({"explicit_mpmd_schedule_mode": "transfer_priority"}, "default"),
        ({"explicit_mpmd_stage_task_microbatch_group_size": 2}, "group size 1"),
        ({"microbatches": 1}, "microbatches > 1"),
    ),
)
def test_fused_expert_gradient_accumulation_rejects_unsupported_pipeline(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _pipeline_config(**overrides)


def test_fused_expert_gradient_accumulation_requires_ring_and_expert_parallelism() -> None:
    data = LmDataConfig(tokenizer="passthrough", vocab_size=128, components={})
    resources = ResourceConfig.with_cpu()

    with pytest.raises(ValueError, match="exact bulk-ring"):
        GrugRunConfig(
            model=_model_config("ring_fused"),
            data=data,
            resources=resources,
            trainer=GrugTrainerConfig(expert_axis_size=2, pipeline=_pipeline_config()),
        )
    with pytest.raises(ValueError, match="expert_axis_size greater than 1"):
        GrugRunConfig(
            model=_model_config(),
            data=data,
            resources=resources,
            trainer=GrugTrainerConfig(expert_axis_size=1, pipeline=_pipeline_config()),
        )


def test_may_launcher_reads_expert_gradient_accumulation_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("PP_IMPLEMENTATION", "explicit_mpmd")
    monkeypatch.setenv("PP_SCHEDULE", "std_1f1b")
    monkeypatch.setenv("PP_STAGES", "4")
    monkeypatch.setenv("PP_MPMD_DIM", "4")
    monkeypatch.setenv("PP_MICROBATCHES", "4")
    monkeypatch.setenv("PP_EXPERT_GRADIENT_ACCUMULATION", "fused_fp32_data_local")

    config = launch_cw_jaxpp_may_d2560.build_pipeline_config()

    assert config.expert_gradient_accumulation == "fused_fp32_data_local"


def test_shell_launcher_forwards_expert_gradient_accumulation(tmp_path) -> None:
    result = subprocess.run(
        (
            "bash",
            str(_RUN_SCRIPT),
            "--run-id",
            "fused-expert-gradient-test",
            "--implementation",
            "explicit_mpmd",
            "--expert-gradient-accumulation",
            "fused_fp32_data_local",
            "--expert-axis",
            "2",
            "--microbatches",
            "4",
        ),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )

    assert "expert_gradient_accumulation: fused_fp32_data_local" in result.stdout
