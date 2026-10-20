"""CPU unit tests for the single-turn delphi math-RLVR launcher path.

Pins the `configs/delphi_math_rl.yaml` contract and the TP-divides-heads guard so a
future edit that drops the `environment` flatten, the parquet data-kind routing, the 4k
cap, or the TP-42 guard fails here instead of silently at rollout time on a GPU gang.

Run:
    python -m pytest cloud/iris/tests/test_delphi_math_rl_config.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.rl_config_translation import (  # noqa: E402
    compose_skyrl_config,
    parse_rl_config,
)

_CONFIG = "cloud/iris/configs/delphi_math_rl.yaml"


@dataclass
class _HPCStub:
    gpus_per_node: int = 8


def test_delphi_config_composes_environment_and_caps_into_hydra_config():
    parsed = parse_rl_config(_CONFIG)
    exp_args = {"job_name": "delphi-math-rl-test", "experiments_dir": "/tmp/exp", "num_nodes": 4}
    cfg = compose_skyrl_config(parsed, exp_args, _HPCStub()).config
    assert cfg.environment.env_class == "aime"
    assert cfg.generator.engine_init_kwargs.max_model_len == 4096
    assert cfg.trainer.algorithm.advantage_estimator == "grpo"
    assert "kind" not in cfg.data
    assert "policy_chat_template" not in cfg
    assert "model_num_attention_heads" not in cfg


def test_launch_without_hub_destination_does_not_publish():
    parsed = parse_rl_config(_CONFIG)

    cfg = compose_skyrl_config(
        parsed,
        {
            "job_name": "artifact-only-run",
            "export_hf_artifact": True,
            "hf_hub_repo_id": None,
            "num_nodes": 4,
        },
        _HPCStub(),
    ).config

    assert cfg.trainer.export_hf_artifact is True
    assert cfg.trainer.get("hf_hub_repo_id") is None


def test_iris_derives_durable_training_trajectory_path():
    parsed = parse_rl_config(_CONFIG)
    cfg = compose_skyrl_config(
        parsed,
        {"job_name": "retained-run", "experiments_dir": "s3://bucket/iris/", "num_nodes": 4},
        _HPCStub(),
    ).config
    assert (
        cfg.generator.trajectory_retention.output_path
        == "s3://bucket/iris/retained-run/trace_jobs/training_trajectories"
    )


def test_model_source_locator_reaches_trainer_config():
    parsed = parse_rl_config(_CONFIG)
    exp_args = {
        "job_name": "exportable-run",
        "model_path": "/tmp/materialized-model",
        "model_source_uri": "s3://models/policy",
        "model_source_identity": "policy@abc123",
        "num_nodes": 4,
    }

    cfg = compose_skyrl_config(parsed, exp_args, _HPCStub()).config
    assert cfg.trainer.policy.model.source_uri == "s3://models/policy"
    assert cfg.trainer.policy.model.source_identity == "policy@abc123"


def test_policy_revision_override_reaches_trainer_config():
    parsed = parse_rl_config(_CONFIG)
    revision = "68c46c4b3498877f3ef123c856ecfde50c39f404"

    cfg = compose_skyrl_config(
        parsed,
        {
            "job_name": "revision-pinned-run",
            "model_path": "Qwen/Qwen3.5-9B-Base",
            "model_revision": revision,
            "num_nodes": 4,
        },
        _HPCStub(),
    ).config
    assert cfg.trainer.policy.model.revision == revision


def test_parse_rejects_bad_tp_against_declared_heads(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "entrypoint: standard\n"
        "model_num_attention_heads: 42\n"
        "context_budget:\n"
        "  request_window_tokens: 4096\n"
        "  max_new_tokens_per_turn: 3584\n"
        "  max_turns: 1\n"
        "generator:\n"
        "  inference_engine_tensor_parallel_size: 8\n"
    )
    with pytest.raises(ValueError, match="does not divide"):
        parse_rl_config(str(bad))
