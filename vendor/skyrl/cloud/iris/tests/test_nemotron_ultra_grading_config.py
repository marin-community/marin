"""Launch-time checks for skipped Nemotron Ultra grading."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from rolloutengine.models import ModelTurn
from skyrl_gym.nemotron_tasks import NemotronTaskSession
from skyrl_gym.task_records import grade_skipped
from skyrl_train.dataset.tasks import source_row_task

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config  # noqa: E402

_BASE_CONFIG = _REPO_ROOT / "cloud/iris/configs/qwen_megatron_smoke.yaml"


@dataclass
class _HPCStub:
    gpus_per_node: int = 8


def _skipped_grading_config(tmp_path: Path, *, eval_interval: int) -> Path:
    raw = yaml.safe_load(_BASE_CONFIG.read_text())
    raw["environment"]["env_class"] = "nemotron_ultra"
    trainer = raw["trainer"]
    trainer["eval_before_train"] = False
    trainer["eval_interval"] = eval_interval
    trainer["algorithm"]["advantage_estimator"] = "uniform"
    trainer["algorithm"]["distillation"] = {
        "objective": "sampled_reverse_kl",
        "routing_plan": "opd",
        "coefficient": 1.0,
        "reward_mode": "replace",
    }
    raw["teachers"] = {
        "primary": {
            "source": "openai_compatible",
            "placement": "external",
            "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
            "endpoints": [{"url": "https://teacher.example/v1", "max_concurrency": 8}],
            "tokenizer_fingerprint": f"sha256:{'a' * 64}",
            "max_sequence_length": 32768,
            "request_timeout_seconds": 120,
            "evidence": "chosen_token",
        }
    }
    raw["teacher_routing"] = {
        "opd": {"revision": "route-revision", "routes": {"default": {"teacher": "primary", "weight": 1.0}}}
    }
    raw.setdefault("environment", {}).setdefault("task_sessions", {}).setdefault("nemotron_ultra", {})["grading"] = (
        "skip"
    )
    path = tmp_path / "skipped_grading.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.asyncio
async def test_pure_distillation_session_omits_verifier_credit(tmp_path):
    parsed = parse_rl_config(str(_skipped_grading_config(tmp_path, eval_interval=-1)))
    cfg = compose_skyrl_config(parsed, {"job_name": "grading-test", "num_nodes": 1}, _HPCStub()).config
    lowered = source_row_task(
        {
            "env_class": "nemotron_ultra",
            "prompt": [{"role": "user", "content": "Choose A."}],
            "extra_info": {
                "nemotron_ultra": {
                    "agent": "mcqa_simple_agent",
                    "route": "task_session",
                    "record_json": json.dumps({"expected_answer": "A"}),
                    "request_json": json.dumps({"input": []}),
                }
            },
        },
        0,
        source_name="skip-grading",
        environment_configs=OmegaConf.to_container(cfg.environment.task_sessions, resolve=True),
    )
    session = NemotronTaskSession(lowered, None)
    try:
        await session.prepare()
        result = await session.advance(
            ModelTurn({"role": "assistant", "content": r"\boxed{A}"}, (1,), (2,), (-0.1,), "stop")
        )
        assert result.done and result.reward == 0.0
        assert result.grade is not None and grade_skipped(result.grade)
    finally:
        await session.close()


def test_launcher_rejects_skipped_grading_with_eval_before_submission(tmp_path):
    parsed = parse_rl_config(str(_skipped_grading_config(tmp_path, eval_interval=5)))

    with pytest.raises(ValueError, match="eval_interval<=0"):
        compose_skyrl_config(parsed, {"job_name": "grading-test", "num_nodes": 1}, _HPCStub())
