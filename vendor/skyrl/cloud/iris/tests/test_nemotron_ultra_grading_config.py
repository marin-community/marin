"""Launch-time checks for skipped Nemotron Ultra grading."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config  # noqa: E402

_BASE_CONFIG = _REPO_ROOT / "cloud/iris/configs/nemotron_ultra_rlvr_acceptance.yaml"


@dataclass
class _HPCStub:
    gpus_per_node: int = 8


def _skipped_grading_config(tmp_path: Path, *, eval_interval: int) -> Path:
    raw = yaml.safe_load(_BASE_CONFIG.read_text())
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


def test_launcher_composes_skipped_grading_for_pure_distillation(tmp_path):
    parsed = parse_rl_config(str(_skipped_grading_config(tmp_path, eval_interval=-1)))

    cfg = compose_skyrl_config(parsed, {"job_name": "grading-test", "num_nodes": 1}, _HPCStub()).config

    assert cfg.environment.task_sessions.nemotron_ultra.grading == "skip"


def test_launcher_rejects_skipped_grading_with_eval_before_submission(tmp_path):
    parsed = parse_rl_config(str(_skipped_grading_config(tmp_path, eval_interval=5)))

    with pytest.raises(ValueError, match="eval_interval<=0"):
        compose_skyrl_config(parsed, {"job_name": "grading-test", "num_nodes": 1}, _HPCStub())
