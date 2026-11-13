import os
import subprocess
from pathlib import Path

import pytest
import torch
from ci.marin_nightly import opd as nightly_opd
from examples.cat_count.cpu_canary import build_tokenizer
from examples.cat_count.gpu_opd import opd_config, parse_args
from skyrl_train.objective.teacher import teacher_advantages


RUNNER = Path(__file__).parents[2] / "ci" / "marin_nightly" / "run_cat_count_opd_h100.sh"


def test_gpu_opd_runner_requires_an_explicit_spec_for_evens_only():
    env = {**os.environ, "TEACHER_ROUTES": "evens"}
    env.pop("SPEC", None)
    result = subprocess.run(
        ["bash", str(RUNNER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2


@pytest.mark.parametrize(("flags", "expected"), [([], 5.0), (["--advantage-clip", "none"], 10.0)])
def test_opd_clip_option_controls_teacher_credit(tmp_path, flags, expected):
    model = tmp_path / "policy"
    build_tokenizer().save_pretrained(model)
    args = parse_args(
        [
            "--model",
            str(model),
            "--output",
            str(tmp_path / "run"),
            "--teacher",
            "cat=http://127.0.0.1:18080/v1",
            "--teacher-revision",
            "test",
            *flags,
        ]
    )
    cfg = opd_config(args)
    advantages, _ = teacher_advantages(
        torch.tensor([[-1.0]]),
        torch.tensor([[-11.0]]),
        torch.tensor([[True]]),
        torch.ones(1, 1),
        clip=cfg.trainer.algorithm.distillation.get("advantage_clip"),
    )
    assert advantages.item() == expected


@pytest.mark.parametrize(
    ("source", "staleness", "loss"),
    [
        ("local_inference", 0, "regular"),
        ("local_inference", 1, "behavior_clip"),
        ("openai_compatible", 0, "regular"),
        ("openai_compatible", 1, "behavior_clip"),
    ],
)
def test_nightly_opd_selects_teacher_placement_and_policy_loss(tmp_path, source, staleness, loss):
    args = nightly_opd.parse_args(
        [
            "--model",
            "fixture/policy",
            "--teacher-model",
            "fixture/teacher",
            "--teacher-revision",
            "fixture",
            "--teacher-source",
            source,
            "--tokenizer-fingerprint",
            "f" * 64,
            "--data-dir",
            str(tmp_path / "data"),
            "--steps",
            "2",
            "--max-staleness-steps",
            str(staleness),
        ]
    )
    cfg = nightly_opd.opd_config(args)
    teacher = cfg.teachers.primary
    assert cfg.trainer.algorithm.policy_loss_type == loss
    if source == "local_inference":
        assert teacher.placement == "pinned"
        assert teacher.resources.gpus_per_node == 1
        assert teacher.resources.colocation_group == "teacher"
        assert "endpoints" not in teacher
    else:
        assert teacher.placement == "external"
        assert teacher.endpoints[0].url == "http://127.0.0.1:18080/v1"
        assert teacher.tokenizer_fingerprint == "f" * 64
        assert "resources" not in teacher
