import os
import subprocess
from pathlib import Path

import pytest
import torch
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
def test_gpu_opd_clip_option_controls_teacher_credit(tmp_path, flags, expected):
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
        clip=cfg.trainer.algorithm.distillation.advantage_clip,
    )
    assert advantages.item() == expected
