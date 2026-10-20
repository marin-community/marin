import pytest
import torch
from examples.cat_count.cpu_canary import build_tokenizer
from examples.cat_count.gpu_opd import opd_config, parse_args
from skyrl_train.objective.teacher import teacher_advantages


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
            "--teacher-url",
            "http://127.0.0.1:18080/v1",
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
