# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from marin.rl.kl_regularization import KLConfig, KLMode
from marin.rl.opd_losses import HybridRLOOOPDSampledTokenReverseKLLoss, OPDSampledTokenReverseKLLoss
from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.teacher import TeacherConfig
from marin.rl.train_worker import TrainWorker


def _job_config(*, rl_loss, teacher: TeacherConfig | None) -> RLJobConfig:
    return RLJobConfig(
        model=object(),
        trainer=SimpleNamespace(device_mesh=None, compute_axis_mapping={}, parameter_axis_mapping={}, seed=0),
        train_params=TrainParams(optimizer=object(), rl_loss=rl_loss),
        curriculum=SimpleNamespace(
            lessons={"lesson": SimpleNamespace(sampling_params=SimpleNamespace(n_generations_per_prompt=1))},
            max_seq_len=8,
        ),
        tokenizer=SimpleNamespace(vocab_size=8, pad_token_id=0, eos_token_id=0),
        inference_type="vllm",
        inference_config=SimpleNamespace(),
        initial_checkpoint="student-checkpoint",
        teacher=teacher,
    )


def test_rl_job_worker_config_loads_required_teacher(monkeypatch):
    calls = []

    def fake_load_model_from_checkpoint(**kwargs):
        calls.append(kwargs)
        return f"model:{kwargs['checkpoint']}"

    monkeypatch.setattr("marin.rl.train_worker.load_model_from_checkpoint", fake_load_model_from_checkpoint)

    teacher = TeacherConfig(checkpoint="teacher-checkpoint")
    job = RLJob(_job_config(rl_loss=OPDSampledTokenReverseKLLoss(), teacher=teacher))

    train_config, _rollout_config = job.to_worker_configs()
    worker = TrainWorker.__new__(TrainWorker)
    worker.config = train_config
    worker.tokenizer = train_config.tokenizer
    worker.loss_module = train_config.loss

    worker._build_models()

    assert [call["checkpoint"] for call in calls] == ["student-checkpoint", "teacher-checkpoint"]
    assert worker.teacher_model == "model:teacher-checkpoint"


@pytest.mark.parametrize(
    "rl_loss",
    [
        OPDSampledTokenReverseKLLoss(),
        HybridRLOOOPDSampledTokenReverseKLLoss(
            kl=KLConfig(mode=KLMode.NONE, beta=0.0),
            opd_coef=0.1,
        ),
    ],
)
def test_rl_job_requires_teacher_config_for_teacher_loss(rl_loss):
    job = RLJob(_job_config(rl_loss=rl_loss, teacher=None))

    with pytest.raises(ValueError, match="TeacherConfig is required"):
        job.to_worker_configs()


def test_rl_job_rejects_teacher_config_for_non_teacher_loss():
    job = RLJob(
        _job_config(
            rl_loss=RLOOLoss(kl=KLConfig(mode=KLMode.NONE, beta=0.0)),
            teacher=TeacherConfig(checkpoint="teacher-checkpoint"),
        )
    )

    with pytest.raises(ValueError, match="does not use a teacher"):
        job.to_worker_configs()
