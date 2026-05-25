# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from marin.rl.kl_regularization import KLConfig, KLMode
from marin.rl.opd_losses import HybridRLOOOPDSampledTokenReverseKLLoss, OPDSampledTokenReverseKLLoss
from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.teacher import TeacherConfig


def _job_config(*, rl_loss, teacher: TeacherConfig | None) -> RLJobConfig:
    return RLJobConfig(
        model=object(),
        trainer=SimpleNamespace(device_mesh=None, compute_axis_mapping={}, seed=0),
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
def test_rl_job_threads_teacher_config_to_train_worker(rl_loss):
    teacher = TeacherConfig(checkpoint="teacher-checkpoint")
    job = RLJob(_job_config(rl_loss=rl_loss, teacher=teacher))

    train_config, _rollout_config = job.to_worker_configs()

    assert train_config.teacher == teacher


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
