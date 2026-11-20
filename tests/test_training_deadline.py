# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace

import pytest
from fray.types import ResourceConfig
from levanter.main import train_lm
from levanter.main.train_lm import TrainLmConfig
from levanter.trainer import TrainerConfig
from levanter.training_budget import TrainingBudgetConfig
from marin.training import levanter as training
from marin.training.training import TrainLmOnPodConfig
from rigging.timing import Timestamp


@pytest.mark.parametrize("prior_deadline,expected", [(None, 72_010), (70_000, 70_000)])
def test_worker_binds_deadline_to_earliest_iris_execution_start(tmp_path, monkeypatch, prior_deadline, expected):
    policy = TrainingBudgetConfig(timedelta(hours=20), timedelta(minutes=40), timedelta(minutes=10), prior_deadline)
    config = TrainLmConfig(trainer=TrainerConfig(num_train_steps=746, training_budget=policy))
    pod = TrainLmOnPodConfig(
        train_config=config, resources=ResourceConfig(cpu=1, ram="1GB", disk="1GB"), output_path=str(tmp_path)
    )
    monkeypatch.setattr(training, "_prepare_training_run", lambda pod: (pod, pod.train_config, {}))
    tasks = [SimpleNamespace(started_at=value) for value in (Timestamp.from_ms(30_000), None, Timestamp.from_ms(10_000))]
    monkeypatch.setattr(
        training,
        "iris_ctx",
        lambda: SimpleNamespace(client=SimpleNamespace(list_tasks=lambda job: tasks), job_id="owned-training-job"),
    )
    received = []
    monkeypatch.setattr(train_lm, "main", received.append)
    training.run_levanter_train_lm(pod)
    assert received[0].trainer.training_budget.deadline_epoch_seconds == expected
    assert config.trainer.training_budget == policy
