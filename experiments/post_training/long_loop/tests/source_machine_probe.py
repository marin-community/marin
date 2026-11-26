# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise source admission without inference or sandbox allocation."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from skyrl_train.dataset.tasks import canonical_task_spec
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.long_loop.validate_launch import configure_worker_environment, validate_source_tasks


def main(tmp_path: Path) -> None:
    os.environ.pop("VERIFYIT_JUDGE_MODEL", None)
    plain = TaskSpec(
        id="plain",
        context=ConversationInput(events=(TextMessage(role="user", content="Question"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="Machine admission fixture"),
        source=Source(dataset="plain-source", revision="1", row="0", importer_revision="1"),
    )
    actor = plain.model_copy(
        update={
            "id": "thin-actor",
            "source": plain.source.model_copy(update={"dataset": "OpenResearcher/OpenResearcher-Dataset"}),
        }
    ).model_copy(
        update={
            "grader": VerifyitGrader(
                mode="judge",
                parameters={
                    "references": ["12"],
                    "rubric": "labels",
                    "system_prompt": "Grade the response.",
                    "prompt_template": "Reference: {reference}\nCandidate: {candidate}",
                    "label_scores": {"PASS": 1.0, "FAIL": 0.0},
                    "model": "",
                },
            ),
            "environment_requirements": EnvironmentRequirements.model_validate(
                {
                    "command_semantics": "linux_process",
                    "docker_image": "fixture@sha256:" + "0" * 64,
                }
            ),
        }
    )
    inputs = []
    for task in (plain, actor):
        path = tmp_path / f"{task.id}.parquet"
        pq.write_table(pa.table({"task_spec": [task.model_dump_json()]}), path)
        inputs.append({"uri": str(tmp_path), "relative_path": path.name})
    config_path = (
        Path(__file__).resolve().parents[4]
        / "vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml"
    )
    config = OmegaConf.to_container(OmegaConf.load(config_path).task_spec, resolve=True)
    launch = OmegaConf.create(
        {
            "inputs": {"train_data": inputs, "validation_data": []},
            "runtime": {"task_env": {"VERIFYIT_JUDGE_MODEL": "fixture-judge"}, "entrypoint": "training"},
            "skyrl": {
                "environment": {"task_sessions": {"task_spec": config}},
                "terminal_bench_config": {"harbor": {}},
            },
        }
    )

    def allocate():
        raise AssertionError("CPU admission must not allocate a Daytona sandbox")

    worker_config = OmegaConf.create({})
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, SimpleNamespace()),
        SimpleNamespace(),
        {"daytona": DaytonaMachineFactory(client_factory=allocate)},
        shutdown_timeout=5,
        inference_queue_timeout=2280,
        retrieval_service="http://retrieval.invalid",
        judge_connection=JudgeConnection("http://judge.invalid", "fixture-key"),
    )
    configure_worker_environment(launch)
    try:
        validate_source_tasks(launch, worker)
    except ValueError as error:
        failure = str(error)
        assert "OpenResearcher/OpenResearcher-Dataset/thin-actor" in failure
        assert "selected machine" in failure
    else:
        raise AssertionError("Unbound actor passed source admission")
    machine = {
        actor.source.dataset: {
            "backend": "daytona",
            "network": "deny",
            "cpus": 1,
            "memory_mb": 2048,
            "storage_mb": 2048,
            "gpus": 0,
            "user": None,
            "startup_timeout": 600,
            "cleanup_timeout": 120,
        }
    }
    # A retained campaign row includes both raw and prepared columns. Production
    # consumes the prepared task even when the launch has no dataset override.
    bound = canonical_task_spec(actor, {**config, "task_machine_by_dataset": machine})
    pq.write_table(
        pa.table({"task_spec": [actor.model_dump_json()], "task_rollout_spec": [bound.model_dump_json()]}),
        tmp_path / f"{actor.id}.parquet",
    )
    receipts = validate_source_tasks(launch, worker)
    assert [row["task_id"] for row in receipts] == ["plain", "thin-actor"]
    assert receipts[0]["task_machine"] is None
    assert receipts[1]["task_machine"]["backend"] == "daytona"
    assert receipts[1]["task_machine"]["network"] == "deny"

    print(json.dumps({"unbound_error": failure, "receipts": receipts}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
