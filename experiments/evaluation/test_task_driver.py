# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the maintained driver's Iris submission through its RPC serialization boundary."""

from types import SimpleNamespace

from iris.client.client import IrisClient
from iris.cluster.client import remote_client
from iris.cluster.types import ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from marin.evaluation.task_driver import EvaluationRunConfig, ModelIdentity
from rolloutengine.models import RolloutLimits

from experiments.evaluation.task_driver import SOLVER_API_KEY, SOLVER_ENDPOINT, submit_evaluation


def test_submission_serializes_iris_timeout_and_separates_connection_secrets(monkeypatch):
    requests = []

    def launch(request, **_kwargs):
        requests.append(request)
        return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

    monkeypatch.setattr(
        remote_client,
        "ControllerServiceClientSync",
        lambda **_kwargs: SimpleNamespace(launch_job=launch, close=lambda: None),
    )
    config = EvaluationRunConfig(
        model=ModelIdentity(
            name="solver",
            revision="model-pin",
            tokenizer="solver",
            tokenizer_revision="tokenizer-pin",
            context_window=1024,
            serving={"backend": "fixture"},
        ),
        max_tokens=16,
        concurrency=2,
        limits=RolloutLimits(
            max_turns=1,
            max_total_tokens=None,
            model_turn_timeout=10,
            tool_turn_timeout=10,
            total_turn_timeout=20,
            attempt_timeout=30,
            verifier_timeout=10,
            cleanup_timeout=10,
        ),
    )
    connections = {SOLVER_ENDPOINT: "https://solver.invalid/scoped/secret-token", SOLVER_API_KEY: "secret-solver-key"}
    with IrisClient.remote("http://controller.invalid") as client:
        job = submit_evaluation(
            client,
            ("s3://fixture/corpus",),
            "s3://fixture/results",
            config,
            name="evaluation",
            resources=ResourceSpec(cpu=2, memory="4g", disk="8g"),
            timeout=73,
            secrets=connections,
        )
    (request,) = requests
    assert job.job_id.to_wire() == request.name
    assert request.timeout.milliseconds == 73_000
    assert request.container_profile == job_pb2.CONTAINER_PROFILE_PRIVILEGED
    assert not request.resources.HasField("device")
    for name, value in connections.items():
        assert request.environment.secrets[name] == value
        assert value.encode() not in request.entrypoint.SerializeToString()
