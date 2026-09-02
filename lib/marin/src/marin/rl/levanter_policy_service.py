# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris lifecycle for a gang-scheduled Levanter policy service."""

import contextlib
import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace

import jax
import requests
from fray.client import JobHandle
from fray.current_client import current_client
from fray.types import Entrypoint, EnvironmentConfig, JobRequest, JobStatus, ResourceConfig
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from levanter.trainer import TrainerConfig
from marin.inference.backend import ModelSpec
from marin.inference.config import LevanterEngineConfig, ServedModelConfig
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background
from marin.inference.levanter_backend import LevanterBackend
from marin.rl.levanter_policy import (
    LevanterPolicy,
    LevanterPolicyClient,
    LevanterPolicyGroupClient,
    TorchDistributedWeightPublisher,
    build_levanter_policy_app,
)
from marin.rl.torch_distributed import init_custom_process_group
from rigging.timing import Deadline

_ENDPOINT_POLL_SECONDS = 2.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevanterPolicyServiceConfig:
    model: ServedModelConfig
    endpoint_name: str
    learning_rate: float
    clip_epsilon: float = 0.2
    tensor_parallel_size: int | None = None
    timeout_hours: float = 24.0
    port_name: str | None = "http"


@dataclass(frozen=True)
class RemoteLevanterPolicyConfig:
    service: LevanterPolicyServiceConfig
    resources: ResourceConfig
    environment: EnvironmentConfig
    endpoint_ready_timeout_seconds: float = 1800.0
    max_retries_failure: int = 1
    max_retries_preemption: int = 10
    priority: int = 0


@dataclass(frozen=True)
class RemoteLevanterPolicySession:
    policy: LevanterPolicyGroupClient
    job: JobHandle
    endpoint_name: str

    def check_alive(self) -> None:
        status = self.job.status()
        if JobStatus.finished(status):
            raise RuntimeError(f"Levanter policy job {self.job.job_id} finished unexpectedly with status {status}")


def run_iris_levanter_policy_service(config: LevanterPolicyServiceConfig) -> None:
    """Load a Levanter policy and register one private API endpoint per gang process."""
    trainer_config = TrainerConfig(require_accelerator=True)
    trainer_config.initialize()

    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("The Levanter policy service must run inside an Iris job")

    model = config.model
    backend = LevanterBackend(LevanterEngineConfig())
    spec = ModelSpec(
        weights=model.weights,
        api_model=model.model_id,
        num_chips=jax.device_count(),
        tensor_parallel_size=config.tensor_parallel_size,
        dtype=model.dtype,
        max_model_len=model.max_model_len,
        chat_template_content=model.chat_template_content,
        revision=model.revision,
    )
    with backend.load_model(spec) as loaded:
        policy = LevanterPolicy(
            loaded.model,
            learning_rate=config.learning_rate,
            clip_epsilon=config.clip_epsilon,
        )

        def configure_weight_sync(payload: dict) -> TorchDistributedWeightPublisher:
            if jax.process_index() != 0:
                raise RuntimeError("Only Levanter process zero can source the NCCL weight broadcast")
            process_group = init_custom_process_group(
                backend=payload["backend"],
                master_addr=payload["master_addr"],
                master_port=int(payload["master_port"]),
                world_size=int(payload["world_size"]),
                rank=0,
                group_name=payload["group_name"],
            )
            bridge = payload["bridge_url"].rstrip("/")

            def announce(name: str, dtype: str, shape: tuple[int, ...]) -> None:
                response = requests.post(
                    f"{bridge}/weights/update",
                    json={
                        "names": [name],
                        "dtypes": [dtype],
                        "shapes": [list(shape)],
                        "extras": [],
                        "packed": False,
                    },
                    timeout=config.timeout_hours * 3600,
                )
                response.raise_for_status()

            def complete() -> None:
                response = requests.post(f"{bridge}/weights/wait", timeout=config.timeout_hours * 3600)
                response.raise_for_status()

            logger.info("Levanter policy joined SkyRL weight-sync group")
            return TorchDistributedWeightPublisher(process_group, announce, complete)

        ctx = iris_ctx()
        port = ctx.get_port(config.port_name) if config.port_name is not None else 0
        socket = bind_serving_socket(job_info.advertise_host, port)
        address = f"http://{job_info.advertise_host}:{port}"
        metadata = {
            "kind": "levanter-policy",
            "process_index": str(jax.process_index()),
            "process_count": str(jax.process_count()),
        }
        with (
            serve_app_background(
                build_levanter_policy_app(policy, configure_weight_sync), socket, name="levanter-policy"
            ),
            ctx.registry.registered(config.endpoint_name, address, metadata),
        ):
            deadline = Deadline.from_seconds(config.timeout_hours * 3600)
            while not deadline.expired():
                time.sleep(min(_ENDPOINT_POLL_SECONDS, deadline.remaining_seconds()))


def _wait_for_policy_clients(
    job: JobHandle,
    endpoint_name: str,
    expected_processes: int,
    timeout_seconds: float,
) -> tuple[LevanterPolicyClient, ...]:
    deadline = Deadline.from_seconds(timeout_seconds)
    while True:
        status = job.status()
        if JobStatus.finished(status):
            raise RuntimeError(f"Levanter policy job finished before registering endpoints: {status}")
        instances = iris_ctx().client.list_endpoint_instances(endpoint_name)
        by_process = {
            int(instance.metadata["process_index"]): LevanterPolicyClient(instance.address)
            for instance in instances
            if instance.metadata.get("kind") == "levanter-policy"
        }
        if len(by_process) == expected_processes:
            return tuple(by_process[index] for index in range(expected_processes))
        deadline.raise_if_expired(f"Levanter policy registered {len(by_process)}/{expected_processes} process endpoints")
        time.sleep(min(_ENDPOINT_POLL_SECONDS, deadline.remaining_seconds()))


@contextlib.contextmanager
def remote_levanter_policy(config: RemoteLevanterPolicyConfig) -> Iterator[RemoteLevanterPolicySession]:
    """Launch a gang-scheduled Levanter policy and yield its SkyRL-shaped client."""
    endpoint_name = config.service.endpoint_name or f"/policy/levanter-{uuid.uuid4().hex}"
    service = replace(config.service, endpoint_name=endpoint_name)
    job = current_client().submit(
        JobRequest(
            name=f"levanter-policy-{uuid.uuid4().hex[:8]}",
            entrypoint=Entrypoint.from_callable(run_iris_levanter_policy_service, args=(service,)),
            resources=config.resources,
            replicas=config.resources.replicas,
            environment=config.environment,
            max_retries_failure=config.max_retries_failure,
            max_retries_preemption=config.max_retries_preemption,
            priority=config.priority,
        )
    )
    try:
        clients = _wait_for_policy_clients(
            job,
            endpoint_name,
            config.resources.replicas,
            config.endpoint_ready_timeout_seconds,
        )
        yield RemoteLevanterPolicySession(LevanterPolicyGroupClient(clients), job, endpoint_name)
    finally:
        job.terminate()
