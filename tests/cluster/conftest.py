# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for smokes that submit a job to a standing Marin cluster.

These tests carry the ``cluster`` marker and are deselected by default (see
``addopts`` in the root ``pyproject.toml``). They run against a long-lived cluster
that is already up -- the GCP ``marin`` cluster for TPU work, ``cw-us-east-02a``
for CoreWeave GPU work -- not an ephemeral cluster the test starts itself (those
are the ``requires_cluster`` Iris smokes in ``lib/iris``).

Authentication uses whatever credential is ambient: a cached desktop login,
``$MARIN_CLUSTER_TOKEN``, or a service account (GCE metadata, a key, or an
impersonating ADC such as the CI ``IRIS_CI_GCP_SA_KEY``). When no credential can
reach the cluster the fixtures skip rather than error:

- CoreWeave (kube-fronted): opening the client loads the kubeconfig and raises
  ``ConfigException`` when it is absent.
- ``marin`` (IAP-fronted): the client connects lazily, so a missing credential
  would otherwise surface mid-submit. The IAP edge token is minted up front; with
  no ambient service-account creds that raises ``IapCredentialsUnavailable``.
"""

import contextlib
import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import JobRequest, JobStatus
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.test_util import wait_for_condition
from rigging.auth import IapCredentialsUnavailable
from rigging.filesystem.cluster_config import load_cluster_config, use_data_config
from rigging.timing import Duration

# kubernetes ships with iris[controller]. A kube-fronted cluster (CoreWeave) needs it to discover
# the controller, so its absence there means the cluster is unreachable. The IAP-fronted marin
# cluster never touches it. It is also absent from the plain unit-test env, which imports this
# module before deselecting the cluster tests, so the import must not be a hard requirement.
try:
    from kubernetes.config.config_exception import ConfigException as _KubeConfigException
except ImportError:
    _KubeConfigException = None

# Exceptions that mean "no credential/route reaches this cluster": skip rather than error.
# With kubernetes installed a missing kubeconfig raises ConfigException; without it, reaching a
# kube-fronted cluster raises ModuleNotFoundError instead, which is equally "unreachable" -> skip.
_MISSING_CREDENTIAL_ERRORS: tuple[type[BaseException], ...] = (
    (IapCredentialsUnavailable, _KubeConfigException)
    if _KubeConfigException is not None
    else (IapCredentialsUnavailable, ModuleNotFoundError)
)

# tests/cluster/conftest.py -> repo root is two parents up.
MARIN_ROOT = Path(__file__).resolve().parents[2]

# The standing GCP cluster (TPU pools) and the standing CoreWeave peer (H100).
MARIN_TPU_CLUSTER = "marin"
# CoreWeave holds the H100s but its controller is kube-fronted: reaching it directly needs a kubectl
# port-forward, which credentials without pods/portforward (the agent token, the CI kubeconfig) cannot
# open. GPU jobs instead submit through the IAP-fronted marin hub with target_cluster set to this peer,
# so the marin controller federates the work to CoreWeave without a direct tunnel.
MARIN_GPU_CLUSTER = "cw-us-east-02a"

# Region the TPU smokes pin. v6e lives in us-east5-b, so the slice and its artifacts colocate here.
MARIN_SMOKE_REGION = "us-east5"


@contextlib.contextmanager
def open_cluster_client(cluster_name: str) -> Iterator[IrisClient]:
    """Open an ``IrisClient`` for a standing Marin cluster, or skip if no credential reaches it.

    Resolves the cluster config and credentials, verifies a credential is available (see the
    module docstring), then yields a connected client bundling ``MARIN_ROOT`` as the job workspace.
    """
    with contextlib.ExitStack() as stack:
        try:
            endpoint = stack.enter_context(connect_controller(cluster_name=cluster_name))
            # Mint the IAP edge token now so a credential-less environment skips here rather than
            # erroring at submit time. No-op for kube-fronted clusters, which carry no IAP provider.
            if endpoint.credentials.iap_provider is not None:
                endpoint.credentials.iap_provider.get_token()
        except _MISSING_CREDENTIAL_ERRORS as exc:
            pytest.skip(f"Marin cluster {cluster_name!r} unavailable (no credentials): {exc}")
        client = stack.enter_context(
            IrisClient.remote(endpoint.url, workspace=MARIN_ROOT, credentials=endpoint.credentials)
        )
        yield client


@pytest.fixture
def iris_client() -> Iterator[IrisClient]:
    """The standing ``marin`` TPU cluster, also installed as the current Fray client.

    ``StepRunner`` smokes (evalchemy, SFT) submit through Fray's ``current_client()``; binding it
    here routes their jobs to the ``marin`` controller.
    """
    with open_cluster_client(MARIN_TPU_CLUSTER) as client:
        with set_current_client(FrayIrisClient.from_iris_client(client)):
            yield client


@pytest.fixture
def smoke_region() -> Iterator[str]:
    """The region the TPU smokes pin, as the single source of truth for colocation.

    The test pins the slice to this region (``ResourceConfig.regions``); this fixture binds the
    storage root to the same ``gs://marin-<region>`` for the duration, so ``marin_prefix()`` resolves
    output paths there and the job reads and writes region-locally -- no cross-region I/O. Binding the
    ``DataConfig`` root (rather than exporting ``MARIN_PREFIX``) keeps the region the only knob and
    still resolves on a metadata-less CI runner, where ``marin_prefix()`` would otherwise fall back to
    a local path.
    """
    config = dataclasses.replace(load_cluster_config(), root=f"gs://marin-{MARIN_SMOKE_REGION}")
    with use_data_config(config):
        yield MARIN_SMOKE_REGION


@pytest.fixture
def marin_gpu_client() -> Iterator[IrisClient]:
    """The IAP-fronted marin hub for the GPU vLLM e2es.

    The H100s live on the CoreWeave peer, but its controller is only reachable by a kubectl
    port-forward that agent and CI credentials cannot open. Tests submit through this hub client and
    set ``target_cluster=MARIN_GPU_CLUSTER`` on each H100 job, so the marin controller federates the
    work to CoreWeave without a direct tunnel.
    """
    with open_cluster_client(MARIN_TPU_CLUSTER) as client:
        yield client


def _task_left_queue(client: IrisClient, task_id: JobName) -> bool:
    """Whether a task has left the queue, tolerating the federation handoff window.

    A job with ``target_cluster`` set federates to a peer cluster, and its task does not exist on
    the hub until the handoff completes, so ``task_status`` raises NOT_FOUND for the first seconds
    after submit. Treat that as still queued rather than a failure.
    """
    try:
        state = client.task_status(task_id).state
    except ConnectError as exc:
        if exc.code == Code.NOT_FOUND:
            return False
        raise
    return state not in (
        job_pb2.TASK_STATE_PENDING,
        job_pb2.TASK_STATE_ASSIGNED,
        job_pb2.TASK_STATE_BUILDING,
    )


@pytest.fixture
def run_test_job() -> Callable[..., None]:
    """Return a helper that submits a ``JobRequest``, bounds its queue/runtime waits, and cleans up.

    Call it as ``run_test_job(client, request, pending_timeout=..., runtime_timeout=...)``. The job
    is terminated on interruption if it has not already finished.
    """

    def _run(
        client: IrisClient,
        request: JobRequest,
        *,
        pending_timeout: float,
        runtime_timeout: float,
    ) -> None:
        job = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
        try:
            task_id = JobName.from_string(job.job_id).task(0)
            wait_for_condition(
                lambda: _task_left_queue(client, task_id),
                timeout=Duration.from_seconds(pending_timeout),
                poll_interval=5,
            )
            job.wait(timeout=runtime_timeout, stream_logs=True)
        finally:
            if not JobStatus.finished(job.status()):
                job.terminate()

    return _run
