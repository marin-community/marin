# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a model, run evalchemy against its OpenAI URL, tear the server down.

The eval is decoupled from the model backend by an OpenAI-compatible URL (issue #4827): a marin-serve
child job (``VllmBackend`` or ``LevanterBackend``, on a TPU/GPU slice) exposes an endpoint, and an
evalchemy child job (the ``:evalchemy-tpu`` container on a CPU slice) hits it with
``eval.eval --model local-completions``. Evalchemy is the sole eval client.

Topology — one parent orchestrator job spawns a serve child, then one eval child per eval unit
against the same endpoint (:func:`run_eval_units` serves once for a whole suite)::

    parent (CPU, marin)  ──serve child──▶  marin-serve backend (TPU/GPU)  ──▶ OpenAI endpoint
                         ──eval child(s)──▶  :evalchemy-tpu (CPU)  ──local-completions──▶ endpoint

:func:`serve_model` submits the serve child, waits for its endpoint to register, and yields the
served backend's in-cluster address. The eval child is pinned to the serve region, so it reaches
that address straight over the same-cluster VPC -- no controller proxy or capability token. Iris
auto-cleans child jobs when the parent ends, so leaving the ``with`` block (or the parent exiting)
tears the server down; the context manager also stops it eagerly for promptness.

The two children take different entrypoints. The serve child runs in the marin image, whose synced
venv can deserialize a cloudpickled callable, so it uses ``Entrypoint.from_callable``. The eval child
runs in the ``:evalchemy-tpu`` image, whose default interpreter is a bare python with no cloudpickle
-- only ``/opt/openthoughts/.venv`` carries ``eval``/``lm_eval``/``fsspec`` -- so it runs
:mod:`experiments.evals.evalchemy.run_evalchemy_client` as a plain command under that interpreter,
with its config passed as JSON in an env var.

Top-level imports are kept light: the serve child's ``VllmBackend``/``LevanterBackend``
construction (which pulls levanter + vLLM) happens lazily inside :func:`_serve_for_eval`, so the CPU
parent that references it for cloudpickling never imports the serving stack.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum

from iris.client import Job, JobFailedError, iris_ctx
from iris.cluster.constraints import region_constraint
from iris.cluster.setup_scripts import default_setup_script
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    gpu_device,
    is_job_finished,
    tpu_device,
)
from marin.evaluation.evaluation_config import EvalTaskConfig

from experiments.evals.evalchemy.image import EVALCHEMY_IMAGE, EVALCHEMY_PYTHON
from experiments.evals.evalchemy.run_evalchemy_client import CONFIG_ENV_KEY

logger = logging.getLogger(__name__)

# How long to wait for the served endpoint to register before giving up (model download + compile).
ENDPOINT_READY_TIMEOUT_SECONDS = 2400
_ENDPOINT_POLL_SECONDS = 10.0
# The served model self-stops after this wall-clock lifetime, a backstop in case the parent dies
# before it can tear the child down.
SERVE_TIMEOUT_HOURS = 4.0
# 512 matches the quick-serve default: on the current TPU vLLM stack the prompt-logprobs path kills
# the whole engine within minutes of MCQ traffic at 2048 (five out of five serves died on first
# logprob bursts; generation-only traffic was unaffected), so larger prefill budgets are not safe
# until the fork's prompt-logprobs handling is fixed.
EVAL_SERVE_MAX_NUM_BATCHED_TOKENS = 512
# lm-eval's local-completions client concurrency (parallel in-flight requests to the endpoint).
DEFAULT_NUM_CONCURRENT = 16
# Credentials the child jobs need (HF model/dataset downloads, wandb logging); the parent propagates
# whichever of these it holds, since a child does not inherit the parent's process env.
_PROPAGATED_ENV_KEYS = ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT")
# How many trailing log lines of a failed child ride along in EvalPipelineError (and from there into
# the run's failure record).
LOG_TAIL_LINES = 100


class PipelineStage(StrEnum):
    """Which stage of the serve/eval pipeline a failure came from.

    ``EVAL`` means the model was served and the eval client itself failed -- a result about the model.
    ``SERVE`` and ``ARTIFACTS`` are pipeline failures: the endpoint never came up, or the eval's
    results tree never reached durable storage.
    """

    SERVE = "serve"
    EVAL = "eval"
    ARTIFACTS = "artifacts"


class EvalPipelineError(RuntimeError):
    """A serve/eval pipeline failure carrying the child jobs' identities and final log lines.

    ``jobs`` maps pipeline role (``serve``/``eval``) to iris job path for every child submitted before
    the failure; ``log_tails`` maps the failed role(s) to their last :data:`LOG_TAIL_LINES` log lines,
    so the failure record is diagnosable without cluster access.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: PipelineStage,
        jobs: dict[str, str],
        log_tails: dict[str, tuple[str, ...]],
    ):
        super().__init__(message)
        self.stage = stage
        self.jobs = jobs
        self.log_tails = log_tails


def job_log_tail(job: Job, limit: int = LOG_TAIL_LINES) -> tuple[str, ...]:
    """The last ``limit`` log lines across ``job``'s tasks (the pipeline children run one task each).

    Diagnostics for a failure already being raised: a log-fetch error is logged and yields ``()``
    rather than masking the original failure.
    """
    try:
        entries = [entry for task in job.tasks() for entry in task.logs()]
    except Exception:
        logger.warning("could not fetch log tail for %s", job, exc_info=True)
        return ()
    return tuple(entry.data.rstrip("\n") for entry in entries[-limit:])


def _propagated_env(**extra: str) -> dict[str, str]:
    env = {key: os.environ[key] for key in _PROPAGATED_ENV_KEYS if os.environ.get(key)}
    env.update(extra)
    return env


class ServeBackend(StrEnum):
    """Which marin-serve backend serves the model under eval. Both expose the same OpenAI API, so the
    eval client is identical either way."""

    VLLM = "vllm"
    LEVANTER = "levanter"


@dataclass(frozen=True)
class ServeSpec:
    """Which backend serves the model under eval, and on what slice.

    Exactly one of ``tpu_type`` / (``gpu_type``, ``gpu_count``) is set.
    """

    backend: ServeBackend = ServeBackend.VLLM
    tpu_type: str | None = "v6e-8"
    gpu_type: str | None = None
    gpu_count: int | None = None
    max_model_len: int | None = None
    tensor_parallel_size: int | None = None
    dtype: str = "bfloat16"
    region: str | None = None
    serve_cpu: float = 8.0
    serve_memory: str = "64g"
    serve_disk: str = "100g"
    timeout_hours: float = SERVE_TIMEOUT_HOURS
    """Serve-child self-stop lifetime, a backstop in case the parent dies before tearing it down.
    Must cover the whole eval suite the session runs against the endpoint; the launcher scales it
    with the number of evals in the group."""
    max_num_batched_tokens: int = EVAL_SERVE_MAX_NUM_BATCHED_TOKENS
    """vLLM prefill budget per engine step. The 512 default is conservative: 2048 boots on the current
    TPU stack but prompt-logprobs traffic then kills the engine within minutes."""
    vllm_extra_args: tuple[str, ...] = ()
    """Extra flags forwarded to ``vllm serve`` (``VllmBackend.extra_args``); empty for the common case.
    Use it for models the portable defaults miss:

    - 256-expert Grug MoE export: ``--data-parallel-size N --enable-expert-parallel
      --model-loader-extra-config '{"distributed":true}'`` with ``tensor_parallel_size=1``; the per-head
      TP heuristic cannot infer this.
    - Qwen gated-delta-net models (``qwen_gdn_linear_attn``: ``Qwen/Qwen3.5-35B-A3B``,
      ``Qwen/Qwen3-Next-80B-A3B``): ``--gdn-prefill-backend triton``. The GPU serve env runs without
      ``nvcc``, so the default FlashInfer GDN prefill kernel — JIT-compiled at warmup — fails; triton
      needs no compiler."""
    chat_template_content: str | None = None
    """Chat template (jinja) served so ``/v1/chat/completions`` templates server-side, required when the
    eval uses ``--apply_chat_template`` and the model's own repo does not carry a vLLM-loadable template.
    Passed straight to :class:`~marin.inference.quick_serve.QuickServeConfig`."""


@dataclass(frozen=True)
class ServedEndpoint:
    """A live OpenAI-compatible endpoint an evalchemy client can call over the same-cluster VPC."""

    base_url: str
    """OpenAI API root (ends in ``/v1``); the served backend's in-cluster address."""
    model_id: str
    """The model id the served backend reports (matches ``ServeSpec``'s model)."""
    tokenizer: str
    """HF tokenizer id/path the eval client uses to build prompts."""
    job: str
    """Iris job path of the serve child behind the endpoint."""
    handle: Job
    """Live handle to the serve child, for liveness checks and log tails between evals."""
    name: str
    """Registry endpoint name the serve child registers; a restarted serve attempt re-registers it
    at its new address, so resolving the name again finds the live server after a preemption."""


@dataclass(frozen=True)
class EvalchemyEvalConfig:
    """Everything the parent orchestrator needs to serve a model and eval it. Cloudpickled into the job."""

    model: str
    """HF repo id or object-store (``gs://``) path of the model to serve and eval."""
    tasks: tuple[EvalTaskConfig, ...]
    out_path: str | None = None
    """Object-store destination for the eval child's ``results_*.json`` tree, read back by
    :class:`~marin.evaluation.eval_result.EvalchemyResult`. Resolved by
    :func:`_resolve_durable_out_path`: an object-store path is used verbatim; ``None`` or a pod-local
    path is routed under the cluster's ``marin_prefix()`` so results survive pod teardown."""
    serve: ServeSpec = field(default_factory=ServeSpec)
    tokenizer: str | None = None
    """HF tokenizer id the eval client loads to build prompts; defaults to ``model``. Set it when
    ``model`` is a path the eval image cannot load a tokenizer from (e.g. a ``gs://`` checkpoint)."""
    max_gen_toks: int = 2048
    apply_chat_template: bool = False
    max_eval_instances: int | None = None
    num_concurrent: int = DEFAULT_NUM_CONCURRENT
    eval_image: str = EVALCHEMY_IMAGE
    eval_cpu: float = 8.0
    eval_memory: str = "32g"
    eval_disk: str = "50g"


# --------------------------------------------------------------------------------------------------
# Serve child: build the marin-serve config + backend and block serving (runs in the marin image).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ServeParams:
    """The serve child's inputs, cloudpickled into it. Flattened from ``ServeSpec`` so the child never
    imports this module's serving-stack helpers just to read a config."""

    model: str
    endpoint_name: str
    backend: ServeBackend
    tpu_type: str | None
    gpu_type: str | None
    gpu_count: int | None
    dtype: str
    max_model_len: int | None
    tensor_parallel_size: int | None
    timeout_hours: float
    startup_timeout_seconds: int
    max_num_batched_tokens: int
    vllm_extra_args: tuple[str, ...]
    chat_template_content: str | None


def _serve_for_eval(params: _ServeParams) -> None:
    """Serve-child entrypoint: construct the backend + quick-serve config and serve until stopped.

    Runs in the marin task image on the serving slice. Imports the serving stack lazily so the CPU
    parent that references this function for cloudpickling never pulls levanter/vLLM.
    """
    from marin.inference.quick_serve import (  # noqa: PLC0415  # lazy: keep vLLM/levanter out of the CPU parent
        QuickServeConfig,
        serve_in_job,
    )
    from marin.inference.serving_backend import (  # noqa: PLC0415  # lazy: keep vLLM/levanter out of the CPU parent
        LevanterBackend,
        VllmBackend,
    )
    from marin.inference.vllm_server import (  # noqa: PLC0415  # lazy: keep vLLM out of the CPU parent
        IsolatedCudaVllm,
        VllmType,
    )
    from rigging.log_setup import configure_logging  # noqa: PLC0415  # lazy: only needed in the serve child

    configure_logging()
    if params.backend == ServeBackend.VLLM:
        # GPU serves the Marin vLLM fork (grug_moe et al.) from an isolated uvx env; TPU serves the
        # workspace TPU-vLLM stack already on PATH (WorkspaceVllm, the default launcher).
        launcher = IsolatedCudaVllm(source=VllmType.MARIN_FORK) if params.gpu_count is not None else None
        backend = VllmBackend(
            launcher=launcher,
            startup_timeout_seconds=params.startup_timeout_seconds,
            max_num_batched_tokens=params.max_num_batched_tokens,
            # warning: an lm-eval suite makes ~10^5 completions requests, and uvicorn's per-request
            # INFO access lines would drown the serve log (errors and tracebacks still print).
            extra_args=(*params.vllm_extra_args, "--uvicorn-log-level", "warning"),
        )
    elif params.backend == ServeBackend.LEVANTER:
        backend = LevanterBackend()
    else:
        raise ValueError(f"unknown serve backend {params.backend!r}")

    config = QuickServeConfig(
        model=params.model,
        endpoint_name=params.endpoint_name,
        backend=backend,
        tpu_type=params.tpu_type,
        gpu_type=params.gpu_type,
        gpu_count=params.gpu_count,
        dtype=params.dtype,
        max_model_len=params.max_model_len,
        tensor_parallel_size=params.tensor_parallel_size,
        chat_template_content=params.chat_template_content,
        timeout_hours=params.timeout_hours,
    )
    serve_in_job(config)


def _serve_environment(spec: ServeSpec) -> EnvironmentSpec:
    """Worker environment for the serve child, by slice and backend.

    - GPU + vLLM: base ``marin-core`` only. The fork, its precompiled/cu130 build, the Run:ai
      streamer, and the virtual-hosted S3 addressing are all owned by ``IsolatedCudaVllm(MARIN_FORK)``
      (uvx), so the worker venv just needs enough to run ``serve_in_job``.
    - GPU + Levanter: the ``gpu`` build (the JAX/Levanter GPU stack) suffices; Levanter serves in-process.
    - TPU + vLLM: the ``tpu``+``vllm`` build. TPU + Levanter: only jax (the ``tpu`` extra).
    """
    if spec.gpu_count is not None:
        if spec.backend == ServeBackend.LEVANTER:
            return EnvironmentSpec(extras=("gpu",), env_vars=_propagated_env())
        return EnvironmentSpec(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars=_propagated_env(),
        )
    if spec.backend == ServeBackend.LEVANTER:
        extras = ("tpu",)
    else:
        extras = ("tpu", "vllm")
    return EnvironmentSpec(extras=extras, env_vars=_propagated_env())


def _serve_device(spec: ServeSpec):
    if spec.gpu_count is not None:
        return gpu_device(spec.gpu_type or "H100", spec.gpu_count)
    if spec.tpu_type is None:
        raise ValueError("ServeSpec needs tpu_type (TPU path) or gpu_type/gpu_count (GPU path).")
    return tpu_device(spec.tpu_type)


@contextmanager
def serve_model(model: str, tokenizer: str, spec: ServeSpec) -> Iterator[ServedEndpoint]:
    """Serve ``model`` on a marin-serve child job and yield its in-cluster OpenAI URL.

    Submits the serve child, waits for its endpoint to register, and yields a :class:`ServedEndpoint`.
    On exit the serve child is stopped eagerly; Iris also auto-cleans it when the parent job ends, so
    the server never outlives the eval.
    """
    ctx = iris_ctx()
    if ctx is None or ctx.client is None:
        raise RuntimeError("serve_model must run inside an Iris job (no ambient IrisClient).")
    client = ctx.client

    run_id = uuid.uuid4().hex[:8]
    endpoint_name = f"/serve/eval-{run_id}"
    params = _ServeParams(
        model=model,
        endpoint_name=endpoint_name,
        backend=spec.backend,
        tpu_type=spec.tpu_type,
        gpu_type=spec.gpu_type,
        gpu_count=spec.gpu_count,
        dtype=spec.dtype,
        max_model_len=spec.max_model_len,
        tensor_parallel_size=spec.tensor_parallel_size,
        timeout_hours=spec.timeout_hours,
        startup_timeout_seconds=int(ENDPOINT_READY_TIMEOUT_SECONDS),
        max_num_batched_tokens=spec.max_num_batched_tokens,
        vllm_extra_args=spec.vllm_extra_args,
        chat_template_content=spec.chat_template_content,
    )
    constraints = [region_constraint([spec.region])] if spec.region else None

    serve_job = client.submit(
        entrypoint=Entrypoint.from_callable(_serve_for_eval, params),
        name=f"eval-serve-{run_id}",
        resources=ResourceSpec(
            cpu=spec.serve_cpu, memory=spec.serve_memory, disk=spec.serve_disk, device=_serve_device(spec)
        ),
        environment=_serve_environment(spec),
        ports=["http"],
        constraints=constraints,
        # One retry: first placements are occasionally poisoned by host-global port collisions
        # (libtpu's fixed :8431 grabbed by a co-tenant) that a reschedule escapes. The endpoint
        # wait tracks job state, so a retry is transparent to it.
        max_retries_failure=1,
    )
    logger.info("Submitted serve job %s (backend=%s) for endpoint %s", serve_job, spec.backend, endpoint_name)
    serve_path = str(serve_job.job_id)
    try:
        try:
            _wait_for_endpoint(client, serve_job, endpoint_name)
        except (RuntimeError, TimeoutError) as exc:
            raise EvalPipelineError(
                str(exc),
                stage=PipelineStage.SERVE,
                jobs={"serve": serve_path},
                log_tails={"serve": job_log_tail(serve_job)},
            ) from exc
        # In-cluster, resolve_endpoint returns the served backend's direct address (its dashboard's
        # OpenAI reverse proxy); the eval child, colocated in the same region/VPC, calls it straight
        # -- no controller proxy or capability token needed for a same-cluster consumer.
        base_url = client.resolve_endpoint(endpoint_name).rstrip("/") + "/v1"
        logger.info("Serve endpoint %s ready at %s", endpoint_name, base_url)
        yield ServedEndpoint(
            base_url=base_url,
            model_id=model,
            tokenizer=tokenizer,
            job=serve_path,
            handle=serve_job,
            name=endpoint_name,
        )
    finally:
        with suppress(Exception):
            serve_job.terminate()
            logger.info("Terminated serve job %s", serve_job)


def _wait_for_endpoint(client, serve_job, endpoint_name: str) -> None:
    """Poll the controller registry until the serve endpoint registers, or the serve job dies."""
    deadline = time.monotonic() + ENDPOINT_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if is_job_finished(serve_job.state):
            raise RuntimeError(f"serve job {serve_job} finished before registering endpoint {endpoint_name!r}.")
        endpoints = client._cluster_client.list_endpoints(endpoint_name, exact=True)
        if endpoints:
            return
        time.sleep(_ENDPOINT_POLL_SECONDS)
    raise TimeoutError(
        f"timed out after {ENDPOINT_READY_TIMEOUT_SECONDS}s waiting for endpoint {endpoint_name!r} to register."
    )


# --------------------------------------------------------------------------------------------------
# Eval child: run evalchemy as an OpenAI client (runs in the :evalchemy-tpu image on a CPU slice).
# --------------------------------------------------------------------------------------------------

# The eval child runs run_evalchemy_client.py under the image's own interpreter (the only one with
# eval/lm_eval/fsspec). $IRIS_WORKDIR is exported into the job env and holds the synced workspace.
_EVAL_CLIENT_SCRIPT = "experiments/evals/evalchemy/run_evalchemy_client.py"


def _task_dir(task: EvalTaskConfig) -> str:
    """The per-task upload subdirectory, unique per task-config.

    The alias, when the author set one, is the full identity (they encode shots there when they run one
    task at several shot counts, e.g. ``hellaswag_0shot`` vs ``hellaswag_10shot``); otherwise it is the
    task name plus its shot count. Distinct dirs keep shot variants of one task from overwriting each
    other, since lm-eval keys its own results by the bare task name -- see
    :class:`~marin.evaluation.eval_result.EvalchemyResult`.
    """
    return task.task_alias or f"{task.name}_{task.num_fewshot}shot"


def _client_config_json(session: EvalSession, unit: EvalUnit, endpoint: ServedEndpoint) -> str:
    """The evalchemy client's config as a JSON string, passed to the eval child in an env var.

    A plain JSON payload (not a cloudpickled object) so the image's bare interpreter can read it
    without any marin/cloudpickle import -- see :mod:`experiments.evals.evalchemy.run_evalchemy_client`.
    """
    return json.dumps(
        {
            "base_url": endpoint.base_url,
            "model_id": endpoint.model_id,
            "tokenizer": endpoint.tokenizer,
            "tasks": [
                {
                    "name": t.name,
                    "num_fewshot": t.num_fewshot,
                    "dir": _task_dir(t),
                    "generation": t.generation,
                    "unsafe_code": t.unsafe_code,
                    "completion_only": t.completion_only,
                }
                for t in unit.tasks
            ],
            "out_path": unit.out_path,
            "apply_chat_template": session.apply_chat_template,
            "max_gen_toks": unit.max_gen_toks,
            "max_eval_instances": unit.max_eval_instances,
            "num_concurrent": session.num_concurrent,
        }
    )


# --------------------------------------------------------------------------------------------------
# Parent orchestrator: serve once, eval N units against the URL, tear down (marin image, CPU).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSession:
    """The shared serving context for a group of evals: one model boot, N evals against it."""

    model: str
    """HF repo id or object-store (``gs://``) path of the model to serve."""
    serve: ServeSpec = field(default_factory=ServeSpec)
    tokenizer: str | None = None
    """HF tokenizer id the eval client loads to build prompts; defaults to ``model``. Required when
    ``model`` is a path the eval image cannot load a tokenizer from (e.g. a ``gs://`` checkpoint)."""
    apply_chat_template: bool = False
    num_concurrent: int = DEFAULT_NUM_CONCURRENT
    eval_image: str = EVALCHEMY_IMAGE
    eval_cpu: float = 8.0
    eval_memory: str = "32g"
    eval_disk: str = "50g"


@dataclass(frozen=True)
class EvalUnit:
    """One independently recorded eval within a session: its tasks, limits, and durable destination."""

    name: str
    """Short label for the unit (the launcher's eval key); used in child job names and logs."""
    tasks: tuple[EvalTaskConfig, ...]
    out_path: str
    """Object-store destination for this unit's ``results_*.json`` tree and sample parquets."""
    max_gen_toks: int = 2048
    max_eval_instances: int | None = None


@dataclass(frozen=True)
class EvalUnitOutcome:
    """One unit's result: the child jobs it ran, by role, and the failure, if any.

    ``error is None`` means the unit's eval child succeeded, its results tree was verified durable,
    and its per-sample parquets were exported.
    """

    unit: EvalUnit
    jobs: dict[str, str]
    error: EvalPipelineError | None


# How long the between-units endpoint probe waits before declaring the served endpoint dead.
_ENDPOINT_PROBE_TIMEOUT_SECONDS = 15.0
_CHILD_WAIT_SLICE_SECONDS = 60.0


def _refresh_endpoint(endpoint: ServedEndpoint) -> ServedEndpoint | None:
    """The endpoint at its currently registered address, or None when the serve is gone.

    The serve slice is preemptible: a preempted attempt restarts on a new host and re-registers its
    endpoint name at a new address. Resolving the name again (waiting out a mid-restart gap) keeps
    later units pointed at the live server instead of the address captured at group start.
    """
    client = iris_ctx().client
    if is_job_finished(endpoint.handle.state):
        return None
    try:
        _wait_for_endpoint(client, endpoint.handle, endpoint.name)
        base_url = client.resolve_endpoint(endpoint.name).rstrip("/") + "/v1"
    except (ConnectionError, RuntimeError, TimeoutError):
        # ConnectionError: the registration vanished between the wait and the resolve.
        return None
    if base_url == endpoint.base_url:
        return endpoint
    logger.info("serve endpoint %s moved %s -> %s (attempt restarted)", endpoint.name, endpoint.base_url, base_url)
    return replace(endpoint, base_url=base_url)


def _serve_death_outcome(unit: EvalUnit, endpoint: ServedEndpoint, serve_tail: tuple[str, ...]) -> EvalUnitOutcome:
    dead = EvalPipelineError(
        f"serve endpoint died before eval {unit.name!r} ran",
        stage=PipelineStage.SERVE,
        jobs={"serve": endpoint.job},
        log_tails={"serve": serve_tail},
    )
    return EvalUnitOutcome(unit=unit, jobs=dict(dead.jobs), error=dead)


def _endpoint_departed(endpoint: ServedEndpoint) -> bool:
    """Whether the serve provably left the address a running eval child was configured with.

    True when the serve job is finished or its endpoint name now resolves to a different address
    (the attempt restarted after preemption). A transient resolve failure returns False: a
    mid-restart registration gap resolves to the new address on a later poll.
    """
    client = iris_ctx().client
    if is_job_finished(endpoint.handle.state):
        return True
    try:
        base_url = client.resolve_endpoint(endpoint.name).rstrip("/") + "/v1"
    except (ConnectionError, RuntimeError, TimeoutError):
        # ConnectionError: nothing registered right now (a mid-restart gap) -- not yet proof the
        # serve left this address.
        return False
    return base_url != endpoint.base_url


def _endpoint_alive(endpoint: ServedEndpoint) -> bool:
    """Whether the served endpoint still answers: serve job running and ``/v1/models`` returning 200.

    Probed after a unit fails so the failure is attributed correctly: a dead server fails the
    remaining units as serve failures instead of a string of misleading eval failures.
    """
    if is_job_finished(endpoint.handle.state):
        return False
    try:
        with urllib.request.urlopen(f"{endpoint.base_url}/models", timeout=_ENDPOINT_PROBE_TIMEOUT_SECONDS) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _run_one_unit(session: EvalSession, unit: EvalUnit, endpoint: ServedEndpoint) -> EvalUnitOutcome:
    """Run one unit's eval child against ``endpoint``, verify its artifacts, and export its parquets."""
    jobs: dict[str, str] = {"serve": endpoint.job}
    try:
        jobs["eval"] = _submit_eval_child(session, unit, endpoint)
    except EvalPipelineError as exc:
        exc.jobs = {**jobs, **exc.jobs}
        return EvalUnitOutcome(unit=unit, jobs=dict(exc.jobs), error=exc)
    try:
        _verify_durable_artifacts(unit.out_path)
        # Lazy: pyarrow/pydantic are only needed for this post-processing step.
        from marin.evaluation.samples import export_lm_eval_samples  # noqa: PLC0415

        parquets = export_lm_eval_samples(unit.out_path)
    except Exception as exc:
        error = EvalPipelineError(str(exc), stage=PipelineStage.ARTIFACTS, jobs=dict(jobs), log_tails={})
        error.__cause__ = exc
        return EvalUnitOutcome(unit=unit, jobs=dict(jobs), error=error)
    logger.info("unit %s: wrote %d per-task sample parquet file(s) under %s", unit.name, len(parquets), unit.out_path)
    return EvalUnitOutcome(unit=unit, jobs=dict(jobs), error=None)


def run_eval_units(session: EvalSession, units: Sequence[EvalUnit]) -> Iterator[EvalUnitOutcome]:
    """Serve ``session.model`` once and evaluate each unit against the endpoint, in order.

    Yields one outcome per unit as it finishes, so callers can record results progressively across an
    hours-long suite. One unit failing does not stop the session unless the served endpoint itself
    died; then the remaining units are yielded as serve-stage failures without running. The serve
    child is torn down when the generator finishes, so callers must consume it fully.
    """
    if not units:
        raise ValueError("run_eval_units requires at least one unit")
    for unit in units:
        if not unit.tasks:
            raise ValueError(f"eval unit {unit.name!r} has no tasks")
        if "://" not in unit.out_path:
            raise ValueError(f"eval unit {unit.name!r} out_path {unit.out_path!r} is not an object-store path")
    # The served backend loads the model from any fsspec path (gs://...), but the eval client loads
    # its tokenizer through HF, which cannot read an object-store path. Fail fast rather than let
    # every eval child die deep in lm-eval.
    if session.tokenizer is None and "://" in session.model:
        raise ValueError(
            f"model {session.model!r} is an object-store path the eval image cannot load a tokenizer "
            "from; set EvalSession.tokenizer to the base model's HF id."
        )
    stack = ExitStack()
    try:
        endpoint = stack.enter_context(serve_model(session.model, session.tokenizer or session.model, session.serve))
    except EvalPipelineError as exc:
        for unit in units:
            yield EvalUnitOutcome(unit=unit, jobs=dict(exc.jobs), error=exc)
        return
    with stack:
        pending = list(units)
        restart_retried: set[str] = set()
        while pending:
            unit = pending.pop(0)
            live = _refresh_endpoint(endpoint)
            if live is None:
                serve_tail = job_log_tail(endpoint.handle)
                for rest in [unit, *pending]:
                    yield _serve_death_outcome(rest, endpoint, serve_tail)
                return
            endpoint = live
            outcome = _run_one_unit(session, unit, endpoint)
            if outcome.error is None:
                yield outcome
                continue
            live = _refresh_endpoint(endpoint)
            if live is not None and live.base_url != endpoint.base_url and unit.name not in restart_retried:
                # The serve attempt restarted mid-unit (preemption) and came back elsewhere; the
                # failure is the stale address, not the eval. Run the unit once more against the
                # live server before believing any of its failures.
                logger.info("unit %s: serve moved mid-run; retrying it against %s", unit.name, live.base_url)
                restart_retried.add(unit.name)
                endpoint = live
                pending.insert(0, unit)
                continue
            if live is not None and _endpoint_alive(live):
                endpoint = live
                yield outcome
                continue
            # The server died under this unit: re-stage its failure as a serve failure with the serve
            # tail attached, and fail the rest without running them.
            serve_tail = job_log_tail(endpoint.handle)
            failed = EvalPipelineError(
                f"{outcome.error}; the serve endpoint is dead after this failure",
                stage=PipelineStage.SERVE,
                jobs={**outcome.error.jobs, "serve": endpoint.job},
                log_tails={**outcome.error.log_tails, "serve": serve_tail},
            )
            failed.__cause__ = outcome.error
            yield EvalUnitOutcome(unit=unit, jobs=dict(failed.jobs), error=failed)
            for rest in pending:
                yield _serve_death_outcome(rest, endpoint, serve_tail)
            return


def _resolve_durable_out_path(out_path: str | None, run_id: str) -> str:
    """Resolve the object-store destination for the eval child's artifacts.

    An object-store path (contains ``://``) is returned verbatim. Anything else -- ``None`` or a
    pod-local path, which is garbage-collected when the eval pod ends -- is routed under the active
    cluster's ``marin_prefix()``, keyed by ``run_id``. ``marin_prefix()`` resolves the region-local
    store, so no bucket is hardcoded.
    """
    if out_path and "://" in out_path:
        return out_path.rstrip("/")
    from rigging.filesystem import marin_prefix, prefix_join  # noqa: PLC0415  # lazy: keep rigging out of the parent

    durable = prefix_join(marin_prefix(), f"eval/evalchemy/{run_id}")
    if out_path:
        logger.warning(
            "out_path %r is not an object-store path; routing eval artifacts to durable storage %r.",
            out_path,
            durable,
        )
    return durable


def _verify_durable_artifacts(out_path: str) -> None:
    """Raise if no ``results_*.json`` reached ``out_path``.

    Runs in the parent (marin image, which carries rigging + s3fs) after the eval child finishes, so a
    succeeded child cannot report success over an empty prefix.
    """
    from rigging.filesystem import url_to_fs  # noqa: PLC0415  # lazy: keep rigging out of the CPU parent's import

    fs, path = url_to_fs(out_path)
    objects = fs.find(path)
    results = [p for p in objects if p.rsplit("/", 1)[-1].startswith("results_") and p.endswith(".json")]
    logger.info(
        "Durable evalchemy artifacts under %s: %d object(s), %d results_*.json", out_path, len(objects), len(results)
    )
    if not results:
        raise RuntimeError(f"no evalchemy results_*.json landed under {out_path!r}; eval artifacts were lost")


@dataclass(frozen=True)
class ServeAndEvalRun:
    """A completed pipeline: the durable results path and the child jobs it ran, by role."""

    out_path: str
    jobs: dict[str, str]


def serve_and_eval(config: EvalchemyEvalConfig) -> ServeAndEvalRun:
    """Parent entrypoint for a single-record run: serve the model, run all tasks as one eval, tear down.

    The single-unit wrapper over :func:`run_eval_units`, kept as the composable pipeline path's
    entrypoint (one ``EvalchemyResult`` artifact per step). The eval child's artifacts are routed to
    durable object storage (:func:`_resolve_durable_out_path`) and verified back before the parent
    returns. Failures raise :class:`EvalPipelineError` carrying the child jobs submitted so far and
    the failed child's log tail.
    """
    if not config.tasks:
        raise ValueError("serve_and_eval requires at least one task")
    run_id = uuid.uuid4().hex[:8]
    durable_out_path = _resolve_durable_out_path(config.out_path, run_id)
    logger.info("evalchemy artifacts for this run route to durable storage %s", durable_out_path)
    session = EvalSession(
        model=config.model,
        serve=config.serve,
        tokenizer=config.tokenizer,
        apply_chat_template=config.apply_chat_template,
        num_concurrent=config.num_concurrent,
        eval_image=config.eval_image,
        eval_cpu=config.eval_cpu,
        eval_memory=config.eval_memory,
        eval_disk=config.eval_disk,
    )
    unit = EvalUnit(
        name="eval",
        tasks=config.tasks,
        out_path=durable_out_path,
        max_gen_toks=config.max_gen_toks,
        max_eval_instances=config.max_eval_instances,
    )
    (outcome,) = tuple(run_eval_units(session, (unit,)))
    if outcome.error is not None:
        raise outcome.error
    return ServeAndEvalRun(out_path=durable_out_path, jobs=outcome.jobs)


def _submit_eval_child(session: EvalSession, unit: EvalUnit, endpoint: ServedEndpoint) -> str:
    """Submit the evalchemy client child job for ``unit`` against ``endpoint``, block until it
    finishes, and return its iris job path."""
    client = iris_ctx().client
    child_id = uuid.uuid4().hex[:8]
    # Colocate the eval client with the serving slice so it reaches the served address over the
    # same-region VPC.
    constraints = [region_constraint([session.serve.region])] if session.serve.region else None
    # A command entrypoint, not from_callable: the eval image's synced interpreter has no cloudpickle,
    # so the client runs under EVALCHEMY_PYTHON (which does) with its config passed as JSON in an env var.
    command = f'exec {EVALCHEMY_PYTHON} "$IRIS_WORKDIR/{_EVAL_CLIENT_SCRIPT}"'
    eval_job = client.submit(
        entrypoint=Entrypoint.from_command("bash", "-c", command),
        name=f"eval-{unit.name.replace('.', '-')}-{child_id}",
        resources=ResourceSpec(cpu=session.eval_cpu, memory=session.eval_memory, disk=session.eval_disk),
        # The evalchemy image runs as a pure HTTP client here, so keep JAX off the (TPU-tagged) image's
        # TPU init path and let humaneval-style code_eval tasks execute generated code.
        environment=EnvironmentSpec(
            env_vars=_propagated_env(
                JAX_PLATFORMS="cpu",
                HF_ALLOW_CODE_EVAL="1",
                # lm-eval attaches OPENAI_API_KEY as a bearer header on every request and its retry
                # logging dumps the header; the served endpoint ignores auth, so pin a dummy here so
                # no real key reaches request logs.
                OPENAI_API_KEY="local-endpoint",
                # Throttle tqdm redraws (finelog captures each redraw as a log line).
                TQDM_MININTERVAL="30",
                **{CONFIG_ENV_KEY: _client_config_json(session, unit, endpoint)},
            )
        ),
        task_image=session.eval_image,
        constraints=constraints,
        max_retries_failure=0,
    )
    logger.info("Submitted evalchemy client job %s for unit %s against %s", eval_job, unit.name, endpoint.model_id)
    eval_path = str(eval_job.job_id)
    try:
        # Wait in slices, watching the serve registration between them: lm-eval retries connection
        # errors indefinitely, so a child pointed at a preempted serve address wedges rather than
        # fails. When the serve provably left the child's address, kill the child so the caller's
        # restart-retry reruns the unit against the live address immediately.
        while True:
            try:
                # wait(raise_on_failure=True) raises JobFailedError on any non-SUCCESS terminal state.
                eval_job.wait(timeout=_CHILD_WAIT_SLICE_SECONDS)
                break
            except TimeoutError:
                pass
            if _endpoint_departed(endpoint):
                logger.info(
                    "unit %s: serve endpoint %s left %s mid-eval; terminating child %s",
                    unit.name,
                    endpoint.name,
                    endpoint.base_url,
                    eval_path,
                )
                eval_job.terminate()
                raise EvalPipelineError(
                    f"evalchemy client job {eval_path} terminated: the serve endpoint left {endpoint.base_url}",
                    stage=PipelineStage.EVAL,
                    jobs={"eval": eval_path},
                    log_tails={"eval": job_log_tail(eval_job)},
                )
    except JobFailedError as exc:
        raise EvalPipelineError(
            f"evalchemy client job {eval_path} failed: {exc}",
            stage=PipelineStage.EVAL,
            jobs={"eval": eval_path},
            log_tails={"eval": job_log_tail(eval_job)},
        ) from exc
    return eval_path
