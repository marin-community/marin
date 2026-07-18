# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a model, run evalchemy against its OpenAI URL, tear the server down.

The eval is decoupled from the model backend by an OpenAI-compatible URL (issue #4827): a marin-serve
child job (``VllmBackend`` or ``LevanterBackend``, on a TPU/GPU slice) exposes an endpoint, and an
evalchemy child job (the ``:evalchemy-tpu`` container on a CPU slice) hits it with
``eval.eval --model local-completions``. Evalchemy is the sole eval client.

Topology — one parent orchestrator job spawns two children::

    parent (CPU, marin)  ──serve child──▶  marin-serve backend (TPU/GPU)  ──▶ OpenAI endpoint
                         ──eval  child──▶  :evalchemy-tpu (CPU)  ──local-completions──▶ endpoint

:func:`serve_model` submits the serve child, waits for its endpoint to register, and yields the
served backend's in-cluster address. The eval child is pinned to the serve region, so it reaches
that address straight over the same-cluster VPC -- no controller proxy or capability token. Iris
auto-cleans child jobs when the parent ends, so leaving the ``with`` block (or the parent exiting)
tears the server down; the context manager also stops it eagerly for promptness.

The two children take different entrypoints for a reason. The serve child runs in the marin image,
whose synced venv can deserialize a cloudpickled callable, so it uses ``Entrypoint.from_callable``.
The eval child runs in the ``:evalchemy-tpu`` image, whose default interpreter is a bare python with
no cloudpickle -- only ``/opt/openthoughts/.venv`` carries ``eval``/``lm_eval``/``fsspec`` -- so it
runs :mod:`experiments.evals.evalchemy.run_evalchemy_client` as a plain *command* under that
interpreter, with its config passed as JSON in an env var.

Top-level imports are kept light on purpose: the serve child's ``VllmBackend``/``LevanterBackend``
construction (which pulls levanter + vLLM) happens lazily inside :func:`_serve_for_eval`, so the CPU
parent that references it for cloudpickling never imports the serving stack.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from enum import StrEnum

from iris.client import iris_ctx
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
# lm-eval's local-completions client concurrency (parallel in-flight requests to the endpoint).
DEFAULT_NUM_CONCURRENT = 16
# Credentials the child jobs need (HF model/dataset downloads, wandb logging); the parent propagates
# whichever of these it holds, since a child does not inherit the parent's process env.
_PROPAGATED_ENV_KEYS = ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT")


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
    vllm_extra_args: tuple[str, ...] = ()
    """Raw flags forwarded verbatim to ``vllm serve`` on the vLLM serve path (``VllmBackend.extra_args``).
    Needed for models the portable defaults cannot serve: e.g. a 256-expert Grug MoE export shards its
    experts with data + expert parallelism (``--data-parallel-size N --enable-expert-parallel
    --model-loader-extra-config '{"distributed":true}'``, with ``tensor_parallel_size=1``), which the
    per-head TP heuristic cannot infer. Empty for the common single-model case."""
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


@dataclass(frozen=True)
class EvalchemyEvalConfig:
    """Everything the parent orchestrator needs to serve a model and eval it. Cloudpickled into the job."""

    model: str
    """HF repo id or object-store (``gs://``) path of the model to serve and eval."""
    tasks: tuple[EvalTaskConfig, ...]
    out_path: str
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
            extra_args=params.vllm_extra_args,
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
        timeout_hours=SERVE_TIMEOUT_HOURS,
        startup_timeout_seconds=int(ENDPOINT_READY_TIMEOUT_SECONDS),
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
        max_retries_failure=0,
    )
    logger.info("Submitted serve job %s (backend=%s) for endpoint %s", serve_job, spec.backend, endpoint_name)
    try:
        _wait_for_endpoint(client, serve_job, endpoint_name)
        # In-cluster, resolve_endpoint returns the served backend's direct address (its dashboard's
        # OpenAI reverse proxy); the eval child, colocated in the same region/VPC, calls it straight
        # -- no controller proxy or capability token needed for a same-cluster consumer.
        base_url = client.resolve_endpoint(endpoint_name).rstrip("/") + "/v1"
        logger.info("Serve endpoint %s ready at %s", endpoint_name, base_url)
        yield ServedEndpoint(base_url=base_url, model_id=model, tokenizer=tokenizer)
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


def _client_config_json(config: EvalchemyEvalConfig, endpoint: ServedEndpoint) -> str:
    """The evalchemy client's config as a JSON string, passed to the eval child in an env var.

    A plain JSON payload (not a cloudpickled object) so the image's bare interpreter can read it
    without any marin/cloudpickle import -- see :mod:`experiments.evals.evalchemy.run_evalchemy_client`.
    """
    return json.dumps(
        {
            "base_url": endpoint.base_url,
            "model_id": endpoint.model_id,
            "tokenizer": endpoint.tokenizer,
            "tasks": [{"name": t.name, "num_fewshot": t.num_fewshot, "dir": _task_dir(t)} for t in config.tasks],
            "out_path": config.out_path,
            "apply_chat_template": config.apply_chat_template,
            "max_gen_toks": config.max_gen_toks,
            "max_eval_instances": config.max_eval_instances,
            "num_concurrent": config.num_concurrent,
        }
    )


# --------------------------------------------------------------------------------------------------
# Parent orchestrator: serve, eval against the URL, tear down (runs in the marin image, CPU).
# --------------------------------------------------------------------------------------------------


def serve_and_eval(config: EvalchemyEvalConfig) -> None:
    """Parent entrypoint: serve the model, run evalchemy against its OpenAI URL, tear the server down.

    Runs as a CPU orchestrator job. Serving and eval are separate child jobs (different container
    images), tied together by the served OpenAI URL and by Iris's parent/child auto-cleanup.
    """
    if not config.tasks:
        raise ValueError("serve_and_eval requires at least one task")
    # The served backend loads the model from any fsspec path (gs://...), but the eval client loads its
    # tokenizer through HF, which cannot read an object-store path. Fail fast rather than let the eval
    # child die deep in lm-eval; the caller sets tokenizer to the base model's HF id for a gs:// model.
    if config.tokenizer is None and "://" in config.model:
        raise ValueError(
            f"model {config.model!r} is an object-store path the eval image cannot load a tokenizer "
            "from; set EvalchemyEvalConfig.tokenizer (EvalGroup.tokenizer) to the base model's HF id."
        )
    with serve_model(config.model, config.tokenizer or config.model, config.serve) as endpoint:
        _submit_eval_child(config, endpoint)


def _submit_eval_child(config: EvalchemyEvalConfig, endpoint: ServedEndpoint) -> None:
    """Submit the evalchemy client child job against ``endpoint`` and block until it finishes."""
    client = iris_ctx().client
    run_id = uuid.uuid4().hex[:8]
    # Colocate the eval client with the serving slice so it reaches the served address over the
    # same-region VPC.
    constraints = [region_constraint([config.serve.region])] if config.serve.region else None
    # A command entrypoint, not from_callable: the eval image's synced interpreter has no cloudpickle,
    # so the client runs under EVALCHEMY_PYTHON (which does) with its config passed as JSON in an env var.
    command = f'exec {EVALCHEMY_PYTHON} "$IRIS_WORKDIR/{_EVAL_CLIENT_SCRIPT}"'
    eval_job = client.submit(
        entrypoint=Entrypoint.from_command("bash", "-c", command),
        name=f"eval-evalchemy-{run_id}",
        resources=ResourceSpec(cpu=config.eval_cpu, memory=config.eval_memory, disk=config.eval_disk),
        # The evalchemy image runs as a pure HTTP client here, so keep JAX off the (TPU-tagged) image's
        # TPU init path and let humaneval-style code_eval tasks execute generated code.
        environment=EnvironmentSpec(
            env_vars=_propagated_env(
                JAX_PLATFORMS="cpu",
                HF_ALLOW_CODE_EVAL="1",
                **{CONFIG_ENV_KEY: _client_config_json(config, endpoint)},
            )
        ),
        task_image=config.eval_image,
        constraints=constraints,
        max_retries_failure=0,
    )
    logger.info("Submitted evalchemy client job %s against %s", eval_job, endpoint.model_id)
    # wait(raise_on_failure=True) raises JobFailedError on any non-SUCCESS terminal state.
    eval_job.wait(timeout=float("inf"))
