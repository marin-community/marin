# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import click
from fray.types import ResourceConfig
from iris.cli.connect import open_controller_endpoint
from iris.client import IrisClient
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from marin.evaluation.lm_eval import (
    LM_EVAL_UV_PACKAGES,
    LmEvalRun,
    run_iris_brokered_lm_eval,
    run_local_brokered_lm_eval,
)
from marin.execution.lazy import Artifact, ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    BrokeredVllmSystemConfig,
    InferenceWorkerConfig,
)
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

VLLM_WORKER_ENV_VARS = (
    ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
    ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
    ("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1"),
    ("VLLM_TPU_SKIP_PRECOMPILE", "1"),
)


@dataclass(frozen=True)
class ServedLmEvalBenchmark:
    tasks: Sequence[str]
    output_path: str
    job_name: str
    confirm_run_unsafe_code: bool = False
    parent_env_vars: Mapping[str, str] = field(default_factory=dict)
    model: str = "Qwen/Qwen3-0.6B-Base"
    tokenizer: str = "Qwen/Qwen3-0.6B"
    tpu_type: str = "v5litepod-4"
    worker_ram: str = "96g"
    parent_cpu: float = 0.5
    parent_ram: str = "6g"
    parent_disk: str = "16g"
    region: str | None = "us-west4"
    workers: int = 1
    num_concurrent: int = DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER


class LmEvalLauncher(StrEnum):
    IRIS = "iris"
    LOCAL = "local"


@dataclass(frozen=True)
class BrokeredLmEvalArtifactConfig:
    model: str
    tokenizer: str | None
    max_model_len: int
    max_num_batched_tokens: int
    workers: InferenceWorkerConfig
    worker_env_vars: tuple[tuple[str, str], ...]
    ignored_request_fields: tuple[str, ...]
    lm_eval_uv_packages: tuple[str, ...]
    eval_run: LmEvalRun


def _run_brokered_lm_eval_artifact(
    inference: BrokeredVllmSystemConfig,
    eval_run: LmEvalRun,
) -> None:
    output_path = eval_run.output_path
    with tempfile.TemporaryDirectory() as local_output:
        run_iris_brokered_lm_eval(inference, replace(eval_run, output_path=local_output))
        StoragePath(output_path).upload_from(local_output + "/", recursive=True)


def brokered_lm_eval_step(
    inference: BrokeredVllmSystemConfig,
    eval_run: LmEvalRun,
    *,
    name: str,
    version: str,
    parent_resources: ResourceConfig,
    parent_env_vars: Mapping[str, str],
) -> ArtifactStep[Artifact]:
    """Build a lazy artifact containing lm-eval metrics and samples."""
    if inference.worker_resources is None:
        raise ValueError("inference.worker_resources must be set for a brokered lm-eval artifact")

    def build_config(ctx: StepContext) -> BrokeredLmEvalArtifactConfig:
        return BrokeredLmEvalArtifactConfig(
            model=inference.model,
            tokenizer=inference.tokenizer,
            max_model_len=inference.server.max_model_len,
            max_num_batched_tokens=inference.server.max_num_batched_tokens,
            workers=inference.workers,
            worker_env_vars=tuple(sorted(inference.worker_env_vars.items())),
            ignored_request_fields=inference.proxy.ignored_request_fields,
            lm_eval_uv_packages=LM_EVAL_UV_PACKAGES,
            eval_run=replace(eval_run, output_path=ctx.output_path),
        )

    def run_step(cfg: BrokeredLmEvalArtifactConfig) -> None:
        remote(
            _run_brokered_lm_eval_artifact,
            name=name,
            resources=parent_resources,
            env_vars=dict(parent_env_vars),
        )(inference, cfg.eval_run)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=run_step,
        build_config=build_config,
        deps=(),
        runtime_args={
            "parent_resources": parent_resources,
            "worker_resources": inference.worker_resources,
        },
    )


def run_iris_lm_eval_job(
    inference: BrokeredVllmSystemConfig,
    run: LmEvalRun,
    *,
    job_name: str,
    iris_config_path: str,
    parent_cpu: float,
    parent_ram: str,
    parent_disk: str,
    parent_env_vars: Mapping[str, str],
    region: str | None,
    priority: int,
) -> None:
    """Submit a non-preemptible CPU parent that runs brokered lm-eval."""

    def run_parent() -> None:
        configure_logging()
        run_iris_brokered_lm_eval(inference, run)

    constraints = [preemptible_constraint(False)]
    if region is not None:
        constraints.append(region_constraint([region]))
    with open_controller_endpoint(config_file=Path(iris_config_path)) as endpoint:
        with IrisClient.remote(
            endpoint.url,
            workspace=Path.cwd(),
            credentials=endpoint.credentials,
        ) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(run_parent),
                name=job_name,
                resources=ResourceSpec(cpu=parent_cpu, memory=parent_ram, disk=parent_disk),
                environment=EnvironmentSpec(env_vars=dict(parent_env_vars)),
                constraints=constraints,
                priority_band=priority,
            )
            print(f"Submitted Iris parent job {job.job_id}", flush=True)
            job.wait(timeout=float("inf"))


def brokered_lm_eval_configs(
    benchmark: ServedLmEvalBenchmark,
    *,
    limit: int | None,
    timeout_seconds: int | None,
    priority: int,
) -> tuple[BrokeredVllmSystemConfig, LmEvalRun]:
    inference = brokered_vllm_config(
        model=benchmark.model,
        tokenizer=benchmark.tokenizer,
        workers=benchmark.workers,
        num_concurrent=benchmark.num_concurrent,
        timeout_seconds=timeout_seconds,
    )
    inference = replace(
        inference,
        worker_resources=ResourceConfig.with_tpu(
            benchmark.tpu_type,
            ram=benchmark.worker_ram,
            regions=[benchmark.region] if benchmark.region is not None else None,
        ),
        worker_env_vars=dict(VLLM_WORKER_ENV_VARS),
        priority=priority,
    )
    eval_run = LmEvalRun(
        tasks=benchmark.tasks,
        output_path=benchmark.output_path,
        limit=limit,
        confirm_run_unsafe_code=benchmark.confirm_run_unsafe_code,
        extra_model_args={
            "num_concurrent": inference.workers.max_in_flight_per_worker,
            "timeout": int(inference.proxy.request_timeout_seconds),
        },
    )
    return inference, eval_run


def served_lm_eval_command(help_text: str, benchmark: ServedLmEvalBenchmark) -> click.Command:
    """Build a CLI for a fixed lm-eval task set served by brokered vLLM."""

    @click.command(help=help_text, context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
    @click.option(
        "--limit",
        type=click.IntRange(min=0),
        default=8,
        help="Limit documents per task for a fast run. Use 0 to run every document.",
    )
    @click.option("--output-path", default=benchmark.output_path, help="Directory for lm-eval samples and metrics.")
    @click.option(
        "--timeout-seconds",
        type=click.IntRange(min=1),
        help="Override vLLM startup and proxy/lm-eval request timeouts for slow manual runs.",
    )
    @click.option(
        "--num-concurrent",
        type=click.IntRange(min=1),
        default=benchmark.num_concurrent,
        help="lm-eval request concurrency and per-worker vLLM request limit.",
    )
    @click.option(
        "--workers",
        type=click.IntRange(min=1),
        default=benchmark.workers,
        help="Number of child vLLM worker jobs. Local mode requires one.",
    )
    @click.option("--model", default=benchmark.model, help="Model served by each vLLM worker.")
    @click.option("--tokenizer", default=benchmark.tokenizer, help="Tokenizer used by vLLM and lm-eval.")
    @click.option("--tpu-type", default=benchmark.tpu_type, help="TPU type for child vLLM workers.")
    @click.option("--worker-ram", default=benchmark.worker_ram, help="Host RAM for each child vLLM worker.")
    @click.option(
        "--parent-ram", default=benchmark.parent_ram, help="Host RAM for the CPU parent running lm-eval and the proxy."
    )
    @click.option("--region", default=benchmark.region, help="Region for the parent and worker jobs.")
    @click.option("--job-name", default=benchmark.job_name, help="Iris parent job name.")
    @click.option("--priority", type=click.Choice(PRIORITY_BAND_NAMES), help="Iris priority band for all jobs.")
    @click.option(
        "--launcher",
        type=click.Choice([launcher.value for launcher in LmEvalLauncher]),
        default=LmEvalLauncher.IRIS.value,
        help="Run through Iris or use a single local dev TPU worker.",
    )
    def command(
        limit: int,
        output_path: str,
        timeout_seconds: int | None,
        num_concurrent: int,
        workers: int,
        model: str,
        tokenizer: str,
        tpu_type: str,
        worker_ram: str,
        parent_ram: str,
        region: str | None,
        job_name: str,
        priority: str | None,
        launcher: str,
    ) -> None:
        configure_logging()
        launcher_mode = LmEvalLauncher(launcher)
        if launcher_mode is LmEvalLauncher.LOCAL and workers > 1:
            raise click.UsageError("The local launcher only supports --workers 1; use --num-concurrent for fanout")

        iris_priority = priority_band_value(priority) if priority else job_pb2.PRIORITY_BAND_UNSPECIFIED
        configured_benchmark = replace(
            benchmark,
            model=model,
            tokenizer=tokenizer,
            workers=workers,
            num_concurrent=num_concurrent,
            output_path=output_path,
            tpu_type=tpu_type,
            worker_ram=worker_ram,
            parent_ram=parent_ram,
            region=region,
        )
        inference, run = brokered_lm_eval_configs(
            configured_benchmark,
            limit=limit if limit > 0 else None,
            timeout_seconds=timeout_seconds,
            priority=iris_priority,
        )
        if launcher_mode is LmEvalLauncher.LOCAL:
            run_local_brokered_lm_eval(inference, run)
            return

        run_iris_lm_eval_job(
            inference,
            run,
            job_name=job_name,
            iris_config_path="lib/iris/config/marin.yaml",
            parent_cpu=benchmark.parent_cpu,
            parent_ram=parent_ram,
            parent_disk=benchmark.parent_disk,
            parent_env_vars=benchmark.parent_env_vars,
            region=region,
            priority=iris_priority,
        )

    return command


def brokered_vllm_config(
    *,
    model: str,
    tokenizer: str,
    workers: int,
    num_concurrent: int,
    timeout_seconds: int | None,
) -> BrokeredVllmSystemConfig:
    config = BrokeredVllmSystemConfig(
        model=model,
        tokenizer=tokenizer,
        workers=InferenceWorkerConfig(count=workers, max_in_flight_per_worker=num_concurrent),
    )
    timeout = timeout_seconds or config.server.timeout_seconds
    return replace(
        config,
        server=replace(config.server, timeout_seconds=timeout),
        proxy=replace(
            config.proxy,
            request_timeout_seconds=timeout,
            readiness_timeout_seconds=timeout,
            ignored_request_fields=("seed",),
        ),
    )
