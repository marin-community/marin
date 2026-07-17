# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import click
import fsspec
from fray.types import ResourceConfig
from iris.cli.connect import open_controller_endpoint
from iris.client import IrisClient
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from marin.evaluation.lm_eval import LmEvalRun, run_iris_brokered_lm_eval, run_local_brokered_lm_eval
from marin.execution.lazy import Artifact, ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    BrokeredVllmSystemConfig,
    InferenceWorkerConfig,
)
from rigging.log_setup import configure_logging

VLLM_WORKER_ENV_VARS: dict[str, str] = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


@dataclass(frozen=True)
class ServedLmEvalBenchmark:
    tasks: Sequence[str]
    output_path: str
    job_name: str
    confirm_run_unsafe_code: bool = False
    parent_env_vars: Mapping[str, str] = field(default_factory=dict)


def _run_brokered_lm_eval_artifact(
    inference: BrokeredVllmSystemConfig,
    eval_run: LmEvalRun,
    output_path: str,
) -> None:
    output_fs, _ = fsspec.core.url_to_fs(output_path)
    with tempfile.TemporaryDirectory() as local_output:
        run_iris_brokered_lm_eval(inference, replace(eval_run, output_path=local_output))
        output_fs.put(local_output + "/", output_path, recursive=True)


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

    def build_config(ctx: StepContext) -> dict:
        return {
            "model": inference.model,
            "tokenizer": inference.tokenizer,
            "max_model_len": inference.server.max_model_len,
            "max_num_batched_tokens": inference.server.max_num_batched_tokens,
            "worker_count": inference.workers.count,
            "max_in_flight_per_worker": inference.workers.max_in_flight_per_worker,
            "worker_env_vars": dict(inference.worker_env_vars),
            "ignored_request_fields": list(inference.proxy.ignored_request_fields),
            "tasks": list(eval_run.tasks),
            "adapter": eval_run.adapter.value,
            "apply_chat_template": eval_run.apply_chat_template,
            "limit": eval_run.limit,
            "num_fewshot": eval_run.num_fewshot,
            "batch_size": eval_run.batch_size,
            "confirm_run_unsafe_code": eval_run.confirm_run_unsafe_code,
            "uv_packages": list(eval_run.uv_packages),
            "extra_model_args": dict(eval_run.extra_model_args),
            "out": ctx.output_path,
        }

    def run_step(cfg: dict) -> None:
        remote(
            _run_brokered_lm_eval_artifact,
            name=name,
            resources=parent_resources,
            env_vars=dict(parent_env_vars),
        )(inference, eval_run, cfg["out"])

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


def submit_iris_lm_eval(
    inference: BrokeredVllmSystemConfig,
    run: LmEvalRun,
    *,
    job_name: str,
    iris_config_path: str,
    parent_ram: str,
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
                resources=ResourceSpec(cpu=0.5, memory=parent_ram, disk="16g"),
                environment=EnvironmentSpec(env_vars=dict(parent_env_vars)),
                constraints=constraints,
                priority_band=priority,
            )
            print(f"Submitted Iris parent job {job.job_id}", flush=True)
            job.wait(timeout=float("inf"))


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
        default=DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
        help="lm-eval request concurrency and per-worker vLLM request limit.",
    )
    @click.option(
        "--workers",
        type=click.IntRange(min=1),
        default=1,
        help="Number of child vLLM worker jobs. Local mode requires one.",
    )
    @click.option("--model", default="Qwen/Qwen3-0.6B-Base", help="Model served by each vLLM worker.")
    @click.option("--tokenizer", default="Qwen/Qwen3-0.6B", help="Tokenizer used by vLLM and lm-eval.")
    @click.option("--tpu-type", default="v5litepod-4", help="TPU type for child vLLM workers.")
    @click.option("--worker-ram", default="96g", help="Host RAM for each child vLLM worker.")
    @click.option("--parent-ram", default="6g", help="Host RAM for the CPU parent running lm-eval and the proxy.")
    @click.option("--region", default="us-west4", help="Region for the parent and worker jobs.")
    @click.option("--job-name", default=benchmark.job_name, help="Iris parent job name.")
    @click.option("--priority", type=click.Choice(PRIORITY_BAND_NAMES), help="Iris priority band for all jobs.")
    @click.option("--local", is_flag=True, help="Run a single local dev TPU worker instead of submitting to Iris.")
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
        local: bool,
    ) -> None:
        configure_logging()
        if local and workers > 1:
            raise click.UsageError("--local only supports --workers 1; use --num-concurrent for request fanout")

        inference = brokered_vllm_config(
            model=model,
            tokenizer=tokenizer,
            workers=workers,
            num_concurrent=num_concurrent,
            timeout_seconds=timeout_seconds,
        )
        run = LmEvalRun(
            tasks=benchmark.tasks,
            output_path=output_path,
            limit=limit if limit > 0 else None,
            confirm_run_unsafe_code=benchmark.confirm_run_unsafe_code,
            extra_model_args={
                "num_concurrent": inference.workers.max_in_flight_per_worker,
                "timeout": int(inference.proxy.request_timeout_seconds),
            },
        )
        if local:
            run_local_brokered_lm_eval(inference, run)
            return

        iris_priority = priority_band_value(priority) if priority else job_pb2.PRIORITY_BAND_UNSPECIFIED
        worker_regions = [region] if region is not None else None
        inference = replace(
            inference,
            worker_resources=ResourceConfig.with_tpu(
                tpu_type,
                ram=worker_ram,
                regions=worker_regions,
            ),
            worker_env_vars=VLLM_WORKER_ENV_VARS,
            priority=iris_priority,
        )
        submit_iris_lm_eval(
            inference,
            run,
            job_name=job_name,
            iris_config_path="lib/iris/config/marin.yaml",
            parent_ram=parent_ram,
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
