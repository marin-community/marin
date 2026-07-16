# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run HumanEval against Qwen3 through brokered vLLM serving.

Default mode submits one Iris CPU parent job that runs lm-eval plus a local
OpenAI-compatible proxy. The parent starts a broker actor and TPU vLLM worker
child jobs. Local mode runs the same broker/proxy/worker loop in the current
process for single-node checks.

lm-eval runs in an isolated uv environment so the eval client does not share the
TPU serving environment's `vllm` dependency set.

\b
Examples:
  uv run python experiments/evals/served_qwen3_humaneval.py
  uv run python experiments/evals/served_qwen3_humaneval.py --priority production
  uv run python experiments/evals/served_qwen3_humaneval.py --local
"""

import shlex
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import click
from fray.types import ResourceConfig
from iris.client import IrisClient
from iris.cluster.composer import provider_bundle
from iris.cluster.config import load_config
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from marin.inference.types import RunningModel
from marin.inference.vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    BrokeredVllmSystemConfig,
    InferenceWorkerConfig,
    start_iris_brokered_vllm,
    start_local_brokered_vllm,
)
from rigging.log_setup import configure_logging

# Inherited TPU vLLM settings from the repo's generation-eval recipes.
# TODO(yonromai): double-check whether all inherited TPU vLLM settings are still needed.
VLLM_WORKER_ENV_VARS: dict[str, str] = {
    # Keep vLLM's API server and engine in one process so the TPU is claimed once.
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    # Lets overridden --max-model-len exceed model metadata; not always needed.
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    # TPU workaround from Harbor/eval recipes; not needed for non-TPU vLLM.
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    # Speeds up small HumanEval runs by avoiding TPU precompile; not needed off TPU.
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


@dataclass(frozen=True)
class HumanEvalRunConfig:
    output_path: str
    limit: int | None
    num_concurrent: int
    request_timeout_seconds: int


def _run_lm_eval(
    *,
    base_url: str,
    model: str,
    tokenizer: str,
    output_path: str,
    limit: int | None,
    num_concurrent: int,
    request_timeout_seconds: int,
) -> None:
    model_args = ",".join(
        [
            f"model={model}",
            f"base_url={base_url.rstrip('/')}/completions",
            f"tokenizer={tokenizer}",
            "tokenized_requests=False",
            f"num_concurrent={num_concurrent}",
            f"timeout={request_timeout_seconds}",
        ]
    )
    cmd = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        # Pin away from lm-eval 0.5.0.dev1, which breaks HumanEval scoring.
        "lm-eval[api]==0.4.9.1",
        "--with",
        # HumanEval's code_eval metric imports evaluate.
        "evaluate",
        "--with",
        # lm-eval 0.4.x imports symbols removed in transformers 5.
        "transformers<5",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        "humaneval",
        "--output_path",
        output_path,
        # Persist per-sample completions for debugging failed generations.
        "--log_samples",
        # HumanEval scoring executes generated Python.
        "--confirm_run_unsafe_code",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    # dev_tpu SSH buffers this script's stdout behind child-process logs without an explicit flush.
    print(f"Running lm-eval CLI: {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_humaneval(
    running_model: RunningModel,
    eval_config: HumanEvalRunConfig,
) -> None:
    tokenizer = running_model.tokenizer or running_model.endpoint.model
    print(
        "Running lm-eval HumanEval against",
        f"{running_model.endpoint.base_url} model={running_model.endpoint.model}",
        flush=True,
    )
    _run_lm_eval(
        base_url=running_model.endpoint.base_url,
        model=running_model.endpoint.model,
        tokenizer=tokenizer,
        output_path=eval_config.output_path,
        limit=eval_config.limit,
        num_concurrent=eval_config.num_concurrent,
        request_timeout_seconds=eval_config.request_timeout_seconds,
    )

    print(f"lm-eval output written under {eval_config.output_path}", flush=True)
    output_dir = Path(eval_config.output_path)
    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file()) if output_dir.exists() else []
    if output_files:
        print("\n".join(str(path) for path in output_files), flush=True)


def run_local_humaneval(inference_config: BrokeredVllmSystemConfig, eval_config: HumanEvalRunConfig) -> None:
    with start_local_brokered_vllm(inference_config) as running_model:
        run_humaneval(running_model, eval_config)


def run_iris_humaneval(
    inference_config: BrokeredVllmSystemConfig,
    eval_config: HumanEvalRunConfig,
) -> None:
    with start_iris_brokered_vllm(inference_config) as running_model:
        run_humaneval(running_model, eval_config)


def submit_iris_humaneval(
    inference_config: BrokeredVllmSystemConfig,
    eval_config: HumanEvalRunConfig,
    *,
    job_name: str,
    iris_config_path: str,
    parent_ram: str,
    region: str | None,
    priority: int,
) -> None:
    def _run_parent() -> None:
        configure_logging()
        run_iris_humaneval(inference_config, eval_config)

    iris_config = load_config(iris_config_path)
    controller = provider_bundle(iris_config).controller
    controller_address = iris_config.controller_address() or controller.discover_controller(iris_config.controller)
    constraints = [preemptible_constraint(False)]
    if region is not None:
        constraints.append(region_constraint([region]))
    with controller.tunnel(address=controller_address) as controller_url:
        with IrisClient.remote(controller_url, workspace=Path.cwd()) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(_run_parent),
                name=job_name,
                resources=ResourceSpec(cpu=0.5, memory=parent_ram, disk="16g"),
                environment=EnvironmentSpec(env_vars={"HF_ALLOW_CODE_EVAL": "1"}),
                constraints=constraints,
                priority_band=priority,
            )
            print(f"Submitted Iris parent job {job.job_id}", flush=True)
            job.wait(timeout=float("inf"))


def _build_inference_config(
    *,
    workers: int,
    num_concurrent: int,
    timeout_seconds: int | None,
) -> BrokeredVllmSystemConfig:
    config = BrokeredVllmSystemConfig(
        model="Qwen/Qwen3-0.6B-Base",
        tokenizer="Qwen/Qwen3-0.6B",
        workers=InferenceWorkerConfig(count=workers, max_in_flight_per_worker=num_concurrent),
    )
    if timeout_seconds is None:
        return config
    return replace(
        config,
        server=replace(config.server, timeout_seconds=timeout_seconds),
        proxy=replace(
            config.proxy,
            request_timeout_seconds=timeout_seconds,
            readiness_timeout_seconds=timeout_seconds,
        ),
    )


@click.command(help=__doc__, context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option(
    "--limit",
    type=click.IntRange(min=0),
    default=8,
    help="Limit HumanEval docs for a fast run. Use 0 to run the full task.",
)
@click.option(
    "--output-path",
    default="/tmp/served-qwen3-humaneval",
    help="Directory where lm-eval writes samples and metrics.",
)
@click.option(
    "--timeout-seconds",
    type=click.IntRange(min=1),
    help="Override vLLM startup and proxy/lm-eval request timeouts for slow manual runs.",
)
@click.option(
    "--num-concurrent",
    type=click.IntRange(min=1),
    default=DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    help="lm-eval local-completions concurrency and per-worker vLLM request limit.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    help="Number of child vLLM worker jobs in Iris mode. Local mode requires 1.",
)
@click.option("--tpu-type", default="v5litepod-4", help="TPU type for child vLLM workers.")
@click.option("--worker-ram", default="96g", help="Host RAM request for each child vLLM worker.")
@click.option(
    "--parent-ram",
    default="6g",
    help="Host RAM request for the CPU parent job that runs lm-eval and the proxy.",
)
@click.option(
    "--region",
    default=None,
    help="Optional region for the Iris parent job; broker and worker jobs inherit it.",
)
@click.option("--job-name", default="served-qwen3-humaneval", help="Iris parent job name.")
@click.option(
    "--priority",
    type=click.Choice(PRIORITY_BAND_NAMES),
    help="Iris priority band for the parent, broker, and worker jobs.",
)
@click.option(
    "--local",
    is_flag=True,
    help="Run the single-node dev TPU path in the current process instead of submitting an Iris parent job.",
)
def main(
    limit: int,
    output_path: str,
    timeout_seconds: int | None,
    num_concurrent: int,
    workers: int,
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
        raise click.UsageError("--local only supports --workers 1; use --num-concurrent for local request fanout")

    try:
        inference_config = _build_inference_config(
            workers=workers,
            num_concurrent=num_concurrent,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    eval_config = HumanEvalRunConfig(
        output_path=output_path,
        limit=limit if limit > 0 else None,
        num_concurrent=inference_config.workers.max_in_flight_per_worker,
        request_timeout_seconds=int(inference_config.proxy.request_timeout_seconds),
    )

    if local:
        run_local_humaneval(inference_config, eval_config)
        return

    iris_priority = priority_band_value(priority) if priority else job_pb2.PRIORITY_BAND_UNSPECIFIED
    inference_config = replace(
        inference_config,
        worker_resources=ResourceConfig.with_tpu(tpu_type, ram=worker_ram),
        worker_env_vars=VLLM_WORKER_ENV_VARS,
        priority=iris_priority,
    )
    submit_iris_humaneval(
        inference_config,
        eval_config,
        job_name=job_name,
        iris_config_path="lib/iris/config/marin.yaml",
        parent_ram=parent_ram,
        region=region,
        priority=iris_priority,
    )


if __name__ == "__main__":
    main()
