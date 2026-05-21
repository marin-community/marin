# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Run HumanEval against Qwen3 through brokered vLLM serving.

Default mode submits one Iris CPU parent job that runs lm-eval plus a local
OpenAI-compatible proxy. The parent starts a broker actor and TPU vLLM worker
child jobs. Local mode runs the same broker/proxy/worker loop in the current
process for single-node checks.

lm-eval runs in an isolated uv environment so the eval client does not share the
TPU serving environment's `vllm-tpu` dependency set.

    uv run python experiments/evals/served_qwen3_humaneval.py
    uv run python experiments/evals/served_qwen3_humaneval.py --priority production
    uv run python experiments/evals/served_qwen3_humaneval.py --local
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from fray.types import ResourceConfig
from iris.client import IrisClient
from iris.cluster.config import IrisConfig
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from iris.rpc.proto_utils import PRIORITY_BAND_NAMES, priority_band_value
from marin.inference.brokered_vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    BrokeredVllmSystemConfig,
    IrisBrokeredVllmRuntimeConfig,
    VllmWorkerConfig,
    start_iris_brokered_vllm,
    start_local_brokered_vllm,
)
from marin.inference.types import RunningModel
from rigging.log_setup import configure_logging

# These match the repo's existing TPU generation-eval recipe in
# experiments/exp1337_eval_suite.py.
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
    runtime_config: IrisBrokeredVllmRuntimeConfig,
) -> None:
    with start_iris_brokered_vllm(inference_config, runtime_config) as running_model:
        run_humaneval(running_model, eval_config)


def submit_iris_humaneval(
    inference_config: BrokeredVllmSystemConfig,
    eval_config: HumanEvalRunConfig,
    runtime_config: IrisBrokeredVllmRuntimeConfig,
    *,
    job_name: str,
    iris_config_path: str,
    parent_ram: str,
    region: str,
    priority_band: job_pb2.PriorityBand,
) -> None:
    def _run_parent() -> None:
        configure_logging()
        run_iris_humaneval(inference_config, eval_config, runtime_config)

    iris_config = IrisConfig.load(iris_config_path)
    controller = iris_config.provider_bundle().controller
    controller_address = iris_config.controller_address() or controller.discover_controller(iris_config.proto.controller)
    with controller.tunnel(address=controller_address) as controller_url:
        with IrisClient.remote(controller_url, workspace=Path.cwd()) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(_run_parent),
                name=job_name,
                resources=ResourceSpec(cpu=0.5, memory=parent_ram, disk="16g"),
                environment=EnvironmentSpec(env_vars={"HF_ALLOW_CODE_EVAL": "1"}),
                constraints=[preemptible_constraint(False), region_constraint([region])],
                priority_band=priority_band,
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
        workers=VllmWorkerConfig(count=workers, max_in_flight_per_worker=num_concurrent),
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


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, BrokeredVllmSystemConfig]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Limit HumanEval docs for a fast run (default: 8). Use 0 to run the full task.",
    )
    parser.add_argument(
        "--output-path",
        default="/tmp/served-qwen3-humaneval",
        help="Directory where lm-eval writes samples and metrics (default: /tmp/served-qwen3-humaneval).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Override vLLM startup and proxy/lm-eval request timeouts for slow manual runs.",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
        help=(
            "lm-eval local-completions concurrency and per-worker vLLM request limit "
            f"(default: brokered worker default, currently {DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER})."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of child vLLM worker jobs in Iris mode (default: 1). Local mode requires 1.",
    )
    parser.add_argument(
        "--tpu-type",
        default="v5litepod-4",
        help="TPU type for child vLLM workers (default: v5litepod-4).",
    )
    parser.add_argument(
        "--worker-ram",
        default="96g",
        help="Host RAM request for each child vLLM worker (default: 96g).",
    )
    parser.add_argument(
        "--parent-ram",
        default="6g",
        help="Host RAM request for the CPU parent job that runs lm-eval and the proxy (default: 6g).",
    )
    parser.add_argument(
        "--tpu-region",
        "--tpu-regions",
        dest="tpu_region",
        default="us-west4",
        help="Region for the CPU parent, broker actor, and TPU worker jobs (default: us-west4).",
    )
    parser.add_argument(
        "--job-name",
        default="served-qwen3-humaneval",
        help="Iris parent job name (default: served-qwen3-humaneval).",
    )
    parser.add_argument(
        "--priority",
        choices=PRIORITY_BAND_NAMES,
        help="Iris priority band for the parent, broker, and worker jobs (default: Iris cluster default).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run the single-node dev TPU path in the current process instead of submitting an Iris parent job.",
    )
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.num_concurrent < 1:
        parser.error("--num-concurrent must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.local and args.workers > 1:
        parser.error("--local only supports --workers 1; use --num-concurrent for local request fanout")
    if args.timeout_seconds is not None and args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1")
    tpu_region = args.tpu_region.strip()
    if not tpu_region:
        parser.error("--tpu-region must be non-empty")
    if "," in tpu_region:
        parser.error("--tpu-region accepts one region; rerun separately for another region")
    args.limit = args.limit if args.limit > 0 else None
    args.tpu_region = tpu_region

    try:
        return args, _build_inference_config(
            workers=args.workers,
            num_concurrent=args.num_concurrent,
            timeout_seconds=args.timeout_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args, inference_config = _parse_args(argv)

    eval_config = HumanEvalRunConfig(
        output_path=args.output_path,
        limit=args.limit,
        num_concurrent=inference_config.workers.max_in_flight_per_worker,
        request_timeout_seconds=int(inference_config.proxy.request_timeout_seconds),
    )

    if args.local:
        run_local_humaneval(inference_config, eval_config)
        return 0

    priority_band = priority_band_value(args.priority) if args.priority else job_pb2.PRIORITY_BAND_UNSPECIFIED
    runtime_config = IrisBrokeredVllmRuntimeConfig(
        region=args.tpu_region,
        worker_resources=ResourceConfig.with_tpu(args.tpu_type, ram=args.worker_ram),
        worker_env_vars=VLLM_WORKER_ENV_VARS,
        priority_band=priority_band,
    )
    submit_iris_humaneval(
        inference_config,
        eval_config,
        runtime_config,
        job_name=args.job_name,
        iris_config_path="lib/iris/config/marin.yaml",
        parent_ram=args.parent_ram,
        region=args.tpu_region,
        priority_band=priority_band,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
