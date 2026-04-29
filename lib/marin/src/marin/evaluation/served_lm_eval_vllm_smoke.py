# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from fray import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from rigging.filesystem import open_url, url_to_fs

from marin.evaluation.utils import is_remote_path
from marin.evaluation.served_lm_eval import LmEvalRun, run_lm_eval
from marin.inference.served_model import ModelDeployment, VllmModelLauncher
from marin.inference.vllm_server import VLLM_NATIVE_PIP_PACKAGES, resolve_vllm_mode
from marin.utils import remove_tpu_lockfile_on_exit

TOKENIZER_FILENAMES: tuple[str, ...] = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "config.json",
)
MAX_IRIS_JOB_NAME_COMPONENT_LENGTH = 80


def run_vllm_lm_eval_smoke(
    *,
    model: str,
    model_name: str,
    tokenizer: str | None,
    task: str,
    output_path: str,
    limit: int,
    num_fewshot: int,
    mode: Literal["docker", "native"] | None,
    docker_image: str | None,
    port: int,
    load_format: str | None,
    max_model_len: int | None,
    api_timeout: int,
) -> None:
    """Run the RFC 1 served-model lm-eval slice against a real vLLM server."""
    engine_kwargs: dict[str, object] = {}
    if load_format is not None:
        engine_kwargs["load_format"] = load_format
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len

    with _resolved_tokenizer(model, tokenizer) as resolved_tokenizer:
        deployment = ModelDeployment(
            model_name=model_name,
            model_path=model,
            tokenizer=resolved_tokenizer,
            engine_kwargs=engine_kwargs,
        )
        launcher = VllmModelLauncher(
            mode=mode,
            port=port,
            timeout_seconds=3600,
            docker_image=docker_image,
        )
        lm_eval_run = LmEvalRun(
            tasks=[task],
            output_path=output_path,
            limit=limit,
            num_fewshot=num_fewshot,
            batch_size=1,
            extra_model_args={
                "max_retries": 1,
                "timeout": api_timeout,
            },
        )

        with launcher.launch(deployment) as running_model:
            run_lm_eval(running_model, lm_eval_run)
        print_lm_eval_output_summary(output_path)


@contextmanager
def _resolved_tokenizer(model: str, tokenizer: str | None):
    if tokenizer is not None:
        yield tokenizer
        return
    if not is_remote_path(model):
        yield model
        return

    with tempfile.TemporaryDirectory(prefix="marin-tokenizer-") as local_dir:
        copied = _stage_remote_tokenizer_dir(model, local_dir)
        if not copied:
            raise ValueError(
                "lm-eval local-completions requires a Hugging Face tokenizer name/path, "
                f"but no tokenizer files were found under remote model path {model!r}. "
                "Pass --tokenizer with an HF tokenizer id or a local tokenizer path."
            )
        print(f"staged tokenizer files from {model} into {local_dir}")
        yield local_dir


def _stage_remote_tokenizer_dir(remote_dir: str, local_dir: str) -> bool:
    copied_any = False
    for filename in TOKENIZER_FILENAMES:
        remote_path = f"{remote_dir.rstrip('/')}/{filename}"
        fs, fs_path = url_to_fs(remote_path)
        if not fs.exists(fs_path):
            continue
        local_path = os.path.join(local_dir, filename)
        with open_url(remote_path, "rb") as src:
            data = src.read()
        with open(local_path, "wb") as dst:
            dst.write(data)
        copied_any = True
    return copied_any


def print_lm_eval_output_summary(output_path: str) -> None:
    output_dir = Path(output_path)
    result_files = sorted(output_dir.glob("*/results_*.json"))
    sample_files = sorted(output_dir.glob("*/samples_*.jsonl"))

    print("lm-eval output summary:")
    print(f"  output_path: {output_path}")
    print(f"  result_files: {len(result_files)}")
    for path in result_files:
        print(f"    {path}")
    print(f"  sample_files: {len(sample_files)}")
    for path in sample_files:
        print(f"    {path}")

    if not result_files:
        raise RuntimeError(f"lm-eval produced no results_*.json files under {output_path!r}.")

    _print_result_metrics(result_files[-1])
    if sample_files:
        _print_first_sample_shape(sample_files[0])


def _print_result_metrics(path: Path) -> None:
    with path.open() as f:
        payload = json.load(f)
    results = payload.get("results", {})
    print(f"  latest_result: {path}")
    for task_name, metrics in results.items():
        metric_items = []
        for key, value in metrics.items():
            if key.endswith("_stderr") or key.endswith(",stderr"):
                continue
            if isinstance(value, int | float | str | bool):
                metric_items.append(f"{key}={value}")
        print(f"  metrics[{task_name}]: {', '.join(metric_items)}")


def _print_first_sample_shape(path: Path) -> None:
    with path.open() as f:
        first_line = f.readline().strip()
    if not first_line:
        print(f"  first_sample[{path.name}]: <empty>")
        return
    sample = json.loads(first_line)
    print(f"  first_sample[{path.name}].keys: {sorted(sample.keys())}")


def _resource_config(args: argparse.Namespace) -> ResourceConfig:
    regions = _parse_csv(args.regions)
    kwargs: dict[str, object] = {
        "cpu": args.cpu,
        "ram": args.ram,
        "disk": args.disk,
    }
    if regions:
        kwargs["regions"] = regions

    if args.resource == "gpu":
        return ResourceConfig.with_gpu(args.gpu_type, count=args.gpu_count, **kwargs)
    if args.resource == "tpu":
        return ResourceConfig.with_tpu(args.tpu_type, **kwargs)
    raise ValueError(f"Unknown resource kind: {args.resource!r}")


def _environment_extras(resource: str) -> list[str]:
    if resource == "tpu":
        return ["eval", "tpu"]
    return ["eval", "cpu"]


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_with_optional_tpu_cleanup(resource: str, fn: Callable[[], None]) -> None:
    if resource == "tpu":
        with remove_tpu_lockfile_on_exit():
            fn()
        return
    fn()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manual smoke for RFC 1: launch real vLLM, expose RunningModel, and run real lm-eval. "
            "This is intentionally not a normal pytest test."
        )
    )
    parser.add_argument("--model", default="gpt2", help="HF id or object-store path served by vLLM.")
    parser.add_argument("--model-name", default=None, help="Logical deployment name. Defaults to --model.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "HF tokenizer id/path for lm-eval. Defaults to --model for local/HF models; "
            "remote object-store models stage tokenizer files from --model."
        ),
    )
    parser.add_argument("--task", default="arc_easy", help="lm-eval task name.")
    parser.add_argument("--limit", type=int, default=1, help="lm-eval limit.")
    parser.add_argument("--num-fewshot", type=int, default=0, help="lm-eval fewshot count.")
    parser.add_argument("--output-path", default="/tmp/marin-served-lm-eval-vllm-smoke", help="lm-eval output path.")
    parser.add_argument("--timeout", type=int, default=300, help="lm-eval API timeout.")
    parser.add_argument("--mode", choices=["docker", "native"], default="native", help="vLLM launch mode.")
    parser.add_argument("--docker-image", default=None, help="Optional vLLM Docker image override.")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port.")
    parser.add_argument(
        "--load-format",
        choices=["runai_streamer", "runai_streamer_sharded"],
        default=None,
        help="Optional vLLM load format for object-store checkpoints.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048, help="vLLM max model length.")
    parser.add_argument("--resource", choices=["gpu", "tpu"], default="tpu", help="Manual Iris resource kind.")
    parser.add_argument("--gpu-type", default="H100", help="GPU type for Iris manual job.")
    parser.add_argument("--gpu-count", type=int, default=1, help="GPU count for Iris manual job.")
    parser.add_argument("--tpu-type", default="v5p-8", help="TPU type for Iris manual job.")
    parser.add_argument("--cpu", type=int, default=16, help="CPU request for the manual job.")
    parser.add_argument("--ram", default="128g", help="RAM request for the manual job.")
    parser.add_argument("--disk", default="128g", help="Disk request for the manual job.")
    parser.add_argument("--regions", default=None, help="Optional comma-separated Iris regions.")
    parser.add_argument("--local", action="store_true", help="Run in the current process instead of submitting Iris.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to set in the submitted job. Can be repeated.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    mode = resolve_vllm_mode(args.mode)
    if mode == "native" and args.resource != "tpu":
        raise ValueError("Native vLLM mode in this repo is TPU-oriented; use --resource tpu or --mode docker.")

    model_name = args.model_name or args.model
    tokenizer = args.tokenizer

    def _run() -> None:
        start = time.time()
        try:
            run_vllm_lm_eval_smoke(
                model=args.model,
                model_name=model_name,
                tokenizer=tokenizer,
                task=args.task,
                output_path=args.output_path,
                limit=args.limit,
                num_fewshot=args.num_fewshot,
                mode=mode,
                docker_image=args.docker_image,
                port=args.port,
                load_format=args.load_format,
                max_model_len=args.max_model_len,
                api_timeout=args.timeout,
            )
        except Exception:
            traceback.print_exc()
            raise
        elapsed = time.time() - start
        print(f"served lm-eval vLLM smoke completed in {elapsed:.1f}s; output_path={args.output_path}")

    if args.local:
        _run_with_optional_tpu_cleanup(args.resource, _run)
        return 0

    env_vars = dict(_parse_env(args.env))
    if args.docker_image is not None:
        env_vars["MARIN_VLLM_DOCKER_IMAGE"] = args.docker_image
    env_vars["MARIN_VLLM_MODE"] = mode

    client = current_client()
    job_request = JobRequest(
        name=_smoke_job_name(args.resource, model_name),
        entrypoint=Entrypoint.from_callable(lambda: _run_with_optional_tpu_cleanup(args.resource, _run)),
        resources=_resource_config(args),
        environment=create_environment(
            extras=_environment_extras(args.resource),
            pip_packages=VLLM_NATIVE_PIP_PACKAGES if mode == "native" else (),
            env_vars=env_vars or None,
        ),
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)
    return 0


def _parse_env(items: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--env key must be non-empty, got {item!r}")
        parsed.append((key, value))
    return parsed


def _smoke_job_name(resource: str, model_name: str) -> str:
    return f"served-lm-eval-vllm-smoke-{resource}-{_job_name_component(model_name)}"


def _job_name_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not component:
        return "model"
    return component[:MAX_IRIS_JOB_NAME_COMPONENT_LENGTH].rstrip("-") or "model"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
