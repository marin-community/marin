# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark GPT-OSS 120B Stage 3 extraction from a seeded Stage 2 checkpoint.

This experiment seeds a fresh prompt-generation output directory with Stage 1
and Stage 2 checkpoints from the earlier full-spec GPT-OSS 120B run and then
invokes the normal prompt-generation entrypoint. Because the seeded output has
no Stage 3 extraction shards, Marin skips Stages 1 and 2 and runs extraction
only.

The source checkpoint came from:
  - job: `/ahmed/goss-120b-full-spec-ckpt-resume-east5-20260328-restart1`
  - prompts artifact: `gs://marin-us-east5/align/goss_120b_full_spec/prompts-c623f9`

Run on an east5 TPU worker. Zone selection still comes from the Iris CLI:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --extra marin:tpu \
        --tpu <slice> \
        --region us-east5 \
        --zone us-east5-b \
        -e MARIN_PREFIX gs://marin-us-east5 \
        -- python experiments/benchmark_gpt_oss_120b_stage3_from_stage2.py \
        --name goss_120b_stage3_bench_east5b \
        --tpu-type v6e-128

The TPU layout is intentionally exposed as CLI args because GPT-OSS 120B has
only been validated on some slice layouts so far. When `--tpu-type` is a v6e
slice and `--tensor-parallel-size` is omitted, the script uses the host-aware
defaults in `experiments/gpt_oss_120b_v6e_config.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from iris.marin_fs import url_to_fs
from zephyr import load_jsonl, write_jsonl_file

from experiments.gpt_oss_120b_tpu import GPT_OSS_TPU_DEFAULT_MAX_TOKENS, gpt_oss_120b_tpu_vllm_config
from experiments.gpt_oss_120b_v6e_config import gpt_oss_120b_v6e_vllm_config
from marin.alignment.align import _llm_env_vars
from marin.alignment.generate_prompts import PromptGenConfig, generate_prompts_from_spec
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.execution.remote import remote

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")
SOURCE_JOB_ID = "/ahmed/goss-120b-full-spec-ckpt-resume-east5-20260328-restart1"
# Relative path under the marin prefix — resolved at runtime via MARIN_PREFIX
_SOURCE_PROMPTS_RELATIVE = "align/goss_120b_full_spec/prompts-c623f9"
_CHECKPOINT_DIRNAME = "artifacts/checkpoints"
_STAGE1_CHECKPOINT_FILENAME = "understandings.jsonl.gz"
_STAGE2_CHECKPOINT_FILENAME = "ideations.jsonl.gz"
_STAGE3_CHECKPOINT_DIRNAME = "extractions"
_STAGE_STATUS_FILENAME = "stage_status.json"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="benchmark_gpt_oss_120b_stage3_from_stage2")
    parser.add_argument("--source-output-path", default=None)
    parser.add_argument("--local-serve-batch-size", type=int, default=256)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--statement-id", action="append", dest="statement_ids", default=None)
    parser.add_argument("--uncensored", action="store_true", help="Use the uncensored GPT-OSS 120B model")
    parser.add_argument("--extract-max-tokens", type=int, default=None, help="Override extract_max_tokens")
    return parser.parse_known_args()


_SOURCE_OUTPUT_PATH_ENV = "STAGE3_BENCH_SOURCE_OUTPUT_PATH"


@dataclass(frozen=True)
class _Stage3BenchmarkConfig:
    prompt_config: PromptGenConfig


def _checkpoint_base_path(output_path: str) -> str:
    return f"{output_path}/{_CHECKPOINT_DIRNAME}"


def _stage1_checkpoint_path(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE1_CHECKPOINT_FILENAME}"


def _stage2_checkpoint_path(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE2_CHECKPOINT_FILENAME}"


def _stage3_checkpoint_dir(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE3_CHECKPOINT_DIRNAME}"


def _stage_status_path(output_path: str) -> str:
    return f"{_checkpoint_base_path(output_path)}/{_STAGE_STATUS_FILENAME}"


def _load_filtered_checkpoint_records(
    checkpoint_path: str,
    *,
    record_key: str,
    selected_statement_ids: set[str] | None,
) -> list[dict[str, Any]]:
    records = list(load_jsonl(checkpoint_path))
    if selected_statement_ids is None:
        return records
    return [record for record in records if record[record_key] in selected_statement_ids]


def _write_json(path: str, payload: dict[str, Any]) -> None:
    fs, fs_path = url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _remove_path_if_present(path: str) -> None:
    fs, fs_path = url_to_fs(path)
    if fs.exists(fs_path):
        fs.rm(fs_path, recursive=True)


def _remove_prompt_shards(output_path: str) -> None:
    fs, fs_path = url_to_fs(output_path)
    for shard in fs.glob(f"{fs_path}/shard_*.jsonl.gz"):
        fs.rm(shard)


def _seed_stage12_checkpoints(config: _Stage3BenchmarkConfig) -> None:
    import os

    source_output_path = os.environ.get(_SOURCE_OUTPUT_PATH_ENV, "")
    if not source_output_path:
        raise ValueError(f"Environment variable {_SOURCE_OUTPUT_PATH_ENV} must be set to the source prompts path")

    output_path = config.prompt_config.output_path
    stage1_target = _stage1_checkpoint_path(output_path)
    stage2_target = _stage2_checkpoint_path(output_path)
    target_fs, target_stage1_path = url_to_fs(stage1_target)
    _, target_stage2_path = url_to_fs(stage2_target)
    if target_fs.exists(target_stage1_path) and target_fs.exists(target_stage2_path):
        return

    selected_statement_ids = (
        set(config.prompt_config.statement_ids) if config.prompt_config.statement_ids is not None else None
    )
    source_stage1 = _stage1_checkpoint_path(source_output_path)
    source_stage2 = _stage2_checkpoint_path(source_output_path)
    stage1_records = _load_filtered_checkpoint_records(
        source_stage1,
        record_key="statement_id",
        selected_statement_ids=selected_statement_ids,
    )
    stage2_records = _load_filtered_checkpoint_records(
        source_stage2,
        record_key="statement_id",
        selected_statement_ids=selected_statement_ids,
    )
    if not stage1_records:
        raise ValueError(f"No Stage 1 checkpoint records found in {source_stage1}")
    if not stage2_records:
        raise ValueError(f"No Stage 2 checkpoint records found in {source_stage2}")
    if {record["statement_id"] for record in stage1_records} != {record["statement_id"] for record in stage2_records}:
        raise ValueError("Seeded Stage 1 and Stage 2 checkpoints contain different statement ID sets.")
    if selected_statement_ids is not None:
        seeded_statement_ids = {record["statement_id"] for record in stage2_records}
        missing_statement_ids = sorted(selected_statement_ids - seeded_statement_ids)
        if missing_statement_ids:
            raise ValueError(f"Requested statement IDs are missing from the source checkpoint: {missing_statement_ids}")

    write_jsonl_file(stage1_records, stage1_target)
    write_jsonl_file(stage2_records, stage2_target)
    _remove_path_if_present(_stage3_checkpoint_dir(output_path))
    _remove_prompt_shards(output_path)
    stage_status = {
        "understanding": {"complete": True, "num_statements": len(stage1_records)},
        "concretize": {"complete": True, "num_statements": len(stage2_records)},
        "extract": {"complete": False, "completed_items": 0},
    }
    _write_json(_stage_status_path(output_path), stage_status)
    _write_json(
        f"{output_path}/artifacts/stage3_benchmark_seed.json",
        {
            "source_job_id": SOURCE_JOB_ID,
            "source_output_path": source_output_path,
            "statement_ids": sorted(record["statement_id"] for record in stage2_records),
        },
    )


def _run_stage3_benchmark(config: _Stage3BenchmarkConfig) -> None:
    _seed_stage12_checkpoints(config)
    generate_prompts_from_spec(config.prompt_config)


def build_steps(
    *,
    name: str,
    source_output_path: str,
    local_serve_batch_size: int,
    statement_ids: list[str] | None,
    tpu_type: str,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    uncensored: bool = False,
    extract_max_tokens: int | None = None,
) -> list[ExecutorStep]:
    if tpu_type.startswith("v6e-"):
        gpt_oss_vllm = gpt_oss_120b_v6e_vllm_config(
            tpu_type=tpu_type,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=8192,
            gpu_memory_utilization=0.9 if gpu_memory_utilization is None else gpu_memory_utilization,
            ram="400g",
            model_impl_type="vllm",
            prefer_jax_for_bootstrap=False,
        )
    else:
        gpt_oss_vllm = gpt_oss_120b_tpu_vllm_config(
            tpu_type=tpu_type,
            tensor_parallel_size=4 if tensor_parallel_size is None else tensor_parallel_size,
            max_model_len=8192,
            gpu_memory_utilization=0.9 if gpu_memory_utilization is None else gpu_memory_utilization,
            ram="400g",
            model_impl_type="vllm",
            prefer_jax_for_bootstrap=False,
        )
    if uncensored:
        from experiments.models import gpt_oss_120b_abliterated_vllm

        gpt_oss_vllm = dataclasses.replace(gpt_oss_vllm, model=output_path_of(gpt_oss_120b_abliterated_vllm))

    # Let the child inherit region from the parent job's Iris constraint
    # instead of hardcoding east5.
    tpu_resources = gpt_oss_vllm.resources

    prompt_config = PromptGenConfig(
        spec_path=SPEC_PATH,
        output_path=this_output_path(),
        ideation_model=gpt_oss_vllm,
        extract_model=gpt_oss_vllm,
        covering_strength=versioned(2),
        covering_seed=versioned(42),
        local_serve_batch_size=versioned(local_serve_batch_size),
        ideation_workers=1,
        concretize_workers=1,
        extract_workers=1,
        understanding_max_tokens=versioned(GPT_OSS_TPU_DEFAULT_MAX_TOKENS),
        understanding_temperature=versioned(1.0),
        understanding_max_attempts=versioned(5),
        concretize_temperature=versioned(1.0),
        concretize_max_tokens=versioned(GPT_OSS_TPU_DEFAULT_MAX_TOKENS),
        extract_max_tokens=versioned(extract_max_tokens or GPT_OSS_TPU_DEFAULT_MAX_TOKENS),
        concretize_max_attempts=versioned(5),
        extract_max_attempts=versioned(5),
        statement_ids=versioned(statement_ids),
    )

    prompts_step = ExecutorStep(
        name=f"align/{name}/prompts",
        description="Benchmark GPT-OSS 120B Stage 3 extraction by seeding Stage 1/2 checkpoints from an earlier run",
        fn=remote(
            _run_stage3_benchmark,
            resources=tpu_resources,
            pip_dependency_groups=gpt_oss_vllm.pip_dependency_groups,
            pip_packages=gpt_oss_vllm.pip_packages,
            env_vars={**_llm_env_vars(), _SOURCE_OUTPUT_PATH_ENV: source_output_path},
        ),
        config=_Stage3BenchmarkConfig(
            prompt_config=prompt_config,
        ),
    )

    return [prompts_step]


if __name__ == "__main__":
    args, executor_args = parse_args()
    resolved_source = args.source_output_path or f"gs://marin-us-east5/{_SOURCE_PROMPTS_RELATIVE}"
    sys.argv = [sys.argv[0], *executor_args]
    executor_main(
        steps=build_steps(
            name=args.name,
            source_output_path=resolved_source,
            local_serve_batch_size=args.local_serve_batch_size,
            statement_ids=args.statement_ids,
            tpu_type=args.tpu_type,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            uncensored=args.uncensored,
            extract_max_tokens=args.extract_max_tokens,
        ),
        description="Benchmark GPT-OSS 120B prompt-generation Stage 3 from a seeded Stage 2 checkpoint",
    )
