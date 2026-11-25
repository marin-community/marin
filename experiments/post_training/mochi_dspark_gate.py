# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify Mochi DSpark before measuring matched long-thinking decode windows."""

import argparse
import asyncio
import hashlib
import json
import logging
import tempfile
from enum import StrEnum
from pathlib import Path
from statistics import mean

from iris.cluster.client.job_info import get_job_info
from marin.inference.backend import ModelSpec
from marin.inference.config import ServingGeometry, VllmCompilationCacheMode, VllmEngineConfig
from marin.inference.iris_vllm import (
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_backend import VllmBackend
from marin.inference.vllm_server import VllmEnvironment
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.io import io

from experiments.post_training.long_loop.throughput import graph_capture_observed, workload_requests, write_json
from experiments.post_training.mochi_correctness import (
    AUTOREGRESSIVE_ARTIFACT,
    MAX_ROUTE_CHOICE_NOISE,
    decode_probes,
    prefix_scores,
    sampling_by_prompt,
    sampling_difference,
    score_difference,
)
from experiments.post_training.mochi_decode_window import DecodeWindowConfig, steady_decode_window

logger = logging.getLogger(__name__)


class GatePurpose(StrEnum):
    QUALIFICATION = "qualification"
    BENCHMARK = "benchmark"
    PLAIN_BATCHING = "plain-batching"


def checked_json(uri: str, sha256: str) -> dict:
    payload = StoragePath(uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"Pinned JSON changed: {uri}")
    return json.loads(payload)


def engine_config(spec: dict, draft: Path | None) -> VllmEngineConfig:
    arguments = (
        "--enable-expert-parallel",
        "--attention-backend",
        "FLASH_ATTN",
        "--moe-backend",
        "triton",
        "--enable-return-routed-experts",
        "--load-format",
        "runai_streamer",
        "--model-loader-extra-config",
        json.dumps({"distributed": False, "concurrency": 4, "memory_limit": 8589934592}),
        "--gpu-memory-utilization",
        str(spec["gpu_memory_utilization"]),
        "--enable-prefix-caching",
        "--scheduling-policy",
        "priority",
        "--compilation-config",
        json.dumps(spec["compilation_config"]),
    )
    if draft is not None:
        arguments += (
            "--speculative-config",
            json.dumps(
                {
                    "method": "dspark",
                    "model": str(draft),
                    "num_speculative_tokens": 4,
                    "draft_sample_method": "probabilistic",
                    "rejection_sample_method": "standard",
                    "max_model_len": spec["model"]["max_model_len"],
                }
            ),
        )
    return VllmEngineConfig(
        compilation_cache=VllmCompilationCacheMode.CALLER_MANAGED,
        startup_timeout_seconds=7200,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        extra_args=arguments,
        extra_metric_families=frozenset(
            {
                "vllm:spec_decode_num_drafts",
                "vllm:spec_decode_num_draft_tokens",
                "vllm:spec_decode_num_accepted_tokens",
            }
        ),
    )


def require_difference(spec: dict, name: str, reference: dict, scores: dict, bounds: dict) -> None:
    difference = score_difference(reference, scores)
    passed = all(difference[key] <= bound for key, bound in bounds.items())
    write_json(
        spec["output"], f"{name}.json", {"passed": passed, "difference": difference, "bounds": bounds, "scores": scores}
    )
    if not passed:
        raise ValueError(f"Strict correctness failed: {name}; no performance measurements permitted")


async def qualify(environment: VllmEnvironment, spec: dict, stage: str, calibration: dict) -> None:
    base = environment.server_url
    if stage == "plain-rescore":
        fixed_reference = json.loads((StoragePath(spec["output"]) / "plain-check/fixed-reference.json").read_text())
        teacher = await prefix_scores(base, calibration["probes"], 16)
        require_difference(
            spec, "plain-rescore/fixed-control", fixed_reference["scores"], teacher, calibration["fixed_bounds"]
        )
        for arm in ("plain-check", "dspark-check"):
            recorded = json.loads((StoragePath(spec["output"]) / f"{arm}/{AUTOREGRESSIVE_ARTIFACT}").read_text())
            teacher = await prefix_scores(base, list(recorded["scores"].values()), 16)
            require_difference(
                spec, f"{arm}/autoregressive-correctness", recorded["scores"], teacher, calibration["decode_bounds"]
            )
        write_json(
            spec["output"],
            "correctness-passed.json",
            {
                "passed": True,
                "calibration_sha256": spec["calibration_sha256"],
                "teacher_model": spec["model"]["weights"],
                "reference": "independent plain decoding of the same policy",
            },
        )
        return
    scores = await prefix_scores(base, calibration["probes"], 16)
    if stage == "plain-check":
        write_json(spec["output"], "plain-check/fixed-reference.json", {"scores": scores})
        for repeat in range(2):
            teacher = await prefix_scores(base, calibration["probes"], 16)
            require_difference(
                spec, f"plain-check/independent-control-{repeat}", scores, teacher, calibration["fixed_bounds"]
            )
    else:
        reference = json.loads((StoragePath(spec["output"]) / "plain-check/fixed-reference.json").read_text())
        require_difference(spec, f"{stage}/fixed-correctness", reference["scores"], scores, calibration["fixed_bounds"])
    decoded = await decode_probes(base, calibration["probes"], spec["correctness_seeds"])
    write_json(spec["output"], f"{stage}/{AUTOREGRESSIVE_ARTIFACT}", decoded)
    grouped = sampling_by_prompt(decoded["trajectories"])
    if stage == "plain-check":
        # Rebind the teacher to this policy without changing the frozen noise limits.
        sampling_reference = {
            **calibration["sampling"],
            "by_prompt": grouped,
            "metrics": {
                metric: {
                    **bounds,
                    "reference": mean(mean(row[metric] for row in rows) for rows in grouped.values()),
                }
                for metric, bounds in calibration["sampling"]["metrics"].items()
            },
        }
        write_json(spec["output"], "plain-check/sampling-reference.json", sampling_reference)
        teacher = await prefix_scores(base, list(decoded["scores"].values()), 16)
        require_difference(
            spec, f"{stage}/fresh-teacher-control", decoded["scores"], teacher, calibration["decode_bounds"]
        )
        return
    reference = json.loads((StoragePath(spec["output"]) / "plain-check/sampling-reference.json").read_text())
    sampling = sampling_difference(grouped, reference)
    write_json(spec["output"], f"{stage}/sampling-correctness.json", sampling)
    if not sampling["passed"]:
        raise ValueError(f"Strict sampled-distribution check failed: {stage}")


async def investigate_plain_batching(environment: VllmEnvironment, spec: dict, calibration: dict) -> None:
    """Retain plain-server controls without treating diagnostic completion as qualification."""
    # Each pair repeats the same inventory; adjacent pairs change only request
    # concurrency or the requested DP rank. Engine options remain gate-exact.
    conditions = (
        ("auto-c16", 16, None),
        ("auto-c1", 1, None),
        ("rank0-c1", 1, 0),
        ("rank1-c1", 1, 1),
        ("rank0-c16", 16, 0),
    )
    scores_by_condition = {}
    summary = []
    for name, concurrency, rank in conditions:
        first = None
        for repeat in range(2):
            scores = await prefix_scores(
                environment.server_url, calibration["probes"], concurrency, data_parallel_rank=rank
            )
            if first is None:
                first = scores
            scores_by_condition[name] = scores
            difference = score_difference(first, scores)
            passed = all(difference[key] <= bound for key, bound in calibration["fixed_bounds"].items())
            record = {
                "condition": name,
                "repeat": repeat,
                "request_concurrency": concurrency,
                "requested_data_parallel_rank": rank,
                "scores": scores,
                "difference_from_same_condition_first": difference,
                "within_frozen_bounds": passed,
                "bounds": calibration["fixed_bounds"],
            }
            write_json(spec["output"], f"plain-batching/{name}-{repeat}.json", record)
            summary.append({key: value for key, value in record.items() if key != "scores"})
            logger.info("Plain batching condition %s repeat %s: %s", name, repeat, difference)
    comparisons = {}
    for reference, candidate in (
        ("auto-c16", "auto-c1"),
        ("auto-c1", "rank0-c1"),
        ("rank0-c1", "rank1-c1"),
        ("rank0-c1", "rank0-c16"),
    ):
        comparisons[f"{reference}-vs-{candidate}"] = score_difference(
            scores_by_condition[reference], scores_by_condition[candidate]
        )
    write_json(
        spec["output"],
        "plain-batching/investigation-complete.json",
        {
            "teacher_model": spec["model"]["weights"],
            "conditions": summary,
            "comparisons": comparisons,
            "engine_options_unchanged": True,
            "prefix_caching_enabled": True,
            "cache_state": "Later sweeps share a cache touched by the inventory; eviction is not controlled",
            "comparison_reference": "Second sweep of each condition; first sweep retained separately",
            "limitation": "Request concurrency is controlled, not an exact realized GPU microbatch shape",
            "qualification_passed": False,
            "RL_launch_authorized": False,
            "calibration_sha256": spec["calibration_sha256"],
        },
    )


async def benchmark(environment: VllmEnvironment, spec: dict, stage: str) -> None:
    if not json.loads((StoragePath(spec["output"]) / "correctness-passed.json").read_text())["passed"]:
        raise ValueError("All correctness stages must pass before measuring load")
    for running in (16, 32, 48):
        config = DecodeWindowConfig(**spec["decode_window"], running_per_engine=running)
        result = await steady_decode_window(
            environment,
            workload_requests(spec["performance_workload"], config.seed),
            config,
            speculative=stage == "dspark-performance",
        )
        write_json(spec["output"], f"{stage}/running{running}.json", result)
        if not result["valid"]:
            raise ValueError(f"Unqualified occupancy/counters/KV preemptions: {stage} running{running}")
        logger.info("Measured %s running%s: %s", stage, running, result["summary"])


def run(spec: dict, purpose: GatePurpose) -> None:
    info = get_job_info()
    if info is None or (info.worker_region is not None and info.worker_region.casefold() != "us-east-08a"):
        raise ValueError("Run this gate only in the registered 08a Iris gang")
    calibration = checked_json(spec["calibration"], spec["calibration_sha256"])
    if not calibration["sensitivity_passed"] or any(
        calibration[name]["mean_prompt_route_choice_mismatch_fraction"] > MAX_ROUTE_CHOICE_NOISE
        for name in ("fixed_bounds", "decode_bounds")
    ):
        raise ValueError("The frozen route check lacks required sensitivity")
    geometry = ServingGeometry(**spec["geometry"])
    if geometry.tensor_parallel_size != 1 or geometry.data_parallel_size != 16 or geometry.total_gpus != 16:
        raise ValueError("This approval covers one TP1/EP16, 16-GB200 gang")
    stages = (
        ("plain-batching",)
        if purpose == GatePurpose.PLAIN_BATCHING
        else ("plain-check", "dspark-check", "plain-rescore")
    )
    if purpose == GatePurpose.BENCHMARK:
        stages += ("plain-performance", "dspark-performance")
    with tempfile.TemporaryDirectory(prefix="mochi-dspark-gate-") as scratch:
        draft = Path(scratch) / "draft"
        if purpose != GatePurpose.PLAIN_BATCHING:
            io.download_directory(spec["draft"], str(draft))
            manifest = checked_json(spec["draft_manifest"], spec["draft_manifest_sha256"])
            for entry in manifest["files"]:
                local = draft / entry["path"]
                if (
                    local.stat().st_size != entry["size"]
                    or hashlib.sha256(local.read_bytes()).hexdigest() != entry["sha256"]
                ):
                    raise ValueError("Staged draft differs from the verified-copy SHA manifest")
        for stage in stages:
            speculative = stage.startswith("dspark")
            with iris_vllm_launch(geometry, instance_name=f"vllm-{stage}") as launch:
                with VllmBackend(engine_config(spec, draft if speculative else None)).start(
                    ModelSpec(**spec["model"]),
                    extra_args=launch.extra_cli_args,
                    subprocess_env=launch.subprocess_env,
                ) as environment:
                    try:
                        if launch.is_leader:
                            with iris_vllm_followers(launch):
                                environment.wait_until_ready()
                                if stage == "plain-batching":
                                    asyncio.run(investigate_plain_batching(environment, spec, calibration))
                                elif stage.endswith("performance"):
                                    asyncio.run(benchmark(environment, spec, stage))
                                else:
                                    asyncio.run(qualify(environment, spec, stage, calibration))
                        else:
                            wait_for_iris_vllm_shutdown(launch, environment.check_alive)
                    finally:
                        logs = environment.logs()
                        (StoragePath(spec["output"]) / stage / f"task-{info.task_index}-vllm.log").write_text(logs)
                    if not graph_capture_observed(logs):
                        raise ValueError(f"CUDA graph capture unconfirmed for {stage} task {info.task_index}")
                if not launch.is_leader:
                    notify_iris_vllm_stopped(launch)
        write_json(
            spec["output"],
            f"task-{info.task_index}-complete.json",
            {"task": str(info.task_id), "source_revision": spec["source_revision"]},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--purpose", type=GatePurpose, choices=list(GatePurpose), required=True)
    args = parser.parse_args()
    spec = json.loads(StoragePath(args.spec).read_text())
    spec["source_revision"] = args.source_revision
    run(spec, args.purpose)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
