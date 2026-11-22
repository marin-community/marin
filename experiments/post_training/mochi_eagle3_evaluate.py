# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a frozen Mochi draft with target-only serving on the same Iris gang."""

import asyncio
import json
import logging
import math
import tempfile
from pathlib import Path
from statistics import mean, median, variance

import httpx
from iris.cluster.client.job_info import get_job_info
from marin.evaluation.inference_metrics import InferenceMetricWindow
from marin.inference.config import ServingGeometry
from marin.inference.iris_vllm import (
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.load_test import LoadConfig, LoadRequest, run_load
from marin.inference.vllm_backend import VllmBackend
from marin.inference.vllm_server import VllmEnvironment
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.io import io
from skyrl_train.trajectory_runners.routed_experts import decode_routed_experts, normalize_routed_experts

from experiments.post_training.long_loop.throughput import graph_capture_observed, workload_requests, write_json
from experiments.post_training.mochi_eagle3 import STOP_TOKEN_IDS, worker_spec
from experiments.post_training.mochi_eagle3_capture import mochi_engine_config, mochi_model_spec

logger = logging.getLogger(__name__)
PROBE_POSITIONS = (0, 1, 3, 7, 15, 31, 63)
DECODE_POSITIONS = (*PROBE_POSITIONS, 127, 255)
AUTOREGRESSIVE_ARTIFACT = "autoregressive.json"
SAMPLING_STANDARD_ERRORS = 3


async def probe_inventory(base_url: str, requests: list[LoadRequest]) -> list[dict]:
    """Freeze identical-prefix checks from exact target-generated token IDs."""
    probes = []
    async with httpx.AsyncClient(timeout=1800) as client:
        for request in requests:
            response = await client.post(
                f"{base_url}/{request.path}",
                json={
                    **request.body,
                    "model": "mochi",
                    "max_tokens": 64,
                    "temperature": 0,
                    "ignore_eos": True,
                    "return_token_ids": True,
                    "seed": 17,
                },
            )
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            prompt = result["prompt_token_ids"] if request.path == "chat/completions" else choice["prompt_token_ids"]
            generated = choice["token_ids"]
            if len(generated) != 64:
                raise ValueError("The fixed correctness probe did not return 64 exact generated IDs")
            probes.extend(
                {
                    "identity": f"{request.identity}@{position}",
                    "prompt": prompt + generated[: position + 1],
                    "token": generated[position],
                }
                for position in PROBE_POSITIONS
            )
    return probes


async def prefix_scores(base_url: str, probes: list[dict], concurrency: int) -> dict[str, dict]:
    """Score the same known next tokens without retokenizing generated text."""
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=1800) as client:

        async def score(probe: dict) -> tuple[str, dict]:
            async with semaphore:
                response = await client.post(
                    f"{base_url}/completions",
                    json={
                        "model": "mochi",
                        "prompt": probe["prompt"],
                        "max_tokens": 1,
                        "temperature": 0,
                        "prompt_logprobs": 1,
                        "return_token_ids": True,
                        "routed_experts_prompt_start": len(probe["prompt"]) - 2,
                    },
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                entries = choice["prompt_logprobs"][-1]
                top = min(entries, key=lambda token: (-entries[token]["logprob"], int(token)))
                routes = decode_routed_experts(choice["routed_experts"], 2)
                return probe["identity"], {
                    "logprob": entries[str(probe["token"])]["logprob"],
                    "top1": int(top),
                    "top1_logprob": entries[top]["logprob"],
                    "routes": routes[0].tolist(),
                }

        return dict(await asyncio.gather(*(score(probe) for probe in probes)))


def score_difference(reference: dict[str, dict], candidate: dict[str, dict]) -> dict:
    """Measure fixed-token logprob drift and top-one mismatch on identical prefixes."""
    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("Correctness score inventories differ or are empty")
    errors = [abs(row["logprob"] - candidate[key]["logprob"]) for key, row in reference.items()]
    top_errors = [abs(row["top1_logprob"] - candidate[key]["top1_logprob"]) for key, row in reference.items()]
    return {
        "prefixes": len(errors),
        "max_absolute_logprob_error": max(errors),
        "mean_absolute_logprob_error": sum(errors) / len(errors),
        "max_top1_logprob_error": max(top_errors),
        "top1_mismatch_fraction": (
            sum(row["top1"] != candidate[key]["top1"] for key, row in reference.items()) / len(errors)
        ),
        "route_mismatch_fraction": (
            sum(row["routes"] != candidate[key]["routes"] for key, row in reference.items()) / len(errors)
        ),
    }


async def decode_probes(base_url: str, probes: list[dict], seeds: list[int]) -> dict:
    """Retain autoregressive target logprobs and routes across speculative rejections."""
    async with httpx.AsyncClient(timeout=1800) as client:

        async def decode(probe: dict, temperature: float, seed: int) -> dict:
            prompt = probe["prompt"][:-1]
            response = await client.post(
                f"{base_url}/completions",
                json={
                    "model": "mochi",
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "seed": seed,
                    "stop_token_ids": list(STOP_TOKEN_IDS),
                    "return_token_ids": True,
                    "return_tokens_as_token_ids": True,
                    "logprobs": 1,
                    "routed_experts_prompt_start": len(prompt) - 1,
                },
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            tokens = choice["token_ids"]
            routes = normalize_routed_experts(choice["routed_experts"], prompt[-1:], tokens)
            if routes.shape[1:] != (48, 8) or routes.max() >= 384:
                raise ValueError("Autoregressive Mochi routes are not aligned to target output IDs")
            decoded = {}
            for position in DECODE_POSITIONS:
                if position >= len(tokens):
                    continue
                identity = f"{probe['identity']}/seed{seed}/temp{temperature}/decode{position}"
                top = choice["logprobs"]["top_logprobs"][position]
                top_token = min(top, key=lambda token: (-top[token], int(token.removeprefix("token_id:"))))
                decoded[identity] = {
                    "logprob": choice["logprobs"]["token_logprobs"][position],
                    "top1": int(top_token.removeprefix("token_id:")),
                    "top1_logprob": top[top_token],
                    "routes": routes[position].tolist(),
                    "prompt": prompt + tokens[: position + 1],
                    "token": tokens[position],
                    "identity": identity,
                }
            return {
                "scores": decoded,
                "trajectory": {
                    "identity": probe["identity"],
                    "seed": seed,
                    "temperature": temperature,
                    "choice": choice,
                },
            }

        results = await asyncio.gather(
            *(
                decode(probe, temperature, seed)
                for probe in probes
                if probe["identity"].endswith("@0")
                for temperature in (0, 0.7)
                for seed in seeds
            )
        )
    return {
        "scores": {key: row for result in results for key, row in result["scores"].items()},
        "trajectories": [result["trajectory"] for result in results],
    }


def sampling_by_prompt(trajectories: list[dict]) -> dict[str, list[dict]]:
    """Keep the prompt as the sampling unit rather than treating correlated tokens as independent."""
    grouped: dict[str, list[dict]] = {}
    for row in trajectories:
        if row["temperature"] == 0:
            continue
        choice = row["choice"]
        logprobs = choice["logprobs"]
        grouped.setdefault(row["identity"], []).append(
            {
                "seed": row["seed"],
                "mean_logprob": mean(logprobs["token_logprobs"]),
                "mean_top1_probability": mean(math.exp(max(top.values())) for top in logprobs["top_logprobs"]),
                "tokens": len(choice["token_ids"]),
                "finish_reason": choice["finish_reason"],
            }
        )
    return grouped


def sampling_calibration(grouped: dict[str, list[dict]], numerical_bounds: dict) -> dict:
    """Predeclare three-standard-error bounds from paired target-only seed repeats."""
    metrics = {"mean_logprob": "max_absolute_logprob_error", "mean_top1_probability": "max_top1_logprob_error"}
    calibrated = {}
    for metric, numerical_metric in metrics.items():
        values = [[row[metric] for row in rows] for rows in grouped.values()]
        standard_error = math.sqrt(sum(2 * variance(group) / len(group) for group in values)) / len(values)
        numerical_bound = numerical_bounds[numerical_metric]
        if metric == "mean_top1_probability":
            numerical_bound = math.expm1(numerical_bound)
        calibrated[metric] = {
            "reference": mean(mean(group) for group in values),
            "maximum_mean_shift": max(numerical_bound, SAMPLING_STANDARD_ERRORS * standard_error),
            "standard_error": standard_error,
        }
    return {"by_prompt": grouped, "metrics": calibrated, "standard_errors": SAMPLING_STANDARD_ERRORS}


def sampling_difference(grouped: dict[str, list[dict]], calibration: dict) -> dict:
    if grouped.keys() != calibration["by_prompt"].keys():
        raise ValueError("Sampling-check prompt inventories differ")
    for prompt, rows in grouped.items():
        if {row["seed"] for row in rows} != {row["seed"] for row in calibration["by_prompt"][prompt]}:
            raise ValueError("Sampling-check seed inventories differ")
    metrics = {
        metric: {
            "observed": mean(mean(row[metric] for row in rows) for rows in grouped.values()),
            **bounds,
        }
        for metric, bounds in calibration["metrics"].items()
    }
    return {
        "passed": all(abs(row["observed"] - row["reference"]) <= row["maximum_mean_shift"] for row in metrics.values()),
        "metrics": metrics,
        "by_prompt": grouped,
    }


async def steady_counters(environment: VllmEnvironment, config: LoadConfig, speculative: bool) -> dict:
    await asyncio.sleep(config.warmup_seconds)
    window = await asyncio.to_thread(InferenceMetricWindow.start, environment, speculative=speculative)
    await asyncio.sleep(config.measurement_seconds)
    result = await asyncio.to_thread(window.finish)
    return result.model_dump(mode="json")


async def benchmark(environment: VllmEnvironment, spec: dict, arm: dict) -> None:
    """Retain request latency, actual per-engine occupancy, and steady-window acceptance."""
    geometry = ServingGeometry(**spec["geometry"])
    for workload in spec["workloads"]:
        for seed in arm["seeds"]:
            requests = workload_requests(workload, seed)
            for running in spec["running_per_engine"]:
                config = LoadConfig(
                    concurrency=running * geometry.data_parallel_size,
                    seed=seed,
                    gpu_count=geometry.total_gpus,
                    warmup_seconds=spec["warmup_seconds"],
                    measurement_seconds=spec["measurement_seconds"],
                    request_timeout_seconds=spec["request_timeout_seconds"],
                )
                load, counters = await asyncio.gather(
                    run_load(environment.server_url, environment.metrics_url, "mochi", requests, config),
                    steady_counters(environment, config, speculative=arm["draft"] is not None),
                )
                occupancy = load["server"]["gauges"]["vllm:num_requests_running"]
                observed = None if occupancy is None else occupancy["rank_sample_distribution"]["p50"]
                engine_samples: dict[tuple, list[float]] = {}
                for sample in load["metrics"]:
                    for row in sample["series"]:
                        if row["name"] == "vllm:num_requests_running":
                            engine_samples.setdefault(tuple(sorted(row["labels"].items())), []).append(row["value"])
                engine_medians = [
                    {"labels": dict(labels), "median": median(values)} for labels, values in engine_samples.items()
                ]
                achieved = (
                    len(engine_medians) == geometry.data_parallel_size
                    and observed is not None
                    and all(abs(row["median"] - running) <= max(0.5, 0.1 * running) for row in engine_medians)
                )
                name = f"{arm['name']}/{workload['name']}-seed{seed}-running{running}.json"
                write_json(
                    spec["output"],
                    name,
                    {
                        "source_revision": spec["source_revision"],
                        "arm": arm,
                        "workload": workload,
                        "requested_running_per_engine": running,
                        "observed_median_running_per_engine": observed,
                        "observed_engine_medians": engine_medians,
                        "occupancy_achieved": achieved,
                        "load": load,
                        "counter_window": counters,
                    },
                )
                logger.info("Measured %s: occupancy=%s valid=%s", name, observed, load["valid"])
                if not load["valid"] or not achieved or load["summary"]["request_decode_tokens_per_second"] is None:
                    raise ValueError(f"Invalid load or unachieved occupancy; retain {name} and stop")


async def qualify_arm(environment: VllmEnvironment, spec: dict, arm: dict) -> None:
    root = StoragePath(spec["output"])
    if arm["name"] == spec["arms"][0]["name"]:
        if arm["draft"] is not None:
            raise ValueError("The first arm must establish target-only numerical noise")
        requests = workload_requests(spec["workloads"][0], arm["seeds"][0])[: spec["probe_requests"]]
        probes = await probe_inventory(environment.server_url, requests)
        reference = await prefix_scores(environment.server_url, probes, 1)
        repeat = await prefix_scores(environment.server_url, probes, 16)
        noise = score_difference(reference, repeat)
        autoregressive = await decode_probes(environment.server_url, probes, spec["correctness_seeds"])
        write_json(spec["output"], f"{arm['name']}/{AUTOREGRESSIVE_ARTIFACT}", autoregressive)
        teacher = await prefix_scores(environment.server_url, list(autoregressive["scores"].values()), 16)
        decode_noise = score_difference(autoregressive["scores"], teacher)
        # Accepted design: predeclare the numerical bound from baseline repeats,
        # before observing any candidate scores.
        calibration = {
            "probes": probes,
            "reference": reference,
            "noise": noise,
            "max_logprob_error": max(1e-4, 2 * noise["max_absolute_logprob_error"]),
            "max_top1_mismatch_fraction": noise["top1_mismatch_fraction"] + 1 / len(probes),
            "max_route_mismatch_fraction": noise["route_mismatch_fraction"] + 1 / len(probes),
            "decode_noise": decode_noise,
            "decode_bounds": {
                "max_absolute_logprob_error": max(1e-4, 2 * decode_noise["max_absolute_logprob_error"]),
                "max_top1_logprob_error": max(1e-4, 2 * decode_noise["max_top1_logprob_error"]),
                "top1_mismatch_fraction": decode_noise["top1_mismatch_fraction"] + 1 / len(teacher),
                "route_mismatch_fraction": decode_noise["route_mismatch_fraction"] + 1 / len(teacher),
            },
        }
        calibration["sampling"] = sampling_calibration(
            sampling_by_prompt(autoregressive["trajectories"]), calibration["decode_bounds"]
        )
        write_json(spec["output"], "correctness-calibration.json", calibration)
        return
    calibration = json.loads((root / "correctness-calibration.json").read_text())
    scores = await prefix_scores(environment.server_url, calibration["probes"], 16)
    difference = score_difference(calibration["reference"], scores)
    passed = (
        difference["max_absolute_logprob_error"] <= calibration["max_logprob_error"]
        and difference["top1_mismatch_fraction"] <= calibration["max_top1_mismatch_fraction"]
        and difference["route_mismatch_fraction"] <= calibration["max_route_mismatch_fraction"]
    )
    write_json(
        spec["output"], f"{arm['name']}/correctness.json", {"passed": passed, "difference": difference, "scores": scores}
    )
    if not passed:
        raise ValueError("Fixed-prefix correctness exceeded the predeclared baseline-noise bound")
    autoregressive = await decode_probes(environment.server_url, calibration["probes"], spec["correctness_seeds"])
    write_json(spec["output"], f"{arm['name']}/{AUTOREGRESSIVE_ARTIFACT}", autoregressive)
    sampling = sampling_difference(sampling_by_prompt(autoregressive["trajectories"]), calibration["sampling"])
    write_json(spec["output"], f"{arm['name']}/sampling-correctness.json", sampling)
    if not sampling["passed"]:
        raise ValueError("Sampled logprob/top-one mass shifted beyond predeclared target-only seed variation")
    if arm["name"] != spec["arms"][-1]["name"]:
        return
    for candidate in spec["arms"][1:]:
        recorded = json.loads((root / candidate["name"] / AUTOREGRESSIVE_ARTIFACT).read_text())
        teacher = await prefix_scores(environment.server_url, list(recorded["scores"].values()), 16)
        difference = score_difference(recorded["scores"], teacher)
        passed = all(difference[name] <= bound for name, bound in calibration["decode_bounds"].items())
        write_json(
            spec["output"],
            f"{candidate['name']}/autoregressive-correctness.json",
            {"passed": passed, "difference": difference, "teacher_scores": teacher},
        )
        if not passed:
            raise ValueError(f"Autoregressive logprobs/routes exceeded baseline noise for {candidate['name']}")


def run(spec: dict) -> None:
    """Run paired arms serially on one allocation, preserving identical serving controls."""
    info = get_job_info()
    if info is None:
        raise RuntimeError("Run Mochi paired serving evaluation inside Iris")
    if spec["arms"][0]["draft"] is not None or spec["arms"][-1]["draft"] is not None:
        raise ValueError("Target-only arms must bracket candidates to qualify their autoregressive outputs")
    if len({arm["name"] for arm in spec["arms"]}) != len(spec["arms"]):
        raise ValueError("Each arm requires a distinct serving coordinator name")
    if not spec["output"].startswith("s3://marin-us-east-08a/"):
        raise ValueError("Mochi evaluation artifacts must remain in regional 08a S3")
    if len(set(spec["correctness_seeds"])) < 2 or len(set(spec["correctness_seeds"])) != len(spec["correctness_seeds"]):
        raise ValueError("Sampling calibration requires independent target-only seed repeats")
    geometry = ServingGeometry(**spec["geometry"])
    with tempfile.TemporaryDirectory(prefix="mochi-eagle3-evaluate-") as scratch:
        for arm in spec["arms"]:
            draft = None
            if arm["draft"] is not None:
                draft = Path(scratch) / arm["name"]
                io.download_directory(arm["draft"], str(draft))
            engine = mochi_engine_config(geometry, draft)
            with iris_vllm_launch(geometry, instance_name=f"vllm-{arm['name']}") as launch:
                with VllmBackend(engine).start(
                    mochi_model_spec(),
                    extra_args=launch.extra_cli_args,
                    subprocess_env={**launch.subprocess_env, "VLLM_USE_V2_MODEL_RUNNER": "0"},
                ) as environment:
                    if launch.is_leader:
                        with iris_vllm_followers(launch):
                            environment.wait_until_ready()
                            asyncio.run(qualify_arm(environment, spec, arm))
                            asyncio.run(benchmark(environment, spec, arm))
                    else:
                        wait_for_iris_vllm_shutdown(launch, environment.check_alive)
                    logs = environment.logs()
                    (StoragePath(spec["output"]) / arm["name"] / f"task-{info.task_index}-vllm.log").write_text(logs)
                    if not graph_capture_observed(logs):
                        raise ValueError(f"CUDA graph capture unconfirmed for arm {arm['name']} task {info.task_index}")
                if not launch.is_leader:
                    notify_iris_vllm_stopped(launch)


def main() -> None:
    run(worker_spec(__doc__))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
