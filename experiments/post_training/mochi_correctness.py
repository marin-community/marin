# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen Mochi logprob, routed-expert and sampling checks from the EAGLE3 gate."""

import asyncio
import math
from statistics import mean, variance

import httpx
import numpy as np
from marin.inference.load_test import LoadRequest
from skyrl_train.trajectory_runners.routed_experts import decode_routed_experts, normalize_routed_experts

STOP_TOKEN_IDS = (128001, 128009)
PROBE_POSITIONS = (0, 1, 3, 7, 15, 31, 63)
DECODE_POSITIONS = (*PROBE_POSITIONS, 127, 255)
AUTOREGRESSIVE_ARTIFACT = "autoregressive.json"
CALIBRATION_STANDARD_ERRORS = 3
ROUTED_LAYERS = 48
ROUTED_EXPERTS_PER_TOKEN = 8
ROUTED_EXPERT_COUNT = 384
MAX_ROUTE_CHOICE_NOISE = 0.05


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


async def prefix_scores(
    base_url: str, probes: list[dict], concurrency: int, *, data_parallel_rank: int | None = None
) -> dict[str, dict]:
    """Score the same known next tokens without retokenizing generated text."""
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=1800) as client:

        async def score(probe: dict) -> tuple[str, dict]:
            async with semaphore:
                prompt = probe["prompt"][:-1]
                response = await client.post(
                    f"{base_url}/completions",
                    headers=None if data_parallel_rank is None else {"X-data-parallel-rank": str(data_parallel_rank)},
                    json={
                        "model": "mochi",
                        "prompt": prompt,
                        "max_tokens": 1,
                        "temperature": 0,
                        "logprobs": 1,
                        "logprob_token_ids": [probe["token"]],
                        "ignore_eos": True,
                        "return_token_ids": True,
                        "return_tokens_as_token_ids": True,
                        "routed_experts_prompt_start": len(prompt) - 1,
                    },
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                entries = choice["logprobs"]["top_logprobs"][0]
                top = choice["token_ids"][0]
                routes = decode_routed_experts(choice["routed_experts"], 1)
                return probe["identity"], {
                    "logprob": entries[f"token_id:{probe['token']}"],
                    "top1": top,
                    "top1_logprob": entries[f"token_id:{top}"],
                    "routes": routes[0].tolist(),
                }

        return dict(await asyncio.gather(*(score(probe) for probe in probes)))


def score_difference(reference: dict[str, dict], candidate: dict[str, dict]) -> dict:
    """Compare identical-prefix probabilities, expert choices, and ordered route diagnostics."""
    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("Correctness score inventories differ or are empty")
    errors = [abs(row["logprob"] - candidate[key]["logprob"]) for key, row in reference.items()]
    top_errors = [abs(row["top1_logprob"] - candidate[key]["top1_logprob"]) for key, row in reference.items()]
    reference_routes = np.asarray([row["routes"] for row in reference.values()])
    candidate_routes = np.asarray([candidate[key]["routes"] for key in reference])
    expected_shape = (len(errors), ROUTED_LAYERS, ROUTED_EXPERTS_PER_TOKEN)
    if reference_routes.shape != expected_shape or candidate_routes.shape != expected_shape:
        raise ValueError("Mochi correctness scores must retain all 48 layers and eight expert slots")
    for routes in (reference_routes, candidate_routes):
        if (
            not np.issubdtype(routes.dtype, np.integer)
            or routes.min() < 0
            or routes.max() >= ROUTED_EXPERT_COUNT
            or np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0)
        ):
            raise ValueError("Mochi routes must contain eight distinct valid expert IDs per layer")
    route_mismatch = reference_routes != candidate_routes
    # Replay gathers weights by expert ID, so slot permutations preserve the chosen set.
    missing_choices = ~(reference_routes[..., :, None] == candidate_routes[..., None, :]).any(axis=-1)
    choices_by_prompt: dict[str, list[float]] = {}
    for key, rate in zip(reference, missing_choices.mean(axis=(1, 2)), strict=True):
        # Both @position and @0/seed/temperature/decode suffixes follow the request identity.
        choices_by_prompt.setdefault(key.split("@", 1)[0], []).append(float(rate))
    choice_means = {key: mean(values) for key, values in choices_by_prompt.items()}
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
        "route_slot_mismatch_fraction": float(route_mismatch.mean()),
        "route_slot_mismatch_by_layer": route_mismatch.mean(axis=(0, 2)).tolist(),
        "route_choice_mismatch_fraction": float(missing_choices.mean()),
        "route_choice_mismatch_by_prompt": choice_means,
        "mean_prompt_route_choice_mismatch_fraction": mean(choice_means.values()),
    }


def route_choice_bound(difference: dict) -> float:
    """Calibrate expert-choice variation using independent prompts as the sampling unit."""
    values = list(difference["route_choice_mismatch_by_prompt"].values())
    if len(values) < 2:
        raise ValueError("Route-choice calibration requires at least two independent prompt groups")
    standard_error = math.sqrt(variance(values) / len(values))
    one_slot = 1 / (difference["prefixes"] * ROUTED_LAYERS * ROUTED_EXPERTS_PER_TOKEN)
    return mean(values) + CALIBRATION_STANDARD_ERRORS * standard_error + one_slot


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


def sampling_calibration(grouped: dict[str, list[dict]]) -> dict:
    """Predeclare three-standard-error bounds from paired target-only seed repeats."""
    calibrated = {}
    for metric in ("mean_logprob", "mean_top1_probability"):
        values = [[row[metric] for row in rows] for rows in grouped.values()]
        standard_error = math.sqrt(sum(2 * variance(group) / len(group) for group in values)) / len(values)
        calibrated[metric] = {
            "reference": mean(mean(group) for group in values),
            "maximum_mean_shift": CALIBRATION_STANDARD_ERRORS * standard_error,
            "standard_error": standard_error,
        }
    return {"by_prompt": grouped, "metrics": calibrated, "standard_errors": CALIBRATION_STANDARD_ERRORS}


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
