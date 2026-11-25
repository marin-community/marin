# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP checks. Callers own deployment and the measurement window."""

import asyncio
import dataclasses
import gzip
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
from prometheus_client.parser import text_string_to_metric_families
from rigging.filesystem.storage_path import prefix_join

from marin.testing.inference.backend_parity import TokenDistribution, distribution_difference
from marin.testing.inference.goldens import (
    ErrorLimits,
    LookupStatus,
    ModelFixture,
    ReferenceLookup,
    canonical_bytes,
    history_digest,
    lookup_reference,
    read_bytes,
)

TOP_K = 50
MAX_CACHE_PREPARATION_REQUESTS = 3
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=900)
RANK_HEADER = "X-data-parallel-rank"
_CHECKPOINT_CONFIG_FILENAME = "config.json"
_CACHE_PROBE_SUFFIX_TOKENS = 17
_SHORT_HISTORY_LIMIT = 4096


def response_distribution(entries: dict) -> TokenDistribution:
    return TokenDistribution.from_logprob_map(
        {
            int(str(token).removeprefix("token_id:")): float(value["logprob"] if isinstance(value, dict) else value)
            for token, value in entries.items()
        }
    )


def response_top_distribution(entries: dict) -> TokenDistribution:
    """Exclude an appended prompt token below the requested top-k cutoff."""
    scores = response_distribution(entries)
    return TokenDistribution(scores.token_ids[:TOP_K], scores.logprobs[:TOP_K])


def mismatch_summary(rows: list[dict], limits: ErrorLimits) -> dict:
    values = np.asarray([row["max_probability_error"] for row in rows])
    if not len(values):
        return {"status": "pending", "reason": "No matching score positions were checked", "positions": 0}
    statistics = {
        "maximum": float(values.max()),
        "p99": float(np.quantile(values, 0.99)),
        "p90": float(np.quantile(values, 0.9)),
        "fraction_above_0075": float(np.mean(values > 0.075)),
    }
    return {
        "status": "passed" if all(value <= getattr(limits, name) for name, value in statistics.items()) else "failed",
        "positions": len(values),
        "statistics": statistics,
        "limits": limits.model_dump(),
        "worst": sorted(rows, key=lambda row: row["max_probability_error"], reverse=True)[:10],
    }


@dataclass
class EndpointClient:
    base_url: str
    model: str
    output: Path
    session: aiohttp.ClientSession = field(init=False)
    requests: list[dict] = field(default_factory=list, init=False)

    async def __aenter__(self) -> "EndpointClient":
        self.output.mkdir(parents=True, exist_ok=True)
        # Parsing and compressing a large prompt-score response can outlast the
        # server's keepalive timeout. Do not reuse an idle connection afterward.
        self.session = aiohttp.ClientSession(
            timeout=HTTP_TIMEOUT, connector=aiohttp.TCPConnector(limit=0, force_close=True)
        )
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.session.close()

    @property
    def root(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1")

    async def get(self, path: str) -> dict:
        async with self.session.get(f"{self.root}/{path}") as response:
            response.raise_for_status()
            return await response.json()

    async def reset_cache(self) -> None:
        async with self.session.post(f"{self.root}/reset_prefix_cache") as response:
            response.raise_for_status()

    async def metrics(self) -> str:
        async with self.session.get(f"{self.root}/metrics") as response:
            response.raise_for_status()
            return await response.text()

    async def complete(
        self,
        tokens: tuple[int, ...],
        *,
        label: str,
        count: int = 1,
        prompt_scores: bool = False,
        rank: int = 0,
    ) -> dict:
        body = {
            "model": self.model,
            "prompt": tokens,
            "add_special_tokens": False,
            "max_tokens": count,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": TOP_K,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        if prompt_scores:
            body["prompt_logprobs"] = TOP_K
        started = time.time()
        async with self.session.post(
            f"{self.root}/v1/completions",
            json=body,
            headers={RANK_HEADER: str(rank), "X-Request-Id": label},
        ) as response:
            raw = await response.read()
            # Keep failed HTTP responses too. Request IDs and histories are recorded separately.
            path = self.output / f"{len(self.requests):05d}-{label}.json.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(raw)
            self.requests.append(
                {
                    "label": label,
                    "history_sha256": history_digest(tokens),
                    "prompt_tokens": len(tokens),
                    "rank": rank,
                    "requested_tokens": count,
                    "prompt_scores": prompt_scores,
                    "http_status": response.status,
                    "response": path.name,
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "seconds": time.time() - started,
                }
            )
            response.raise_for_status()
        result = json.loads(raw)
        (choice,) = result["choices"]
        if choice.get("prompt_token_ids") != list(tokens):
            raise ValueError(f"The endpoint changed the exact input tokens: {label}")
        if len(choice["token_ids"]) != count:
            raise ValueError(f"The endpoint returned an incomplete continuation: {label}")
        for token, top in zip(choice["token_ids"], choice["logprobs"]["top_logprobs"], strict=True):
            distribution = response_distribution(top)
            if int(token) not in distribution.token_ids or int(token) != distribution.token_ids[0]:
                # Equal maximum scores may have a different tie order.
                scores = dict(zip(distribution.token_ids, distribution.logprobs, strict=True))
                if scores.get(int(token)) != max(scores.values()):
                    raise ValueError(f"The greedy token lacks its finite maximum score: {label}")
        return result


def next_distribution(response: dict) -> TokenDistribution:
    return response_top_distribution(response["choices"][0]["logprobs"]["top_logprobs"][0])


def cached_tokens(response: dict) -> int:
    return int(response["usage"]["prompt_tokens_details"]["cached_tokens"])


def metric_values(text: str, name: str) -> dict[str, float]:
    return {
        sample.labels["engine"]: float(sample.value)
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name and "engine" in sample.labels
    }


async def verify_deployment(client: EndpointClient, fixture: ModelFixture, runtime_source: str) -> dict:
    """Check server-reported identities instead of trusting a caller's labels."""
    information = await client.get("server_info?config_format=json")
    config = information["vllm_config"]
    model = config["model_config"]
    version = await client.get("version")
    identity = fixture.model
    expected = {
        "tokenizer": identity.tokenizer,
        "tokenizer_revision": identity.tokenizer_revision,
        "quantization": fixture.policy.quantization,
    }
    if not identity.weights.startswith(("s3://", "gs://")):
        expected["revision"] = identity.revision
    mismatches: dict = {
        name: {"actual": model.get(name), "expected": value}
        for name, value in expected.items()
        if model.get(name) != value
    }
    # RunAI replaces `model` with its local metadata directory. The runtime's
    # `model_weights` field preserves the source actually used by its loader.
    weights = model.get("model_weights") or model["model"]
    if weights != identity.weights:
        mismatches["weights"] = {"actual": weights, "expected": identity.weights}
    if runtime_source[:12] not in version["version"]:
        mismatches["runtime"] = {"actual": version["version"], "expected": runtime_source}
    if str(model["dtype"]).removeprefix("torch.") != fixture.policy.dtype:
        mismatches["dtype"] = {"actual": str(model["dtype"]), "expected": fixture.policy.dtype}
    hardware = information["system_env"]["nvidia_gpu_models"]
    if not any(device in hardware for device in fixture.policy.hardware):
        mismatches["hardware"] = {"actual": hardware, "expected": fixture.policy.hardware}
    if identity.weights.startswith(("s3://", "gs://")):
        manifest_bytes = await asyncio.to_thread(read_bytes, prefix_join(identity.weights, "export-manifest.json"))
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        if digest != identity.revision:
            mismatches["export_manifest"] = {"actual": digest, "expected": identity.revision}
        config_uri = prefix_join(identity.weights, _CHECKPOINT_CONFIG_FILENAME)
    elif Path(identity.weights).is_absolute():
        config_uri = str(Path(identity.weights) / _CHECKPOINT_CONFIG_FILENAME)
    else:
        config_uri = (
            f"https://huggingface.co/{identity.weights}/resolve/{identity.revision}/{_CHECKPOINT_CONFIG_FILENAME}"
        )
    config_bytes = await asyncio.to_thread(read_bytes, config_uri)
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    if config_digest != identity.config_sha256:
        mismatches["model_config"] = {"actual": config_digest, "expected": identity.config_sha256}
    graph_mode = config["compilation_config"].get("cudagraph_mode")
    if graph_mode in (None, 0, "NONE", "CUDAGraphMode.NONE"):
        mismatches["cuda_graphs"] = {"actual": graph_mode, "expected": "Enabled"}
    if config["cache_config"].get("enable_prefix_caching") is not True:
        mismatches["prefix_caching"] = {"actual": config["cache_config"].get("enable_prefix_caching"), "expected": True}
    cache_dtype = config["cache_config"].get("cache_dtype")
    if cache_dtype not in fixture.policy.kv_cache_dtypes:
        mismatches["kv_cache_dtype"] = {"actual": cache_dtype, "expected": fixture.policy.kv_cache_dtypes}
    eos = model.get("override_generation_config", {}).get("eos_token_id", json.loads(config_bytes).get("eos_token_id"))
    if identity.eos_token_id not in (eos if isinstance(eos, list) else [eos]):
        mismatches["eos_token_id"] = {"actual": eos, "expected": identity.eos_token_id}
    return {
        "status": "failed" if mismatches else "passed",
        "mismatches": mismatches,
        "version": version,
        "hardware": hardware,
        "config_sha256": config_digest,
        "model_config": {
            key: value
            for key, value in model.items()
            if key
            in {
                "model",
                "model_weights",
                "revision",
                "tokenizer",
                "tokenizer_revision",
                "dtype",
                "max_model_len",
                "quantization",
                "generation_config",
                "override_generation_config",
                "trust_remote_code",
            }
        },
        "scheduler_config": config["scheduler_config"],
        "cache_config": config["cache_config"],
        "parallel_config": config["parallel_config"],
        "compilation_config": config["compilation_config"],
    }


async def check_reference(client: EndpointClient, references: ReferenceLookup, fixture: ModelFixture) -> dict:
    expected = {(row.case_id, row.position): row for row in references.scores}
    groups: dict[str, list[dict]] = {name: [] for name in fixture.policy.reference}
    observations = []
    completed = []
    failures = []
    for case in references.cases:
        # A context includes its sampled output token; score the final prediction separately
        # when an existing 262K reference fills the entire configured context.
        try:
            response = await client.complete(case.tokens, label=f"reference-{case.id}", prompt_scores=True)
            (choice,) = response["choices"]
            positions = [*choice["prompt_logprobs"][1:], choice["logprobs"]["top_logprobs"][0]]
            if len(positions) != len(case.tokens):
                raise ValueError(f"Missing prompt log probabilities: {case.id}")
            for position in case.positions:
                row = expected[(case.id, position)]
                actual = response_top_distribution(positions[position])
                reference = TokenDistribution(row.token_ids, row.logprobs)
                mismatch = dataclasses.asdict(distribution_difference(reference, actual)) | {
                    "case": case.id,
                    "position": position,
                }
                groups[case.group].append(mismatch)
                observations.append(mismatch | {"actual": dataclasses.asdict(actual)})
        except Exception as error:
            failures.append({"case": case.id, "error": f"{type(error).__name__}: {error}"})
            continue
        completed.append(case.id)
    summaries = {name: mismatch_summary(rows, fixture.policy.reference[name]) for name, rows in groups.items()}
    return {
        "status": (
            "passed" if not failures and all(value["status"] == "passed" for value in summaries.values()) else "failed"
        ),
        "groups": summaries,
        "observations": observations,
        "completed_cases": completed,
        "total_cases": len(references.cases),
        "failures": failures,
    }


async def check_batch(client: EndpointClient, references: ReferenceLookup, fixture: ModelFixture) -> dict:
    cases = sorted(
        (case for case in references.cases if len(case.tokens) <= _SHORT_HISTORY_LIMIT),
        key=lambda case: len(case.tokens),
    )
    selected = cases[-fixture.policy.batch_size :]
    if len(selected) < fixture.policy.batch_size:
        raise ValueError("The suite has too few distinct cases for the engine batch")
    alone = []
    for index, case in enumerate(selected):
        await client.reset_cache()
        alone.append(await client.complete(case.tokens, label=f"alone-{index}", count=fixture.policy.decode_tokens))
    await client.reset_cache()
    before = await client.metrics()
    tasks = [
        asyncio.create_task(client.complete(case.tokens, label=f"batch-{index}", count=fixture.policy.decode_tokens))
        for index, case in enumerate(selected)
    ]
    running_peak = 0.0
    evidence = []
    while not all(task.done() for task in tasks):
        metrics = await client.metrics()
        running = metric_values(metrics, "vllm:num_requests_running").get("0", 0)
        running_peak = max(running_peak, running)
        if running >= 2:
            evidence.append({"unix": time.time(), "engine": 0, "running": running})
        await asyncio.sleep(0.05)
    batched = await asyncio.gather(*tasks)
    after = await client.metrics()
    (client.output / "batch-before.prom").write_text(before)
    (client.output / "batch-after.prom").write_text(after)
    rows = [
        dataclasses.asdict(distribution_difference(next_distribution(a), next_distribution(b))) | {"case": case.id}
        for case, a, b in zip(selected, alone, batched, strict=True)
    ]
    summary = mismatch_summary(rows, fixture.policy.batch)
    if running_peak < 2:
        summary |= {"status": "failed", "reason": "No simultaneous running requests were observed on engine 0"}
    return summary | {"engine": 0, "running_peak": running_peak, "batch_evidence": evidence, "observations": rows}


async def check_settled_branch(
    client: EndpointClient,
    branch: tuple[int, ...],
    alternate: tuple[int, ...],
    boundary: int,
    block_size: int,
    *,
    label: str,
) -> dict:
    """Compare a branch before and after a sibling, with the complete usable prefix cached."""
    expected = (len(branch) - 1) // block_size * block_size
    shared_checkpoint = boundary // block_size * block_size
    prepared_hits = []
    result: dict = {
        "boundary": boundary,
        "divergence_tokens": boundary,
        "status": "failed",
        "expected_cached_tokens": expected,
        "required_alternate_checkpoint": shared_checkpoint,
        "shared_seed_tokens": shared_checkpoint + 1,
        "full_seed_tokens": expected + 1,
    }
    # Retention preserves existing snapshots. An aligned cold prefill need not
    # create the lower replay point, so materialize the shared state explicitly.
    await client.complete(branch[: shared_checkpoint + 1], label=f"{label}-shared-seed")
    await client.complete(branch[: expected + 1], label=f"{label}-full-seed")
    for attempt in range(MAX_CACHE_PREPARATION_REQUESTS):
        before = await client.complete(branch, label=f"{label}-prepare-{attempt}")
        prepared_hits.append(cached_tokens(before))
        if cached_tokens(before) == expected:
            break
    result["preparation_hits"] = prepared_hits
    if prepared_hits[-1] != expected or expected <= boundary or shared_checkpoint <= 0:
        return result
    sibling = await client.complete(alternate, label=f"{label}-settled-alternate")
    result["alternate_cached_tokens"] = cached_tokens(sibling)
    after = await client.complete(branch, label=f"{label}-settled-repeat")
    result["repeat_cached_tokens"] = cached_tokens(after)
    if cached_tokens(sibling) == shared_checkpoint < expected and cached_tokens(after) == expected:
        result["status"] = "passed"
        result["comparison"] = dataclasses.asdict(
            distribution_difference(next_distribution(before), next_distribution(after))
        ) | {"boundary": boundary, "case": "settled-branch"}
    return result


async def check_cache(
    client: EndpointClient,
    references: ReferenceLookup,
    fixture: ModelFixture,
    cache_config: dict,
    max_model_len: int,
) -> dict:
    # The scoped TP1 fixtures use attention blocks and optional Mamba blocks.
    block_size = math.lcm(cache_config["block_size"], cache_config.get("mamba_block_size") or 1)
    retention = cache_config["prefix_cache_retention_interval"]
    source = max(references.cases, key=lambda case: len(case.tokens)).tokens
    if len(set(source)) < 3:
        raise ValueError("Cache probes require varied token histories")
    same_split, different_split = [], []
    evidence = []
    prefill_reuse = []
    settled_branches = []
    decode_reuse = []
    replayed_decode_reuse = []
    failures = []
    for length in fixture.policy.boundary_lengths:
        observed = {"boundary": length}
        evidence.append(observed)
        try:
            prefix = source[:length]
            branch = prefix + source[length : length + _CACHE_PROBE_SUFFIX_TOKENS]
            if len(branch) <= len(prefix):
                raise ValueError(f"The prompt suite cannot reach cache boundary {length}")
            await client.reset_cache()
            fresh = await client.complete(branch, label=f"cache-{length}-fresh")
            observed["fresh_cached_tokens"] = cached_tokens(fresh)
            if cached_tokens(fresh) != 0:
                raise ValueError("Cache reset did not produce a fresh recomputation")
            await client.reset_cache()
            checkpoint = length // block_size * block_size
            seed = source[: checkpoint + 1]
            await client.complete(seed, label=f"cache-{length}-seed")
            warm = await client.complete(branch, label=f"cache-{length}-warm")
            alternate = prefix + source[length + 31 : length + 48]
            await client.complete(alternate, label=f"cache-{length}-branch")
            repeated = await client.complete(branch, label=f"cache-{length}-repeat")
            hits = [cached_tokens(warm), cached_tokens(repeated)]
            observed["warm_cached_tokens"], observed["repeat_cached_tokens"] = hits
            prefill_reuse.append(
                {
                    "boundary": length,
                    "status": "passed" if min(hits) >= checkpoint > 0 else "failed",
                    "cached_tokens": hits,
                    "seed_tokens": len(seed),
                    "required_checkpoint": checkpoint,
                }
            )
            if min(hits) > 0:
                different_split.append(
                    dataclasses.asdict(distribution_difference(next_distribution(fresh), next_distribution(warm)))
                    | {"boundary": length}
                )
                # A new hybrid-cache junction can split the suffix even when the joint hit is unchanged.
                different_split.append(
                    dataclasses.asdict(distribution_difference(next_distribution(warm), next_distribution(repeated)))
                    | {"boundary": length, "case": "seed-and-branch"}
                )

            other_token = next(token for token in source if token != branch[length])
            isolated_alternate = (*prefix, other_token, *branch[length + 1 :])
            settled = await check_settled_branch(
                client, branch, isolated_alternate, length, block_size, label=f"cache-{length}"
            )
            settled_branches.append(settled)
            if settled["status"] == "passed":
                same_split.append(settled["comparison"])

            # Choose a reachable periodic checkpoint instead of assuming any decode
            # length creates one. Zero retains prompt replay points, not these points.
            decode_prefix = prefix
            interval = block_size if retention is None else retention
            checkpoint = (len(prefix) // interval + 1) * interval if interval else 0
            direct_required = checkpoint > 0 and checkpoint + 5 <= max_model_len
            count = fixture.policy.cache_decode_tokens
            if direct_required and checkpoint > len(prefix) + count - 1:
                decode_prefix = source[: checkpoint - 1]
                count = max(count, checkpoint - len(decode_prefix) + 1)
            count = min(count, max_model_len - len(decode_prefix) - 4)
            await client.reset_cache()
            decoded = await client.complete(decode_prefix, label=f"cache-{length}-decode-seed", count=count)
            history = decode_prefix + tuple(decoded["choices"][0]["token_ids"]) + source[length : length + 3]
            reused = await client.complete(history, label=f"cache-{length}-decode-reuse")
            observed["decode_cached_tokens"] = cached_tokens(reused)
            observed["decode_prompt_tokens"] = len(decode_prefix)
            observed["decode_generated_tokens"] = count
            decode_reuse.append(
                {
                    "boundary": length,
                    "status": (
                        ("passed" if cached_tokens(reused) >= checkpoint else "failed")
                        if direct_required
                        else "not_required"
                    ),
                    "cached_tokens": cached_tokens(reused),
                    "prompt_tokens": len(decode_prefix),
                    "required_checkpoint": checkpoint if direct_required else None,
                    "reason": (
                        "Periodic checkpoint in generated history"
                        if direct_required
                        else "No periodic checkpoint reachable in this retention mode and context"
                    ),
                }
            )
            # Fork after generated tokens, at a materialized checkpoint before the
            # full replay boundary. Reusing only the original prompt is insufficient.
            divergence = (len(history) - 1) // block_size * block_size - block_size
            if divergence <= len(decode_prefix):
                raise ValueError("The generated history cannot reach an intermediate branch checkpoint")
            other_token = next(token for token in source if token != history[divergence])
            replay_alternate = (*history[:divergence], other_token, *history[divergence + 1 :])
            replayed = await check_settled_branch(
                client, history, replay_alternate, divergence, block_size, label=f"cache-{length}-replayed-decode"
            )
            replayed["requested_boundary"] = length
            replayed_decode_reuse.append(replayed)
            if replayed["status"] == "passed":
                same_split.append(replayed["comparison"] | {"case": "replayed-decode-branch"})
            await client.reset_cache()
            recomputed = await client.complete(history, label=f"cache-{length}-decode-fresh")
            different_split.append(
                dataclasses.asdict(distribution_difference(next_distribution(reused), next_distribution(recomputed)))
                | {"boundary": length, "seed": "decode"}
            )
        except Exception as error:
            failures.append({"boundary": length, "error": f"{type(error).__name__}: {error}"})
            continue
    # An exact repeat of a warmed history supplies a matched-split control even if prefix
    # seeding and branching fill different numbers of blocks on this model.
    try:
        branch = source[: max(fixture.policy.boundary_lengths) + _CACHE_PROBE_SUFFIX_TOKENS]
        await client.complete(branch, label="cache-matched-seed")
        a = await client.complete(branch, label="cache-matched-first")
        b = await client.complete(branch, label="cache-matched-second")
        matched = cached_tokens(a) == cached_tokens(b) and cached_tokens(a) > 0
        matched_repeat = {
            "status": "passed" if matched else "failed",
            "cached_tokens": [cached_tokens(a), cached_tokens(b)],
        }
        if matched:
            same_split.append(
                dataclasses.asdict(distribution_difference(next_distribution(a), next_distribution(b)))
                | {"case": "matched-repeat"}
            )
    except Exception as error:
        matched_repeat = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    summaries = {
        "same_split": mismatch_summary(same_split, fixture.policy.same_split),
        "different_split": mismatch_summary(different_split, fixture.policy.different_split),
    }
    coverage = {
        name: {
            "status": (
                "passed"
                if len(rows) == len(fixture.policy.boundary_lengths)
                and all(
                    row["status"] == "passed" or (name == "decode_reuse" and row["status"] == "not_required")
                    for row in rows
                )
                else "failed"
            ),
            "observations": rows,
        }
        for name, rows in (
            ("prefill_reuse", prefill_reuse),
            ("settled_branches", settled_branches),
            ("decode_reuse", decode_reuse),
            ("replayed_decode_reuse", replayed_decode_reuse),
        )
    }
    if coverage["decode_reuse"]["status"] == "passed" and all(row["status"] == "not_required" for row in decode_reuse):
        coverage["decode_reuse"]["status"] = "not_required"
    return {
        "status": (
            "passed"
            if not failures
            and all(row["status"] == "passed" for row in (*summaries.values(), matched_repeat))
            and all(row["status"] in ("passed", "not_required") for row in coverage.values())
            else "failed"
        ),
        **summaries,
        **coverage,
        "matched_repeat": matched_repeat,
        "retention_interval": retention,
        "cache_alignment": block_size,
        "failures": failures,
        "cache_evidence": evidence,
        "observations": {"same_split": same_split, "different_split": different_split},
    }


async def check_decode(client: EndpointClient, references: ReferenceLookup, fixture: ModelFixture) -> dict:
    cases = sorted(
        (case for case in references.cases if len(case.tokens) < _SHORT_HISTORY_LIMIT), key=lambda case: len(case.tokens)
    )
    selected = [cases[0], cases[-1]]
    differences = []
    for index, case in enumerate(selected):
        await client.reset_cache()
        decoded = await client.complete(case.tokens, label=f"decode-{index}", count=fixture.policy.decode_tokens)
        choice = decoded["choices"][0]
        history = case.tokens + tuple(choice["token_ids"])
        await client.reset_cache()
        rescored = await client.complete(history, label=f"decode-rescore-{index}", prompt_scores=True)
        prompt_scores = rescored["choices"][0]["prompt_logprobs"]
        for position, (token, top) in enumerate(
            zip(choice["token_ids"], choice["logprobs"]["top_logprobs"], strict=True)
        ):
            decode_distribution = response_distribution(top)
            prefill_distribution = response_distribution(prompt_scores[len(case.tokens) + position])
            a = dict(zip(decode_distribution.token_ids, decode_distribution.logprobs, strict=True))[token]
            b = dict(zip(prefill_distribution.token_ids, prefill_distribution.logprobs, strict=True))[token]
            differences.append(
                {
                    "case": case.id,
                    "position": position,
                    "token_id": token,
                    "decode_logprob": a,
                    "prefill_logprob": b,
                    "absolute_logprob_error": abs(a - b),
                    "max_probability_error": abs(math.exp(a) - math.exp(b)),
                }
            )
    return mismatch_summary(differences, fixture.policy.decode) | {"observations": differences}


async def check_endpoint(
    base_url: str,
    model: str,
    fixture: ModelFixture,
    runtime_source: str,
    output: Path,
) -> dict:
    """Run the shared suite and retain evidence, including failures and missing coverage."""
    references = lookup_reference(fixture)
    report: dict = {
        "schema_version": 1,
        "status": "pending",
        "model": fixture.model.model_dump(),
        "runtime_source": runtime_source,
        "manifest": fixture.manifest.model_dump(),
        "policy": fixture.policy.model_dump(),
        "policy_sha256": hashlib.sha256(canonical_bytes(fixture.policy.model_dump(mode="json"))).hexdigest(),
        "started_unix": time.time(),
        "reference_lookup": {"status": references.status, "reason": references.reason},
        "checks": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_bytes(canonical_bytes(report))

    if references.status != LookupStatus.FOUND:
        if references.status == LookupStatus.MISMATCHED:
            report["status"] = "failed"
            report["checks"]["reference_identity"] = {"status": "failed", "reason": references.reason}
        save()
        return json.loads(output.read_bytes())
    async with EndpointClient(base_url, model, output.parent / f"{output.stem}-requests") as client:
        stages: tuple[tuple[str, Callable[[], Awaitable[dict]]], ...] = (
            ("deployment", lambda: verify_deployment(client, fixture, runtime_source)),
            ("reference", lambda: check_reference(client, references, fixture)),
            ("batch", lambda: check_batch(client, references, fixture)),
            (
                "cache",
                lambda: check_cache(
                    client,
                    references,
                    fixture,
                    report["checks"]["deployment"]["cache_config"],
                    report["checks"]["deployment"]["model_config"]["max_model_len"],
                ),
            ),
            ("decode", lambda: check_decode(client, references, fixture)),
        )
        for name, stage in stages:
            started = time.time()
            try:
                result = await stage()
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            report["checks"][name] = result | {"seconds": time.time() - started}
            report["requests"] = client.requests
            save()
            if name == "deployment" and result["status"] != "passed":
                break
    report["status"] = (
        "passed"
        if len(report["checks"]) == 5 and all(row["status"] == "passed" for row in report["checks"].values())
        else "failed"
    )
    report["finished_unix"] = time.time()
    save()
    return json.loads(output.read_bytes())
