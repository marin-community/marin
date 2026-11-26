# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four numerical comparisons against an endpoint supplied by the caller."""

import asyncio
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from google.protobuf.json_format import MessageToDict, ParseDict
from iris.cluster.types import ResourceSpec
from iris.rpc import job_pb2
from prometheus_client.parser import text_string_to_metric_families
from pydantic import ConfigDict, Field, JsonValue, field_serializer, field_validator
from rigging.filesystem.storage_path import prefix_join

from marin.evaluation.model_config import ServeBackend
from marin.inference.config import InferenceModelConfig
from marin.testing.inference.backend_parity import TokenDistribution, distribution_difference
from marin.testing.inference.endpoint import VllmEndpoint, response_distribution, response_top_distribution
from marin.testing.inference.goldens import (
    ArtifactRef,
    ErrorLimits,
    LookupStatus,
    ModelIdentity,
    Record,
    ReferenceLookup,
    ReferenceManifest,
    canonical_bytes,
    lookup_reference,
    read_bytes,
    verified_bytes,
)
from marin.testing.inference.results import CheckReport, CheckResult, CheckStatus, ComparisonResult, CoverageResult

MAX_CACHE_PREPARATION_REQUESTS = 3
_CHECKPOINT_CONFIG_FILENAME = "config.json"
_CACHE_PROBE_SUFFIX_TOKENS = 17
_SHORT_HISTORY_LIMIT = 4096


class TestSpec(Record):
    """A shared reference plus caller-owned placement, serving settings and limits.

    ``reference`` pins one immutable manifest across execution variants.
    ``resources`` is the Iris request; reading or running a spec allocates nothing.
    ``engine_kwargs`` uses the existing vLLM model configuration, including dtype.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: Literal[1] = 1
    id: str
    reference: ArtifactRef
    resources: ResourceSpec
    replicas: int = Field(default=1, ge=1)
    serving_engine: ServeBackend
    engine_kwargs: dict[str, JsonValue]
    hardware: tuple[str, ...]
    kv_cache_dtypes: tuple[str, ...] = ("auto", "bfloat16")
    tolerances: dict[str, ErrorLimits]
    tolerance_version: str
    provisional: bool
    rationale: str
    evidence: tuple[str, ...]
    batch_size: int = Field(default=4, ge=2)
    decode_tokens: int = Field(default=32, ge=2)
    cache_decode_tokens: int = Field(default=257, ge=2)
    boundary_lengths: tuple[int, ...] = (127, 128, 129, 255, 256, 257, 2047, 2048, 2049)

    @field_validator("resources", mode="before")
    @classmethod
    def parse_resources(cls, value: Any) -> ResourceSpec:
        if isinstance(value, ResourceSpec):
            return value
        values = dict(value)
        device = values.pop("device", None)
        return ResourceSpec(**values, device=ParseDict(device, job_pb2.DeviceConfig()) if device is not None else None)

    @field_serializer("resources")
    def resource_fields(self, value: ResourceSpec) -> dict:
        return {
            "cpu": value.cpu,
            "memory": value.memory,
            "disk": value.disk,
            "device": (
                MessageToDict(value.device, preserving_proto_field_name=True) if value.device is not None else None
            ),
        }

    @property
    def reference_limits(self) -> dict[str, ErrorLimits]:
        return {
            name.removeprefix("reference."): value
            for name, value in self.tolerances.items()
            if name.startswith("reference.")
        }

    def make_model_config(self) -> InferenceModelConfig:
        """Convert the reference identity to Marin's existing serving model type."""
        manifest = ReferenceManifest.model_validate_json(verified_bytes(self.reference))
        identity = manifest.model
        arguments: dict[str, Any] = dict(self.engine_kwargs)
        arguments.update(
            tokenizer=identity.tokenizer,
            tokenizer_revision=identity.tokenizer_revision,
            override_generation_config=dict(arguments.get("override_generation_config", {}))
            | {"eos_token_id": identity.eos_token_id},
        )
        if not identity.weights.startswith(("s3://", "gs://")):
            arguments["revision"] = identity.revision
        return InferenceModelConfig(name=self.id, path=identity.weights, engine_kwargs=arguments)

    async def run(self, endpoint: VllmEndpoint) -> CheckReport:
        """Run the four comparisons without deploying or saving anything."""
        return CheckReport(
            reference=await check_reference(endpoint, self),
            batch=await check_batch(endpoint, self),
            cache=await check_cache(endpoint, self),
            decode=await check_decode(endpoint, self),
        )


def read_test_spec(path: str) -> TestSpec:
    return TestSpec.model_validate_json(read_bytes(path))


async def check_reference(endpoint: VllmEndpoint, spec: TestSpec) -> CheckResult:
    """Compare serving scores with the pinned reference at the same positions."""
    return await _run_check(endpoint, spec, "reference", _check_reference)


async def check_batch(endpoint: VllmEndpoint, spec: TestSpec) -> CheckResult:
    """Compare isolated requests with requests observed running together."""
    return await _run_check(endpoint, spec, "batch", _check_batch)


async def check_cache(endpoint: VllmEndpoint, spec: TestSpec) -> CheckResult:
    """Compare fresh/reused prefill and exact prepared branch controls."""
    return await _run_check(endpoint, spec, "cache", _check_cache)


async def check_decode(endpoint: VllmEndpoint, spec: TestSpec) -> CheckResult:
    """Compare emitted-token scores with recomputation of their exact prefixes."""
    return await _run_check(endpoint, spec, "decode", _check_decode)


def _comparison(value: dict, limits: ErrorLimits) -> ComparisonResult:
    return ComparisonResult(
        status=CheckStatus.INCOMPLETE if value["status"] == "pending" else CheckStatus(value["status"]),
        positions=value["positions"],
        statistics=value.get("statistics"),
        limits=limits,
        worst=tuple(value.get("worst", ())),
    )


def _measurements(
    name: str, value: dict, spec: TestSpec
) -> tuple[dict[str, ComparisonResult], dict[str, CoverageResult]]:
    if name == "reference":
        differences = {
            f"reference.{group}": _comparison(row, spec.tolerances[f"reference.{group}"])
            for group, row in value["groups"].items()
        }
        coverage = {
            "histories": CoverageResult(
                status=(
                    CheckStatus.PASSED
                    if len(value["completed_cases"]) == value["total_cases"]
                    else CheckStatus.INCOMPLETE
                ),
                observations={"completed_cases": value["completed_cases"], "total_cases": value["total_cases"]},
            )
        }
        return differences, coverage
    if name == "cache":
        differences = {
            "cache.same_execution": _comparison(value["same_split"], spec.tolerances["cache.same_execution"]),
            "cache.changed_prefill": _comparison(value["different_split"], spec.tolerances["cache.changed_prefill"]),
        }
        coverage = {}
        for key in ("prefill_reuse", "settled_branches", "decode_reuse", "replayed_decode_reuse", "matched_repeat"):
            row = value[key]
            status = row["status"] if row["status"] in ("passed", "not_required") else "incomplete"
            coverage[key] = CoverageResult(
                status=status,
                observations=row.get("observations", {field: item for field, item in row.items() if field != "status"}),
            )
        return differences, coverage
    differences = {name: _comparison(value, spec.tolerances[name])}
    coverage = {}
    if name == "batch":
        coverage["overlap"] = CoverageResult(
            status=CheckStatus.PASSED if value["running_peak"] >= 2 else CheckStatus.INCOMPLETE,
            observations={"running_peak": value["running_peak"], "batch_evidence": value["batch_evidence"]},
        )
    return differences, coverage


async def _run_check(
    endpoint: VllmEndpoint,
    spec: TestSpec,
    name: str,
    operation: Callable[[VllmEndpoint, ReferenceLookup, TestSpec], Awaitable[dict]],
) -> CheckResult:
    started = time.time()
    request_start = len(endpoint.requests)
    differences: dict[str, ComparisonResult] = {}
    coverage: dict[str, CoverageResult] = {}
    observations = None
    details: dict = {}
    status = CheckStatus.INCOMPLETE
    try:
        references = await asyncio.to_thread(lookup_reference, spec.reference)
        details["reference_lookup"] = {"status": references.status, "reason": references.reason}
        if references.status == LookupStatus.MISMATCHED:
            status = CheckStatus.FAILED
        elif references.status == LookupStatus.FOUND:
            assert references.manifest is not None
            identity = await _verify_identity(endpoint, spec, references.manifest.model)
            details["identity"] = identity
            if identity["status"] != "passed":
                status = CheckStatus.FAILED
            elif spec.serving_engine != ServeBackend.VLLM:
                details["reason"] = "Only vLLM is implemented"
            elif {case.group for case in references.cases} - spec.reference_limits.keys():
                details["reason"] = "The spec omits a reference comparison limit"
            elif (
                name == "batch"
                and sum(len(case.tokens) <= _SHORT_HISTORY_LIMIT for case in references.cases) < spec.batch_size
            ):
                details["reason"] = "Too few distinct histories for the required batch"
            elif name == "decode" and not any(len(case.tokens) < _SHORT_HISTORY_LIMIT for case in references.cases):
                details["reason"] = "No history fits the decode comparison"
            elif name == "cache" and max(map(len, (case.tokens for case in references.cases))) <= max(
                spec.boundary_lengths
            ):
                details["reason"] = "The histories do not cover every required cache boundary"
            else:
                value = json.loads(canonical_bytes(await operation(endpoint, references, spec)))
                differences, coverage = _measurements(name, value, spec)
                observations = value.get("observations")
                measured = {
                    "status",
                    "positions",
                    "statistics",
                    "limits",
                    "worst",
                    "observations",
                    "groups",
                    "same_split",
                    "different_split",
                    "prefill_reuse",
                    "settled_branches",
                    "decode_reuse",
                    "replayed_decode_reuse",
                    "matched_repeat",
                }
                details.update({key: row for key, row in value.items() if key not in measured})
                if value.get("failures") or any(row.status == CheckStatus.FAILED for row in differences.values()):
                    status = CheckStatus.FAILED
                elif any(row.status == CheckStatus.INCOMPLETE for row in differences.values()) or any(
                    row.status == CheckStatus.INCOMPLETE for row in coverage.values()
                ):
                    status = CheckStatus.INCOMPLETE
                else:
                    status = CheckStatus.PASSED
    except Exception as error:
        status = CheckStatus.FAILED
        details["error"] = f"{type(error).__name__}: {error}"
    return CheckResult(
        status=status,
        test_spec=spec.model_dump(mode="json"),
        run_metadata=dict(endpoint.metadata) | {"started_unix": started, "finished_unix": time.time()},
        differences=differences,
        coverage=coverage,
        observations=observations,
        details=details,
        requests=tuple(endpoint.requests[request_start:]),
        seconds=time.time() - started,
    )


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


async def _verify_identity(client: VllmEndpoint, spec: TestSpec, identity: ModelIdentity) -> dict:
    """Check server-reported identities instead of trusting a caller's labels."""
    information = await client.server_info()
    config = information["vllm_config"]
    model = config["model_config"]
    version = client.metadata["version"]
    expected = {
        "tokenizer": identity.tokenizer,
        "tokenizer_revision": identity.tokenizer_revision,
        "quantization": spec.engine_kwargs.get("quantization"),
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
    if str(model["dtype"]).removeprefix("torch.") != spec.engine_kwargs["dtype"]:
        mismatches["dtype"] = {"actual": str(model["dtype"]), "expected": spec.engine_kwargs["dtype"]}
    hardware = information["system_env"]["nvidia_gpu_models"]
    if not any(device in hardware for device in spec.hardware):
        mismatches["hardware"] = {"actual": hardware, "expected": spec.hardware}
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
    if cache_dtype not in spec.kv_cache_dtypes:
        mismatches["kv_cache_dtype"] = {"actual": cache_dtype, "expected": spec.kv_cache_dtypes}
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


async def _check_reference(client: VllmEndpoint, references: ReferenceLookup, spec: TestSpec) -> dict:
    expected = {(row.case_id, row.position): row for row in references.scores}
    groups: dict[str, list[dict]] = {name: [] for name in spec.reference_limits}
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
    summaries = {name: mismatch_summary(rows, spec.reference_limits[name]) for name, rows in groups.items()}
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


async def _check_batch(client: VllmEndpoint, references: ReferenceLookup, spec: TestSpec) -> dict:
    cases = sorted(
        (case for case in references.cases if len(case.tokens) <= _SHORT_HISTORY_LIMIT),
        key=lambda case: len(case.tokens),
    )
    selected = cases[-spec.batch_size :]
    if len(selected) < spec.batch_size:
        raise ValueError("The suite has too few distinct cases for the engine batch")
    alone = []
    for index, case in enumerate(selected):
        await client.reset_cache()
        alone.append(await client.complete(case.tokens, label=f"alone-{index}", count=spec.decode_tokens))
    await client.reset_cache()
    before = await client.metrics()
    tasks = [
        asyncio.create_task(client.complete(case.tokens, label=f"batch-{index}", count=spec.decode_tokens))
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
    rows = [
        dataclasses.asdict(distribution_difference(next_distribution(a), next_distribution(b))) | {"case": case.id}
        for case, a, b in zip(selected, alone, batched, strict=True)
    ]
    summary = mismatch_summary(rows, spec.tolerances["batch"])
    return summary | {
        "engine": 0,
        "running_peak": running_peak,
        "batch_evidence": evidence,
        "metrics_before": before,
        "metrics_after": after,
        "observations": rows,
    }


async def _check_settled_branch(
    client: VllmEndpoint,
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


async def _check_cache(
    client: VllmEndpoint,
    references: ReferenceLookup,
    spec: TestSpec,
) -> dict:
    information = await client.server_info()
    cache_config = information["vllm_config"]["cache_config"]
    max_model_len = information["vllm_config"]["model_config"]["max_model_len"]
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
    for length in spec.boundary_lengths:
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
            settled = await _check_settled_branch(
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
            count = spec.cache_decode_tokens
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
            replayed = await _check_settled_branch(
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
        branch = source[: max(spec.boundary_lengths) + _CACHE_PROBE_SUFFIX_TOKENS]
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
        "same_split": mismatch_summary(same_split, spec.tolerances["cache.same_execution"]),
        "different_split": mismatch_summary(different_split, spec.tolerances["cache.changed_prefill"]),
    }
    coverage = {
        name: {
            "status": (
                "passed"
                if len(rows) == len(spec.boundary_lengths)
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


async def _check_decode(client: VllmEndpoint, references: ReferenceLookup, spec: TestSpec) -> dict:
    cases = sorted(
        (case for case in references.cases if len(case.tokens) < _SHORT_HISTORY_LIMIT), key=lambda case: len(case.tokens)
    )
    selected = [cases[0], cases[-1]]
    differences = []
    for index, case in enumerate(selected):
        await client.reset_cache()
        decoded = await client.complete(case.tokens, label=f"decode-{index}", count=spec.decode_tokens)
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
    return mismatch_summary(differences, spec.tolerances["decode"]) | {"observations": differences}
