# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded native-suite goldens, owned-server observations and serving proof.

No model restore, server launch or wheel selection occurs here. ``prepare``
freezes existing native scores; ``collect`` calls only the pinned owned endpoint;
``verify`` recomputes evidence from retained source bytes and HTTP responses.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

import draccus
import requests
from marin.inference.backend import ModelSpec
from marin.inference.vllm_release import VllmGpuWheelProvenance
from marin.testing.inference.backend_parity import (
    NextTokenObservation,
    assert_same_rank_repeatability,
    cross_rank_diagnostic,
)
from marin.testing.inference.backend_parity import (
    TokenScore as Probability,
)
from pydantic import Field, model_validator
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.format_eval import EOT_ID, Candidate, Serialization, grade_panel
from experiments.grug.moe_hero_ep.format_sample import NATIVE_BACKEND, native_evaluation_model
from experiments.grug.moe_hero_ep.format_suite import NativeSuiteRequest
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Record,
    SampleResult,
    StopReason,
    TokenScore,
    digest,
)

MAX_RECORD_BYTES = 128 * 1024 * 1024
_VERIFIED_WHEEL = "MARIN_VLLM_WHEEL_VERIFIED="
_TOPOLOGY = re.compile(
    r"GrugMoE effective config: TP=(\d+)/(\d+) PP=(\d+)/(\d+) " r"layers=\[(\d+),(\d+)\) DP=(\d+)/(\d+) EP=(\d+)/(\d+)"
)


class JsonPin(Record):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StopContract(StrEnum):
    EOT = "eot"
    BUDGET = "budget"


class Trajectory(Record):
    panel_id: str
    prompt_id: str
    sample_index: int = Field(ge=0)
    minimum_decode_tokens: int = Field(ge=1, le=63)
    stop: StopContract

    @property
    def case_id(self) -> str:
        return f"{self.panel_id}/{self.prompt_id}/{self.sample_index}"


class ServingParityRequest(Record):
    candidate: Candidate
    suite_manifest: JsonPin
    export_manifest: JsonPin
    context_extension_manifest: JsonPin | None = None
    trajectories: tuple[Trajectory, ...] = Field(min_length=2, max_length=16)
    max_new_tokens: int = Field(ge=2, le=64)
    ranks: tuple[int, ...] = Field(min_length=1, max_length=64)
    waves: int = Field(ge=2, le=4)
    max_probability_error: float = Field(ge=0, le=1, allow_inf_nan=False)
    required_serving_source: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    launch_manifest: JsonPin | None = None

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if len(set(self.ranks)) != len(self.ranks) or min(self.ranks) < 0:
            raise ValueError("Serving ranks must be distinct nonnegative ranks")
        if len({case.case_id for case in self.trajectories}) != len(self.trajectories):
            raise ValueError("Serving trajectories must be distinct")
        if len(self.trajectories) * len(self.ranks) * self.waves > 128:
            raise ValueError("Serving parity is bounded to 128 multi-token HTTP requests")
        if any(case.minimum_decode_tokens >= self.max_new_tokens for case in self.trajectories):
            raise ValueError("Decode coverage must fit after the prefill token within the request budget")
        if (self.candidate == Candidate.E) != (self.context_extension_manifest is not None):
            raise ValueError("Exactly candidate E requires its pinned context-extension manifest")
        return self


class Topology(Record):
    tensor_parallel_size: int = Field(ge=1)
    pipeline_parallel_size: int = Field(ge=1)
    data_parallel_size: int = Field(ge=1)
    expert_parallel_size: int = Field(ge=1)


class OwnedServer(Record):
    execution_id: str = Field(min_length=1)
    server_url: str = Field(pattern=r"^https?://")
    model: ModelSpec
    export_manifest: JsonPin
    qualification_manifest: JsonPin
    startup_log: JsonPin
    wheel: VllmGpuWheelProvenance
    architecture: str
    compute_capability: str
    topology: Topology
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    command: tuple[str, ...] = Field(min_length=1)
    attention_backend: str = Field(min_length=1)
    prefix_caching: bool
    speculative_decoding: bool


class Golden(Record):
    selection: Trajectory
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_scores: tuple[TokenScore, ...]
    native_stop: StopReason


class ServingIdentity(Record):
    candidate: Candidate
    checkpoint: Checkpoint
    model_config_digest: str
    tokenizer: str
    tokenizer_revision: str
    suite_manifest_digest: str
    context_extension_manifest_digest: str | None
    export_id: str


@dataclasses.dataclass(frozen=True)
class Preparation:
    identity: ServingIdentity
    goldens: tuple[Golden, ...]
    launch: OwnedServer | None
    pending: tuple[str, ...]
    files: dict[str, dict[str, str]]


def _read(pin: JsonPin, files: dict[str, dict[str, str]], retained: bool) -> str:
    if pin.path in files:
        if files[pin.path].get("missing") == "not-found":
            raise FileNotFoundError(pin.path)
        raw = files[pin.path]["text"].encode()
    elif retained:
        raise ValueError(f"Serving proof omitted pinned source bytes: {pin.path}")
    else:
        path = StoragePath(pin.path)
        try:
            if path.size() > MAX_RECORD_BYTES:
                raise ValueError("Serving proof source exceeds the bounded metadata budget")
            raw = path.read_bytes()
        except FileNotFoundError:
            files[pin.path] = {"sha256": pin.sha256, "missing": "not-found"}
            raise
        files[pin.path] = {"sha256": pin.sha256, "text": raw.decode()}
        if sum(len(value.get("text", "").encode()) for value in files.values()) > MAX_RECORD_BYTES:
            raise ValueError("Serving proof exceeds the total retained source byte budget")
    if hashlib.sha256(raw).hexdigest() != pin.sha256 or files[pin.path]["sha256"] != pin.sha256:
        raise ValueError(f"Serving proof source bytes changed: {pin.path}")
    return raw.decode()


def _bound(value: dict, key: str) -> str:
    actual = digest({name: item for name, item in value.items() if name != key})
    if actual != value.get(key):
        raise ValueError(f"Serving proof {key} changed")
    return actual


def _runtime_pending(request, server, qualification, startup, hf_config) -> list[str]:
    wheel = server.wheel
    if (
        wheel.source_commit != request.required_serving_source
        or qualification.get("source", {}).get("fork_commit") != request.required_serving_source
    ):
        return ["required serving source lacks a matching qualified wheel"]
    if (
        qualification.get("release", {}).get("status") != "released"
        or qualification.get("validation", {}).get("status") != "passed"
    ):
        return ["final GPU release qualification is unavailable"]
    platforms = [item for item in qualification["platforms"] if item["architecture"] == server.architecture]
    targets = [item for item in qualification["validation"]["targets"] if item["architecture"] == server.architecture]
    if len(platforms) != 1 or len(targets) != 1:
        return ["qualified runtime platform evidence is incomplete"]
    platform, target = platforms[0], targets[0]
    candidate_tag = qualification["release"].get("candidate_tag")
    candidate_wheel = {key: platform["wheel"][key] for key in ("filename", "sha256")}
    candidate_wheel["url"] = wheel.wheel_url.rsplit("/", 2)[0] + f"/{candidate_tag}/{candidate_wheel['filename']}"
    if (
        platform["wheel"]["url"] != wheel.wheel_url
        or platform["wheel"]["sha256"] != wheel.wheel_sha256
        or qualification["release"]["tag"] != wheel.release_tag
        or qualification["distribution"]["version"] != wheel.version
        or target.get("source_commit") != wheel.source_commit
        or target.get("candidate_tag") != candidate_tag
        or not candidate_tag
        or target.get("wheel") != candidate_wheel
        or target.get("result") != "passed"
        or target.get("hardware", {}).get("compute_capability") != server.compute_capability
        or server.compute_capability not in wheel.sm_targets
    ):
        return ["qualified wheel/source/hash/platform differs from the owned server"]
    gates = target.get("gates", {})
    if (
        any(
            gates.get(key, {}).get("status") != "passed"
            for key in (
                "wheel_sha256",
                "distribution_metadata",
                "vllm._C_stable_libtorch",
                "GrugMoeForCausalLM",
                "cumem_allocator",
                "torchaudio.resample",
                "sparse_nccl_contract",
                "serve_smoke",
            )
        )
        or any(item.get("status") not in ("passed", "not_run") for item in gates.values())
        or any(
            not math.isfinite(gates.get("serve_smoke", {}).get("metrics", {}).get(key, 0))
            for key in ("completions", "output_tokens_per_second")
        )
        or gates.get("serve_smoke", {}).get("metrics", {}).get("completions", 0) <= 0
        or gates.get("serve_smoke", {}).get("metrics", {}).get("output_tokens_per_second", 0) <= 0
    ):
        return ["release qualification lacks measured passing runtime gates"]
    verified = [
        json.loads(line.split(_VERIFIED_WHEEL, 1)[1]) for line in startup.splitlines() if _VERIFIED_WHEEL in line
    ]
    expected = dataclasses.asdict(wheel)
    if not verified or any(
        any(value.get(key) != json.loads(json.dumps(item)) for key, item in expected.items())
        or value.get("compute_capability") != server.compute_capability
        or not value.get("extension_path")
        for value in verified
    ):
        return ["actual startup lacks matching verified installed-wheel provenance"]
    expected_arguments = {
        "--served-model-name": server.model.api_model,
        "--tokenizer": server.model.tokenizer,
        "--tokenizer-revision": server.model.tokenizer_revision,
        "--dtype": server.model.dtype,
        "--max-model-len": str(server.model.max_model_len),
        "--tensor-parallel-size": str(server.topology.tensor_parallel_size),
        "--pipeline-parallel-size": str(server.topology.pipeline_parallel_size),
        "--data-parallel-size": str(server.topology.data_parallel_size),
        "--attention-backend": server.attention_backend,
    }
    for flag, value in expected_arguments.items():
        observed = [
            item.partition("=")[2] if "=" in item else server.command[index + 1]
            for index, item in enumerate(server.command)
            if item.partition("=")[0] == flag and ("=" in item or index + 1 < len(server.command))
        ]
        if observed != [value]:
            return [f"owned server command does not bind {flag} to its ModelSpec/runtime"]
    if not any(
        server.command[index : index + 2] == ("serve", server.model.weights) for index in range(len(server.command) - 1)
    ):
        return ["owned server command does not serve the pinned HF export"]
    topology = server.topology
    placements = {tuple(map(int, match.groups())) for match in _TOPOLOGY.finditer(startup)}
    for rank in request.ranks:
        for tensor in range(topology.tensor_parallel_size):
            for pipeline in range(topology.pipeline_parallel_size):
                start = pipeline * hf_config["num_hidden_layers"] // topology.pipeline_parallel_size
                end = (pipeline + 1) * hf_config["num_hidden_layers"] // topology.pipeline_parallel_size
                if not any(
                    row[:8]
                    == (
                        tensor,
                        topology.tensor_parallel_size,
                        pipeline,
                        topology.pipeline_parallel_size,
                        start,
                        end,
                        rank,
                        topology.data_parallel_size,
                    )
                    and row[9] == topology.expert_parallel_size
                    for row in placements
                ):
                    return ["startup does not attest every selected rank and model stage"]
    if (
        server.prefix_caching
        or server.speculative_decoding
        or "--no-enable-prefix-caching" not in server.command
        or any(item.partition("=")[0] == "--speculative-config" for item in server.command)
    ):
        return ["decode proof requires fresh non-speculative KV execution without prefix caching"]
    return []


def prepare_serving(request: ServingParityRequest, *, files: dict[str, dict[str, str]] | None = None) -> Preparation:
    """Freeze completed native-suite goldens and check owned runtime metadata; no arrays or HTTP."""
    retained = files is not None
    snapshots = {} if files is None else files

    def read(pin):
        return json.loads(_read(pin, snapshots, retained))

    manifest = read(request.suite_manifest)
    suite_id = _bound(manifest, "manifest_digest")
    suite = NativeSuiteRequest.model_validate(manifest["request"])
    model = draccus.decode(GrugModelConfig, suite.model)
    if (
        manifest["suite_id"] != suite.suite_id
        or manifest["model_digest"] != digest(draccus.encode(model))
        or manifest["model"] != draccus.encode(model)
        or manifest["checkpoint"] != suite.checkpoint.model_dump()
        or manifest["backend"] != NATIVE_BACKEND
        or manifest["compute_dtype"] != "bfloat16"
        or manifest["routing_control_dtype"] != "float32"
        or manifest["source_revision"] != suite.source_revision
    ):
        raise ValueError("Native suite identity or precision changed")
    exported = read(request.export_manifest)
    export_request = exported["request"]
    if (
        exported["export_id"] != digest(export_request)
        or export_request["export_version"] != 3
        or not exported["shards"]
    ):
        raise ValueError("Serving parity requires a completed version-3 HF export")
    source = export_request["config"]
    export_model = draccus.decode(GrugModelConfig, source["model"])
    if (
        source["checkpoint"] != suite.checkpoint.uri
        or source["metadata_digest"] != suite.checkpoint.metadata_digest
        or native_evaluation_model(export_model) != model
        or digest(export_request["hf_config"]) != digest(export_model.to_hf_config(export_model.vocab_size).to_dict())
        or exported["precision"]["learned_weights"] != "bfloat16"
        or exported["precision"]["effective_routing_bias"] != "float32"
    ):
        raise ValueError("HF export belongs to another native checkpoint, forward model or precision")
    hf_config = export_request["hf_config"]
    export_root = StoragePath(source["destination"])
    # These small files establish the served HF configuration/index identity;
    # shard bytes are already recorded by the completed exporter, not re-read here.
    for name, expected in (("config.json", hf_config), ("export-request.json", export_request)):
        value = read(
            JsonPin(path=str(export_root / name), sha256=_snapshot_sha(export_root / name, snapshots, retained))
        )
        if value != expected:
            raise ValueError("Completed HF export metadata changed")
    index_path = export_root / "model.safetensors.index.json"
    index = read(JsonPin(path=str(index_path), sha256=_snapshot_sha(index_path, snapshots, retained)))
    if index["weight_map"] != {
        name: shard["filename"] for shard in exported["shards"] for name in shard["tensor_names"]
    }:
        raise ValueError("HF index differs from completed export shard records")
    extension_digest = None
    if request.context_extension_manifest is not None:
        extension = read(request.context_extension_manifest)
        extension_digest = _bound(extension, "manifest_digest")
        if (
            extension["candidate"] != "E"
            or extension["source_checkpoint"] != suite.checkpoint.uri
            or extension["source_metadata_digest"] != suite.checkpoint.metadata_digest
            or native_evaluation_model(draccus.decode(GrugModelConfig, extension["model"])) != model
            or extension["hf_config"] != hf_config
            or model.yarn is None
            or model.max_seq_len != 65536
        ):
            raise ValueError("E extension differs from native/serving arrays or forward settings")
    goldens = []
    pending = []
    for selection in request.trajectories:
        entries = [
            (entry, output)
            for entry, output in zip(suite.panels, manifest["panels"], strict=True)
            if entry.panel.panel_id == selection.panel_id and entry.serialization == Serialization.MARIN
        ]
        if len(entries) != 1:
            raise ValueError("Selected native Marin panel is absent or ambiguous")
        entry, output = entries[0]
        if output["result_key"] != entry.result_key or output["panel_id"] != entry.panel.panel_id:
            raise ValueError("Native panel output belongs to another declared panel")
        sample_path = StoragePath(request.suite_manifest.path).parent / output["result_key"] / "samples.json"
        result = SampleResult.model_validate(
            read(JsonPin(path=str(sample_path), sha256=output["file_sha256"]["samples.json"]))
        )
        if (
            result.request.checkpoint != suite.checkpoint
            or result.request.model != suite.model
            or result.request.source_revision != suite.source_revision
            or result.request.spec.temperature != 0
            or result.eos_token_id != EOT_ID
            or result.request.sample_id != output["sample_id"]
            or (result.request.spec.tokenizer, result.request.spec.tokenizer_revision)
            != (suite.tokenizer, suite.tokenizer_revision)
        ):
            raise ValueError("Native goldens differ from suite checkpoint/model/tokenizer/greedy execution")
        grade_panel(entry.panel, result, entry.serialization)
        row = next(row for row in result.completions if row.prompt_id == selection.prompt_id)
        completion = row.samples[selection.sample_index]
        if completion.token_scores is None:
            pending.append(f"{selection.case_id}: native per-token scores unavailable")
            continue
        if any(
            score.logprob != max(token.logprob for token in score.top_tokens)
            or not any(token.token_id == score.token_id and token.logprob == score.logprob for token in score.top_tokens)
            for score in completion.token_scores
        ):
            raise ValueError("Native greedy trajectory disagrees with its retained raw probabilities")
        count = min(request.max_new_tokens, len(completion.token_ids))
        if len(row.prompt_token_ids) + count > model.max_seq_len:
            raise ValueError("Selected raw prompt and completion exceed the native/server context")
        if selection.stop == StopContract.BUDGET and (
            count != request.max_new_tokens or EOT_ID in completion.token_ids[:count]
        ):
            pending.append(f"{selection.case_id}: bounded native non-EOT trajectory unavailable")
        if count < selection.minimum_decode_tokens + 1:
            pending.append(f"{selection.case_id}: native decode coverage unavailable")
        if selection.stop == StopContract.EOT and (
            completion.stop_reason != StopReason.EOS or len(completion.token_ids) > request.max_new_tokens
        ):
            pending.append(f"{selection.case_id}: bounded native EOT trajectory unavailable")
        goldens.append(
            Golden(
                selection=selection,
                prompt_token_ids=row.prompt_token_ids,
                token_ids=completion.token_ids[:count],
                token_scores=completion.token_scores[:count],
                native_stop=completion.stop_reason,
            )
        )
    lengths = [len(case.prompt_token_ids) for case in goldens]
    if not any(length < 16000 for length in lengths) or not any(16000 <= length < 65000 for length in lengths):
        pending.append("actual short and 16K native prompt coverage unavailable")
    if request.candidate == Candidate.E and not any(length >= 65000 for length in lengths):
        pending.append("actual >=65000-token native prompt coverage unavailable")
    if not any(case.selection.stop == StopContract.EOT for case in goldens):
        pending.append("native EOT stop coverage unavailable")
    identity = ServingIdentity(
        candidate=request.candidate,
        checkpoint=suite.checkpoint,
        model_config_digest=manifest["model_digest"],
        tokenizer=suite.tokenizer,
        tokenizer_revision=suite.tokenizer_revision,
        suite_manifest_digest=suite_id,
        context_extension_manifest_digest=extension_digest,
        export_id=exported["export_id"],
    )
    server = None
    if request.launch_manifest is None:
        pending.append("qualified owned-server launch manifest unavailable")
    else:
        try:
            server = OwnedServer.model_validate(read(request.launch_manifest))
            qualification = read(server.qualification_manifest)
            startup = _read(server.startup_log, snapshots, retained)
        except FileNotFoundError:
            pending.append("qualified owned-server source evidence unavailable")
        else:
            if (
                server.export_manifest != request.export_manifest
                or server.model.weights.rstrip("/") != source["destination"].rstrip("/")
                or server.model.tokenizer != suite.tokenizer
                or server.model.tokenizer_revision != suite.tokenizer_revision
                or server.model.tensor_parallel_size != server.topology.tensor_parallel_size
                or server.model.dtype != "bfloat16"
                or server.model.max_model_len != model.max_seq_len
                or max(request.ranks) >= server.topology.data_parallel_size
                or server.model.num_chips
                != server.topology.tensor_parallel_size
                * server.topology.pipeline_parallel_size
                * server.topology.data_parallel_size
            ):
                raise ValueError("Owned server ModelSpec/export/tokenizer/context/topology differs from native proof")
            pending.extend(_runtime_pending(request, server, qualification, startup, hf_config))
    return Preparation(identity, tuple(goldens), server, tuple(pending), snapshots)


def _snapshot_sha(path: StoragePath, files: dict[str, dict[str, str]], retained: bool) -> str:
    if str(path) in files:
        return files[str(path)]["sha256"]
    if retained:
        raise ValueError("Serving proof omitted export metadata bytes")
    if path.size() > MAX_RECORD_BYTES:
        raise ValueError("HF metadata exceeds the bounded byte budget")
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    files[str(path)] = {"sha256": sha256, "text": raw.decode()}
    if sum(len(value.get("text", "").encode()) for value in files.values()) > MAX_RECORD_BYTES:
        raise ValueError("Serving proof exceeds the total retained source byte budget")
    return sha256


def _http_request(request: ServingParityRequest, prepared: Preparation, case: Golden, wave: int, rank: int) -> dict:
    assert prepared.launch is not None
    request_id = (
        f"hero-parity-{digest(request.model_dump(mode='json'))[:16]}-{wave}-{rank}-{digest(case.selection.case_id)[:12]}"
    )
    return {
        "url": prepared.launch.server_url.rstrip("/") + "/completions",
        "headers": {"X-data-parallel-rank": str(rank), "X-Request-Id": request_id},
        "body": {
            "model": prepared.launch.model.api_model,
            "prompt": list(case.prompt_token_ids),
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": len(case.token_ids),
            "logprobs": 50,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "skip_special_tokens": False,
            "ignore_eos": True,
            "stop_token_ids": [EOT_ID],
            "include_stop_str_in_output": True,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    }


def score_serving(request: ServingParityRequest, prepared: Preparation, exchanges: list[dict]) -> dict:
    """Recompute probability, common-prefix decode, stop, context and repeatability evidence."""
    if prepared.pending:
        if exchanges:
            raise ValueError("Pending runtime/source proof cannot contain accepted HTTP observations")
        return {
            "status": "pending",
            "passed": False,
            "reasons": list(prepared.pending),
            "comparisons": [],
            "trajectories": [],
        }
    expected = {
        (case.selection.case_id, wave, rank): case
        for case in prepared.goldens
        for wave in range(request.waves)
        for rank in request.ranks
    }
    indexed = {(row["case_id"], row["wave"], row["rank"]): row for row in exchanges}
    if len(indexed) != len(exchanges) or indexed.keys() != expected.keys():
        raise ValueError("Serving exchanges do not cover the exact declared case/wave/rank grid")
    reasons = []
    comparisons = []
    trajectories = []
    repeat = {}
    for key, case in expected.items():
        exchange = indexed[key]
        if (
            exchange["request"] != _http_request(request, prepared, case, key[1], key[2])
            or exchange["http_status"] != 200
        ):
            raise ValueError("Retained HTTP request/status differs from the pinned raw-token execution")
        response = exchange["response"]
        assert prepared.launch is not None
        if response["model"] != prepared.launch.model.api_model or len(response["choices"]) != 1:
            raise ValueError("HTTP response belongs to another served model")
        choice = response["choices"][0]
        if choice["prompt_token_ids"] != list(case.prompt_token_ids):
            raise ValueError("Serving changed/truncated the exact raw prompt IDs")
        tokens = choice["token_ids"]
        scores = choice["logprobs"]["top_logprobs"]
        if (
            len(tokens) != len(scores)
            or len(tokens) > len(case.token_ids)
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("Serving token IDs/logprob positions exceed the bounded request")
        observations = []
        common = 0
        diverged = False
        for position, (token, values) in enumerate(zip(tokens, scores, strict=True)):
            if any(re.fullmatch(r"token_id:\d+", name) is None for name in values):
                raise ValueError("Serving top probabilities must retain raw token IDs")
            mapping = {int(name.removeprefix("token_id:")): float(value) for name, value in values.items()}
            if not all(math.isfinite(value) and value <= 0 for value in mapping.values()):
                raise ValueError("Serving returned nonfinite or invalid log probabilities")
            observation = NextTokenObservation.from_logprob_map(
                f"{key[0]}@{position}", token, mapping, backend_rank=key[2]
            )
            observations.append(observation)
            if diverged:
                continue
            expected_scores = tuple(
                Probability(logprob=score.logprob, token_id=score.token_id)
                for score in case.token_scores[position].top_tokens
            )
            try:
                parity = observation.parity_against(expected_scores)
                parity.assert_matches(max_probability_error=request.max_probability_error)
            except AssertionError:
                reasons.append(f"{key}: probability/top-token coverage at position {position}")
            else:
                comparisons.append(
                    {
                        "case_id": key[0],
                        "wave": key[1],
                        "rank": key[2],
                        "position": position,
                        **dataclasses.asdict(parity),
                    }
                )
            common += 1
            diverged = token != case.token_ids[position]
        decode = max(0, common - 1)
        if diverged or common != len(case.token_ids) or decode < case.selection.minimum_decode_tokens:
            reasons.append(f"{key}: common-prefix decode coverage lost")
        if case.selection.stop == StopContract.EOT:
            if (
                not tokens
                or tokens[-1] != EOT_ID
                or EOT_ID in tokens[:-1]
                or choice.get("finish_reason") != "stop"
                or choice.get("stop_reason") != EOT_ID
            ):
                reasons.append(f"{key}: actual EOT stop contract failed")
        elif choice.get("finish_reason") != "length" or len(tokens) != len(case.token_ids):
            reasons.append(f"{key}: bounded decode stopped unexpectedly")
        repeat[key] = observations
        trajectories.append(
            {
                "case_id": key[0],
                "wave": key[1],
                "rank": key[2],
                "prompt_tokens": len(case.prompt_token_ids),
                "common_prefix_positions": common,
                "decode_positions": decode,
                "first_divergence": common - 1 if diverged else None,
                "generated_token_ids": tokens,
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
            }
        )
    for case in prepared.goldens:
        for wave in range(1, request.waves):
            for rank in request.ranks:
                first = repeat[(case.selection.case_id, 0, rank)]
                other = repeat[(case.selection.case_id, wave, rank)]
                try:
                    if len(first) != len(other):
                        raise AssertionError("different decode lengths")
                    for left, right in zip(first, other, strict=True):
                        assert_same_rank_repeatability([left], [right])
                except AssertionError:
                    reasons.append(f"{case.selection.case_id}/{rank}: same-rank recomputation changed")
    diagnostics = []
    for case in prepared.goldens:
        for position in range(len(case.token_ids)):
            observations = [
                repeat[(case.selection.case_id, 0, rank)][position]
                for rank in request.ranks
                if len(repeat[(case.selection.case_id, 0, rank)]) > position
            ]
            if observations:
                diagnostic = dataclasses.asdict(cross_rank_diagnostic(observations))
                if not math.isfinite(diagnostic["max_probability_spread"]):
                    diagnostic["max_probability_spread"] = None
                diagnostics.append(diagnostic)
    return {
        "status": "pass" if not reasons else "inconclusive",
        "passed": not reasons,
        "reasons": reasons,
        "comparisons": comparisons,
        "trajectories": trajectories,
        "cross_rank_diagnostics": diagnostics,
        "probability_scope": (
            "native retained top tokens on identical raw prefixes; later divergent positions are unmeasured"
        ),
    }


def _report(request, prepared, exchanges) -> dict:
    body = {
        "schema_version": 1,
        "request": request.model_dump(mode="json"),
        "identity": prepared.identity.model_dump(mode="json"),
        "files": prepared.files,
        "goldens": [case.model_dump(mode="json") for case in prepared.goldens],
        "exchanges": exchanges,
        "measurement": _measurement(request, prepared, exchanges),
        "created_at": datetime.now(UTC).isoformat(),
    }
    body["report_digest"] = digest(body)
    return body


def _measurement(request, prepared, exchanges) -> dict:
    try:
        return score_serving(request, prepared, exchanges)
    except (ValueError, AssertionError, KeyError, TypeError) as error:
        # Preserve malformed or wrong-identity HTTP replies as rejected evidence,
        # then reproduce this same decision in the offline verifier.
        return {"status": "reject", "passed": False, "reasons": [str(error)], "comparisons": [], "trajectories": []}


def _write(destination: str, body: dict) -> None:
    raw = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("Serving proof exceeds bounded retained bytes")
    conditional_object(destination).write(raw, expected_version=None)


def collect_serving(request: ServingParityRequest, destination: str, *, post=requests.post) -> dict:
    """Collect bounded real HTTP observations from the declared owned server, or retain pending proof."""
    request = ServingParityRequest.model_validate(request.model_dump(mode="json"))
    prepared = prepare_serving(request)
    exchanges = []
    if not prepared.pending:
        for case in prepared.goldens:
            for wave in range(request.waves):
                for rank in request.ranks:
                    query = _http_request(request, prepared, case, wave, rank)
                    try:
                        response = post(
                            query["url"], headers=query["headers"], json=query["body"], timeout=(30.0, 300.0)
                        )
                    except requests.RequestException as error:
                        exchanges.append(
                            {
                                "case_id": case.selection.case_id,
                                "wave": wave,
                                "rank": rank,
                                "request": query,
                                "http_status": None,
                                "response": {"transport_error": type(error).__name__},
                            }
                        )
                        body = _report(request, prepared, exchanges)
                        _write(destination, body)
                        return body
                    try:
                        parsed = response.json()
                    except requests.exceptions.JSONDecodeError:
                        parsed = {"non_json_response": response.text[:4096]}
                    exchanges.append(
                        {
                            "case_id": case.selection.case_id,
                            "wave": wave,
                            "rank": rank,
                            "request": query,
                            "http_status": response.status_code,
                            "response": parsed,
                        }
                    )
    body = _report(request, prepared, exchanges)
    _write(destination, body)
    return body


def verify_serving_report(pin: JsonPin, expected: ServingIdentity | None = None) -> dict:
    """Validate pinned raw bytes and recompute every serving acceptance gate without HTTP or arrays."""
    body = json.loads(_read(pin, {}, False))
    _bound(body, "report_digest")
    request = ServingParityRequest.model_validate(body["request"])
    prepared = prepare_serving(request, files=body["files"])
    if body["identity"] != prepared.identity.model_dump(mode="json") or (
        expected is not None and expected != prepared.identity
    ):
        raise ValueError("Serving proof belongs to another candidate/checkpoint/model/tokenizer/suite/export")
    if digest(body["goldens"]) != digest([case.model_dump(mode="json") for case in prepared.goldens]):
        raise ValueError("Frozen goldens differ from their bound native suite bytes")
    measured = _measurement(request, prepared, body["exchanges"])
    if digest(measured) != digest(body["measurement"]):
        raise ValueError("Serving verdict differs from recomputed raw HTTP observations")
    return {
        "identity": prepared.identity.model_dump(mode="json"),
        "measurement": measured,
        "report_digest": body["report_digest"],
        "report_sha256": pin.sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "collect", "verify"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output")
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.operation == "verify":
        pin = JsonPin.model_validate_json(args.input.read_bytes())
        verdict = verify_serving_report(pin)
        _write(args.output, verdict)
        if not verdict["measurement"]["passed"]:
            raise SystemExit(1)
        return
    request = ServingParityRequest.model_validate_json(args.input.read_bytes())
    if args.operation == "prepare":
        prepared = prepare_serving(request)
        _write(
            args.output,
            {
                "identity": prepared.identity.model_dump(mode="json"),
                "pending": list(prepared.pending),
                "goldens": [case.model_dump(mode="json") for case in prepared.goldens],
                "files": prepared.files,
            },
        )
        return
    report = collect_serving(request, args.output)
    if not report["measurement"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
