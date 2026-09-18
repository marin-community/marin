# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Produce or submit the pinned full-Hero BF16 forward golden bundle."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple, Self, TypedDict

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.rpc.proto_display import priority_band_value
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from marin.testing.inference.hero_forward_goldens import ARRAYS_FILENAME, MANIFEST_FILENAME, write_bundle
from pydantic import Field, model_validator
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from rigging.timing import Timer, log_time
from transformers import AutoTokenizer

from experiments.grug.moe_hero_ep import hero_recipe
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    TOP_TOKEN_COUNT,
    Checkpoint,
    Prompt,
    Record,
    digest,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.config import (
    CHECKPOINT_RUNS,
    SAMPLING_GPUS_PER_NODE,
    TARGET_CLUSTER,
    discover_requests,
    sampling_resources,
    sampling_spec,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import IrisSamplingJobs
from experiments.grug.moe_hero_ep.ops.vibe_check.sample import COMPUTE_POLICY, restore_model_state
from experiments.grug.moe_hero_ep.train import DEFAULT_DROPLESS_MOE_IMPLEMENTATION

logger = logging.getLogger(__name__)

CONTROLLER_CLUSTER = "marin"
JOB_USER = "hero-goldens"
STORE_ROOT = "s3://marin-us-east-02a/marin/reference/hero-forward"
REQUIRED_RELEASE = "hero-535b-step108000-bf16-v1"
SMOKE_RELEASE = "hero-535b-step108000-bf16-smoke-v1"
SELECTED_CHECKPOINT_URI = (
    "s3://marin-us-east-02a/marin/grug/hero-ragged_a2a-nccl2307-ep-step81k/" "2026.08.19.2/checkpoints/step-108000"
)
CHECKPOINT_WRITER_REVISION = "04fb3484560711bafcc98b2975d551363ee95067"
CHECKPOINT_WRITER_BUNDLE_ID = "c26e1dae4cd5a3a6c4305653994679c3f3310d0ad30ff82b3316240ac946029d"
CHECKPOINT_WRITER_EVIDENCE_URL = "https://github.com/marin-community/marin/issues/8506#issuecomment-5609812430"
CHECKPOINT_WRITER_MODEL_DIGEST = "d461b6b0832be902ecbc9111edfbbc77a9cfd760470064285c01f6a296b45661"
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
NATIVE_OUTPUT_BOUND = 1e-4
DETERMINISTIC_XLA_FLAGS = "--xla_gpu_deterministic_ops=true"
GOLDEN_MODES = ("smoke", "required")
AUTHORITATIVE_WEIGHT_KEYS = ("master_params", "params")


class GoldenCaseSpec(Record):
    id: str
    source_prompt_id: str
    valid_length: int = Field(gt=1)


class GoldenSpec(Record):
    release: str
    mode: str
    batch_size: int = Field(gt=0)
    cases: tuple[GoldenCaseSpec, ...]
    tokenizer: str
    tokenizer_revision: str
    training_model: dict
    model: dict

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        if len(self.cases) != self.batch_size:
            raise ValueError("The case count must equal the full-Hero batch size")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Golden case IDs must be unique")
        return self


class GoldenRequest(Record):
    checkpoint: Checkpoint
    spec: GoldenSpec
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_cluster: str

    @property
    def bundle_id(self) -> str:
        identity = {"checkpoint": self.checkpoint.model_dump(), "spec": self.spec.model_dump()}
        return f"{self.spec.release}-{digest(identity)[:12]}"


class GoldenCaseManifest(TypedDict):
    id: str
    row: int
    source_prompt_id: str
    valid_length: int
    scored_target_positions: list[int]
    construction: str


class ForwardValues(NamedTuple):
    target_logprobs: jax.Array
    top_token_ids: jax.Array
    top_logprobs: jax.Array
    full_logits: jax.Array


class TracedValues(NamedTuple):
    forward: ForwardValues
    route_expert_ids: jax.Array
    route_combine_weights: jax.Array
    route_cutoff_gaps: jax.Array


class ForwardSelection(NamedTuple):
    score_cases: jax.Array
    prediction_positions: jax.Array
    targets: jax.Array
    full_cases: jax.Array
    full_positions: jax.Array


class NativeDiagnostic(NamedTuple):
    target_logprob_max_abs: float
    top_logprob_max_abs: float
    full_logit_max_abs: float
    top_token_id_changes: int

    def as_dict(self) -> dict[str, float | int]:
        return self._asdict()


def golden_spec(mode: str) -> GoldenSpec:
    training_model = draccus.encode(hero_recipe.HERO_MODEL_CONFIG)
    training_model_digest = digest(training_model)
    if training_model_digest != CHECKPOINT_WRITER_MODEL_DIGEST:
        raise ValueError(
            "Current HERO_MODEL_CONFIG differs from the checkpoint-writing configuration: "
            f"{training_model_digest} != {CHECKPOINT_WRITER_MODEL_DIGEST}. "
            "Resolve the difference and bump the golden release before producing a new bundle."
        )
    inference_model = draccus.encode(
        dataclasses.replace(
            hero_recipe.HERO_MODEL_CONFIG,
            moe_implementation=DEFAULT_DROPLESS_MOE_IMPLEMENTATION,
            expert_chunks=1,
        )
    )
    changes = {
        key: (training_model.get(key), inference_model.get(key))
        for key in training_model.keys() | inference_model.keys()
        if training_model.get(key) != inference_model.get(key)
    }
    if changes != {"moe_implementation": (hero_recipe.RAGGED_MOE_IMPLEMENTATION, DEFAULT_DROPLESS_MOE_IMPLEMENTATION)}:
        raise ValueError(f"Unexpected inference model overrides: {changes}")

    if mode == "smoke":
        base_cases = (("short-fixed-continuation", "add-two-numbers", 64),)
        release = SMOKE_RELEASE
    elif mode == "required":
        base_cases = (
            ("short-fixed-continuation", "add-two-numbers", 32),
            ("padded-code-continuation", "code-unique-in-order", 128),
            ("long-reference", "neuron-associate-grub-zoo", 512),
            ("local-window-minus-one", "neuron-associate-grub-zoo", 2047),
            ("local-window-exact", "neuron-associate-grub-zoo", 2048),
            ("local-window-plus-one", "neuron-associate-grub-zoo", 2049),
            ("context-minus-one", "neuron-associate-grub-zoo", 4095),
            ("context-exact", "neuron-associate-grub-zoo", 4096),
        )
        release = REQUIRED_RELEASE
    else:
        raise ValueError(f"Unknown golden mode: {mode}")

    batch_size = sampling_spec().batch_size
    if batch_size % len(base_cases) != 0:
        raise ValueError("The full-Hero batch size must be a multiple of the logical case count")
    repeats = batch_size // len(base_cases)
    cases = tuple(
        GoldenCaseSpec(id=f"{case_id}-repeat-{repeat}", source_prompt_id=source, valid_length=length)
        for repeat in range(repeats)
        for case_id, source, length in base_cases
    )
    return GoldenSpec(
        release=release,
        mode=mode,
        batch_size=batch_size,
        cases=cases,
        tokenizer=TOKENIZER,
        tokenizer_revision=TOKENIZER_REVISION,
        training_model=training_model,
        model=inference_model,
    )


def pinned_request(mode: str, revision: str) -> GoldenRequest:
    configure_coreweave_s3()
    discovered = discover_requests(CHECKPOINT_RUNS, sampling_spec(), revision, target_cluster=TARGET_CLUSTER)
    selected = [request.checkpoint for request in discovered if request.checkpoint.uri == SELECTED_CHECKPOINT_URI]
    if len(selected) != 1:
        raise ValueError(f"Pinned permanent checkpoint was not discovered exactly once: {SELECTED_CHECKPOINT_URI}")
    return GoldenRequest(
        checkpoint=selected[0],
        spec=golden_spec(mode),
        source_revision=revision,
        target_cluster=TARGET_CLUSTER,
    )


def _source_prompts() -> dict[str, Prompt]:
    path = Path(__file__).with_name("vibe_check") / "prompts.json"
    prompts = tuple(Prompt.model_validate(row) for row in json.loads(path.read_text()))
    return {prompt.id: prompt for prompt in prompts}


def _fixed_tokens(tokenizer, case: GoldenCaseSpec, prompts: dict[str, Prompt]) -> list[int]:
    source = prompts[case.source_prompt_id]
    if source.expected is None:
        raise ValueError(f"Source prompt has no fixed continuation: {case.source_prompt_id}")
    text = source.text + source.expected
    first = tokenizer.encode(text, add_special_tokens=True)
    continuation = tokenizer.encode("\n" + text, add_special_tokens=False)
    if not first or not continuation:
        raise ValueError(f"Source prompt produced no tokens: {case.source_prompt_id}")
    tokens = list(first)
    while len(tokens) < case.valid_length:
        tokens.extend(continuation)
    return tokens[: case.valid_length]


def _score_positions(valid_length: int, sliding_window: int) -> tuple[int, ...]:
    targets = set(range(1, min(valid_length, 9)))
    targets.update(range(max(1, valid_length - 4), valid_length))
    targets.update(position for position in range(sliding_window - 1, sliding_window + 2) if position < valid_length)
    targets.update(position for position in (4094, 4095) if position < valid_length)
    return tuple(sorted(targets))


def build_inputs(request: GoldenRequest, tokenizer) -> tuple[dict[str, np.ndarray], list[GoldenCaseManifest]]:
    prompts = _source_prompts()
    sequence = max(case.valid_length for case in request.spec.cases)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    tokens = np.full((request.spec.batch_size, sequence), eos_token_id, dtype=np.int32)
    validity = np.zeros_like(tokens, dtype=np.bool_)
    segments = np.full_like(tokens, -1, dtype=np.int32)
    score_mask = np.zeros_like(tokens, dtype=np.bool_)
    cases: list[GoldenCaseManifest] = []
    model = draccus.decode(hero_recipe.GrugModelConfig, request.spec.model)
    for row, case in enumerate(request.spec.cases):
        row_tokens = _fixed_tokens(tokenizer, case, prompts)
        tokens[row, : case.valid_length] = row_tokens
        validity[row, : case.valid_length] = True
        segments[row, : case.valid_length] = 0
        scored_targets = _score_positions(case.valid_length, model.sliding_window)
        score_mask[row, list(scored_targets)] = True
        cases.append(
            {
                "id": case.id,
                "row": row,
                "source_prompt_id": case.source_prompt_id,
                "valid_length": case.valid_length,
                "scored_target_positions": list(scored_targets),
                "construction": (
                    "tokenize prompt plus fixed expected continuation, then repeat it with a newline and truncate"
                ),
            }
        )
    positions = np.broadcast_to(np.arange(sequence, dtype=np.int32), tokens.shape).copy()
    score_indices = np.argwhere(score_mask)

    first_repeat_rows = [row for row, case in enumerate(request.spec.cases) if case.id.endswith("repeat-0")]
    full_rows = []
    for row in first_repeat_rows:
        length = request.spec.cases[row].valid_length
        full_rows.append((row, length - 2))
        if length > model.sliding_window + 1:
            full_rows.extend((row, position) for position in range(model.sliding_window - 2, model.sliding_window + 1))
    full_rows = list(dict.fromkeys(full_rows))
    arrays = {
        "tokens": tokens,
        "token_validity": validity,
        "segment_ids": segments,
        "positions": positions,
        "valid_lengths": validity.sum(axis=1).astype(np.int32),
        "score_mask": score_mask,
        "score_case_indices": score_indices[:, 0].astype(np.int32),
        "prediction_positions": (score_indices[:, 1] - 1).astype(np.int32),
        "target_token_ids": tokens[score_indices[:, 0], score_indices[:, 1]],
        "full_logit_case_indices": np.asarray([row for row, _ in full_rows], dtype=np.int32),
        "full_logit_prediction_positions": np.asarray([position for _, position in full_rows], dtype=np.int32),
    }
    return arrays, cases


def _project_forward(
    model,
    hidden: jax.Array,
    selection: ForwardSelection,
) -> ForwardValues:
    selected = hidden.at[selection.score_cases, selection.prediction_positions].get(out_sharding=P())
    targets = jax.sharding.reshard(selection.targets, P())

    def project_score(inputs):
        state, target = inputs
        logits = jnp.einsum("h,hv->v", state, model.output_proj, preferred_element_type=jnp.float32)
        logprobs = jax.nn.log_softmax(jax.sharding.reshard(logits, P()), axis=-1)
        top_values, top_ids = jax.lax.top_k(logprobs, TOP_TOKEN_COUNT)
        return logprobs[target], top_ids, top_values

    target_logprobs, top_token_ids, top_logprobs = jax.lax.map(project_score, (selected, targets))
    full_hidden = hidden.at[selection.full_cases, selection.full_positions].get(out_sharding=P())
    full_logits = jnp.einsum(
        "rh,hv->rv", full_hidden, model.output_proj, preferred_element_type=jnp.float32, out_sharding=P()
    )
    return ForwardValues(target_logprobs, top_token_ids, top_logprobs, full_logits)


@eqx.filter_jit
def ordinary_forward(
    model,
    tokens: jax.Array,
    segment_ids: jax.Array,
    selection: ForwardSelection,
) -> ForwardValues:
    mask = AttentionMask.causal().with_segment_ids(segment_ids)
    hidden, _ = model(tokens, mask=mask)
    return _project_forward(model, hidden, selection)


@eqx.filter_jit
def traced_forward(
    model,
    tokens: jax.Array,
    segment_ids: jax.Array,
    selection: ForwardSelection,
) -> TracedValues:
    mask = AttentionMask.causal().with_segment_ids(segment_ids)
    hidden, metrics = model(tokens, mask=mask, trace_routes=True)
    forward = _project_forward(model, hidden, selection)
    # Process zero writes the bundle. Replication makes each global trace fully addressable there.
    return TracedValues(
        forward=forward,
        route_expert_ids=jax.sharding.reshard(metrics["route_expert_ids"], P()),
        route_combine_weights=jax.sharding.reshard(metrics["route_combine_weights"], P()),
        route_cutoff_gaps=jax.sharding.reshard(metrics["route_cutoff_gaps"], P()),
    )


def _numeric_diagnostic(first: ForwardValues, second: ForwardValues) -> NativeDiagnostic:
    return NativeDiagnostic(
        target_logprob_max_abs=float(np.max(np.abs(first.target_logprobs - second.target_logprobs))),
        top_logprob_max_abs=float(np.max(np.abs(first.top_logprobs - second.top_logprobs))),
        full_logit_max_abs=float(np.max(np.abs(first.full_logits - second.full_logits))),
        top_token_id_changes=int(np.count_nonzero(first.top_token_ids != second.top_token_ids)),
    )


def _validate_native_diagnostic(name: str, diagnostic: NativeDiagnostic) -> None:
    numeric_changes = (
        diagnostic.target_logprob_max_abs,
        diagnostic.top_logprob_max_abs,
        diagnostic.full_logit_max_abs,
    )
    if diagnostic.top_token_id_changes or any(change > NATIVE_OUTPUT_BOUND for change in numeric_changes):
        raise ValueError(f"{name} changed native outputs beyond the fixed diagnostic bound: {diagnostic}")


def _runtime_versions() -> dict[str, str]:
    packages = ("jax", "jaxlib", "equinox", "flash-attn-4", "numpy", "transformers")
    return {package: importlib.metadata.version(package) for package in packages}


def _validate_authoritative_weights(model, weights_key: str) -> tuple[str, ...]:
    if weights_key not in AUTHORITATIVE_WEIGHT_KEYS:
        raise ValueError(f"Unsupported checkpoint weight tree: {weights_key}")
    dtypes = tuple(sorted({str(leaf.dtype) for leaf in jax.tree.leaves(model) if eqx.is_inexact_array(leaf)}))
    if dtypes != ("float32",):
        raise ValueError(f"Authoritative checkpoint weights must be FP32, found {dtypes}")
    return dtypes


def _upload_bundle(local: Path, remote_root: str) -> None:
    root = StoragePath(remote_root)
    manifest_target = conditional_object(str(root / MANIFEST_FILENAME))
    if manifest_target.version() is not None or (root / ARRAYS_FILENAME).exists():
        raise FileExistsError(f"Refusing to overwrite an existing or partial golden bundle: {remote_root}")
    with (local / ARRAYS_FILENAME).open("rb") as source, (root / ARRAYS_FILENAME).open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    manifest_target.write((local / MANIFEST_FILENAME).read_bytes(), expected_version=None)


def produce(request: GoldenRequest, store_root: str) -> None:
    timer = Timer()
    DistributedConfig().initialize()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    configure_coreweave_s3()
    if jax.default_backend() != "gpu" or jax.device_count() != request.spec.batch_size:
        raise ValueError("Full-Hero golden production requires one GB200 per batch row")
    logger.info("Load tokenizer %s at %s", request.spec.tokenizer, request.spec.tokenizer_revision)
    tokenizer = AutoTokenizer.from_pretrained(request.spec.tokenizer, revision=request.spec.tokenizer_revision)
    input_arrays, cases = build_inputs(request, tokenizer)
    model_config = draccus.decode(hero_recipe.GrugModelConfig, request.spec.model)
    if input_arrays["tokens"].shape[1] > model_config.max_seq_len:
        raise ValueError("Required golden sequence exceeds the unchanged Hero model context")

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        with log_time("Checkpoint restore"):
            restored = restore_model_state(request, mesh)
        weight_dtypes = _validate_authoritative_weights(restored.model, restored.weights_key)
        pending_qb_betas = np.asarray(jax.sharding.reshard(restored.pending_qb_betas, P()))
        with log_time("BF16 weight conversion"):
            model = COMPUTE_POLICY.cast_to_compute(restored.model)
            jax.block_until_ready(model)
        effective_router_bias = np.asarray(
            jax.sharding.reshard(model.stacked_blocks.stacked.mlp.router_bias.astype(jnp.float32), P())
        )

        batch_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))
        replicated = NamedSharding(mesh, P())

        def batch_array(value: np.ndarray) -> jax.Array:
            return jax.make_array_from_callback(value.shape, batch_sharding, lambda index: value[index])

        def replicated_array(value: np.ndarray) -> jax.Array:
            return jax.device_put(value, replicated)

        selection = ForwardSelection(
            score_cases=replicated_array(input_arrays["score_case_indices"]),
            prediction_positions=replicated_array(input_arrays["prediction_positions"]),
            targets=replicated_array(input_arrays["target_token_ids"]),
            full_cases=replicated_array(input_arrays["full_logit_case_indices"]),
            full_positions=replicated_array(input_arrays["full_logit_prediction_positions"]),
        )
        args = (
            model,
            batch_array(input_arrays["tokens"]),
            batch_array(input_arrays["segment_ids"]),
            selection,
        )
        with log_time("Ordinary forward and compilation"):
            first = ordinary_forward(*args)
            jax.block_until_ready(first)
        with log_time("Ordinary forward repeat"):
            second = ordinary_forward(*args)
            jax.block_until_ready(second)
        with log_time("Traced forward and compilation"):
            traced = traced_forward(*args)
            jax.block_until_ready(traced)

    if jax.process_index() != 0:
        return
    first_host = jax.tree.map(np.asarray, first)
    second_host = jax.tree.map(np.asarray, second)
    traced_host = jax.tree.map(np.asarray, traced)
    repeatability = _numeric_diagnostic(first_host, second_host)
    instrumentation = _numeric_diagnostic(first_host, traced_host.forward)
    _validate_native_diagnostic("ordinary repeat", repeatability)
    _validate_native_diagnostic("traced versus ordinary", instrumentation)

    arrays = {
        **input_arrays,
        "target_logprobs": traced_host.forward.target_logprobs.astype(np.float32),
        "top_token_ids": traced_host.forward.top_token_ids.astype(np.int32),
        "top_logprobs": traced_host.forward.top_logprobs.astype(np.float32),
        "full_logits": traced_host.forward.full_logits.astype(np.float32),
        "route_expert_ids": traced_host.route_expert_ids.astype(np.int32),
        "route_combine_weights": traced_host.route_combine_weights.astype(np.float32),
        "route_cutoff_gaps": traced_host.route_cutoff_gaps.astype(np.float32),
        "pending_qb_betas": pending_qb_betas.astype(np.float32),
        "effective_router_bias": effective_router_bias.astype(np.float32),
    }
    remote_root = prefix_join(store_root, request.bundle_id)
    manifest = {
        "bundle_id": request.bundle_id,
        "asset_root": remote_root,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": {
            **request.checkpoint.model_dump(),
            "writer_revision": CHECKPOINT_WRITER_REVISION,
            "writer_bundle_id": CHECKPOINT_WRITER_BUNDLE_ID,
            "writer_evidence_url": CHECKPOINT_WRITER_EVIDENCE_URL,
            "writer_run_version": "2026.08.19.2",
            "retention": "permanent checkpoint metadata has is_temporary=false",
            "restored_weights": restored.weights_key,
            "restored_weight_dtypes": list(weight_dtypes),
            "master_parameter_layout": (
                "Hero used MasterParamMode.DEVICE: the authoritative FP32 master copy is stored directly under "
                "params; this checkpoint has no separate master_params tree"
            ),
            "checkpoint_wrapped": restored.wrapped,
            "weight_conversion": "authoritative FP32 params cast once to BF16 compute parameters",
            "pending_query_bias": "pending_qb_betas applied once with the native restore rule, then held fixed",
            "training_model": request.spec.training_model,
            "training_model_digest": CHECKPOINT_WRITER_MODEL_DIGEST,
        },
        "producer": {
            "revision": request.source_revision,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _runtime_versions(),
            },
            "compute_policy": str(COMPUTE_POLICY),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "target_cluster": request.target_cluster,
            "accelerator_kind": "GPU",
            "accelerator_variant": "GB200",
            "task_id": os.environ.get("IRIS_TASK_ID"),
            "process_count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "global_device_count": jax.device_count(),
            "mesh": dict(mesh.shape),
            "regeneration_command": (
                "python -m experiments.grug.moe_hero_ep.ops.forward_goldens "
                f"--request request.json --store-root {store_root}"
            ),
            "request": request.model_dump(mode="json"),
        },
        "model": {
            "resolved_config": request.spec.model,
            "intentional_inference_overrides": {
                "moe_implementation": {
                    "training": hero_recipe.RAGGED_MOE_IMPLEMENTATION,
                    "golden": DEFAULT_DROPLESS_MOE_IMPLEMENTATION,
                    "reason": "dropless reference computes every selected expert contribution",
                }
            },
            "num_layers": model_config.num_layers,
            "num_experts": model_config.num_experts,
            "experts_per_token": model_config.num_experts_per_token,
            "configuration_cross_check": (
                "checkpoint-writing and producer training configs have the pinned identical digest; "
                "the inference config differs only by the declared dropless MoE implementation"
            ),
        },
        "tokenizer": {"name": request.spec.tokenizer, "revision": request.spec.tokenizer_revision},
        "cases": cases,
        "mask": {
            "kind": "causal_segment_ids",
            "segment_ids_array": "segment_ids",
            "padding_segment_id": -1,
            "token_validity_array": "token_validity",
            "score_mask_array": "score_mask",
            "score_mask_is_attention_mask": False,
        },
        "indexing": {
            "token": "tokens[case, token_position] with zero-based positions",
            "score": (
                "target_token_ids[i] is predicted from hidden state " "[score_case_indices[i], prediction_positions[i]]"
            ),
            "full_logit": "full_logits[row, vocabulary_token_id]",
            "route": "route_expert_ids[layer, case, token_position, ordered_route_slot]",
        },
        "scoring": {
            "normalization": "float32 log_softmax over the full 128256-token vocabulary",
            "top_k": TOP_TOKEN_COUNT,
            "full_logits": "float32 pre-softmax rows",
            "cross_backend_tolerances": "not yet qualified; consumers must provide explicit provisional bounds",
        },
        "routing": {
            "expert_ids": "ordered global expert IDs selected from biased router logits",
            "combine_weights": "actual BF16 expert-combine values, widened to float32 for storage",
            "cutoff_gap": "float32 biased Kth score minus biased (K+1)th score",
            "padding": {"expert_id": -1, "combine_weight": 0, "cutoff_gap": 0},
        },
        "native_validation": {
            "gate": f"max absolute output change <= {NATIVE_OUTPUT_BOUND:g} and no top-token ID changes",
            "ordinary_repeat": repeatability.as_dict(),
            "traced_versus_ordinary": instrumentation.as_dict(),
        },
        "optional_context_diagnostics": {
            "8192": (
                "skipped: the fixed checkpoint-writing model config has max_seq_len=4096; running 8192 "
                "would require a model-semantic configuration change"
            ),
            "16384": (
                "skipped: the fixed checkpoint-writing model config has max_seq_len=4096; running 16384 "
                "would require a model-semantic configuration change"
            ),
        },
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
    }
    with TemporaryDirectory(prefix="hero-forward-goldens-") as directory:
        local = Path(directory) / request.bundle_id
        with log_time("Local bundle validation and compression"):
            write_bundle(local, manifest, arrays)
        with log_time("CW object-storage upload"):
            _upload_bundle(local, remote_root)
    logger.info("Golden bundle completed in %.1f seconds: %s", timer.elapsed_seconds(), remote_root)


def submit(mode: str, store_root: str) -> None:
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], check=True, stdout=subprocess.DEVNULL)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    request = pinned_request(mode, revision)
    configure_coreweave_s3()
    remote_root = prefix_join(store_root, request.bundle_id)
    if conditional_object(prefix_join(remote_root, MANIFEST_FILENAME)).version() is not None:
        raise FileExistsError(f"Golden bundle already exists: {remote_root}")
    name = f"hero-forward-{mode}-{digest(request.model_dump(mode='json'))[:12]}"
    with connect_controller(cluster_name=CONTROLLER_CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            jobs = IrisSamplingJobs(
                client,
                endpoint,
                Path.cwd(),
                store_root,
                sampling_resources(),
                SAMPLING_GPUS_PER_NODE,
                sampler_module="experiments.grug.moe_hero_ep.ops.forward_goldens",
                user=JOB_USER,
                environment_overrides={"XLA_FLAGS": DETERMINISTIC_XLA_FLAGS},
            )
            jobs.submit(request, name, priority_band_value("interactive"))
    print(f"Submitted /{JOB_USER}/{name} for {remote_root}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        parser = argparse.ArgumentParser(description="Submit the full-Hero golden producer")
        parser.add_argument("submit")
        parser.add_argument("--mode", choices=GOLDEN_MODES, required=True)
        parser.add_argument("--store-root", default=STORE_ROOT)
        args = parser.parse_args()
        submit(args.mode, args.store_root)
        return
    configure_logging(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--store-root", required=True)
    args = parser.parse_args()
    produce(GoldenRequest.model_validate_json(args.request.read_bytes()), args.store_root)


if __name__ == "__main__":
    main()
