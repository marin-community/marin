# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture Snowball native-Levanter next-token observations on GPU or TPU.

This module is both the non-pytest golden/measurement entry point and the shared
implementation used by the standing-cluster gate. It intentionally knows nothing
about Iris submission so persistent dev accelerators can run it directly.
"""

import argparse
import dataclasses
import hashlib
import json
import uuid
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax.partitioning import set_mesh
from huggingface_hub import snapshot_download
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from rigging.filesystem import StoragePath

from tests.cluster.vllm import backend_parity as backend_parity_module
from tests.cluster.vllm import snowball as snowball_module
from tests.cluster.vllm import snowball_checkpoint as snowball_checkpoint_module
from tests.cluster.vllm.backend_parity import (
    GoldenTokenObservation,
    NextTokenObservation,
    ObservationReport,
    ParityContract,
    RunProvenance,
    TokenScore,
    assert_report_matches_contract,
    source_digest,
)
from tests.cluster.vllm.snowball import (
    PROMPT_FIXTURE_SHA256,
    REPRESENTATIVE_GOLDEN_SHA256,
    SNOWBALL_NATIVE_GPU,
    SNOWBALL_NATIVE_TPU,
    TOKENIZER_FILE_PATTERNS,
    NativeLevanterCell,
    PromptBatch,
    RepresentativeGolden,
    read_prompt_fixture,
    read_representative_goldens,
)
from tests.cluster.vllm.snowball_checkpoint import (
    VendoredTransformer,
    apply_pending_qb_betas,
    decode_vendored_config,
    load_checkpoint,
    logical_array_digest,
    prepare_bf16_parameters,
    read_executor_info,
)

OBSERVATION_TOP_K = 50


class ScoredTokenBatch(NamedTuple):
    top_logprobs: jax.Array
    top_token_ids: jax.Array
    golden_logprobs: jax.Array
    golden_ranks: jax.Array
    capacity_overflow: jax.Array
    has_nonfinite: jax.Array


class CanonicalTop25Observations(NamedTuple):
    golden_logprobs: np.ndarray
    golden_ranks: np.ndarray
    greedy_token_ids: np.ndarray


def _last_token_logprobs(
    model: VendoredTransformer,
    hidden: jax.Array,
    last_token_indices: jax.Array,
) -> jax.Array:
    last_hidden = hidden.at[jnp.arange(hidden.shape[0]), last_token_indices].get(
        out_sharding=P(("replica_dcn", "data", "expert"))
    )
    logits = jnp.einsum(
        "bh,hv->bv",
        last_hidden,
        model.output_proj,
        out_sharding=P(("replica_dcn", "data", "expert")),
    )
    assert logits.dtype == jnp.bfloat16
    return jax.nn.log_softmax(logits.astype(jnp.float32))


@eqx.filter_jit
def score_next_token(
    model: VendoredTransformer,
    token_ids: jax.Array,
    last_token_indices: jax.Array,
    golden_token_ids: jax.Array,
) -> ScoredTokenBatch:
    """Score only each row's final real token while retaining parity diagnostics."""
    hidden, router_metrics = model(token_ids)
    logprobs = _last_token_logprobs(model, hidden, last_token_indices)
    top_logprobs, top_token_ids = jax.lax.top_k(logprobs, OBSERVATION_TOP_K)
    golden_logprobs = jnp.take_along_axis(logprobs, golden_token_ids, axis=-1)
    golden_ranks = jnp.sum(
        logprobs[:, None, :] > golden_logprobs[:, :, None],
        axis=-1,
        dtype=jnp.int32,
    )
    capacity_overflow = router_metrics["capacity_overflow_per_layer"]
    has_nonfinite = jnp.any(~jnp.isfinite(logprobs), axis=-1)
    return ScoredTokenBatch(
        top_logprobs=top_logprobs,
        top_token_ids=top_token_ids,
        golden_logprobs=golden_logprobs,
        golden_ranks=golden_ranks,
        capacity_overflow=capacity_overflow,
        has_nonfinite=has_nonfinite,
    )


@eqx.filter_jit
def score_gpu_canonical_top25(
    model: VendoredTransformer,
    pending_qb_betas: jax.Array,
    token_ids: jax.Array,
    last_token_indices: jax.Array,
    policy: jmp.Policy,
) -> tuple[jax.Array, jax.Array]:
    """Preserve the exact canonical GPU scoring graph for its observations."""
    model = apply_pending_qb_betas(model, pending_qb_betas)
    model = policy.cast_to_compute(model)
    hidden, _ = model(token_ids)
    logprobs = _last_token_logprobs(model, hidden, last_token_indices)
    return jax.lax.top_k(logprobs, snowball_module.TOP_K)


def apply_canonical_top25(
    golden_token_ids: np.ndarray,
    diagnostic_logprobs: np.ndarray,
    diagnostic_ranks: np.ndarray,
    canonical_logprobs: np.ndarray,
    canonical_token_ids: np.ndarray,
) -> CanonicalTop25Observations:
    """Use baseline-compatible GPU scores for canonical tokens, retaining top-50 diagnostics."""
    canonical_golden_logprobs = diagnostic_logprobs.copy()
    canonical_golden_ranks = diagnostic_ranks.copy()
    for row in range(golden_token_ids.shape[0]):
        canonical_by_id = {
            int(token_id): (float(logprob), rank)
            for rank, (logprob, token_id) in enumerate(
                zip(canonical_logprobs[row], canonical_token_ids[row], strict=True)
            )
        }
        for column, token_id in enumerate(golden_token_ids[row]):
            canonical = canonical_by_id.get(int(token_id))
            if canonical is not None:
                canonical_golden_logprobs[row, column] = canonical[0]
                canonical_golden_ranks[row, column] = canonical[1]
    return CanonicalTop25Observations(
        golden_logprobs=canonical_golden_logprobs,
        golden_ranks=canonical_golden_ranks,
        greedy_token_ids=canonical_token_ids[:, 0],
    )


def build_batch_observations(
    batch: PromptBatch,
    *,
    repeat_index: int,
    top_logprobs: np.ndarray,
    top_token_ids: np.ndarray,
    golden_logprobs: np.ndarray,
    golden_ranks: np.ndarray,
    capacity_overflow: np.ndarray,
    has_nonfinite: np.ndarray,
    greedy_token_ids: np.ndarray | None = None,
) -> tuple[NextTokenObservation, ...]:
    """Convert device arrays into the versioned backend-neutral schema."""
    rows = len(batch.cases)
    expected_row_shapes = {
        "top_logprobs": top_logprobs.shape[0],
        "top_token_ids": top_token_ids.shape[0],
        "golden_logprobs": golden_logprobs.shape[0],
        "golden_ranks": golden_ranks.shape[0],
        "has_nonfinite": has_nonfinite.shape[0],
    }
    if any(actual != rows for actual in expected_row_shapes.values()):
        raise ValueError(f"Batch rows do not match {rows} cases: {expected_row_shapes}")

    overflow = tuple(float(value) for value in capacity_overflow)
    observations = []
    for row, case in enumerate(batch.cases):
        if golden_logprobs.shape[1] != len(case.top_logprobs) or golden_ranks.shape[1] != len(case.top_logprobs):
            raise ValueError(f"Golden diagnostic width does not match {case.id}")
        observations.append(
            NextTokenObservation(
                case_id=case.id,
                bucket_max_tokens=batch.max_tokens,
                repeat_index=repeat_index,
                backend_index=row,
                greedy_token_id=int(top_token_ids[row, 0] if greedy_token_ids is None else greedy_token_ids[row]),
                top_logprobs=tuple(
                    TokenScore(token_id=int(token_id), logprob=float(logprob))
                    for logprob, token_id in zip(top_logprobs[row], top_token_ids[row], strict=True)
                ),
                golden_tokens=tuple(
                    GoldenTokenObservation(
                        token_id=expected.token_id,
                        logprob=float(logprob),
                        rank=int(rank),
                    )
                    for expected, logprob, rank in zip(
                        case.top_logprobs,
                        golden_logprobs[row],
                        golden_ranks[row],
                        strict=True,
                    )
                ),
                capacity_overflow=overflow,
                has_nonfinite=bool(has_nonfinite[row]),
            )
        )
    return tuple(observations)


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# By-value remote serialization preserves this value without reopening controller
# worktree paths on the worker. A directly imported CLI computes it on that host.
CAPTURE_SOURCE_DIGEST = source_digest(
    backend_parity_module.__file__,
    snowball_module.__file__,
    snowball_checkpoint_module.__file__,
    __file__,
)


def _parameter_dtype(cell: NativeLevanterCell):
    if cell.parameter_dtype is None:
        return None
    if cell.parameter_dtype == "bfloat16":
        return jnp.bfloat16
    raise ValueError(f"Unsupported parameter dtype {cell.parameter_dtype!r}")


def capture_native_levanter(
    cell: NativeLevanterCell,
    *,
    same_process_repeats: int = 1,
    batch_indices: tuple[int, ...] | None = None,
    goldens: tuple[RepresentativeGolden, ...] | None = None,
) -> ObservationReport:
    """Load one regional checkpoint and capture selected batches repeatedly."""
    if same_process_repeats <= 0:
        raise ValueError("same_process_repeats must be positive")
    if jax.default_backend() != cell.location.name:
        raise RuntimeError(f"Expected {cell.location.name} backend, found {jax.default_backend()}")

    goldens = read_representative_goldens() if goldens is None else goldens
    prompt_fixture = read_prompt_fixture(goldens, fixture_uri=cell.location.prompt_fixture_uri)
    executor_info = read_executor_info(cell.location)
    if executor_info["config"]["data"]["tokenizer"] != prompt_fixture.tokenizer:
        raise ValueError("Checkpoint and prompt fixture use different tokenizers")
    inference_model_config = dataclasses.replace(
        decode_vendored_config(executor_info),
        moe_implementation=cell.requested_moe,
        attention_implementation=cell.requested_attention,
    )
    gpu_policy = jmp.get_policy(executor_info["config"]["mp"]) if cell.location.name == "gpu" else None
    tokenizer = load_tokenizer(
        snapshot_download(
            prompt_fixture.tokenizer,
            revision=prompt_fixture.tokenizer_revision,
            allow_patterns=list(TOKENIZER_FILE_PATTERNS),
        )
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Snowball tokenizer must define eos_token_id")

    mesh = compact_grug_mesh()
    if mesh.shape.get("expert", 1) != 1:
        raise ValueError(f"Native parity expects EP1, got mesh {mesh.shape}")
    selected_indices = tuple(range(len(prompt_fixture.batches))) if batch_indices is None else batch_indices
    if len(set(selected_indices)) != len(selected_indices) or any(
        index < 0 or index >= len(prompt_fixture.batches) for index in selected_indices
    ):
        raise ValueError(f"Invalid batch indices {selected_indices}")

    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(
            inference_model_config,
            mesh,
            location=cell.location,
            parameter_dtype=_parameter_dtype(cell),
        )
        canonical_gpu_outputs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        if gpu_policy is not None:
            for repeat_index in range(same_process_repeats):
                for batch_index in selected_indices:
                    batch = prompt_fixture.batches[batch_index]
                    token_ids, last_token_indices = snowball_module.pad_prompt_batch(batch, tokenizer.eos_token_id)
                    canonical_outputs = score_gpu_canonical_top25(
                        params,
                        pending_qb_betas,
                        jnp.asarray(token_ids),
                        jnp.asarray(last_token_indices),
                        gpu_policy,
                    )
                    canonical_gpu_outputs[repeat_index, batch_index] = tuple(
                        np.asarray(jax.device_get(output)) for output in canonical_outputs
                    )

        model = prepare_bf16_parameters(params, pending_qb_betas)
        del params, pending_qb_betas
        parameter_dtypes = {leaf.dtype for leaf in jax.tree.leaves(model) if eqx.is_inexact_array(leaf)}
        if parameter_dtypes != {jnp.dtype(jnp.bfloat16)}:
            raise ValueError(f"Expected only BF16 parameters, got {parameter_dtypes}")
        parameter_digest = logical_array_digest(model)

        observations = []
        for repeat_index in range(same_process_repeats):
            for batch_index in selected_indices:
                batch = prompt_fixture.batches[batch_index]
                token_ids, last_token_indices = snowball_module.pad_prompt_batch(batch, tokenizer.eos_token_id)
                golden_token_ids = np.asarray(
                    [[score.token_id for score in case.top_logprobs] for case in batch.cases],
                    dtype=np.int32,
                )
                outputs = score_next_token(
                    model,
                    jnp.asarray(token_ids),
                    jnp.asarray(last_token_indices),
                    jnp.asarray(golden_token_ids),
                )
                device_outputs = jax.tree.map(lambda output: np.asarray(jax.device_get(output)), outputs)
                golden_logprobs = device_outputs.golden_logprobs
                golden_ranks = device_outputs.golden_ranks
                greedy_token_ids = None
                if gpu_policy is not None:
                    canonical_logprobs, canonical_token_ids = canonical_gpu_outputs[repeat_index, batch_index]
                    canonical = apply_canonical_top25(
                        golden_token_ids,
                        golden_logprobs,
                        golden_ranks,
                        canonical_logprobs,
                        canonical_token_ids,
                    )
                    golden_logprobs = canonical.golden_logprobs
                    golden_ranks = canonical.golden_ranks
                    greedy_token_ids = canonical.greedy_token_ids
                observations.extend(
                    build_batch_observations(
                        batch,
                        repeat_index=repeat_index,
                        top_logprobs=device_outputs.top_logprobs,
                        top_token_ids=device_outputs.top_token_ids,
                        golden_logprobs=golden_logprobs,
                        golden_ranks=golden_ranks,
                        capacity_overflow=device_outputs.capacity_overflow,
                        has_nonfinite=device_outputs.has_nonfinite,
                        greedy_token_ids=greedy_token_ids,
                    )
                )

    device_kind = ",".join(sorted({device.device_kind for device in jax.devices()}))
    provenance = RunProvenance(
        backend="levanter-native",
        platform=cell.location.name,
        process_id=uuid.uuid4().hex,
        code_digest=CAPTURE_SOURCE_DIGEST,
        parameter_digest=parameter_digest,
        model_config_digest=_sha256_json(executor_info["config"]["model"]),
        prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
        requested_attention=cell.requested_attention,
        effective_attention=cell.effective_attention,
        requested_moe=cell.requested_moe,
        effective_moe=cell.effective_moe,
        mesh_shape=tuple((name, int(size)) for name, size in mesh.shape.items()),
        device_kind=device_kind,
        golden_digest=REPRESENTATIVE_GOLDEN_SHA256,
    )
    return ObservationReport(provenance=provenance, observations=tuple(observations))


def assert_report_sane(report: ObservationReport) -> None:
    for observation in report.observations:
        if observation.has_nonfinite:
            raise AssertionError(f"Nonfinite logprobs for {observation.case_id}")
        if any(value != 0.0 for value in observation.capacity_overflow):
            raise AssertionError(f"MoE capacity overflow for {observation.case_id}: {observation.capacity_overflow}")


def assert_native_tpu_contract(
    report: ObservationReport,
    goldens: tuple[RepresentativeGolden, ...],
    contract: ParityContract,
) -> None:
    """Apply the frozen native-TPU provenance and numerical holdout gate."""
    expected_by_id = {golden.id: golden.top_logprobs for golden in goldens}
    assert_report_matches_contract(report, expected_by_id, contract)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("gpu", "tpu"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--same-process-repeats", type=int, default=1)
    parser.add_argument("--batch-index", action="append", type=int)
    args = parser.parse_args()

    cell = SNOWBALL_NATIVE_GPU if args.platform == "gpu" else SNOWBALL_NATIVE_TPU
    report = capture_native_levanter(
        cell,
        same_process_repeats=args.same_process_repeats,
        batch_indices=None if args.batch_index is None else tuple(args.batch_index),
    )
    StoragePath(args.output).write_bytes(report.to_json_bytes())
    assert_report_sane(report)


if __name__ == "__main__":
    main()
