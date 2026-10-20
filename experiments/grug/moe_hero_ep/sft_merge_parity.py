# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare pinned compact SFT state with its ordinary candidate-D export.

Run ``python -m experiments.grug.moe_hero_ep.sft_merge_parity request.json output``
inside an explicitly provisioned evaluation job. This command submits no jobs.
It writes the measured report even when declared parity tolerances fail, then
returns a nonzero CLI status. No serving implementation is involved.
"""

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from marin.testing.inference.backend_parity import NextTokenParity, TokenScore, parity_from_logprob_row
from pydantic import Field
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.averaging import _DONORS, read_prepared_provenance
from experiments.grug.moe_hero_ep.format_measure import NativePtRequest, PtPanel, read_pt_panel
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, Record, digest
from experiments.grug.moe_hero_ep.sft import (
    BaseReference,
    SftConfig,
    initialize_from_checkpoint,
    restore_sft_checkpoint,
    sft_logits,
    sft_manifest,
)
from experiments.grug.moe_hero_ep.train import (
    RAGGED_MOE_IMPLEMENTATION,
    _apply_hero_ep_runtime_defaults,
    verify_ragged_pjrt,
)
from experiments.grug.moe_hero_ep.weights import cast_weights, restore_weights

logger = logging.getLogger(__name__)

_BATCH_AXES = ("replica_dcn", "data", "expert")


class MergeProbe(Record):
    row_index: int = Field(ge=0)
    prediction_position: int = Field(ge=0)


class MergeParityRequest(Record):
    reference: BaseReference
    sft: SftConfig
    model: dict
    compact_checkpoint: str
    compact_metadata_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    merged_checkpoint: Checkpoint
    panel_path: str
    panel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    probes: tuple[MergeProbe, ...] = Field(min_length=1, max_length=32)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    expert_axis_size: int = Field(ge=1)
    replica_axis_size: int = Field(ge=1)
    max_absolute_logit_error: float = Field(ge=0, allow_inf_nan=False)
    max_absolute_probability_error: float = Field(ge=0, le=1, allow_inf_nan=False)
    max_greedy_disagreements: int = Field(ge=0)


class MergeMeasurement(Record):
    probe: MergeProbe
    arrays_sha256: str
    target_token_id: int
    compact_greedy_id: int
    merged_greedy_id: int
    max_absolute_logit_error: float
    max_absolute_probability_error: float
    compact_target_logprob: float
    merged_target_logprob: float
    compact_top_scores: tuple[TokenScore, ...]
    merged_top_scores: tuple[TokenScore, ...]
    winner_contract: bool
    next_token_parity: NextTokenParity


def _metadata(path: str, expected_digest: str) -> dict:
    value = json.loads((StoragePath(path) / "metadata.json").read_text())
    if value.get("is_temporary") is not False or digest(value) != expected_digest:
        raise ValueError("Merge parity requires exact permanent checkpoint metadata")
    return value


def validate_merge_request(request: MergeParityRequest) -> tuple[GrugModelConfig, PtPanel, dict]:
    """Validate C/compact/D lineage, configuration and real token rows before array I/O."""
    config = draccus.decode(GrugModelConfig, request.model)
    if config.moe_implementation in ("scatter", "sonic", "sonic_cute") and request.expert_axis_size != 1:
        raise ValueError("Local merge-parity backends require a collapsed expert axis")
    if config.num_experts % request.expert_axis_size:
        raise ValueError("Merge-parity expert mesh must divide the model's expert count")
    reference = request.reference
    provenance = read_prepared_provenance(reference.checkpoint, reference.metadata_digest)
    if provenance["candidate"] != "C" or provenance["donors"] != [list(row) for row in _DONORS]:
        raise ValueError("Merge parity requires the donor-initialized candidate C")
    saved_model = draccus.encode(draccus.decode(GrugModelConfig, provenance["model"]))
    if digest(saved_model) != digest(draccus.encode(config)):
        raise ValueError("Merge parity model differs from the exact candidate-C forward configuration")
    if provenance["tokenizer_pin"] != {"name": reference.tokenizer, "revision": reference.tokenizer_revision}:
        raise ValueError("Merge parity tokenizer differs from candidate C")
    if request.sft.token_ids != tuple(row[1] for row in provenance["donors"]):
        raise ValueError("Merge parity selected vocabulary differs from donor initialization")
    parent = provenance["selected_parent"]
    source_pt = parent["path"]
    if reference.selected_parent == "B":
        averaged = read_prepared_provenance(parent["path"], parent["metadata_digest"])
        if averaged["candidate"] != "B":
            raise ValueError("Declared averaged parent is not candidate B")
        source_pt = averaged["request"]["checkpoints"][-1]["path"]
    elif "averaging_digest" in provenance["source_metadata"][0]:
        raise ValueError("An averaged parent cannot be declared candidate A")
    if (reference.source_pt_checkpoint, reference.source_pt_step) != (source_pt, provenance["source_pt_step"]):
        raise ValueError("Merge parity source PT ancestry differs from candidate C")
    compact = _metadata(request.compact_checkpoint, request.compact_metadata_digest)
    merged = _metadata(request.merged_checkpoint.uri, request.merged_checkpoint.metadata_digest)
    expected = digest(sft_manifest(reference, request.sft))
    if any(digest(metadata.get("hero_sft")) != expected for metadata in (compact, merged)):
        raise ValueError("Compact state and ordinary D do not share the pinned SFT configuration and ancestry")
    if compact["step"] != merged["step"] or merged["step"] != request.merged_checkpoint.step:
        raise ValueError("Compact state and ordinary D belong to different SFT updates")
    if compact["hero_sft_counters"]["sft_step"] != compact["step"]:
        raise ValueError("Compact completion counters disagree with the saved SFT update")
    if (merged.get("merge_base"), merged.get("merge_accumulator_dtype"), merged.get("serving_dtype")) != (
        "BF16-rounded candidate C",
        "float32",
        "bfloat16",
    ):
        raise ValueError("Ordinary D must use the declared BF16-base/FP32-accumulator merge policy")
    # The shared reader validates immutable panel bytes and every typed array's
    # byte digest. It does not load tokenizers or checkpoint arrays.
    panel = read_pt_panel(
        NativePtRequest(
            panel_path=request.panel_path,
            panel_sha256=request.panel_sha256,
            checkpoint=request.merged_checkpoint,
            model=request.model,
            tokenizer=reference.tokenizer,
            tokenizer_revision=reference.tokenizer_revision,
            source_revision=request.source_revision,
            expert_axis_size=1,
            replica_axis_size=request.replica_axis_size,
        )
    )
    if panel.sequence_length > config.max_seq_len:
        raise ValueError("Merge parity panel exceeds the pinned model context")
    if len({(probe.row_index, probe.prediction_position) for probe in request.probes}) != len(request.probes):
        raise ValueError("Merge parity probes must be distinct")
    rows = {row.row_index: row for row in panel.rows}
    for probe in request.probes:
        row = rows[probe.row_index]
        if max(row.tokens) >= config.vocab_size or probe.prediction_position >= panel.sequence_length - 1:
            raise ValueError("Merge parity token or prediction position is outside the pinned model")
        if row.loss_weight[probe.prediction_position] <= 0:
            raise ValueError("Merge parity probes must predict actual weighted held-out targets")
    return config, panel, compact


def _logprobs(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits)
    shifted = logits.astype(np.float64) - maximum
    return shifted - np.log(np.exp(shifted).sum())


def compare_merge_logits(
    request: MergeParityRequest,
    compact: np.ndarray,
    merged: np.ndarray,
    panel: PtPanel,
    *,
    trained_input_token_ids: tuple[int, ...],
) -> dict:
    """Measure full-vocabulary discrepancies; derive acceptance from declared tolerances."""
    if compact.shape != merged.shape or compact.ndim != 2 or compact.shape[0] != len(request.probes):
        raise ValueError("Merge parity logits must align with each pinned probe")
    if not np.all(np.isfinite(compact)) or not np.all(np.isfinite(merged)):
        raise ValueError("Nonfinite compact or ordinary-D logits")
    rows = {row.row_index: row for row in panel.rows}
    input_counts = {str(token): 0 for token in request.sft.token_ids}
    for probe in request.probes:
        row = rows[probe.row_index]
        segment = row.segment_ids[probe.prediction_position]
        for token, token_segment in zip(
            row.tokens[: probe.prediction_position + 1], row.segment_ids[: probe.prediction_position + 1], strict=True
        ):
            if token_segment == segment and token_segment >= 0 and str(token) in input_counts:
                input_counts[str(token)] += 1
    covered = [token for token in trained_input_token_ids if input_counts[str(token)] > 0]
    missing = [token for token in trained_input_token_ids if input_counts[str(token)] == 0]
    measurements = []
    for probe, original, exported in zip(request.probes, compact, merged, strict=True):
        original_logprobs, exported_logprobs = _logprobs(original), _logprobs(exported)
        top_ids = np.argsort(original)[-min(5, original.size) :][::-1]
        expected = tuple(TokenScore(logprob=float(original_logprobs[token]), token_id=int(token)) for token in top_ids)
        case_id = f"row-{probe.row_index}-position-{probe.prediction_position}"
        parity = parity_from_logprob_row(case_id, expected, exported_logprobs, backend_rank=0)
        try:
            parity.assert_matches(max_probability_error=request.max_absolute_probability_error)
            winner_contract = True
        except AssertionError:
            winner_contract = False
        target = rows[probe.row_index].tokens[probe.prediction_position + 1]
        measurements.append(
            MergeMeasurement(
                probe=probe,
                arrays_sha256=rows[probe.row_index].arrays_sha256,
                target_token_id=target,
                compact_greedy_id=int(original.argmax()),
                merged_greedy_id=int(exported.argmax()),
                max_absolute_logit_error=float(np.max(np.abs(original.astype(np.float64) - exported))),
                max_absolute_probability_error=float(
                    np.max(np.abs(np.exp(original_logprobs) - np.exp(exported_logprobs)))
                ),
                compact_target_logprob=float(original_logprobs[target]),
                merged_target_logprob=float(exported_logprobs[target]),
                compact_top_scores=expected,
                merged_top_scores=tuple(
                    TokenScore(token_id=int(token), logprob=float(exported_logprobs[token]))
                    for token in np.argsort(exported)[-min(5, exported.size) :][::-1]
                ),
                winner_contract=winner_contract,
                next_token_parity=parity,
            )
        )
    maximum_logit = max(row.max_absolute_logit_error for row in measurements)
    maximum_probability = max(row.max_absolute_probability_error for row in measurements)
    disagreements = sum(row.compact_greedy_id != row.merged_greedy_id for row in measurements)
    numerical_parity_passed = (
        maximum_logit <= request.max_absolute_logit_error
        and maximum_probability <= request.max_absolute_probability_error
        and disagreements <= request.max_greedy_disagreements
        and all(row.winner_contract for row in measurements)
    )
    return {
        "probes": [row.model_dump() for row in measurements],
        "max_absolute_logit_error": maximum_logit,
        "max_absolute_probability_error": maximum_probability,
        "greedy_disagreements": disagreements,
        "selected_input_token_counts": input_counts,
        "trained_input_token_ids": list(trained_input_token_ids),
        "covered_trained_input_token_ids": covered,
        "missing_trained_input_token_ids": missing,
        "numerical_parity_passed": numerical_parity_passed,
        "passed": numerical_parity_passed and not missing,
    }


def run_merge_parity(request: MergeParityRequest, destination: str) -> dict:
    """Restore exact compact/D checkpoints, compare selected logits, and retain the report."""
    request = MergeParityRequest.model_validate(request.model_dump(mode="json"))
    config, panel, compact_metadata = validate_merge_request(request)
    root = StoragePath(destination)
    if root.exists():
        raise FileExistsError("Choose a new immutable merge-parity destination")
    mesh = compact_grug_mesh(expert_axis_size=request.expert_axis_size, replica_axis_size=request.replica_axis_size)
    batch_multiple = int(np.prod([mesh.shape[axis] for axis in _BATCH_AXES]))
    batch_size = max(2, ((len(request.probes) + batch_multiple - 1) // batch_multiple) * batch_multiple)
    shape = (batch_size, panel.sequence_length)
    tokens, segments = np.zeros(shape, np.int32), np.full(shape, -1, np.int32)
    positions = np.zeros(batch_size, np.int32)
    rows = {row.row_index: row for row in panel.rows}
    for index, probe in enumerate(request.probes):
        row = rows[probe.row_index]
        tokens[index], segments[index], positions[index] = row.tokens, row.segment_ids, probe.prediction_position
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
        ids, segment = [
            jax.make_array_from_callback(shape, sharding, lambda index, array=array: array[index])
            for array in (tokens, segments)
        ]
        last = jax.make_array_from_callback(
            positions.shape, NamedSharding(mesh, P(_BATCH_AXES)), lambda index: positions[index]
        )
        mask = AttentionMask.causal().with_segment_ids(segment)
        logger.info("Restore candidate C and compact SFT state for %d parity probes", len(request.probes))
        frozen, template = initialize_from_checkpoint(request.reference, config, mesh, request.sft)
        state = restore_sft_checkpoint(request.compact_checkpoint, template, request.reference, request.sft, mesh)
        actual_counters = {
            "sft_step": int(state.step),
            "processed_tokens": int(state.processed_tokens),
            "supervised_tokens": int(state.supervised_tokens),
        }
        if actual_counters != compact_metadata["hero_sft_counters"]:
            raise ValueError("Restored compact-state counters differ from permanent completion metadata")
        input_delta = np.asarray(state.params.input_delta)
        if not np.all(np.isfinite(input_delta)):
            raise ValueError("Restored compact input deltas are nonfinite")
        trained_input_ids = tuple(
            token for token, delta in zip(request.sft.token_ids, input_delta, strict=True) if np.any(delta != 0)
        )

        @eqx.filter_jit
        def compact_scores(base, parameters):
            scores = sft_logits(base, parameters, ids, request.sft, mask, positions=last)
            return jax.sharding.reshard(scores, P())

        logger.info("Measure compact-state logits")
        compact = np.asarray(compact_scores(frozen, state.params))[: len(request.probes)]
        # Retain only B*V host scores before restoring the ordinary checkpoint;
        # two copies of Hero's full weights need not coexist on the accelerator.
        del frozen, template, state
        logger.info("Compact scores retained; restore ordinary D")
        ordinary = cast_weights(
            restore_weights(request.merged_checkpoint.uri, request.merged_checkpoint.metadata_digest, config, mesh),
            jnp.bfloat16,
        )

        @eqx.filter_jit
        def merged_scores(model):
            hidden, _ = model(ids, mask)
            selected = hidden.at[jnp.arange(batch_size), last].get(out_sharding=P())
            return jax.sharding.reshard((selected @ model.output_proj).astype(jnp.float32), P())

        logger.info("Measure ordinary-D logits")
        merged = np.asarray(merged_scores(ordinary))[: len(request.probes)]
    report = {
        "request": request.model_dump(mode="json"),
        "request_digest": digest(request.model_dump(mode="json")),
        "panel_digest": digest(panel.model_dump()),
        "compact_counters": actual_counters,
        "model_digest": digest(draccus.encode(config)),
        "backend": jax.default_backend(),
        "devices": jax.device_count(),
        "processes": jax.process_count(),
        "device_kinds": sorted({device.device_kind for device in jax.devices()}),
        "attention_implementation": config.attention_implementation,
        "moe_implementation": config.moe_implementation,
        "expert_chunks": config.expert_chunks,
        "mesh": dict(mesh.shape),
        "batch_size": batch_size,
        "filler_rows": batch_size - len(request.probes),
        "compared_output_columns": config.vocab_size,
        "runtime_base_dtype": "bfloat16",
        "routing_control_dtype": "float32",
        "merged_dtype": "bfloat16",
        "projection_policy": "BF16 projections with FP32 compact vocabulary correction; no full sequence logits",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("sft_merge_parity.py", "sft.py", "model.py", "weights.py")
        },
        "completed_at": datetime.now(UTC).isoformat(),
        **compare_merge_logits(request, compact, merged, panel, trained_input_token_ids=trained_input_ids),
    }
    report["report_digest"] = digest(report)
    if jax.process_index() == 0:
        root.mkdirs()
        conditional_object(str(root / "report.json")).write(
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(), expected_version=None
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("destination")
    args = parser.parse_args()
    request = MergeParityRequest.model_validate_json(args.request.read_bytes())
    config = draccus.decode(GrugModelConfig, request.model)
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False, moe_implementation=config.moe_implementation, remat_mode=config.remat_mode
    )
    DistributedConfig().initialize()
    configure_coreweave_s3()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    if config.moe_implementation == RAGGED_MOE_IMPLEMENTATION:
        verify_ragged_pjrt()
    install(cutlass_kernel_cache())
    report = run_merge_parity(request, args.destination)
    logger.info("Merge parity complete: passed=%s report=%s/report.json", report["passed"], args.destination)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
