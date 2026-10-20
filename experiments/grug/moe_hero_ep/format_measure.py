# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize bounded PT-heldout panels and measure native Hero loss/routing.

Both CLI operations are explicit: materialization reads selected existing dataset
caches; measurement restores one pinned checkpoint. Neither submits jobs.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from pydantic import Field, model_validator
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.format_sample import NATIVE_BACKEND, native_evaluation_model
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, Record, digest
from experiments.grug.moe_hero_ep.weights import cast_weights, restore_weights

MAX_PANEL_TOKENS = 4 * 1024 * 1024
MAX_PANEL_BYTES = 128 * 1024 * 1024
_BATCH_AXES = ("replica_dcn", "data", "expert")


def token_bytes_digest(tokens: tuple[int, ...], weights: tuple[float, ...], segments: tuple[int, ...]) -> str:
    """Hash the exact little-endian int32/float32 arrays consumed by evaluation."""
    arrays = (np.asarray(tokens, dtype="<i4"), np.asarray(weights, dtype="<f4"), np.asarray(segments, dtype="<i4"))
    return hashlib.sha256(b"".join(array.tobytes() for array in arrays)).hexdigest()


class PtRow(Record):
    row_index: int = Field(ge=0)
    tokens: tuple[int, ...] = Field(min_length=2, max_length=65536)
    loss_weight: tuple[float, ...]
    segment_ids: tuple[int, ...]
    arrays_sha256: str

    @model_validator(mode="after")
    def aligned_arrays(self) -> Self:
        if len(self.tokens) != len(self.loss_weight) or len(self.tokens) != len(self.segment_ids):
            raise ValueError("PT token, next-token weight and segment arrays must align")
        tokens, weights, segments = (
            np.asarray(self.tokens),
            np.asarray(self.loss_weight, dtype=np.float32),
            np.asarray(self.segment_ids),
        )
        if (
            np.any(tokens < 0)
            or np.any(tokens > np.iinfo(np.int32).max)
            or np.any(segments < -1)
            or np.any(segments > np.iinfo(np.int32).max)
        ):
            raise ValueError("PT tokens/segments must fit their canonical int32 representation")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not weights.sum() > 0:
            raise ValueError("PT loss weights must be finite, nonnegative and nonempty")
        invalid = (segments < 0) | (np.roll(segments, -1) < 0)
        invalid[-1] = True
        if np.any(weights[invalid] != 0):
            raise ValueError("Padding and the last token must have zero next-token loss weight")
        if token_bytes_digest(self.tokens, self.loss_weight, self.segment_ids) != self.arrays_sha256:
            raise ValueError("Pinned PT token/weight/segment bytes changed")
        return self


class PtPanel(Record):
    source_data_config: dict
    source_data_config_digest: str
    component: str
    tags: tuple[str, ...]
    tokenizer: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sequence_length: int = Field(ge=2, le=65536)
    rows: tuple[PtRow, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def pinned_selection(self) -> Self:
        if digest(self.source_data_config) != self.source_data_config_digest:
            raise ValueError("PT source data configuration changed")
        if len(self.rows) * self.sequence_length > MAX_PANEL_TOKENS:
            raise ValueError("PT panel exceeds the bounded token budget")
        if len({row.row_index for row in self.rows}) != len(self.rows):
            raise ValueError("PT panel row indices must be distinct")
        if any(len(row.tokens) != self.sequence_length for row in self.rows):
            raise ValueError("PT rows must have the pinned sequence length")
        return self


class PtMaterializeRequest(Record):
    data: dict
    data_digest: str
    component: str
    tag: str
    row_indices: tuple[int, ...] = Field(min_length=1, max_length=256)
    sequence_length: int = Field(ge=2, le=65536)
    tokenizer: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def bounded_selection(self) -> Self:
        if digest(self.data) != self.data_digest:
            raise ValueError("PT materialization source configuration changed")
        if len(self.row_indices) * self.sequence_length > MAX_PANEL_TOKENS:
            raise ValueError("Requested PT materialization exceeds the bounded token budget")
        if min(self.row_indices) < 0 or len(set(self.row_indices)) != len(self.row_indices):
            raise ValueError("PT materialization needs distinct nonnegative row indices")
        return self


async def panel_from_dataset(
    dataset: AsyncDataset[LmExample], request: PtMaterializeRequest, tags: tuple[str, ...]
) -> PtPanel:
    """Freeze only explicitly selected rows, preserving existing causal masks."""
    rows = []
    # One row at a time bounds dataset-side materialization independently of panel size.
    for row_index in request.row_indices:
        example = grug_lm_example_from_named(await dataset.getitem_async(row_index))
        mask = example.attn_mask
        if not mask.is_causal or mask.sliding_window is not None or mask.fa4_bounds is not None:
            raise ValueError("PT panel requires ordinary causal/document masks; model defines local/global windows")
        tokens = tuple(int(value) for value in np.asarray(example.tokens))
        weights = tuple(float(value) for value in np.asarray(example.loss_weight, dtype=np.float32))
        segments = (0,) * len(tokens)
        if mask.segment_ids is not None:
            query, key = mask.segment_ids
            if not np.array_equal(query, key):
                raise ValueError("PT panel requires matching query/key document segments")
            segments = tuple(int(value) for value in np.asarray(query))
        rows.append(
            PtRow(
                row_index=row_index,
                tokens=tokens,
                loss_weight=weights,
                segment_ids=segments,
                arrays_sha256=token_bytes_digest(tokens, weights, segments),
            )
        )
    return PtPanel(
        source_data_config=request.data,
        source_data_config_digest=request.data_digest,
        component=request.component,
        tags=tags,
        tokenizer=request.tokenizer,
        tokenizer_revision=request.tokenizer_revision,
        sequence_length=request.sequence_length,
        rows=tuple(rows),
    )


async def materialize_pt_panel(request: PtMaterializeRequest) -> PtPanel:
    """Read the selected saved LmDataConfig validation component via its tagged dataset."""
    if digest(request.data) != request.data_digest:
        raise ValueError("PT materialization source configuration changed")
    data = draccus.decode(LmDataConfig, request.data)
    if data.tokenizer != request.tokenizer:
        raise ValueError("Saved PT data tokenizer differs from the panel tokenizer")
    component = data.components[request.component]
    validation_sequences = data.num_validation_sequences
    if validation_sequences is not None:
        validation_sequences = {
            name: count for name, count in validation_sequences.items() if name == request.component
        } or None
    data = dataclasses.replace(
        data,
        components={request.component: component},
        train_weights=None,
        num_validation_sequences=validation_sequences,
        auto_build_caches=False,
        tokenizer=f"{request.tokenizer}@{request.tokenizer_revision}",
    )
    tagged = data.tagged_eval_sets(Axis("position", request.sequence_length))
    matches = [(dataset, tuple(tags)) for dataset, tags in tagged if request.tag in tags]
    if len(matches) != 1:
        raise ValueError("PT tag must select exactly one validation dataset from the chosen component")
    dataset, tags = matches[0]
    return await panel_from_dataset(dataset, request, tags)


class NativePtRequest(Record):
    panel_path: str
    panel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint: Checkpoint
    model: dict
    tokenizer: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    expert_axis_size: int = Field(ge=1, le=1)
    replica_axis_size: int = Field(ge=1)


class PtMeasurementRow(Record):
    row_index: int
    arrays_sha256: str
    loss_sum: float
    loss_weight: float
    loss: float
    valid_tokens: int


class PtRoutingBatch(Record):
    row_indices: tuple[int, ...]
    filler_rows: int
    valid_tokens: int
    routing_counts_per_layer: list[list[int]]
    routing_entropy_mean: float
    router_z_loss: float


def read_pt_panel(request: NativePtRequest) -> PtPanel:
    """Validate exact panel bytes before any checkpoint arrays are opened."""
    fs, path = filesystem_for(request.panel_path)
    with fs.open(path, "rb") as stream:
        raw = stream.read(MAX_PANEL_BYTES + 1)
    if len(raw) > MAX_PANEL_BYTES:
        raise ValueError("PT panel JSON exceeds the bounded byte budget")
    if hashlib.sha256(raw).hexdigest() != request.panel_sha256:
        raise ValueError("PT panel bytes changed")
    panel = PtPanel.model_validate_json(raw)
    if (panel.tokenizer, panel.tokenizer_revision) != (request.tokenizer, request.tokenizer_revision):
        raise ValueError("PT panel tokenizer pin differs from measurement request")
    return panel


def measure_pt_model(model: Transformer, panel: PtPanel, mesh: Mesh) -> dict:
    """Compute per-row weighted next-token CE and padding-free routing on the actual model."""
    # The production wrapper requires sonic_cute, matching format_sample. Scatter
    # exercises this same computation with a real dropless CPU kernel in tests.
    if model.config.moe_implementation not in ("sonic_cute", "scatter") or model.config.expert_chunks != 1:
        raise ValueError("PT evaluation requires a dropless local expert backend")
    if panel.sequence_length > model.config.max_seq_len or any(
        max(row.tokens) >= model.config.vocab_size for row in panel.rows
    ):
        raise ValueError("PT panel exceeds the actual model context or vocabulary")
    # Keep a non-unit batch dimension even on one device: the segment-boundary
    # kernel concatenates slices whose explicit batch sharding vanishes at B=1.
    batch_size = max(2, int(np.prod([mesh.shape[axis] for axis in _BATCH_AXES])))
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    @eqx.filter_jit
    def kernel(current, tokens, weights, segments):
        loss, metrics = current.next_token_loss(
            tokens,
            weights,
            mask=AttentionMask.causal().with_segment_ids(segments),
            reduction="none",
            logsumexp_weight=None,
            return_router_metrics=True,
        )
        values = (
            jnp.sum(loss, axis=-1),
            metrics["train/router/routing_counts_per_layer"],
            metrics["train/router/routing_entropy_mean"],
            metrics["train/router/router_z_loss"],
        )
        return jax.tree.map(lambda value: jax.sharding.reshard(value, P()), values)

    rows = []
    routing_batches = []
    for start in range(0, len(panel.rows), batch_size):
        batch = panel.rows[start : start + batch_size]
        shape = (batch_size, panel.sequence_length)
        tokens = np.zeros(shape, np.int32)
        weights = np.zeros(shape, np.float32)
        segments = np.full(shape, -1, np.int32)
        for index, row in enumerate(batch):
            tokens[index], weights[index], segments[index] = row.tokens, row.loss_weight, row.segment_ids
        arrays = [
            jax.make_array_from_callback(shape, sharding, lambda index, array=array: array[index])
            for array in (tokens, weights, segments)
        ]
        losses, counts, entropy, z_loss = jax.device_get(kernel(model, *arrays))
        if not all(np.all(np.isfinite(value)) for value in (losses, counts, entropy, z_loss)):
            raise ValueError(f"Nonfinite native PT measurement in batch starting at panel offset {start}")
        valid_tokens = int(np.count_nonzero(segments >= 0))
        expected_assignments = valid_tokens * model.config.num_experts_per_token
        if not np.all(np.asarray(counts).sum(axis=-1) == expected_assignments):
            raise ValueError("Routing assignment totals include padding or dropped real tokens")
        if np.any(losses[len(batch) :] != 0):
            raise ValueError("Filler rows contributed next-token loss")
        for row, loss in zip(batch, losses[: len(batch)], strict=True):
            count = float(np.asarray(row.loss_weight, dtype=np.float32).sum(dtype=np.float64))
            rows.append(
                PtMeasurementRow(
                    row_index=row.row_index,
                    arrays_sha256=row.arrays_sha256,
                    loss_sum=float(loss),
                    loss_weight=count,
                    loss=float(loss) / count,
                    valid_tokens=sum(segment >= 0 for segment in row.segment_ids),
                )
            )
        routing_batches.append(
            PtRoutingBatch(
                row_indices=tuple(row.row_index for row in batch),
                filler_rows=batch_size - len(batch),
                valid_tokens=valid_tokens,
                routing_counts_per_layer=counts.tolist(),
                routing_entropy_mean=float(entropy),
                router_z_loss=float(z_loss),
            )
        )
    loss_sum = sum(row.loss_sum for row in rows)
    weight = sum(row.loss_weight for row in rows)
    valid = sum(row.valid_tokens for row in rows)
    return {
        "rows": [row.model_dump() for row in rows],
        "routing_batches": [batch.model_dump() for batch in routing_batches],
        "batch_size": batch_size,
        "forward_passes": len(routing_batches),
        "loss_sum": loss_sum,
        "loss_weight": weight,
        "loss": loss_sum / weight,
        "valid_tokens": valid,
        "routing_counts_per_layer": (
            np.sum([batch.routing_counts_per_layer for batch in routing_batches], axis=0).tolist()
        ),
        "routing_entropy_mean": (
            sum(batch.routing_entropy_mean * batch.valid_tokens for batch in routing_batches) / valid
        ),
        "router_z_loss": sum(batch.router_z_loss * batch.valid_tokens for batch in routing_batches) / valid,
        "routing_policy": "all non-padding input tokens; segment=-1 filler excluded; counts summed across batches",
        "routing_aggregation": "entropy and z-loss are valid-token-weighted means of actual per-batch measurements",
    }


def measure_native(request: NativePtRequest, destination: str) -> dict:
    """Measure one A/B/C/D/E native checkpoint and retain its full measurement identity."""
    panel = read_pt_panel(request)
    config = draccus.decode(GrugModelConfig, request.model)
    if config != native_evaluation_model(config):
        raise ValueError("Use native_evaluation_model to select dropless evaluation settings")
    if panel.sequence_length > config.max_seq_len or any(max(row.tokens) >= config.vocab_size for row in panel.rows):
        raise ValueError("PT panel exceeds pinned context or vocabulary before weight I/O")
    root = StoragePath(destination)
    if root.exists():
        raise FileExistsError("Choose a new immutable PT measurement destination")
    mesh = compact_grug_mesh(expert_axis_size=request.expert_axis_size, replica_axis_size=request.replica_axis_size)
    with jax.set_mesh(mesh):
        model = cast_weights(
            restore_weights(request.checkpoint.uri, request.checkpoint.metadata_digest, config, mesh), jnp.bfloat16
        )
        measured = measure_pt_model(model, panel, mesh)
    report = {
        "request": request.model_dump(),
        "panel_digest": digest(panel.model_dump()),
        "model_digest": digest(draccus.encode(config)),
        "backend": NATIVE_BACKEND,
        "compute_dtype": "bfloat16",
        "routing_control_dtype": "float32",
        "attention_implementation": config.attention_implementation,
        "moe_implementation": config.moe_implementation,
        "device_platforms": sorted({device.platform for device in jax.devices()}),
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("format_measure.py", "model.py", "weights.py")
        },
        "devices": jax.device_count(),
        "processes": jax.process_count(),
        "mesh": dict(mesh.shape),
        "completed_at": datetime.now(UTC).isoformat(),
        **measured,
    }
    report["report_digest"] = digest(report)
    if jax.process_index() == 0:
        root.mkdirs()
        (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("materialize", "measure"))
    parser.add_argument("request", type=Path)
    parser.add_argument("destination")
    args = parser.parse_args()
    if args.operation == "materialize":
        request = PtMaterializeRequest.model_validate_json(args.request.read_bytes())
        panel = asyncio.run(materialize_pt_panel(request))
        serialized = panel.model_dump_json(indent=2) + "\n"
        if len(serialized.encode()) > MAX_PANEL_BYTES:
            raise ValueError("Materialized PT panel JSON exceeds the bounded byte budget")
        with Path(args.destination).open("x") as output:
            output.write(serialized)
        return
    DistributedConfig().initialize()
    configure_coreweave_s3()
    measure_native(NativePtRequest.model_validate_json(args.request.read_bytes()), args.destination)


if __name__ == "__main__":
    main()
