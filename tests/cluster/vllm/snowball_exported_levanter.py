# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture Snowball next-token observations after loading the canonical HF export."""

import argparse
import hashlib
import uuid
from typing import NamedTuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from jax.sharding import PartitionSpec as P
from levanter.models.snowball import SnowballLMHeadModel
from marin.inference.backend import ModelSpec
from marin.inference.config import LevanterEngineConfig
from marin.inference.levanter_backend import LevanterBackend
from marin.inference.model_preparation import read_attention_heads, select_tensor_parallel_size
from rigging.filesystem import StoragePath

from tests.cluster.vllm import backend_parity as backend_parity_module
from tests.cluster.vllm import snowball as snowball_module
from tests.cluster.vllm.backend_parity import ObservationReport, RunProvenance, source_digest
from tests.cluster.vllm.snowball import (
    BATCH_SIZE,
    PROMPT_FIXTURE_SHA256,
    REPRESENTATIVE_GOLDEN_SHA256,
    SNOWBALL,
    SNOWBALL_EXPORTED_GPU,
    SNOWBALL_EXPORTED_TPU,
    VLLM_MAX_MODEL_LEN,
    ExportedLevanterCell,
    RepresentativeGolden,
    read_prompt_fixture,
    read_representative_goldens,
)
from tests.cluster.vllm.snowball_levanter import OBSERVATION_TOP_K, build_batch_observations


class ScoredExportedTokenBatch(NamedTuple):
    top_logprobs: jax.Array
    top_token_ids: jax.Array
    golden_logprobs: jax.Array
    golden_ranks: jax.Array
    has_nonfinite: jax.Array


CAPTURE_SOURCE_DIGEST = source_digest(backend_parity_module.__file__, snowball_module.__file__, __file__)


def capture_exported_levanter(
    cell: ExportedLevanterCell,
    *,
    same_process_repeats: int = 1,
    batch_indices: tuple[int, ...] | None = None,
    goldens: tuple[RepresentativeGolden, ...] | None = None,
) -> ObservationReport:
    """Load the canonical export and capture selected prompt batches."""
    if same_process_repeats <= 0:
        raise ValueError("same_process_repeats must be positive")
    if jax.default_backend() != cell.location.name:
        raise RuntimeError(f"Expected {cell.location.name} backend, found {jax.default_backend()}")
    if cell.location.export_uri is None:
        raise ValueError(f"No verified export is configured for {cell.location.name}")

    goldens = read_representative_goldens() if goldens is None else goldens
    prompt_fixture = read_prompt_fixture(goldens, fixture_uri=cell.location.prompt_fixture_uri)
    export_uri = cell.location.export_uri
    config_bytes = (StoragePath(export_uri) / "config.json").read_bytes()
    num_heads, num_kv_heads = read_attention_heads(export_uri)
    num_chips = jax.device_count()
    tensor_parallel_size = select_tensor_parallel_size(num_heads, num_chips, num_kv_heads)
    selected_indices = tuple(range(len(prompt_fixture.batches))) if batch_indices is None else batch_indices
    if len(set(selected_indices)) != len(selected_indices) or any(
        index < 0 or index >= len(prompt_fixture.batches) for index in selected_indices
    ):
        raise ValueError(f"Invalid batch indices {selected_indices}")

    spec = ModelSpec(
        model=SNOWBALL.model_name,
        model_path=export_uri,
        num_chips=num_chips,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        chat_template_content=None,
    )
    config_overrides = {
        "moe_implementation": cell.requested_moe,
        "attention_implementation": cell.requested_attention,
    }
    with LevanterBackend(LevanterEngineConfig()).load_model(spec, config_overrides=config_overrides) as loaded:
        model = loaded.model
        if not isinstance(model, SnowballLMHeadModel):
            raise TypeError(f"Expected SnowballLMHeadModel, found {type(model).__name__}")
        parameter_dtypes = {leaf.dtype for leaf in jax.tree.leaves(model) if eqx.is_inexact_array(leaf)}
        if parameter_dtypes != {jnp.dtype(jnp.bfloat16)}:
            raise ValueError(f"Expected only BF16 parameters, got {parameter_dtypes}")
        if loaded.trainer.data_axis_size != BATCH_SIZE:
            raise ValueError(f"Expected data axis {BATCH_SIZE}, got {loaded.trainer.data_axis_size}")
        if loaded.tokenizer.eos_token_id is None:
            raise ValueError("Snowball tokenizer must define eos_token_id")

        @hax.named_jit(axis_resources=loaded.trainer.compute_axis_mapping)
        def score_next_token(model, input_ids, last_positions, golden_token_ids):
            hidden = model.activations(input_ids).rearrange(("batch", "position", "embed")).array
            last_hidden = hidden.at[jnp.arange(hidden.shape[0]), last_positions.array].get(out_sharding=P("data"))
            logits = jnp.einsum("bh,hv->bv", last_hidden, model.transformer.output_proj, out_sharding=P("data"))
            if logits.dtype != jnp.bfloat16:
                raise ValueError(f"Expected BF16 logits, got {logits.dtype}")
            logprobs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            top_logprobs, top_token_ids = jax.lax.top_k(logprobs, OBSERVATION_TOP_K)
            golden_logprobs = jnp.take_along_axis(logprobs, golden_token_ids, axis=-1)
            golden_ranks = jnp.sum(
                logprobs[:, None, :] > golden_logprobs[:, :, None],
                axis=-1,
                dtype=jnp.int32,
            )
            has_nonfinite = jnp.any(~jnp.isfinite(logprobs), axis=-1)
            return ScoredExportedTokenBatch(
                top_logprobs=top_logprobs,
                top_token_ids=top_token_ids,
                golden_logprobs=golden_logprobs,
                golden_ranks=golden_ranks,
                has_nonfinite=has_nonfinite,
            )

        observations = []
        Batch = Axis("batch", BATCH_SIZE)
        for repeat_index in range(same_process_repeats):
            for batch_index in selected_indices:
                batch = prompt_fixture.batches[batch_index]
                token_ids, last_token_indices = snowball_module.pad_prompt_batch(batch, loaded.tokenizer.eos_token_id)
                golden_token_ids = np.asarray(
                    [[score.token_id for score in case.top_logprobs] for case in batch.cases],
                    dtype=np.int32,
                )
                Pos = Axis("position", batch.max_tokens)
                outputs = score_next_token(
                    model,
                    hax.named(jnp.asarray(token_ids), (Batch, Pos)),
                    hax.named(jnp.asarray(last_token_indices), (Batch,)),
                    jnp.asarray(golden_token_ids),
                )
                device_outputs = jax.tree.map(lambda output: np.asarray(jax.device_get(output)), outputs)
                observations.extend(
                    build_batch_observations(
                        batch,
                        repeat_index=repeat_index,
                        top_logprobs=device_outputs.top_logprobs,
                        top_token_ids=device_outputs.top_token_ids,
                        golden_logprobs=device_outputs.golden_logprobs,
                        golden_ranks=device_outputs.golden_ranks,
                        capacity_overflow=np.empty((0,), dtype=np.float32),
                        has_nonfinite=device_outputs.has_nonfinite,
                    )
                )

        mesh_shape = tuple((name, int(size)) for name, size in loaded.trainer.device_mesh.shape.items())

    provenance = RunProvenance(
        backend="levanter-exported",
        platform=cell.location.name,
        process_id=uuid.uuid4().hex,
        code_digest=CAPTURE_SOURCE_DIGEST,
        # Exact export integrity is established separately; this binds the
        # loaded cell to those already verified, content-addressed bytes.
        parameter_digest=SNOWBALL.export_sha256,
        model_config_digest=hashlib.sha256(config_bytes).hexdigest(),
        prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
        requested_attention=cell.requested_attention,
        effective_attention=cell.effective_attention,
        requested_moe=cell.requested_moe,
        effective_moe=cell.effective_moe,
        mesh_shape=mesh_shape,
        device_kind=",".join(sorted({device.device_kind for device in jax.devices()})),
        golden_digest=REPRESENTATIVE_GOLDEN_SHA256,
    )
    return ObservationReport(provenance=provenance, observations=tuple(observations))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("gpu", "tpu"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--same-process-repeats", type=int, default=1)
    parser.add_argument("--batch-index", action="append", type=int)
    args = parser.parse_args()

    cell = SNOWBALL_EXPORTED_GPU if args.platform == "gpu" else SNOWBALL_EXPORTED_TPU
    report = capture_exported_levanter(
        cell,
        same_process_repeats=args.same_process_repeats,
        batch_indices=None if args.batch_index is None else tuple(args.batch_index),
    )
    StoragePath(args.output).write_bytes(report.to_json_bytes())


if __name__ == "__main__":
    main()
