# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict, weights-only native sampling of one pinned hero checkpoint."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import tensorstore as ts
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import load_checkpoint
from levanter.checkpoint_manifest import read_manifest
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tensorstore_serialization import build_kvstore_spec
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.grug.checkpointing import LEGACY_STATE_KEY, MASTER_PARAMS_KEY
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    TOP_TOKEN_COUNT,
    SampleRequest,
    SampleResult,
    SampleStore,
    digest,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.generation import generate, score_expected
from experiments.grug.moe_hero_ep.train import _apply_qb_betas

COMPUTE_POLICY = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")


def restore_model(request: SampleRequest, mesh: jax.sharding.Mesh) -> Transformer:
    """Restore authoritative weights and pending router bias, with no optimizer or fallback checkpoint."""
    checkpoint = request.checkpoint.uri
    checkpoint_path = StoragePath(checkpoint)
    metadata = json.loads((checkpoint_path / "metadata.json").read_text())
    if digest(metadata) != request.checkpoint.metadata_digest or metadata.get("is_temporary") is not False:
        raise ValueError("Checkpoint metadata changed or checkpoint is not permanent")
    config = draccus.decode(GrugModelConfig, request.spec.model)
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    manifest = read_manifest(checkpoint)
    if manifest is not None:
        wrapped = any(path.startswith(f"{LEGACY_STATE_KEY}/") for path in manifest.array_paths)
        prefix = f"{LEGACY_STATE_KEY}/" if wrapped else ""
        master = any(path.startswith(f"{prefix}{MASTER_PARAMS_KEY}/") for path in manifest.array_paths)
    elif (checkpoint_path / "manifest.ocdbt").exists():
        # Probe known metadata keys inside the database. Filesystem directories cannot reveal this layout.
        kvstore = ts.KvStore.open({"driver": "ocdbt", "base": build_kvstore_spec(checkpoint)}).result()
        wrapped = kvstore.read(f"{LEGACY_STATE_KEY}/pending_qb_betas/zarr.json").result().state == "value"
        prefix = f"{LEGACY_STATE_KEY}/" if wrapped else ""
        master = kvstore.read(f"{prefix}{MASTER_PARAMS_KEY}/token_embed/zarr.json").result().state == "value"
    else:
        # Pre-manifest checkpoints use directory-backed arrays. Probe layout directories only.
        wrapped = (checkpoint_path / LEGACY_STATE_KEY).exists()
        prefix = f"{LEGACY_STATE_KEY}/" if wrapped else ""
        master = (checkpoint_path / f"{prefix}{MASTER_PARAMS_KEY}").exists()
    weights_key = MASTER_PARAMS_KEY if master else "params"
    state_template: dict[str, Any] = {
        weights_key: template,
        "pending_qb_betas": jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
    }
    state = load_checkpoint(
        {LEGACY_STATE_KEY: state_template} if wrapped else state_template,
        checkpoint,
        mesh=mesh,
        allow_partial=False,
    )
    if wrapped:
        state = state[LEGACY_STATE_KEY]
    jax.block_until_ready(state)
    return _apply_qb_betas(state[weights_key], state["pending_qb_betas"])


@eqx.filter_jit
def next_logits(model: Transformer, tokens: jax.Array, positions: jax.Array) -> jax.Array:
    hidden, _ = model(tokens)
    last = hidden.at[jnp.arange(tokens.shape[0]), positions].get(out_sharding=P())
    scores = jnp.einsum("bh,hv->bv", last, model.output_proj, preferred_element_type=jnp.float32)
    return jax.sharding.reshard(scores, P())


@eqx.filter_jit
def expected_logprobs(
    model: Transformer, tokens: jax.Array, positions: jax.Array, targets: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Score all reference positions with one causal model pass and a bounded vocabulary projection."""
    hidden, _ = model(tokens)
    selected = hidden.at[jnp.arange(tokens.shape[0])[:, None], positions].get(out_sharding=P())
    targets = jax.sharding.reshard(targets, P())

    def project(inputs):
        state, target = inputs
        logits = jnp.einsum("bh,hv->bv", state, model.output_proj, preferred_element_type=jnp.float32)
        values = jax.nn.log_softmax(jax.sharding.reshard(logits, P()), axis=-1)
        top_values, top_ids = jax.lax.top_k(values, min(TOP_TOKEN_COUNT, values.shape[-1]))
        chosen = jnp.take_along_axis(values, target[:, None], axis=-1)[:, 0]
        return chosen, top_ids, top_values

    # Project one position at a time to keep the full sequence of vocabulary logits out of memory.
    scores = jax.lax.map(project, (jnp.swapaxes(selected, 0, 1), jnp.swapaxes(targets, 0, 1)))
    return tuple(jax.sharding.reshard(jnp.swapaxes(value, 0, 1), P()) for value in scores)


def sample(request: SampleRequest, store_root: str) -> None:
    """Run one rack of native inference and commit one validated result on process zero."""
    DistributedConfig().initialize()
    install(cutlass_kernel_cache())
    if jax.device_count() != request.spec.batch_size or jax.default_backend() != "gpu":
        raise ValueError("The hero sampler requires one GPU per batch row")
    tokenizer = AutoTokenizer.from_pretrained(request.spec.tokenizer, revision=request.spec.tokenizer_revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    prompt_ids = [tokenizer.encode(prompt.text, add_special_tokens=True) for prompt in request.spec.prompts]
    if any(not ids or len(ids) > request.spec.context_length for ids in prompt_ids):
        raise ValueError("Prompt does not fit the context")
    expected_ids = []
    for prompt, ids in zip(request.spec.prompts, prompt_ids, strict=True):
        if not prompt.expected:
            raise ValueError(f"Prompt has no expected completion: {prompt.id}")
        expected = tokenizer.encode(prompt.expected, add_special_tokens=False)
        if not expected or len(ids) + len(expected) > request.spec.context_length:
            raise ValueError(f"Expected completion does not fit the context: {prompt.id}")
        expected_ids.append(expected)

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model = COMPUTE_POLICY.cast_to_compute(restore_model(request, mesh))
        jax.block_until_ready(model)
        sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))

        def logits(tokens: np.ndarray, positions: np.ndarray) -> np.ndarray:
            ids = jax.make_array_from_callback(tokens.shape, sharding, lambda index: tokens[index])
            last = jax.make_array_from_callback(positions.shape, sharding, lambda index: positions[index])
            return np.asarray(next_logits(model, ids, last))

        def logprobs(
            tokens: np.ndarray, positions: np.ndarray, targets: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            arrays = [
                jax.make_array_from_callback(value.shape, sharding, lambda index, value=value: value[index])
                for value in (tokens, positions, targets)
            ]
            values, top_ids, top_values = expected_logprobs(model, *arrays)
            return np.asarray(values), np.asarray(top_ids), np.asarray(top_values)

        expected_scores = score_expected(
            request.spec,
            prompt_ids,
            expected_ids,
            eos_token_id=tokenizer.eos_token_id,
            logprobs=logprobs,
            decode=decode,
        )
        completions = generate(
            request.spec,
            prompt_ids,
            eos_token_id=tokenizer.eos_token_id,
            logits=logits,
            decode=decode,
        )
        completions = tuple(
            completion.model_copy(update={"expected_scores": scores})
            for completion, scores in zip(completions, expected_scores, strict=True)
        )
        if jax.process_index() == 0:
            SampleStore(store_root).save_result(
                SampleResult(
                    request=request,
                    completions=completions,
                    completed_at=datetime.now(UTC).isoformat(),
                    eos_token_id=tokenizer.eos_token_id,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--store-root", required=True)
    args = parser.parse_args()
    sample(SampleRequest.model_validate_json(args.request.read_bytes()), args.store_root)


if __name__ == "__main__":
    main()
