# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weights-only, compact-parameter format SFT for the current Hero architecture.

Only ``SftParameters`` is differentiated or handed to Adam. The frozen BF16 base
(including effective QB biases) is an ordinary nondifferentiated argument, never
an optimizer parameter or a full-model master copy.
"""

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, get_abstract_mesh, reshard
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import load_checkpoint, save_checkpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.grug import checkpointing
from experiments.grug.moe_hero_ep.averaging import _DONORS, read_prepared_provenance
from experiments.grug.moe_hero_ep.model import (
    _BATCH_AXES,
    LoraProjection,
    Transformer,
    _partition_spec_of,
    _seq_axis,
    _summarize_router_metrics,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.sft_hlo import audit_gradient_operations
from experiments.grug.moe_hero_ep.weights import cast_weights, restore_weights


@dataclass(frozen=True)
class SftConfig:
    """Explicit adapter, precision, loss tiling and fresh-Adam schedule choices."""

    rank: int
    alpha: float
    token_ids: tuple[int, ...]
    adapter_learning_rate: float
    vocab_learning_rate: float
    num_steps: int
    warmup_steps: int
    seed: int
    vocab_block_size: int = 4096
    token_block_size: int = 256

    def __post_init__(self):
        if self.rank < 1 or self.alpha <= 0 or not self.token_ids or len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("Require positive rank/alpha and distinct selected token IDs")
        if self.num_steps <= self.warmup_steps or self.warmup_steps < 0:
            raise ValueError("Require 0 <= warmup_steps < num_steps")
        if min(self.adapter_learning_rate, self.vocab_learning_rate) <= 0:
            raise ValueError("Both Adam learning rates must be positive")
        if self.vocab_block_size < 1 or self.token_block_size < 1:
            raise ValueError("Loss tile dimensions must be positive")


@dataclass(frozen=True)
class BaseReference:
    """Exact candidate C, distinct from the original PT or averaged parent."""

    checkpoint: str
    metadata_digest: str
    source_pt_checkpoint: str
    source_pt_step: int
    selected_parent: str
    tokenizer: str
    tokenizer_revision: str

    def __post_init__(self):
        if self.selected_parent not in ("A", "B"):
            raise ValueError("Candidate C must record its selected A/B parent")
        if not all((self.checkpoint, self.metadata_digest, self.source_pt_checkpoint, self.tokenizer_revision)):
            raise ValueError("Pin candidate-C metadata, original PT provenance, and tokenizer revision")


class SftParameters(eqx.Module):
    adapters: Transformer
    input_delta: jax.Array
    output_delta: jax.Array


class SftState(eqx.Module):
    step: jax.Array
    params: SftParameters
    adapter_opt_state: Any
    vocab_opt_state: Any
    processed_tokens: jax.Array
    supervised_tokens: jax.Array


def _attach_block(block, key, config: SftConfig):
    projections = [(block.attn, name) for name in ("q", "k", "v", "o")]
    for projection, name in projections:
        key, subkey = jax.random.split(key)
        adapter = LoraProjection.init(getattr(projection, f"w_{name}"), config.rank, config.alpha, key=subkey)
        block = eqx.tree_at(
            lambda b, name=name: getattr(b.attn, f"lora_{name}"), block, adapter, is_leaf=lambda x: x is None
        )
    if block.shared is not None:
        for index, shared in enumerate(block.shared):
            for name in ("gate", "up", "down"):
                key, subkey = jax.random.split(key)
                adapter = LoraProjection.init(getattr(shared, f"w_{name}"), config.rank, config.alpha, key=subkey)
                block = eqx.tree_at(
                    lambda b, index=index, name=name: getattr(b.shared[index], f"lora_{name}"),
                    block,
                    adapter,
                    is_leaf=lambda x: x is None,
                )
    return block


def initialize_sft(base: Transformer, config: SftConfig) -> tuple[Transformer, SftState]:
    """Round a prepared FP32 model once; partition before gradients or Adam exist."""
    if min(config.token_ids) < 0 or max(config.token_ids) >= base.config.vocab_size:
        raise ValueError("Selected vocabulary entries must already exist")
    base = cast_weights(base, jnp.bfloat16)
    keys = jax.random.split(jax.random.PRNGKey(config.seed), base.config.num_layers)
    stacked = eqx.filter_vmap(partial(_attach_block, config=config))(base.stacked_blocks.stacked, keys)
    model = eqx.tree_at(lambda m: m.stacked_blocks.stacked, base, stacked)

    def is_adapter(x):
        return isinstance(x, LoraProjection)

    adapters, frozen = eqx.partition(model, is_adapter, is_leaf=is_adapter)
    params = SftParameters(
        adapters=adapters,
        input_delta=reshard(jnp.zeros((len(config.token_ids), base.config.hidden_dim), jnp.float32), P(None, None)),
        output_delta=reshard(jnp.zeros((base.config.hidden_dim, len(config.token_ids)), jnp.float32), P(None, None)),
    )
    adapter_optimizer, vocab_optimizer = sft_optimizers(config)
    return frozen, SftState(
        step=reshard(jnp.array(0, jnp.int32), P()),
        params=params,
        adapter_opt_state=adapter_optimizer.init(params.adapters),
        vocab_opt_state=vocab_optimizer.init((params.input_delta, params.output_delta)),
        processed_tokens=reshard(jnp.array(0, jnp.int32), P()),
        supervised_tokens=reshard(jnp.array(0, jnp.int32), P()),
    )


def initialize_from_checkpoint(reference: BaseReference, model_config, mesh: Mesh, config: SftConfig):
    """Restore authoritative weights/effective QB state; never load the PT optimizer or step."""
    provenance = read_prepared_provenance(reference.checkpoint, reference.metadata_digest)
    if provenance["candidate"] != "C" or provenance["donors"] != [list(row) for row in _DONORS]:
        raise ValueError("SFT must initialize from the exact donor-initialized candidate C")
    normalized_saved = draccus.encode(draccus.decode(type(model_config), provenance["model"]))
    if digest(normalized_saved) != digest(draccus.encode(model_config)):
        raise ValueError("SFT forward configuration differs from the prepared checkpoint")
    if tuple(row[1] for row in provenance["donors"]) != config.token_ids:
        raise ValueError("Trainable vocabulary entries differ from candidate-C donor initialization")
    if provenance["tokenizer_pin"] != {"name": reference.tokenizer, "revision": reference.tokenizer_revision}:
        raise ValueError("SFT tokenizer differs from the candidate-C tokenizer pin")
    parent = provenance["selected_parent"]
    source_pt = parent["path"]
    if reference.selected_parent == "B":
        parent_provenance = read_prepared_provenance(parent["path"], parent["metadata_digest"])
        if parent_provenance["candidate"] != "B":
            raise ValueError("The declared averaged parent is not candidate B")
        source_pt = parent_provenance["request"]["checkpoints"][-1]["path"]
    elif "averaging_digest" in provenance["source_metadata"][0]:
        raise ValueError("An averaged parent was incorrectly declared as candidate A")
    if reference.source_pt_checkpoint != source_pt or reference.source_pt_step != provenance["source_pt_step"]:
        raise ValueError("SFT source PT checkpoint or step differs from prepared-checkpoint ancestry")
    return initialize_sft(restore_weights(reference.checkpoint, reference.metadata_digest, model_config, mesh), config)


def sft_optimizers(config: SftConfig):
    def optimizer(peak):
        # Format SFT uses a finite linear cooldown, not the PT Muon schedule.
        if config.warmup_steps:
            schedule = optax.join_schedules(
                [
                    optax.linear_schedule(0, peak, config.warmup_steps),
                    optax.linear_schedule(peak, 0, config.num_steps - config.warmup_steps),
                ],
                [config.warmup_steps],
            )
        else:
            schedule = optax.linear_schedule(peak, 0, config.num_steps)
        return optax.adam(schedule)

    return optimizer(config.adapter_learning_rate), optimizer(config.vocab_learning_rate)


def sft_logits(frozen: Transformer, params: SftParameters, token_ids, config: SftConfig, mask=None, *, positions=None):
    """Diagnostic logits, using the same rounded base and runtime precision as training."""
    model = eqx.combine(params.adapters, frozen, is_leaf=lambda x: isinstance(x, LoraProjection))
    selected = jnp.asarray(config.token_ids, jnp.int32)
    hidden, _ = model(token_ids, mask, embedding_delta=(selected, params.input_delta))
    if positions is not None:
        # The existing native sampler gathers prediction states before projecting
        # the vocabulary. Diagnostics must not materialize B*S*V logits.
        hidden = hidden.at[jnp.arange(token_ids.shape[0]), positions].get(out_sharding=P())
    logits = hidden @ model.output_proj
    correction = hidden @ params.output_delta.astype(hidden.dtype)
    return (
        logits.astype(jnp.float32)
        .at[..., selected]
        .add(correction.astype(jnp.float32), out_sharding=_partition_spec_of(logits))
    )


def _local_cross_entropy(hidden, head, labels, weights, delta, selected, *, vocab_block_size, token_block_size):
    """Stream both axes; AD has no head cotangent because the head is frozen."""
    width, vocab = head.shape
    tokens = labels.size
    tblock = min(token_block_size, tokens)
    vblock = min(vocab_block_size, vocab)
    padded_tokens = ((tokens + tblock - 1) // tblock) * tblock
    padded_vocab = ((vocab + vblock - 1) // vblock) * vblock
    h = jnp.pad(hidden.reshape(-1, width), ((0, padded_tokens - tokens), (0, 0)))
    target = jnp.pad(labels.reshape(-1), (0, padded_tokens - tokens))
    weight = jnp.pad(weights.reshape(-1), (0, padded_tokens - tokens))
    head = jnp.pad(head, ((0, 0), (0, padded_vocab - vocab)))

    def token_loss(total, start):
        x = jax.lax.dynamic_slice(h, (start, 0), (tblock, width))
        y = jax.lax.dynamic_slice(target, (start,), (tblock,))
        w = jax.lax.dynamic_slice(weight, (start,), (tblock,))
        correction = (x @ delta.astype(x.dtype)).astype(jnp.float32)

        def vocab_loss(carry, vstart):
            lse, correct_logit = carry
            columns = vstart + jnp.arange(vblock)
            block = jax.lax.dynamic_slice(head, (0, vstart), (width, vblock))
            logits = (x @ block).astype(jnp.float32)
            # The small equality matrix scatters only selected output corrections.
            logits = logits + correction @ (selected[:, None] == columns[None, :]).astype(jnp.float32)
            logits = jnp.where(columns[None, :] < vocab, logits, -jnp.inf)
            lse = jnp.logaddexp(lse, jax.nn.logsumexp(logits, axis=-1))
            correct_logit = correct_logit + jnp.sum(jnp.where(y[:, None] == columns, logits, 0), axis=-1)
            return (lse, correct_logit), None

        (lse, correct_logit), _ = jax.lax.scan(
            jax.checkpoint(vocab_loss),
            (jnp.full((tblock,), -jnp.inf), jnp.zeros((tblock,), jnp.float32)),
            jnp.arange(0, padded_vocab, vblock),
        )
        return total + jnp.sum((lse - correct_logit) * w), None

    total, _ = jax.lax.scan(jax.checkpoint(token_loss), jnp.array(0, jnp.float32), jnp.arange(0, padded_tokens, tblock))
    return total, jnp.sum(weight)


def compact_cross_entropy(hidden, head, labels, weights, delta, config: SftConfig):
    """Normal full-vocabulary CE with compact selected columns and bounded logits tiles."""
    mesh = get_abstract_mesh()
    token_axes = (_BATCH_AXES, _seq_axis(mesh))
    reduction_axes = (*_BATCH_AXES, *(() if token_axes[1] is None else (token_axes[1],)))

    def local(h, w, y, mask, d):
        total, count = _local_cross_entropy(
            h,
            w,
            y,
            mask,
            d,
            jnp.asarray(config.token_ids, jnp.int32),
            vocab_block_size=config.vocab_block_size,
            token_block_size=config.token_block_size,
        )
        total, count = jax.lax.psum((total, count), reduction_axes)
        return total / jnp.maximum(count, 1)

    # Reuse Hero's existing CE layout: local sequence/token shards and a replicated
    # read-only head. The head has no gradient or optimizer buffer.
    specs = (P(*token_axes, None), P(None, None), P(*token_axes), P(*token_axes), P(None, None))
    args = tuple(reshard(value, spec) for value, spec in zip((hidden, head, labels, weights, delta), specs, strict=True))
    return jax.shard_map(local, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False)(*args)


def sft_loss(params: SftParameters, frozen: Transformer, tokens, loss_weight, config: SftConfig, mask=None):
    model = eqx.combine(params.adapters, frozen, is_leaf=lambda x: isinstance(x, LoraProjection))
    hidden, routing = model(tokens, mask, embedding_delta=(jnp.asarray(config.token_ids, jnp.int32), params.input_delta))
    labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    loss = compact_cross_entropy(hidden, model.output_proj, labels, loss_weight, params.output_delta, config)
    return loss, _summarize_router_metrics(routing)


def make_sft_gradients(config: SftConfig):
    """Compile only the compact cotangents, preserving activation backpropagation."""

    @jax.jit
    def gradients(params: SftParameters, frozen: Transformer, tokens, loss_weight, token_valid, mask=None):
        (loss, routing), grads = jax.value_and_grad(sft_loss, has_aux=True)(
            params, frozen, tokens, loss_weight, config, mask
        )
        return grads, {
            "loss": loss,
            "gradient_norm": optax.global_norm(grads),
            "processed_tokens": jnp.sum(token_valid.astype(jnp.int32)),
            "supervised_tokens": jnp.sum((loss_weight > 0).astype(jnp.int32)),
            **routing,
        }

    return gradients


def make_sft_optimizer_step(config: SftConfig):
    """Compile Adam separately so the learning smoke can measure optimizer time."""
    adapter_optimizer, vocab_optimizer = sft_optimizers(config)

    @jax.jit
    def optimize(state: SftState, grads: SftParameters, processed, supervised):
        updates, adapter_state = adapter_optimizer.update(grads.adapters, state.adapter_opt_state, state.params.adapters)
        vocab = (state.params.input_delta, state.params.output_delta)
        vocab_updates, vocab_state = vocab_optimizer.update(
            (grads.input_delta, grads.output_delta), state.vocab_opt_state, vocab
        )
        input_delta, output_delta = cast(tuple[jax.Array, jax.Array], optax.apply_updates(vocab, vocab_updates))
        params = SftParameters(optax.apply_updates(state.params.adapters, updates), input_delta, output_delta)
        return SftState(
            state.step + 1,
            params,
            adapter_state,
            vocab_state,
            state.processed_tokens + processed,
            state.supervised_tokens + supervised,
        )

    return optimize


def make_sft_step(config: SftConfig):
    """Fused update for correctness/parity; smoke runtime times its two stages separately."""
    gradients = make_sft_gradients(config)
    optimize = make_sft_optimizer_step(config)

    @jax.jit
    def step(state: SftState, frozen: Transformer, tokens, loss_weight, token_valid, mask=None):
        grads, metrics = gradients(state.params, frozen, tokens, loss_weight, token_valid, mask)
        return optimize(state, grads, metrics["processed_tokens"], metrics["supervised_tokens"]), metrics

    return step


def compact_optimizer_report(state: SftState) -> dict:
    """Check Adam's exact compact moment trees, without comparing unrelated frozen shapes.

    This checks the parameter/state boundary and declared buffers. It does not
    infer allocation ancestry from tensor values or certify optimizer kernels.
    """
    errors = []
    if state.step.shape != () or state.step.dtype != jnp.int32:
        raise ValueError("SFT update counter must be a scalar int32")
    step = int(state.step)
    parameters = jax.tree_util.tree_flatten_with_path(state.params)[0]
    for path, value in parameters:
        name = jax.tree_util.keystr(path)
        adapter = name.startswith(".adapters.") and re.search(r"\.lora_(?:q|k|v|o|gate|up|down)\.[ab]$", name)
        if name not in (".input_delta", ".output_delta") and not adapter:
            errors.append(f"Noncompact parameter: {name}")
        if value.dtype != jnp.float32:
            errors.append(f"Non-FP32 parameter: {name}")
    groups = (
        ("adapters", state.params.adapters, state.adapter_opt_state),
        ("vocabulary", (state.params.input_delta, state.params.output_delta), state.vocab_opt_state),
    )
    for group, params, optimizer in groups:
        if not isinstance(optimizer, tuple) or len(optimizer) != 2:
            errors.append(f"Unexpected {group} optimizer state")
            continue
        adam, schedule = optimizer
        if type(adam) is not optax.ScaleByAdamState or type(schedule) is not optax.ScaleByScheduleState:
            errors.append(f"Unexpected {group} Adam/schedule state types")
            continue
        for label, moment in (("mu", adam.mu), ("nu", adam.nu)):
            if jax.tree.structure(moment) != jax.tree.structure(params):
                errors.append(f"{group}.{label} does not match the compact parameter tree")
                continue
            for (path, expected), actual in zip(
                jax.tree_util.tree_flatten_with_path(params)[0], jax.tree.leaves(moment), strict=True
            ):
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    errors.append(f"{group}.{label}{jax.tree_util.keystr(path)} shape/dtype differs")
        if any(count.shape != () or count.dtype != jnp.int32 for count in (adam.count, schedule.count)):
            errors.append(f"Unexpected {group} optimizer counters")
        elif int(adam.count) != step or int(schedule.count) != step:
            errors.append(f"{group} Adam/schedule counters do not equal SFT step {step}")
    parameter_bytes = sum(value.nbytes for _, value in parameters)
    optimizer_bytes = sum(value.nbytes for value in jax.tree.leaves((state.adapter_opt_state, state.vocab_opt_state)))
    expected_bytes = 2 * parameter_bytes + 4 * jnp.dtype(jnp.int32).itemsize
    if optimizer_bytes != expected_bytes:
        errors.append("Optimizer buffers differ from two compact FP32 moments plus four counters")
    return {
        "schema_version": 1,
        "optimizer_step": step,
        "compact_optimizer_only": not errors,
        "errors": errors,
        "trainable_parameter_bytes": parameter_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "expected_optimizer_state_bytes": expected_bytes,
        "scope": "exact compact parameter and Adam moment trees, shapes, dtypes, counters and bytes",
    }


def compiled_gradient_report(compiled, model: Transformer, config: SftConfig | None = None) -> dict:
    """Conservatively audit optimized dot/multiply/scatter/custom-call result shapes.

    Inspect all operations regardless of autodiff metadata or kernel name. In
    particular QuACK Wgrad uses the same generic launcher as activation GEMMs.
    Full-SFT compilation is the negative control. This audits declared outputs;
    it cannot establish the internal allocations of an opaque FFI implementation.
    """
    text = compiled.as_text()
    mesh = get_abstract_mesh()
    shapes: dict[tuple[int, ...], set[str]] = {}
    declared_shapes = {
        tuple(int(dim) for dim in match.split(",")) for match in re.findall(r"(?:bf16|f16|f32|f64)\[([0-9,]+)\]", text)
    }

    def add(path, shape):
        # Grouped GEMM output contract fixes expert/batch as the leading axis;
        # only matrix axes may transpose. Moving expert to a token axis would
        # confuse the ordinary weighted routing reduction with a bank Wgrad.
        layouts = (shape, (*shape[:-2], shape[-1], shape[-2]))
        for layout in layouts:
            shapes.setdefault(tuple(layout), set()).add(path)

    add("token_embed", model.token_embed.shape)
    add("output_proj", model.output_proj.shape)
    head_spec = (*(_partition_spec_of(model.output_proj) or ()), None, None)
    head_local_shape = []
    for dimension, axis in zip(model.output_proj.shape, head_spec[:2], strict=True):
        axes = () if axis is None else (axis,) if isinstance(axis, str) else axis
        head_local_shape.append(dimension // math.prod(mesh.shape[a] for a in axes))
    add("output_proj/local_shard", tuple(head_local_shape))
    if config is not None:
        add(
            "output_proj/vocab_tile",
            (model.output_proj.shape[0], min(config.vocab_block_size, model.output_proj.shape[1])),
        )
    for path, value in jax.tree_util.tree_flatten_with_path(model.stacked_blocks.stacked)[0]:
        name = jax.tree_util.keystr(path)
        short_conv = ".sconv_" in name and name.endswith(".weight")
        if (
            not (short_conv or any(part in name for part in (".w_", ".expert_mlp.", ".router")))
            or "lora_" in name
            or value.ndim < 3
        ):
            continue
        global_shape = tuple(value.shape[1:])
        spec = tuple(_partition_spec_of(value) or ()) + (None,) * value.ndim
        local_shape = []
        for dimension, axis in zip(value.shape, spec, strict=False):
            axes = () if axis is None else (axis,) if isinstance(axis, str) else axis
            local_shape.append(dimension // math.prod(mesh.shape[a] for a in axes))
        candidate_shapes = {global_shape, tuple(local_shape[1:])}
        if short_conv:
            # The joint short-conv kernel declares per-sequence-chunk dW
            # partials [P,W,C], even if only dX is consumed. Channel gathering
            # also makes its C larger than the stored FSDP shard.
            candidate_shapes.update(
                shape for shape in declared_shapes if len(shape) == 3 and shape[1:] in candidate_shapes
            )
        if ".expert_mlp." in name:
            # Context-gathered banks and expert chunks/waves can expose smaller
            # Wgrad outputs than the stored global bank.
            expert_size = int(mesh.shape.get("expert", 1))
            compute_experts = global_shape[0] // expert_size
            for experts in (global_shape[0], local_shape[1], compute_experts):
                for split in (
                    1,
                    model.config.expert_chunks,
                    model.config.num_expert_waves,
                    model.config.expert_chunks * model.config.num_expert_waves,
                ):
                    if experts % split == 0:
                        candidate_shapes.add((experts // split, *global_shape[1:]))
            for candidate in tuple(candidate_shapes):
                if "w_gate" in name or "w_up" in name:
                    candidate_shapes.add((*candidate[:-1], candidate[-1] * 2))
                if candidate[0] == 1:
                    candidate_shapes.add(candidate[1:])
        for candidate in candidate_shapes:
            add(name, candidate)

    audit = audit_gradient_operations(text, shapes)
    memory = compiled.memory_analysis()
    root = Path(__file__).resolve().parents[3]
    sources = (
        "experiments/grug/moe_hero_ep/sft.py",
        "experiments/grug/moe_hero_ep/sft_hlo.py",
        "experiments/grug/moe_hero_ep/model.py",
        "lib/levanter/src/levanter/grug/_moe/sonic_cute.py",
        "lib/levanter/src/levanter/grug/_moe/sonic.py",
        "lib/levanter/src/levanter/grug/_moe/topk.py",
        "lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py",
        "lib/levanter/src/levanter/grug/_moe/quack_moe_cute.py",
        "lib/levanter/src/levanter/kernels/triton/short_conv/api.py",
        "lib/levanter/src/levanter/kernels/triton/short_conv/triton_gpu.py",
        "lib/levanter/src/levanter/kernels/triton/short_conv/reference.py",
    )
    source_digests = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    return {
        "schema_version": 4,
        "backend": jax.default_backend(),
        "hlo_digest": digest(text),
        "model_digest": digest(draccus.encode(model.config)),
        "source_sha256": source_digests,
        **dataclasses.asdict(audit),
        "inspected_projection_shapes": {str(shape): sorted(paths) for shape, paths in sorted(shapes.items())},
        "avoids_full_base_gradients": audit.avoids_full_base_gradients,
        "scope": (
            "declared optimized dot/multiply/scatter/custom-call and convert/bitcast-convert results; "
            "opaque kernel internals and other operations require source audit"
        ),
        "compiled_output_bytes": memory.output_size_in_bytes,
        "compiled_temporary_bytes": memory.temp_size_in_bytes,
    }


def sft_manifest(reference: BaseReference, config: SftConfig) -> dict:
    return {
        "schema_version": 1,
        "candidate": "D",
        "base": dataclasses.asdict(reference),
        "sft": dataclasses.asdict(config),
        "runtime_base_dtype": "bfloat16",
        "trainable_dtype": "float32",
        "routing_control_dtype": "float32",
        "routing_control_policy": "candidate C effective bias and authoritative pending state remain unchanged",
        "adapted_projections": (
            ["stacked_blocks.stacked.attn.w_" + name for name in ("q", "k", "v", "o")]
            + ["stacked_blocks.stacked.shared[*].w_" + name for name in ("gate", "up", "down")]
        ),
        "sharding": {
            "adapter_a": "[base input axis, replicated rank]",
            "adapter_b": "[replicated rank, base output axis]",
            "vocab_deltas": "replicated",
        },
    }


def save_sft_checkpoint(path: str, state: SftState, reference: BaseReference, config: SftConfig):
    """Persist compact optimizer/adapter state; completion metadata binds exact candidate C."""
    if (StoragePath(path) / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite a completed SFT checkpoint: {path}")
    save_checkpoint(
        state,
        int(state.step),
        path,
        is_temporary=False,
        metadata={
            "hero_sft": sft_manifest(reference, config),
            "hero_sft_counters": {
                "sft_step": int(state.step),
                "processed_tokens": int(state.processed_tokens),
                "supervised_tokens": int(state.supervised_tokens),
            },
        },
    )


def restore_sft_checkpoint(path: str, template: SftState, reference: BaseReference, config: SftConfig, mesh: Mesh):
    metadata = json.loads((StoragePath(path) / "metadata.json").read_text())
    if metadata.get("is_temporary") is not False or digest(metadata.get("hero_sft")) != digest(
        sft_manifest(reference, config)
    ):
        raise ValueError("Compact checkpoint does not match the exact base, tokenizer, or SFT configuration")
    return load_checkpoint(template, path, axis_mapping=None, mesh=mesh, allow_partial=False)


def merge_sft(frozen: Transformer, params: SftParameters, config: SftConfig, serving_dtype=jnp.bfloat16) -> Transformer:
    """Merge in FP32 into the BF16-rounded training base, then cast for serving."""
    model = eqx.combine(params.adapters, frozen, is_leaf=lambda x: isinstance(x, LoraProjection))
    block = model.stacked_blocks.stacked
    for name in ("q", "k", "v", "o"):
        adapter = getattr(block.attn, f"lora_{name}")
        weight = getattr(block.attn, f"w_{name}")
        merged = (weight.astype(jnp.float32) + (adapter.a @ adapter.b) * adapter.scale).astype(serving_dtype)
        block = eqx.tree_at(lambda b, name=name: getattr(b.attn, f"w_{name}"), block, merged)
        block = eqx.tree_at(lambda b, name=name: getattr(b.attn, f"lora_{name}"), block, None)
    if block.shared is not None:
        for index, shared in enumerate(block.shared):
            for name in ("gate", "up", "down"):
                adapter = getattr(shared, f"lora_{name}")
                weight = getattr(shared, f"w_{name}")
                merged = (weight.astype(jnp.float32) + (adapter.a @ adapter.b) * adapter.scale).astype(serving_dtype)
                block = eqx.tree_at(
                    lambda b, index=index, name=name: getattr(b.shared[index], f"w_{name}"), block, merged
                )
                block = eqx.tree_at(
                    lambda b, index=index, name=name: getattr(b.shared[index], f"lora_{name}"), block, None
                )
    model = eqx.tree_at(lambda m: m.stacked_blocks.stacked, model, block)
    selected = jnp.asarray(config.token_ids, jnp.int32)
    embedding = (
        model.token_embed.astype(jnp.float32)
        .at[selected]
        .add(params.input_delta, out_sharding=_partition_spec_of(model.token_embed))
        .astype(serving_dtype)
    )
    head = (
        model.output_proj.astype(jnp.float32)
        .at[:, selected]
        .add(params.output_delta, out_sharding=_partition_spec_of(model.output_proj))
        .astype(serving_dtype)
    )
    model = eqx.tree_at(lambda m: (m.token_embed, m.output_proj), model, (embedding, head))
    return cast_weights(model, serving_dtype)


def export_sft(
    path: str,
    frozen: Transformer,
    state: SftState,
    reference: BaseReference,
    config: SftConfig,
    serving_dtype=jnp.bfloat16,
):
    """Write ordinary weights with candidate C's unchanged authoritative QB state."""
    if (StoragePath(path) / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite a completed SFT export: {path}")
    merged = merge_sft(frozen, state.params, config, serving_dtype)
    # apply_qb_betas SETS biases. Zero pending state would erase learned routing
    # biases when a normal consumer reloads the exported checkpoint. Reuse C's
    # small pending array exactly rather than recentering rounded effective bias.
    read_prepared_provenance(reference.checkpoint, reference.metadata_digest)
    pending = checkpointing.load_grug_checkpoint(
        state={
            "pending_qb_betas": jax.ShapeDtypeStruct((merged.config.num_layers, merged.config.num_experts), jnp.float32)
        },
        candidate=reference.checkpoint,
        mesh=jax.sharding.get_mesh(),
        allow_partial=False,
    )["pending_qb_betas"]
    save_checkpoint(
        {
            "params": merged,
            "pending_qb_betas": pending,
        },
        int(state.step),
        path,
        is_temporary=False,
        metadata={
            "hero_sft": sft_manifest(reference, config),
            "merge_base": "BF16-rounded candidate C",
            "merge_accumulator_dtype": "float32",
            "serving_dtype": str(jnp.dtype(serving_dtype)),
        },
    )
