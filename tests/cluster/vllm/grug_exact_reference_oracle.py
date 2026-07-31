# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Build the frozen July 27 GrugMoE serving-semantics fixture.

The parent process archives the exact training-reference commit into a temporary
directory. A clean child process then imports that archive, initializes the
model classes from it, and emits vLLM-layout weights plus immutable numerical
observations. This prevents the current checkout from silently becoming the
reference implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

FROZEN_TRAINING_COMMIT = "fd3e9bc5b428633027f944be7fdf1136567db028"
_WORKER_ENV = "MARIN_GRUG_EXACT_REFERENCE_WORKER"
_FROZEN_ROOT_ENV = "MARIN_GRUG_EXACT_REFERENCE_ROOT"
_ARCHIVE_PATHS = ("experiments/grug", "lib/levanter/src")
_MESH_AXIS_NAMES = ("replica_dcn", "data", "expert", "model")
_SEMANTIC_PROMPT_LENGTH = 33
_BOUNDARY_PROMPT_LENGTH = 514
_ROUTING_SUM = 2.5


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _run_worker(source_repo: Path, output_dir: Path) -> None:
    source_repo = Path(
        subprocess.check_output(
            ["git", "-C", str(source_repo), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )
    resolved_commit = subprocess.check_output(
        ["git", "-C", str(source_repo), "rev-parse", FROZEN_TRAINING_COMMIT],
        text=True,
    ).strip()
    if resolved_commit != FROZEN_TRAINING_COMMIT:
        raise RuntimeError(f"training reference resolved to {resolved_commit}, expected {FROZEN_TRAINING_COMMIT}")

    with tempfile.TemporaryDirectory(prefix="grug-exact-reference-") as temporary_dir_str:
        temporary_dir = Path(temporary_dir_str)
        archive_path = temporary_dir / "source.tar"
        frozen_root = temporary_dir / "source"
        frozen_root.mkdir()
        subprocess.run(
            [
                "git",
                "-C",
                str(source_repo),
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                FROZEN_TRAINING_COMMIT,
                *_ARCHIVE_PATHS,
            ],
            check=True,
        )
        with tarfile.open(archive_path) as archive:
            archive.extractall(frozen_root, filter="data")
        # ``experiments`` is a namespace in the repository. Make it an explicit
        # package in the isolated archive so no editable checkout can shadow it.
        (frozen_root / "experiments" / "__init__.py").touch()

        env = os.environ.copy()
        env[_WORKER_ENV] = "1"
        env[_FROZEN_ROOT_ENV] = str(frozen_root)
        env["PYTHONPATH"] = os.pathsep.join(
            (
                str(frozen_root / "lib" / "levanter" / "src"),
                str(frozen_root),
            )
        )
        for name in tuple(env):
            if name.startswith("SCALE_"):
                del env[name]
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source-repo",
                str(source_repo),
                "--output",
                str(output_dir.resolve()),
            ],
            cwd=temporary_dir,
            env=env,
            check=True,
        )


def _emit_fixture(output_dir: Path) -> None:
    # Imports are deliberately worker-only. Their provenance is checked below.
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, Mesh
    from levanter.grug.attention import (
        AttentionMask,
        align_kv_heads,
        apply_rotary_embedding_fused,
        reference_attention,
    )
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from experiments.grug.moe import model as frozen_grug
    from experiments.grug.moe.model import GrugModelConfig, Transformer

    frozen_root = Path(os.environ[_FROZEN_ROOT_ENV]).resolve()
    imported_model_path = Path(frozen_grug.__file__).resolve()
    expected_model_path = (frozen_root / "experiments" / "grug" / "moe" / "model.py").resolve()
    if imported_model_path != expected_model_path:
        raise RuntimeError(f"loaded Grug implementation from {imported_model_path}, expected {expected_model_path}")

    config = GrugModelConfig(
        vocab_size=64,
        # FA4's non-TMA paged-KV path on GB200 requires a 16-aligned head.
        # Keep four heads so the 2-local/1-global KV-head distinction remains.
        hidden_dim=64,
        intermediate_dim=16,
        shared_expert_intermediate_dim=32,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=7,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=16,
        max_seq_len=1024,
        sliding_window=512,
        global_every=6,
        disable_long_rope=True,
        rope_fraction=0.5,
        rope_fused=True,
        mtp_depth=1,
        mtp_dense=True,
        mtp_intermediate_dim=32,
        over_encoding_vocab_size=0,
        gated_norm=True,
        attn_gate=True,
        xsa=True,
        qb_routing=True,
        sconv=True,
        sconv_kernel=4,
        sconv_sites=("k", "v", "attn", "mlp"),
        attention_implementation="reference",
    )
    devices = np.asarray(jax.local_devices()[:1], dtype=object).reshape((1, 1, 1, 1))
    mesh = Mesh(
        devices,
        _MESH_AXIS_NAMES,
        axis_types=tuple(AxisType.Explicit for _ in _MESH_AXIS_NAMES),
    )
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        model = Transformer.init(config, key=jax.random.key(20260727))
        model = _install_fixture_values(model, eqx=eqx, jnp=jnp, np=np)
        if model.mtp_block is None or model.mtp_proj is None:
            raise AssertionError("the frozen reference must instantiate MTP depth 1")

        semantic_ids = _prompt_ids(_SEMANTIC_PROMPT_LENGTH, np=np, phase=7)
        boundary_ids = _prompt_ids(_BOUNDARY_PROMPT_LENGTH, np=np, phase=19)
        semantic_logprobs, semantic_observations = _evaluate(
            model,
            jnp.asarray(semantic_ids),
            jax=jax,
            jnp=jnp,
            AttentionMask=AttentionMask,
            align_kv_heads=align_kv_heads,
            apply_rotary_embedding_fused=apply_rotary_embedding_fused,
            reference_attention=reference_attention,
            capture=True,
        )
        boundary_logprobs, _ = _evaluate(
            model,
            jnp.asarray(boundary_ids),
            jax=jax,
            jnp=jnp,
            AttentionMask=AttentionMask,
            align_kv_heads=align_kv_heads,
            apply_rotary_embedding_fused=apply_rotary_embedding_fused,
            reference_attention=reference_attention,
            capture=False,
        )
        direct_semantic_logprobs = jax.nn.log_softmax(
            model.logits(jnp.asarray(semantic_ids)[None, :]).astype(jnp.float32),
            axis=-1,
        )[0]
        np.testing.assert_allclose(
            np.asarray(jax.device_get(semantic_logprobs)),
            np.asarray(jax.device_get(direct_semantic_logprobs)),
            rtol=1e-5,
            atol=1e-5,
        )

        host_observations = {name: np.asarray(jax.device_get(value)) for name, value in semantic_observations.items()}
        route_margins = np.asarray(
            [host_observations[f"route_margin.layer.{index}"] for index in range(config.num_layers)]
        )
        if float(route_margins.min()) <= 0.15:
            raise AssertionError(f"routing fixture is not margin-separated: {route_margins}")
        for layer_index in range(config.num_layers):
            weights = host_observations[f"normalized_weights.layer.{layer_index}"]
            np.testing.assert_allclose(weights.sum(axis=-1), _ROUTING_SUM, rtol=1e-6, atol=1e-6)

        state = _to_vllm_state(model, jax=jax, np=np)
        # Keep routing tensors lossless because the fixture compares normalized
        # weights at tight tolerance. Half-precision storage for the remaining
        # tensors keeps this checked-in exact-path fixture small; the live
        # server loads them into FP16 and checks logits at serving tolerance.
        state = {name: value if ".mlp.router." in name else value.astype(np.float16) for name, value in state.items()}
        shared_zero = all(
            not np.any(state[f"model.layers.0.shared_experts.{index}.gate_proj.weight"])
            for index in range(config.num_shared_experts)
        )
        if shared_zero:
            raise AssertionError("shared-expert fixture weights must be nonzero")
        if np.array_equal(
            state["model.layers.0.shared_experts.0.gate_proj.weight"],
            state["model.layers.0.shared_experts.1.gate_proj.weight"],
        ):
            raise AssertionError("the two shared experts must be distinct")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.safetensors"
    observations_path = output_dir / "observations.npz"
    config_path = output_dir / "config.json"
    save_file(state, model_path)
    np.savez_compressed(
        observations_path,
        semantic_input_ids=semantic_ids,
        semantic_logprobs=np.asarray(jax.device_get(semantic_logprobs)),
        boundary_input_ids=boundary_ids,
        boundary_logprobs=np.asarray(jax.device_get(boundary_logprobs[-1])),
        local_layer_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        global_layer_indices=np.asarray([5, 6], dtype=np.int32),
        **host_observations,
    )
    config_path.write_text(json.dumps(_hf_config(config), indent=2, sort_keys=True) + "\n")
    _write_tokenizer(
        output_dir, Tokenizer=Tokenizer, WordLevel=WordLevel, Whitespace=Whitespace, Fast=PreTrainedTokenizerFast
    )

    tracked_files = (
        "config.json",
        "model.safetensors",
        "observations.npz",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    manifest = {
        "schema_version": 1,
        "producer": "tests/cluster/vllm/grug_exact_reference_oracle.py",
        "training_reference_commit": FROZEN_TRAINING_COMMIT,
        "training_reference_model_path": "experiments/grug/moe/model.py",
        "training_reference_model_sha256": _sha256(expected_model_path),
        "imported_training_reference": str(imported_model_path.relative_to(frozen_root)),
        "matmul_precision": "highest",
        "jax_backend": jax.default_backend(),
        "ordinary_serving_path": {
            "trunk_logits": True,
            "mtp_depth": 1,
            "mtp_training_head_excluded": True,
            "over_encoding": False,
        },
        "semantics": {
            "layers": 7,
            "local_layers": [0, 1, 2, 3, 4],
            "global_nope_layers": [5, 6],
            "local_kv_heads": 2,
            "global_kv_heads": 1,
            "stored_kv_heads": 2,
            "rope": "interleaved-pair fused half RoPE on local layers",
            "sliding_window": 512,
            "sconv_kernel": 4,
            "sconv_sites": ["k", "v", "attn", "mlp"],
            "shared_experts": 2,
            "shared_expert_width_each": 16,
            "tensor_storage": "float16 except float32 router tensors",
            "qb_top_k": 2,
            "qb_candidate_count": 3,
            "combine_weight_sum": _ROUTING_SUM,
        },
        "prompts": {
            "semantic_tokens": _SEMANTIC_PROMPT_LENGTH,
            "boundary_tokens": _BOUNDARY_PROMPT_LENGTH,
            "prefix_reuse_tokens": 512,
            "boundary_condition": "append positions 512 and 513 after a cached 512-token prefix",
        },
        "files": {name: _sha256(output_dir / name) for name in tracked_files},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _pattern(shape, *, seed: int, scale: float, offset: float, jnp, np):
    count = int(np.prod(shape))
    index = np.arange(1, count + 1, dtype=np.float64)
    frequency = 0.011 + seed * 0.00037
    values = np.sin(index * frequency + seed * 0.17)
    values += 0.47 * np.cos(index * (frequency * 1.73) - seed * 0.09)
    return jnp.asarray((offset + scale * values).reshape(shape), dtype=jnp.float32)


def _sconv_pattern(shape, *, seed: int, jnp, np):
    base = np.asarray([0.71, 0.19, -0.12, 0.07], dtype=np.float32)
    values = np.broadcast_to(base.reshape(1, 4, 1), shape).copy()
    variation = np.asarray(_pattern(shape, seed=seed, scale=0.012, offset=0.0, jnp=jnp, np=np))
    return jnp.asarray(values + variation, dtype=jnp.float32)


def _install_fixture_values(model, *, eqx, jnp, np):
    model = eqx.tree_at(
        lambda tree: tree.token_embed,
        model,
        _pattern(model.token_embed.shape, seed=1, scale=0.31, offset=0.0, jnp=jnp, np=np),
    )
    model = eqx.tree_at(
        lambda tree: tree.embed_norm.weight,
        model,
        _pattern(model.embed_norm.weight.shape, seed=2, scale=0.06, offset=1.0, jnp=jnp, np=np),
    )
    model = eqx.tree_at(
        lambda tree: tree.final_norm.weight,
        model,
        _pattern(model.final_norm.weight.shape, seed=3, scale=0.06, offset=1.0, jnp=jnp, np=np),
    )
    model = eqx.tree_at(
        lambda tree: tree.output_proj,
        model,
        _pattern(model.output_proj.shape, seed=4, scale=0.8, offset=0.0, jnp=jnp, np=np),
    )

    block = model.stacked_blocks.stacked

    def replace(selector, value):
        nonlocal block
        block = eqx.tree_at(selector, block, value)

    replace(
        lambda tree: tree.rms_attn.weight,
        _pattern(block.rms_attn.weight.shape, seed=10, scale=0.07, offset=1.0, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.rms_mlp.weight,
        _pattern(block.rms_mlp.weight.shape, seed=11, scale=0.07, offset=1.0, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.attn_gated_norm.w_down,
        _pattern(block.attn_gated_norm.w_down.shape, seed=12, scale=0.035, offset=0.0, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.attn_gated_norm.w_up,
        _pattern(block.attn_gated_norm.w_up.shape, seed=13, scale=0.035, offset=0.0, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.mlp_gated_norm.w_down,
        _pattern(block.mlp_gated_norm.w_down.shape, seed=14, scale=0.035, offset=0.0, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.mlp_gated_norm.w_up,
        _pattern(block.mlp_gated_norm.w_up.shape, seed=15, scale=0.035, offset=0.0, jnp=jnp, np=np),
    )
    for seed, field in enumerate(("w_q", "w_k", "w_v", "w_o"), start=20):
        value = getattr(block.attn, field)
        replace(
            lambda tree, name=field: getattr(tree.attn, name),
            _pattern(value.shape, seed=seed, scale=0.055, offset=0.0, jnp=jnp, np=np),
        )
    replace(
        lambda tree: tree.attn.attn_gate,
        _pattern(block.attn.attn_gate.shape, seed=24, scale=0.04, offset=0.0, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.attn.sconv_k.weight,
        _sconv_pattern(block.attn.sconv_k.weight.shape, seed=25, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.attn.sconv_v.weight,
        _sconv_pattern(block.attn.sconv_v.weight.shape, seed=26, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.sconv_attn.weight,
        _sconv_pattern(block.sconv_attn.weight.shape, seed=27, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.sconv_mlp.weight,
        _sconv_pattern(block.sconv_mlp.weight.shape, seed=28, jnp=jnp, np=np),
    )
    replace(
        lambda tree: tree.mlp.router,
        _pattern(block.mlp.router.shape, seed=30, scale=0.021, offset=0.0, jnp=jnp, np=np),
    )
    base_bias = np.asarray([0.9, 0.3, -0.3, -0.9], dtype=np.float32)
    router_bias = np.stack(
        [np.roll(base_bias, layer_index % base_bias.size) for layer_index in range(block.mlp.router_bias.shape[0])]
    )
    replace(lambda tree: tree.mlp.router_bias, jnp.asarray(router_bias))
    for seed, field in enumerate(("w_gate", "w_up", "w_down"), start=31):
        value = getattr(block.mlp.expert_mlp, field)
        replace(
            lambda tree, name=field: getattr(tree.mlp.expert_mlp, name),
            _pattern(value.shape, seed=seed, scale=0.052, offset=0.0, jnp=jnp, np=np),
        )
    for shared_index, shared in enumerate(block.shared):
        for field_index, field in enumerate(("w_gate", "w_up", "w_down")):
            value = getattr(shared, field)
            seed = 40 + shared_index * 4 + field_index
            replace(
                lambda tree, index=shared_index, name=field: getattr(tree.shared[index], name),
                _pattern(value.shape, seed=seed, scale=0.057, offset=0.0, jnp=jnp, np=np),
            )
    return eqx.tree_at(lambda tree: tree.stacked_blocks.stacked, model, block)


def _prompt_ids(length: int, *, np, phase: int):
    positions = np.arange(length, dtype=np.int64)
    # Keep special IDs 0..2 out of the numerical prompts.
    return ((positions * positions * 7 + positions * phase + 11) % 61 + 3).astype(np.int32)


def _rms(x, weight, eps, *, jax, jnp):
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    return (x_float * jax.lax.rsqrt(variance + eps) * weight).astype(x.dtype)


def _nonparametric_rms(x, *, jax, jnp):
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    return (x_float * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)


def _gated_norm(x, module, *, jax, jnp):
    hidden = jnp.einsum("...d,dr->...r", x, module.w_down)
    hidden = jax.nn.silu(hidden)
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", hidden, module.w_up))
    return x * gate.astype(x.dtype)


def _short_conv(x, module, *, jnp):
    output = module.weight[0] * x
    sequence_length = x.shape[1]
    for lag in range(1, module.kernel_size):
        shifted = jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, :sequence_length, :]
        output = output + module.weight[lag] * shifted
    return output


def _dense_mlp(x, module, *, jax, jnp):
    gate = jnp.einsum("bsd,di->bsi", x, module.w_gate)
    up = jnp.einsum("bsd,di->bsi", x, module.w_up)
    return jnp.einsum("bsi,id->bsd", jax.nn.silu(gate) * up, module.w_down)


def _routed_mlp(x, module, *, jax, jnp):
    flat = x.reshape(-1, x.shape[-1])
    logits = jnp.einsum("td,de->te", flat, module.router).astype(jnp.float32)
    top_values, top_indices = jax.lax.top_k(
        logits + module.router_bias,
        module.cfg.num_experts_per_token + 1,
    )
    selected = top_indices[:, :-1]
    selected_logits = jnp.take_along_axis(logits, selected, axis=-1)
    weights = jax.nn.sigmoid(selected_logits)
    weights = weights * (_ROUTING_SUM / (weights.sum(axis=-1, keepdims=True) + 1e-9))

    experts = module.expert_mlp
    gate = jnp.einsum("td,edi->tei", flat, experts.w_gate)
    up = jnp.einsum("td,edi->tei", flat, experts.w_up)
    expert_outputs = jnp.einsum("tei,eid->ted", jax.nn.silu(gate) * up, experts.w_down)
    dispatch = jax.nn.one_hot(selected, module.cfg.num_experts)
    dispatch = jnp.einsum("tke,tk->te", dispatch, weights)
    output = jnp.einsum("te,ted->td", dispatch, expert_outputs).reshape(x.shape)
    margin = jnp.min(top_values[:, -2] - top_values[:, -1])
    return output, selected, weights, margin


def _evaluate(
    model,
    token_ids,
    *,
    jax,
    jnp,
    AttentionMask,
    align_kv_heads,
    apply_rotary_embedding_fused,
    reference_attention,
    capture: bool,
):
    config = model.config
    hidden = model.token_embed[token_ids][None, :, :]
    hidden = _rms(hidden, model.embed_norm.weight, model.embed_norm.eps, jax=jax, jnp=jnp)
    observations: dict[str, Any] = {}

    for layer_index, layer in enumerate(model.stacked_blocks.unstacked()):
        is_global = (layer_index + 1) % config.global_every == 0 or layer_index == config.num_layers - 1
        attn_input = _rms(hidden, layer.rms_attn.weight, layer.rms_attn.eps, jax=jax, jnp=jnp)
        attn_input = _gated_norm(attn_input, layer.attn_gated_norm, jax=jax, jnp=jnp)

        q_flat = jnp.einsum("bsd,dh->bsh", attn_input, layer.attn.w_q)
        k_flat = jnp.einsum("bsd,dh->bsh", attn_input, layer.attn.w_k)
        v_flat = jnp.einsum("bsd,dh->bsh", attn_input, layer.attn.w_v)
        k_flat = _short_conv(k_flat, layer.attn.sconv_k, jnp=jnp)
        v_flat = _short_conv(v_flat, layer.attn.sconv_v, jnp=jnp)

        q = q_flat.reshape(1, token_ids.shape[0], config.num_heads, config.inferred_head_dim)
        k = k_flat.reshape(1, token_ids.shape[0], config.stored_kv_heads, config.inferred_head_dim)
        v = v_flat.reshape(1, token_ids.shape[0], config.stored_kv_heads, config.inferred_head_dim)
        logical_kv_heads = config.global_kv_heads if is_global else config.local_kv_heads
        if logical_kv_heads != config.stored_kv_heads:
            k = align_kv_heads(k[:, :, :logical_kv_heads, :], num_q_heads=config.stored_kv_heads)
            v = align_kv_heads(v[:, :, :logical_kv_heads, :], num_q_heads=config.stored_kv_heads)
        q = _nonparametric_rms(q, jax=jax, jnp=jnp)
        k = _nonparametric_rms(k, jax=jax, jnp=jnp)
        q, k = apply_rotary_embedding_fused(
            q,
            k,
            seq_len=token_ids.shape[0],
            head_dim=config.inferred_head_dim,
            rotary_dim=config.rotary_dim(config.inferred_head_dim),
            rope=config.rope,
            disable_rope=jnp.asarray(is_global),
        )
        q = q * config.qk_mult
        mask = AttentionMask.causal(sliding_window=None if is_global else config.sliding_window)
        attn_output = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

        aligned_value = align_kv_heads(v, num_q_heads=config.num_heads)
        dot = jnp.sum(attn_output * aligned_value, axis=-1, keepdims=True)
        value_norm_sq = jnp.sum(aligned_value * aligned_value, axis=-1, keepdims=True)
        attn_output = attn_output - (dot / (value_norm_sq + 1e-6)) * aligned_value
        attention_gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", attn_input, layer.attn.attn_gate))
        attn_output = attn_output * attention_gate[..., None]
        attn_output = attn_output.reshape(1, token_ids.shape[0], -1)
        attn_output = jnp.einsum("bsh,hd->bsd", attn_output, layer.attn.w_o)
        attn_output = _short_conv(attn_output, layer.sconv_attn, jnp=jnp)
        hidden = hidden + attn_output

        mlp_input = _rms(hidden, layer.rms_mlp.weight, layer.rms_mlp.eps, jax=jax, jnp=jnp)
        mlp_input = _gated_norm(mlp_input, layer.mlp_gated_norm, jax=jax, jnp=jnp)
        routed, selected, weights, route_margin = _routed_mlp(mlp_input, layer.mlp, jax=jax, jnp=jnp)
        mlp_output = routed
        for shared_expert in layer.shared:
            mlp_output = mlp_output + _dense_mlp(mlp_input, shared_expert, jax=jax, jnp=jnp)
        mlp_output = _short_conv(mlp_output, layer.sconv_mlp, jnp=jnp)
        hidden = hidden + mlp_output

        if capture:
            observations[f"mlp_input.layer.{layer_index}"] = mlp_input[0]
            observations[f"selected_experts.layer.{layer_index}"] = selected
            observations[f"normalized_weights.layer.{layer_index}"] = weights
            observations[f"route_margin.layer.{layer_index}"] = route_margin
            observations[f"hidden.layer.{layer_index}"] = hidden[0]

    final_hidden = _rms(hidden, model.final_norm.weight, model.final_norm.eps, jax=jax, jnp=jnp)
    logits = jnp.einsum("bsd,dv->bsv", final_hidden, model.output_proj)
    logprobs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)[0]
    if capture:
        observations["hidden.final"] = final_hidden[0]
        observations["semantic_all_logprobs"] = logprobs
    return logprobs, observations


def _host(value, *, jax, np):
    return np.ascontiguousarray(np.asarray(jax.device_get(value), dtype=np.float32))


def _to_vllm_state(model, *, jax, np) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model.embed_tokens.weight": _host(model.token_embed, jax=jax, np=np),
        "model.embed_norm.weight": _host(model.embed_norm.weight, jax=jax, np=np),
        "model.norm.weight": _host(model.final_norm.weight, jax=jax, np=np),
        "lm_head.weight": _host(model.output_proj.T, jax=jax, np=np),
    }
    for layer_index, layer in enumerate(model.stacked_blocks.unstacked()):
        prefix = f"model.layers.{layer_index}"
        state[f"{prefix}.input_layernorm.weight"] = _host(layer.rms_attn.weight, jax=jax, np=np)
        state[f"{prefix}.post_attention_layernorm.weight"] = _host(layer.rms_mlp.weight, jax=jax, np=np)
        state[f"{prefix}.attn_gated_norm.down_proj.weight"] = _host(layer.attn_gated_norm.w_down.T, jax=jax, np=np)
        state[f"{prefix}.attn_gated_norm.up_proj.weight"] = _host(layer.attn_gated_norm.w_up.T, jax=jax, np=np)
        state[f"{prefix}.mlp_gated_norm.down_proj.weight"] = _host(layer.mlp_gated_norm.w_down.T, jax=jax, np=np)
        state[f"{prefix}.mlp_gated_norm.up_proj.weight"] = _host(layer.mlp_gated_norm.w_up.T, jax=jax, np=np)
        for source_name, target_name in (
            ("w_q", "q_proj"),
            ("w_k", "k_proj"),
            ("w_v", "v_proj"),
            ("w_o", "o_proj"),
        ):
            state[f"{prefix}.self_attn.{target_name}.weight"] = _host(getattr(layer.attn, source_name).T, jax=jax, np=np)
        state[f"{prefix}.self_attn.attn_gate.weight"] = _host(layer.attn.attn_gate.T, jax=jax, np=np)
        state[f"{prefix}.self_attn.sconv_k.weight"] = _host(layer.attn.sconv_k.weight, jax=jax, np=np)
        state[f"{prefix}.self_attn.sconv_v.weight"] = _host(layer.attn.sconv_v.weight, jax=jax, np=np)
        state[f"{prefix}.sconv_attn.weight"] = _host(layer.sconv_attn.weight, jax=jax, np=np)
        state[f"{prefix}.sconv_mlp.weight"] = _host(layer.sconv_mlp.weight, jax=jax, np=np)
        state[f"{prefix}.mlp.router.weight"] = _host(layer.mlp.router.T, jax=jax, np=np)
        state[f"{prefix}.mlp.router.bias"] = _host(layer.mlp.router_bias, jax=jax, np=np)
        state[f"{prefix}.mlp.experts.gate_proj.weight"] = _host(
            np.swapaxes(np.asarray(jax.device_get(layer.mlp.expert_mlp.w_gate)), -1, -2),
            jax=jax,
            np=np,
        )
        state[f"{prefix}.mlp.experts.up_proj.weight"] = _host(
            np.swapaxes(np.asarray(jax.device_get(layer.mlp.expert_mlp.w_up)), -1, -2),
            jax=jax,
            np=np,
        )
        state[f"{prefix}.mlp.experts.down_proj.weight"] = _host(
            np.swapaxes(np.asarray(jax.device_get(layer.mlp.expert_mlp.w_down)), -1, -2),
            jax=jax,
            np=np,
        )
        for shared_index, shared in enumerate(layer.shared):
            shared_prefix = f"{prefix}.shared_experts.{shared_index}"
            state[f"{shared_prefix}.gate_proj.weight"] = _host(shared.w_gate.T, jax=jax, np=np)
            state[f"{shared_prefix}.up_proj.weight"] = _host(shared.w_up.T, jax=jax, np=np)
            state[f"{shared_prefix}.down_proj.weight"] = _host(shared.w_down.T, jax=jax, np=np)
    return state


def _hf_config(config) -> dict[str, Any]:
    return {
        "architectures": ["GrugMoeForCausalLM"],
        "model_type": "grug_moe",
        "vocab_size": config.vocab_size,
        "hidden_dim": config.hidden_dim,
        "hidden_size": config.hidden_dim,
        "intermediate_dim": config.intermediate_dim,
        "intermediate_size": config.intermediate_dim,
        "moe_intermediate_size": config.intermediate_dim,
        "shared_expert_intermediate_dim": config.shared_expert_intermediate_dim,
        "shared_expert_intermediate_size": config.shared_expert_intermediate_dim,
        "num_shared_experts": config.num_shared_experts,
        "num_experts": config.num_experts,
        "num_local_experts": config.num_experts,
        "num_experts_per_token": config.num_experts_per_token,
        "num_experts_per_tok": config.num_experts_per_token,
        "num_layers": config.num_layers,
        "num_hidden_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_attention_heads": config.num_heads,
        "num_kv_heads": config.stored_kv_heads,
        "num_key_value_heads": config.stored_kv_heads,
        "local_kv_heads": config.local_kv_heads,
        "global_kv_heads": config.global_kv_heads,
        "head_dim": config.inferred_head_dim,
        "attention_head_dim": config.inferred_head_dim,
        "max_seq_len": config.max_seq_len,
        "max_position_embeddings": config.max_seq_len,
        "sliding_window": config.sliding_window,
        "global_every": config.global_every,
        "layer_norm_eps": config.layer_norm_eps,
        "rms_norm_eps": config.layer_norm_eps,
        "initializer_std": config.initializer_std,
        "initializer_range": config.initializer_std,
        "qk_mult": config.qk_mult,
        "qk_mult_long_scale": 1.0,
        "disable_pko": True,
        "disable_long_rope": config.disable_long_rope,
        "rope_fraction": config.rope_fraction,
        "rope_fused": config.rope_fused,
        "rope": {"theta": config.rope.theta},
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": config.rope.theta,
        },
        "rope_theta": config.rope.theta,
        "gated_norm": config.gated_norm,
        "attn_gate": config.attn_gate,
        "xsa": config.xsa,
        "qb_routing": config.qb_routing,
        "legacy_input_output_gated_norm": False,
        "mtp_depth": config.mtp_depth,
        "mtp_dense": config.mtp_dense,
        "over_encoding_vocab_size": config.over_encoding_vocab_size,
        "sconv": config.sconv,
        "sconv_kernel": config.sconv_kernel,
        "sconv_sites": list(config.sconv_sites),
        "dtype": "float32",
        "torch_dtype": "float32",
        "tie_word_embeddings": False,
        "use_cache": True,
    }


def _write_tokenizer(output_dir: Path, *, Tokenizer, WordLevel, Whitespace, Fast) -> None:
    vocab = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        **{f"token_{index}": index for index in range(3, 64)},
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = Fast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        model_max_length=1024,
    )
    fast.save_pretrained(output_dir)
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = output_dir / name
        path.write_text(path.read_text().rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get(_WORKER_ENV) == "1":
        _emit_fixture(args.output)
    else:
        _run_worker(args.source_repo, args.output)


if __name__ == "__main__":
    main()
