# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Snowball tests (no dependency on marin ``experiments/``).

These validate the Levanter-side contract: registration/discovery, config <-> HF round-trip,
bidirectional state-dict serialization (with an explicit HF-key/shape manifest so a shared
transpose bug cannot hide), and off-recipe rejection. The Snowball-vs-experiment numerical parity
harness lives on the marin side (``tests/test_snowball_grug_parity.py``) to respect the
levanter -> experiments dependency direction.
"""

import subprocess
import sys
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax
from haliax import Axis
from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict

from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmConfig
from levanter.models.snowball import (
    GRUG_MOE_ARCHITECTURE,
    GRUG_MOE_MODEL_TYPE,
    GrugMoeHfConfig,
    SnowballConfig,
    SnowballLMHeadModel,
    validate_single_name_config,
)


def _tiny_config(**overrides) -> SnowballConfig:
    base = dict(
        vocab_size=48,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=48,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=5,  # layer 3 => i%4==3 long; layer 4 => last long; 0..2 short
        num_heads=4,
        num_kv_heads=2,
        head_dim=12,  # 4*12=48 != hidden(32): q_proj is non-square, catches transpose bugs
        max_seq_len=32,
        sliding_window=4,
        qk_mult=1.37,
        layer_norm_eps=1e-5,
        initializer_std=0.02,
        # These are model-graph correctness tests at tiny seq lengths (8-10). Force the portable,
        # numerically-exact reference kernel so they run identically on CPU/GPU/TPU; the platform
        # default would pick TPU Splash, which requires the KV length to be a multiple of 128.
        attention_implementation="reference",
    )
    base.update(overrides)
    return SnowballConfig(**base)


def _device_batched_ids(vocab_size: int, pos_len: int) -> hax.NamedArray:
    """One identical sequence per device: a ``(batch=device_count, position)`` token grid.

    The grug forward data-parallelizes over the batch axis, so the batch must divide the mesh's
    ``data`` axis. A single unbatched sequence (batch=1) only lowers on one device; on a multi-device
    mesh (e.g. the multi-chip TPU CI) it cannot shard. Batching to the device count matches how the
    real model is scored (one prompt per device) and keeps these tests valid on any device count.
    """
    Batch = Axis("batch", jax.device_count())
    Pos = Axis("position", pos_len)
    row = jnp.arange(pos_len, dtype=jnp.int32) % vocab_size
    return hax.named(jnp.broadcast_to(row, (Batch.size, Pos.size)), (Batch, Pos))


def test_snowball_registered_and_no_arg_constructible():
    choices = LmConfig.get_known_choices()
    assert "snowball" in choices
    assert choices["snowball"] is SnowballConfig
    # HF discovery constructs the config with no args.
    cfg = SnowballConfig()
    assert cfg.vocab_size == 128256 and cfg.num_experts == 256 and cfg.num_experts_per_token == 4


def test_snowball_config_hf_roundtrip():
    cfg = _tiny_config()
    hf = cfg.to_hf_config(cfg.vocab_size)
    assert hf.model_type == GRUG_MOE_MODEL_TYPE
    assert hf.architectures == [GRUG_MOE_ARCHITECTURE]
    back = SnowballConfig.from_hf_config(hf)
    for field in (
        "vocab_size",
        "hidden_dim",
        "intermediate_dim",
        "shared_expert_intermediate_dim",
        "num_experts",
        "num_experts_per_token",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "max_seq_len",
        "sliding_window",
        "qk_mult",
    ):
        assert getattr(back, field) == getattr(cfg, field), field
    assert back.inferred_head_dim == cfg.inferred_head_dim

    # The serialized config carries one canonical name per field and no dropped alias, while
    # from_hf_config still round-trips it (above) through its tolerant fallback tuples.
    validate_single_name_config(hf.to_dict(), cfg)


def test_snowball_hf_converter_matches_config_class():
    # This is exactly the match HFCheckpointConverter.from_hf performs (by HfConfigClass name).
    converter = SnowballConfig().hf_checkpoint_converter()
    assert converter.HfConfigClass is GrugMoeHfConfig


def _expected_state_dict_manifest(cfg: SnowballConfig) -> dict[str, tuple[int, ...]]:
    """Canonical HF keys -> shapes, written out independently of the model's own to_state_dict.

    All Linear weights are stored transposed as HF ``(out, in)``; experts are stacked ``(E, out, in)``.
    """
    d = cfg.hidden_dim
    v = cfg.vocab_size
    n, m, h = cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
    e = cfg.num_experts
    i_moe = cfg.intermediate_dim
    i_sh = cfg.shared_expert_intermediate_dim
    r = 128  # gated-norm rank
    manifest: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (v, d),
        "model.embed_norm.weight": (d,),
        "model.embed_gated_norm.down_proj.weight": (r, d),
        "model.embed_gated_norm.up_proj.weight": (d, r),
        "model.norm.weight": (d,),
        "model.final_gated_norm.down_proj.weight": (r, d),
        "model.final_gated_norm.up_proj.weight": (d, r),
        "lm_head.weight": (v, d),
    }
    for li in range(cfg.num_layers):
        p = f"model.layers.{li}"
        manifest.update(
            {
                f"{p}.input_layernorm.weight": (d,),
                f"{p}.attn_gated_norm.down_proj.weight": (r, d),
                f"{p}.attn_gated_norm.up_proj.weight": (d, r),
                f"{p}.self_attn.q_proj.weight": (n * h, d),
                f"{p}.self_attn.k_proj.weight": (m * h, d),
                f"{p}.self_attn.v_proj.weight": (m * h, d),
                f"{p}.self_attn.o_proj.weight": (d, n * h),
                f"{p}.self_attn.attn_gate.weight": (n, d),
                f"{p}.post_attention_layernorm.weight": (d,),
                f"{p}.mlp_gated_norm.down_proj.weight": (r, d),
                f"{p}.mlp_gated_norm.up_proj.weight": (d, r),
                f"{p}.mlp.router.weight": (e, d),
                f"{p}.mlp.router.bias": (e,),
                f"{p}.mlp.experts.gate_proj.weight": (e, i_moe, d),
                f"{p}.mlp.experts.up_proj.weight": (e, i_moe, d),
                f"{p}.mlp.experts.down_proj.weight": (e, d, i_moe),
                f"{p}.shared_expert.gate_proj.weight": (i_sh, d),
                f"{p}.shared_expert.up_proj.weight": (i_sh, d),
                f"{p}.shared_expert.down_proj.weight": (d, i_sh),
            }
        )
    return manifest


def test_snowball_state_dict_key_and_shape_manifest():
    cfg = _tiny_config()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(0))
        sd = model.to_state_dict()
    expected = _expected_state_dict_manifest(cfg)
    assert set(sd.keys()) == set(
        expected.keys()
    ), f"missing={set(expected) - set(sd)} unexpected={set(sd) - set(expected)}"
    for key, shape in expected.items():
        assert tuple(sd[key].shape) == shape, f"{key}: {tuple(sd[key].shape)} != {shape}"


def test_snowball_state_dict_roundtrip_is_exact():
    cfg = _tiny_config()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        src = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(1))
        dst = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(2))
        sd = src.to_state_dict()
        dst = dst.from_state_dict(sd)

        ids = _device_batched_ids(cfg.vocab_size, 10)
        run = hax.named_jit(lambda m, x: m(x))
        src_logits = np.asarray(run(src, ids).array)
        dst_logits = np.asarray(run(dst, ids).array)
    assert np.array_equal(src_logits, dst_logits), "state-dict round-trip changed logits"


def test_snowball_torch_compatible_state_dict_roundtrip():
    """Exercise the exact serialization path load_pretrained uses (to/from_torch_compatible_state_dict)."""
    cfg = _tiny_config()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(4))
        sd = to_torch_compatible_state_dict(model)
        loaded = from_torch_compatible_state_dict(model, sd)
        loaded_sd = to_torch_compatible_state_dict(loaded)
    assert sd.keys() == loaded_sd.keys()
    for key, value in sd.items():
        np.testing.assert_array_equal(np.asarray(loaded_sd[key]), np.asarray(value))


def test_snowball_requires_explicit_mesh_axes():
    # Snowball reshards with out_sharding= over named specs, which only lower under an explicit
    # mesh; the marin-serve backend reads this to set TrainerConfig.use_explicit_mesh_axes.
    assert SnowballConfig().requires_explicit_mesh_axes is True


def test_snowball_load_pretrained_machinery_is_exact():
    """Snowball survives load_pretrained's eval_shape-template + named_jit(from_state_dict) core.

    HFCheckpointConverter.load_pretrained builds an abstract template with eqx.filter_eval_shape and
    fills it inside haliax.named_jit; Snowball's explicit-mesh reshards must lower under both. This
    guards the marin-serve load path (LevanterBackend.load_model) without an on-disk checkpoint.
    """
    cfg = _tiny_config()
    Vocab = Axis("vocab", cfg.vocab_size)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        src = SnowballLMHeadModel.init(Vocab, cfg, key=jax.random.key(5))
        sd = to_torch_compatible_state_dict(src)
        template = eqx.filter_eval_shape(SnowballLMHeadModel.init, Vocab, cfg, key=jax.random.key(0))
        loaded = hax.named_jit(lambda t, s: from_torch_compatible_state_dict(t, s))(template, sd)

        ids = _device_batched_ids(cfg.vocab_size, 8)
        run = hax.named_jit(lambda m, x: m(x))
        src_logits = np.asarray(run(src, ids).array)
        loaded_logits = np.asarray(run(loaded, ids).array)
    assert np.array_equal(src_logits, loaded_logits), "load_pretrained machinery changed logits"


def test_snowball_forward_shapes_and_finite():
    cfg = _tiny_config()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(3))
        ids = _device_batched_ids(cfg.vocab_size, 8)
        logits = hax.named_jit(lambda m, x: m(x))(model, ids)
    assert logits.axes[-1].name == "vocab" and logits.axes[-1].size == cfg.vocab_size
    assert bool(jnp.all(jnp.isfinite(logits.array)))


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"model_type": "llama"}, "model_type"),
        ({"grugmoe_attention_mode": "experimental"}, "grugmoe_attention_mode"),
        ({"grugmoe_artifact_schema_version": 999}, "schema"),
        ({"disable_pko": False}, "disable_pko"),
        ({"disable_long_rope": False}, "disable_long_rope"),
    ],
)
def test_snowball_rejects_off_recipe(overrides, message):
    cfg = _tiny_config()
    hf = cfg.to_hf_config(cfg.vocab_size)
    for k, val in overrides.items():
        setattr(hf, k, val)
    with pytest.raises(ValueError, match=message):
        SnowballConfig.from_hf_config(hf)


def test_snowball_load_path_multidevice_sharding():
    """The load-path forward must survive a data-sharded mesh (regression for the 67B router_bias).

    ``g()``-loaded leaves (norm weights, router_bias) inherit the sharding of the incoming state
    dict, and a safetensors load auto-shards ``[E]``/``[D]`` tensors over ``data`` when the size
    divides the axis. On a single device this is invisible; with 8 devices, ``router_logits +
    router_bias`` was illegally sharded. Runs in a fresh 8-CPU-device interpreter (XLA device count
    is process-global) and force-shards the state dict like safetensors to reproduce the condition.
    """
    script = textwrap.dedent(
        """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import equinox as eqx
        import haliax as hax
        import jax
        import jax.numpy as jnp
        import numpy as np
        from haliax import Axis
        from haliax.partitioning import set_mesh
        from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict
        from jax.random import PRNGKey
        from jax.sharding import NamedSharding, PartitionSpec as P
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

        assert jax.device_count() == 8
        # All parallel dims divide 8 so they actually shard on data=8 (E=16 => router_bias shards).
        cfg = SnowballConfig(
            vocab_size=128, hidden_dim=64, intermediate_dim=64, shared_expert_intermediate_dim=64,
            num_experts=16, num_experts_per_token=4, num_layers=3, num_heads=8, num_kv_heads=4,
            head_dim=16, max_seq_len=32, sliding_window=4, qk_mult=1.37, layer_norm_eps=1e-5,
            initializer_std=0.02,
        )
        Vocab = Axis("vocab", cfg.vocab_size)
        mesh = compact_grug_mesh(expert_axis_size=1)  # (replica_dcn=1, data=8, expert=1, model=1)
        Batch = Axis("batch", jax.device_count())
        Pos = Axis("position", 8)
        ids = hax.named(
            (jnp.arange(Batch.size * Pos.size, dtype=jnp.int32) % cfg.vocab_size).reshape(Batch.size, Pos.size),
            (Batch, Pos),
        )

        def like_safetensors(v):
            # Auto-shard the leading axis on data when it divides 8, else replicate (mimics the
            # placement of freshly-read safetensors that broke the 67B).
            v = jnp.asarray(v)
            spec = P("data") if v.ndim >= 1 and v.shape[0] % jax.device_count() == 0 else P()
            return jax.device_put(v, NamedSharding(mesh, spec))

        with set_mesh(mesh):
            src = SnowballLMHeadModel.init(Vocab, cfg, key=PRNGKey(1))
            sd = {k: like_safetensors(v) for k, v in to_torch_compatible_state_dict(src).items()}
            ref = np.asarray(hax.named_jit(lambda m, x: m(x))(src, ids).array)
            template = eqx.filter_eval_shape(SnowballLMHeadModel.init, Vocab, cfg, key=PRNGKey(0))
            loaded = hax.named_jit(lambda t, s: from_torch_compatible_state_dict(t, s))(template, sd)
            got = np.asarray(hax.named_jit(lambda m, x: m(x))(loaded, ids).array)
        assert np.array_equal(ref, got), "data-sharded load-path logits differ from the reference"
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout


def test_snowball_fresh_process_hf_discovery(tmp_path):
    """grug_moe must resolve via ``from_hf`` in a fresh interpreter with nothing pre-imported."""
    cfg = _tiny_config()
    hf = cfg.to_hf_config(cfg.vocab_size)
    (tmp_path / "config.json").write_text(__import__("json").dumps(hf.to_dict()))

    script = textwrap.dedent(
        """
        import sys
        # Deliberately do NOT import levanter.models.snowball.
        from transformers import AutoConfig
        from levanter.models.lm_model import LmConfig
        assert "levanter.models.snowball" not in sys.modules
        # from_hf triggers discovery before resolving the HF config; replicate that ordering.
        LmConfig.get_known_choices()
        assert "levanter.models.snowball" in sys.modules, "discovery did not import snowball"
        cfg = AutoConfig.from_pretrained(sys.argv[1])
        assert type(cfg).__name__ == "GrugMoeHfConfig", type(cfg).__name__
        assert cfg.model_type == "grug_moe"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout
