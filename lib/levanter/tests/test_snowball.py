# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Snowball tests (no dependency on marin ``experiments/``).

These validate the Levanter-side contract: registration/discovery, config <-> HF round-trip,
bidirectional state-dict serialization (with an explicit HF-key/shape manifest so a shared
transpose bug cannot hide), and off-recipe rejection. The Snowball-vs-experiment numerical parity
harness lives on the marin side (``tests/test_snowball_grug_parity.py``) to respect the
levanter -> experiments dependency direction.
"""

import collections
import dataclasses
import inspect
import json
import math
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as random
import jmp
import numpy as np
import pytest
from haliax import Axis, named_jit
from haliax.partitioning import set_mesh
from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict
from jax.extend import core as jax_core
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from safetensors.numpy import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from levanter.grug.grug_moe import PreparedMoERouteCapture
from levanter.kernels.pallas.rms_norm import batch_invariant_rms_norm_reference
from levanter.models import snowball as snowball_module
from levanter.models.loss import maybe_fused_next_token_loss, next_token_loss
from levanter.layers.attention import AttentionMask as LmAttentionMask
from levanter.layers.kv_cache import KvPageCache
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.checkpoint import CheckpointerConfig, load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, _reshard_state_dict_for_import
from levanter.field_dtype_overrides import with_field_dtype_overrides
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.layers.attention import AttentionMask
from levanter.main.export_hf_to_lm import ImportHfConfig, main as import_hf
from levanter.main.train_lm import TrainLmConfig, main, preserve_hf_runtime_settings
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.snowball import (
    GRUG_MOE_ARCHITECTURE,
    GRUG_MOE_MODEL_TYPE,
    DenseMLP,
    GatedNorm,
    RMSNorm,
    GrugMoeHfConfig,
    SnowballAttention,
    SnowballConfig,
    SnowballDecodeRouteCaptureBatch,
    SnowballRouteCaptureBatch,
    SnowballTransformer,
    SnowballLMHeadModel,
    SnowballMoEMLP,
    _activation_spec,
    validate_single_name_config,
)
from levanter.optim.config import AdamConfig
from levanter.testing.devices import use_test_devices
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.utils.jax_utils import best_effort_sharding, local_cpu_mesh
from levanter.utils.mesh import MeshConfig
from levanter.testing.devices import run_on_cpu_devices
from levanter.testing.inference import assert_mixed_paged_decode_matches_forward


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


def _single_device_grug_mesh() -> Mesh:
    """Build the explicit Grug mesh used by CPU reference tests, independent of visibility."""
    axes = ("replica_dcn", "data", "context", "expert", "model")
    return Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1, 1)),
        axes,
        axis_types=(AxisType.Explicit,) * len(axes),
    )


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
    assert "rms_norm_implementation" not in hf.to_dict()
    assert "gated_norm_implementation" not in hf.to_dict()


def _bounded_shared_attention_config(**overrides) -> SnowballConfig:
    values = dict(
        vocab_size=23,
        hidden_dim=128,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=128,
        max_seq_len=256,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        shared_attention_implementation="fixed_schedule",
    )
    values.update(overrides)
    return SnowballConfig(**values)


def test_snowball_shared_attention_rejects_public_masks_and_width_outside_fixture():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballTransformer.init(_bounded_shared_attention_config(), key=jax.random.key(0))
        tokens = jnp.zeros((4, 256), dtype=jnp.int32)
        with pytest.raises(ValueError, match="unsegmented causal mask"):
            model(tokens, AttentionMask(is_causal=False))
        with pytest.raises(ValueError, match="unsegmented causal mask"):
            model(tokens, AttentionMask.causal().with_sliding_window(16))

        width64 = _bounded_shared_attention_config(shared_expert_intermediate_dim=64)
        model = SnowballTransformer.init(width64, key=jax.random.key(0))
        with pytest.raises(ValueError, match="bounded fixture"):
            model(tokens)


@pytest.mark.timeout(120)
def test_snowball_shared_attention_public_decode_preserves_explicit_layout_at_page_boundaries():
    run_on_cpu_devices(
        """
        import dataclasses
        import equinox as eqx
        import haliax as hax
        import jax
        import jax.numpy as jnp
        import numpy as np
        from haliax import Axis
        from jax.sharding import AxisType, Mesh, PartitionSpec as P

        from levanter.inference.page_table import PageBatchInfo, PageTableSpec
        from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

        axes = ("replica_dcn", "data", "context", "expert", "model")
        mesh = Mesh(
            np.asarray(jax.devices()).reshape((1, 4, 1, 1, 1)),
            axes,
            axis_types=(AxisType.Explicit,) * len(axes),
        )
        config = SnowballConfig(
            vocab_size=23,
            hidden_dim=128,
            intermediate_dim=32,
            shared_expert_intermediate_dim=32,
            num_experts=2,
            num_experts_per_token=1,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=128,
            max_seq_len=256,
            attention_implementation="reference",
            inference_attention_implementation="reference",
            moe_implementation="scatter",
            shared_attention_implementation="fixed_schedule",
        )
        expected_output = P(("replica_dcn", "data", "expert"), "model")
        page_maps = (
            np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
            np.asarray([[6, 1], [7, 0], [4, 3], [5, 2]], dtype=np.int32),
        )
        slot_ids = np.asarray([2, 0, 3, 1], dtype=np.int32)

        def batch(length, page_map, *, cached_position=None):
            if cached_position is None:
                positions = np.tile(np.arange(length, dtype=np.int32), 4)
                seq_lens = np.full(4, length, dtype=np.int32)
            else:
                positions = np.full(4, cached_position, dtype=np.int32)
                seq_lens = np.full(4, cached_position + 1, dtype=np.int32)
            rows = np.repeat(np.arange(4), positions.size // 4)
            destinations = page_map[rows, positions // 128] * 128 + positions % 128
            return (
                hax.named(jnp.arange(positions.size, dtype=jnp.int32) % 23, "position"),
                hax.named(jnp.asarray(positions), "position"),
                PageBatchInfo(
                    slot_ids=hax.named(jnp.asarray(slot_ids), "seq"),
                    page_indices=hax.named(jnp.asarray(page_map), ("seq", "page")),
                    seq_lens=hax.named(jnp.asarray(seq_lens), "seq"),
                    cu_q_lens=hax.named(jnp.arange(5, dtype=jnp.int32) * (positions.size // 4), "seq"),
                    num_seqs=jnp.asarray(4, dtype=jnp.int32),
                    new_token_dests=hax.named(jnp.asarray(destinations), "position"),
                    page_size=128,
                ),
            )

        with jax.set_mesh(mesh):
            fixed = SnowballLMHeadModel.init(Axis("vocab", 23), config, key=jax.random.key(0))
            native = SnowballLMHeadModel.init(
                Axis("vocab", 23), dataclasses.replace(config, shared_attention_implementation="native"), key=jax.random.key(0)
            )
            cache = fixed.initial_cache(PageTableSpec(8, 128, 4), dtype=jnp.bfloat16)
            for page_map in page_maps:
                for length in (1, 2, 128):
                    tokens, positions, info = batch(length, page_map)
                    for model in (fixed, native):
                        logits, _ = eqx.filter_eval_shape(model.decode, tokens, cache, info, positions)
                        assert logits.array.sharding.spec == expected_output
                for cached_position in (127, 128):
                    tokens, positions, info = batch(1, page_map, cached_position=cached_position)
                    for model in (fixed, native):
                        logits, _ = eqx.filter_eval_shape(model.decode, tokens, cache, info, positions)
                        assert logits.array.sharding.spec == expected_output
        """,
        device_count=4,
    )


@pytest.mark.timeout(120)
def test_snowball_shared_attention_cache_writes_follow_nonidentity_page_map_for_both_kv():
    run_on_cpu_devices(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import haliax as hax
        from haliax import Axis
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from levanter.inference.page_table import PageBatchInfo, PageTableSpec
        from levanter.layers.kv_cache import KvPageCache
        from levanter.models.snowball import SnowballAttention, SnowballConfig

        axes = ("replica_dcn", "data", "context", "expert", "model")
        mesh = Mesh(np.asarray(jax.devices()).reshape((1, 4, 1, 1, 1)), axes, axis_types=(AxisType.Explicit,) * 5)
        cfg = SnowballConfig(
            vocab_size=23, hidden_dim=128, intermediate_dim=32, shared_expert_intermediate_dim=32,
            num_experts=2, num_experts_per_token=1, num_layers=1, num_heads=2, num_kv_heads=1,
            head_dim=128, max_seq_len=256, attention_implementation="reference",
            inference_attention_implementation="reference", shared_attention_implementation="fixed_schedule",
        )
        page_map = jnp.asarray([[6, 1], [7, 0], [4, 3], [5, 2]], dtype=jnp.int32)
        positions = jnp.tile(jnp.asarray([127, 128], dtype=jnp.int32), 4)
        rows = jnp.repeat(jnp.arange(4), 2)
        destinations = page_map[rows, positions // 128] * 128 + positions % 128
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.asarray([2, 0, 3, 1], dtype=jnp.int32), "seq"),
            page_indices=hax.named(page_map, ("seq", "page")),
            seq_lens=hax.named(jnp.full((4,), 129, dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.arange(5, dtype=jnp.int32) * 2, "seq"),
            num_seqs=jnp.asarray(4, dtype=jnp.int32),
            new_token_dests=hax.named(destinations, "position"),
            page_size=128,
        )
        with jax.set_mesh(mesh):
            attention = SnowballAttention.init(cfg, key=jax.random.key(19))
            cache = KvPageCache.init(PageTableSpec(8, 128, 4), Axis("kv_head", 1), Axis("head_size", 128), dtype=jnp.bfloat16)
            sentinel = jax.sharding.reshard(jnp.full((8, 128, 2, 128), 7, dtype=jnp.bfloat16), cache.kv_pages.array.sharding)
            cache = KvPageCache(hax.named(sentinel, cache.kv_pages.axes))
            x = jnp.reshape((jnp.arange(8 * 128, dtype=jnp.float32) + 0.125) / 91, (8, 1, 128))
            x = jax.device_put(x, NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None)))
            _, keys, values = attention._project_qkv(x, disable_rope=True, position_ids=positions[:, None])
            updated = eqx.filter_jit(
                lambda value, state, batch, pos: attention.decode(value, state, batch, pos, use_long=True)[1]
            )(x, cache, info, positions)

        expected = np.full((8, 128, 2, 128), 7, dtype=np.asarray(jnp.asarray(0, dtype=jnp.bfloat16)).dtype)
        for row, destination in enumerate(np.asarray(destinations)):
            page, offset = divmod(int(destination), 128)
            expected[page, offset, 0] = np.asarray(keys[:, 0].astype(jnp.bfloat16))[row]
            expected[page, offset, 1] = np.asarray(values[:, 0].astype(jnp.bfloat16))[row]
        actual = np.asarray(updated.kv_pages.array)
        np.testing.assert_array_equal(actual, expected)
        assert updated.kv_pages.array.dtype == jnp.bfloat16
        assert np.any(np.asarray(keys[:, 0]) != np.asarray(keys[:, 0].astype(jnp.bfloat16)).astype(np.float32))
        assert np.any(np.asarray(values[:, 0]) != np.asarray(values[:, 0].astype(jnp.bfloat16)).astype(np.float32))
        """,
        device_count=4,
    )


def test_snowball_gated_norm_fixed_schedule_matches_reference_and_vjp(monkeypatch):
    def reference_linear(x, weight):
        return jax.lax.dot_general(
            x,
            weight,
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

    monkeypatch.setattr(snowball_module, "fixed_schedule_linear", reference_linear)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        key = jax.random.key(17)
        native = GatedNorm.init(256, 0.02, key=key)
        fixed = GatedNorm.init(256, 0.02, key=key, implementation="fixed_schedule")
        x = jnp.linspace(-1.0, 1.0, 2 * 3 * 256, dtype=jnp.float32).reshape(2, 3, 256)

        def loss(implementation, value, down, up):
            return jnp.sum(GatedNorm(down, up, implementation)(value) ** 2)

        native_value, native_grad = jax.value_and_grad(
            lambda value, down, up: loss("native", value, down, up),
            argnums=(0, 1, 2),
        )(x, native.w_down, native.w_up)
        fixed_value, fixed_grad = jax.value_and_grad(
            lambda value, down, up: loss("fixed_schedule", value, down, up),
            argnums=(0, 1, 2),
        )(x, fixed.w_down, fixed.w_up)
        np.testing.assert_allclose(fixed_value, native_value, rtol=1e-5, atol=1e-5)
        for fixed_component, native_component in zip(fixed_grad, native_grad, strict=True):
            np.testing.assert_allclose(fixed_component, native_component, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fixed(x), native(x), rtol=1e-5, atol=1e-5)


@pytest.mark.timeout(120)
def test_snowball_gated_norm_fixed_schedule_matches_input_layout_on_four_cpu_devices():
    run_on_cpu_devices(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from pathlib import Path
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.models import snowball as snowball_module
        from levanter.models.snowball import GatedNorm

        assert "lib/levanter/src" in str(Path(snowball_module.__file__).resolve())

        # Reference wiring only: this substitutes the contraction arithmetic so CPU can exercise
        # the product layout and autodiff path. It is not evidence for the TPU custom VJP/kernel.
        def reference_linear(x, weight):
            return jax.lax.dot_general(
                x,
                weight,
                (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )

        snowball_module.fixed_schedule_linear = reference_linear
        axes = ("replica_dcn", "data", "context", "expert", "model")
        mesh = Mesh(
            np.asarray(jax.devices()).reshape((1, 4, 1, 1, 1)),
            axes,
            axis_types=(AxisType.Explicit,) * len(axes),
        )
        input_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None))

        with jax.set_mesh(mesh):
            norm = GatedNorm.init(128, 0.02, key=jax.random.key(17), implementation="fixed_schedule")
            for shape in ((4, 256, 128), (512, 1, 128), (4, 1, 128)):
                host_x = np.linspace(-0.5, 0.5, np.prod(shape), dtype=np.float32).reshape(shape)
                x = jax.device_put(jnp.asarray(host_x), input_sharding)
                actual = eqx.filter_jit(norm)(x)
                assert actual.sharding.spec == input_sharding.spec

                down = np.asarray(norm.w_down)
                up = np.asarray(norm.w_up)
                hidden = jax.nn.silu(jnp.asarray(host_x) @ jnp.asarray(down))
                expected = host_x * np.asarray(jax.nn.sigmoid(hidden @ jnp.asarray(up)))
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)

                input_value_and_grad = eqx.filter_jit(jax.value_and_grad(lambda value: jnp.mean(norm(value))))
                _, input_grad = input_value_and_grad(x)
                assert input_grad.sharding.spec == input_sharding.spec
                assert bool(jnp.isfinite(input_grad).all())

            _, norm_grad = eqx.filter_value_and_grad(lambda model: jnp.mean(model(x)))(norm)
            assert bool(jnp.isfinite(norm_grad.w_down).all())
            assert bool(jnp.isfinite(norm_grad.w_up).all())

            cfg = snowball_module.SnowballConfig(
                vocab_size=32,
                hidden_dim=128,
                intermediate_dim=128,
                shared_expert_intermediate_dim=128,
                num_experts=1,
                num_experts_per_token=1,
                num_layers=1,
                num_heads=2,
                num_kv_heads=1,
                head_dim=64,
                projection_implementation="fixed_schedule",
                attention_implementation="reference",
            )
            attention = snowball_module.SnowballAttention.init(cfg, key=jax.random.key(19))
            assert attention.w_q.sharding.spec == P(("data", "context"), "model")
            q, k, v = eqx.filter_jit(lambda value: attention._project_qkv(value, disable_rope=True))(x[:, :1, :])
            assert q.shape == (4, 1, 2, 64)
            assert k.shape == v.shape == (4, 1, 1, 64)
            assert tuple(q.sharding.spec)[0] == input_sharding.spec[0]
            assert all(entry is None for entry in tuple(q.sharding.spec)[1:])
        """,
        device_count=4,
    )


def test_snowball_gated_norm_selector_preserves_state_dict_and_propagates():
    native_cfg = _tiny_config()
    fixed_cfg = dataclasses.replace(native_cfg, gated_norm_implementation="fixed_schedule")
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        native = SnowballLMHeadModel.init(Axis("vocab", native_cfg.vocab_size), native_cfg, key=jax.random.key(23))
        fixed = SnowballLMHeadModel.init(Axis("vocab", fixed_cfg.vocab_size), fixed_cfg, key=jax.random.key(23))
        native_state = native.to_state_dict()
        fixed_state = fixed.to_state_dict()

    assert fixed.transformer.embed_gated_norm.implementation == "fixed_schedule"
    assert fixed.transformer.final_gated_norm.implementation == "fixed_schedule"
    assert all(block.attn_gated_norm.implementation == "fixed_schedule" for block in fixed.transformer.blocks)
    assert all(block.mlp_gated_norm.implementation == "fixed_schedule" for block in fixed.transformer.blocks)

    assert native_state.keys() == fixed_state.keys()
    for name in native_state:
        np.testing.assert_array_equal(native_state[name], fixed_state[name], err_msg=name)


def test_snowball_fixed_schedule_project_output_uses_xsa_for_nonstandard_head_width(monkeypatch):
    """The fixed selector keeps XSA semantics when a portable test uses a small head width."""

    def reference_linear(x, weight):
        return jax.lax.dot_general(
            x,
            weight,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    monkeypatch.setattr(snowball_module, "fixed_schedule_linear", reference_linear)
    native_cfg = _tiny_config(
        hidden_dim=5,
        num_heads=2,
        num_kv_heads=1,
        head_dim=3,
        projection_implementation="native",
    )
    fixed_cfg = dataclasses.replace(native_cfg, projection_implementation="fixed_schedule")
    with jax.set_mesh(_single_device_grug_mesh()):
        native = SnowballAttention.init(native_cfg, key=jax.random.key(31))
        fixed = SnowballAttention.init(fixed_cfg, key=jax.random.key(31))
        gate_weights = jnp.sin(jnp.arange(10, dtype=jnp.float32).reshape(5, 2) + 0.3) / 3
        native = eqx.tree_at(lambda module: module.attn_gate, native, gate_weights)
        fixed = eqx.tree_at(lambda module: module.attn_gate, fixed, gate_weights)
        x = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5) / 17
        attn_out = jnp.arange(2 * 4 * 2 * 3, dtype=jnp.float32).reshape(2, 4, 2, 3) / 13
        value = jnp.arange(2 * 4 * 1 * 3, dtype=jnp.float32).reshape(2, 4, 1, 3) / 11 + 0.2
        native_output = native._project_output(x, attn_out, value)
        fixed_output = fixed._project_output(x, attn_out, value)

    y = np.asarray(attn_out)
    aligned_value = np.repeat(np.asarray(value), 2, axis=2)
    dot = np.sum(y * aligned_value, axis=-1, keepdims=True)
    norm = np.sum(aligned_value * aligned_value, axis=-1, keepdims=True)
    projected = y - dot / (norm + 1e-6) * aligned_value
    logits = np.einsum("bsd,dn->bsn", np.asarray(x), np.asarray(gate_weights))
    gate = 2 / (1 + np.exp(-logits))
    expected = np.einsum("bsh,hd->bsd", (gate[..., None] * projected).reshape(2, 4, 6), np.asarray(fixed.w_o))
    np.testing.assert_allclose(np.asarray(fixed_output), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(native_output), expected, rtol=1e-5, atol=1e-5)
    assert float(np.max(gate) - np.min(gate)) > 0.1


def test_snowball_fixed_schedule_shared_dense_and_lmhead_cover_public_train_prefill_decode(monkeypatch):
    """The opt-in contractions preserve the public paths and parameter identities on CPU."""

    def reference_linear(x, weight):
        return jax.lax.dot_general(
            x,
            weight,
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

    monkeypatch.setattr(snowball_module, "fixed_schedule_linear", reference_linear)
    native_cfg = _tiny_config(
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=48,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        moe_implementation="scatter",
    )
    fixed_cfg = dataclasses.replace(native_cfg, projection_implementation="fixed_schedule")
    ids = hax.named(
        jnp.asarray([[2, 7, 0, 9], [2, 7, 0, 9], [2, 7, 0, 9], [2, 7, 0, 9]], dtype=jnp.int32),
        ("batch", "position"),
    )
    with jax.set_mesh(_single_device_grug_mesh()):
        native = SnowballLMHeadModel.init(Axis("vocab", native_cfg.vocab_size), native_cfg, key=jax.random.key(43))
        fixed = SnowballLMHeadModel.init(Axis("vocab", fixed_cfg.vocab_size), fixed_cfg, key=jax.random.key(43))
        native_logits = native(ids)
        fixed_logits = fixed(ids)
        np.testing.assert_allclose(
            np.asarray(fixed_logits.array), np.asarray(native_logits.array), rtol=1e-5, atol=1e-5
        )

        loss_weight = hax.named(jnp.asarray([[1.0, 0.5, 2.0, 0.0]] * 4, dtype=jnp.float32), ids.axes)
        example = LmExample(tokens=ids, loss_weight=loss_weight, attn_mask=LmAttentionMask.causal())
        native_loss = next_token_loss(
            native.Pos,
            native.Vocab,
            native_logits,
            ids,
            loss_weight=loss_weight,
        )
        fixed_loss = fixed.compute_next_token_loss(example)
        np.testing.assert_allclose(np.asarray(fixed_loss.array), np.asarray(native_loss.array), rtol=1e-5, atol=1e-5)

        def native_model_loss(model):
            return next_token_loss(model.Pos, model.Vocab, model(ids), ids, loss_weight=loss_weight).array

        def fixed_model_loss(model):
            return model.compute_next_token_loss(example).array

        native_grad = eqx.filter_grad(native_model_loss)(native)
        fixed_grad = eqx.filter_grad(fixed_model_loss)(fixed)
        for native_value, fixed_value in (
            (native_grad.transformer.token_embed, fixed_grad.transformer.token_embed),
            (native_grad.transformer.output_proj, fixed_grad.transformer.output_proj),
            (native_grad.transformer.blocks[0].shared.w_gate, fixed_grad.transformer.blocks[0].shared.w_gate),
            (native_grad.transformer.blocks[0].shared.w_up, fixed_grad.transformer.blocks[0].shared.w_up),
            (native_grad.transformer.blocks[0].shared.w_down, fixed_grad.transformer.blocks[0].shared.w_down),
        ):
            np.testing.assert_allclose(np.asarray(fixed_value), np.asarray(native_value), rtol=1e-4, atol=1e-5)
            assert bool(jnp.isfinite(fixed_value).all())

        fixed_grad = eqx.filter_grad(lambda model: jnp.sum(model(ids).array))(fixed)
        assert bool(jnp.isfinite(fixed_grad.transformer.output_proj).all())

        cache = fixed.initial_cache(PageTableSpec(num_pages=4, page_size=2, max_seqs=1), dtype=jnp.float32)
        prefill_ids = hax.named(ids.array[0], "position")
        positions = hax.named(jnp.arange(4, dtype=jnp.int32), "position")
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.asarray([0], dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray([[0, 1, 2]], dtype=jnp.int32), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray([4], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.asarray([0, 4], dtype=jnp.int32), "seq"),
            num_seqs=jnp.asarray(1, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.arange(4, dtype=jnp.int32), "position"),
            page_size=2,
        )
        prefill_logits, prefill_cache = fixed.decode(prefill_ids, cache, info, positions)

        next_id = hax.named(jnp.asarray([5], dtype=jnp.int32), "position")
        decode_info = PageBatchInfo(
            slot_ids=info.slot_ids,
            page_indices=info.page_indices,
            seq_lens=hax.named(jnp.asarray([5], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.asarray([0, 1], dtype=jnp.int32), "seq"),
            num_seqs=jnp.asarray(1, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.asarray([4], dtype=jnp.int32), "position"),
            page_size=2,
        )
        decode_logits, _ = fixed.decode(next_id, prefill_cache, decode_info, hax.named(jnp.asarray([4]), "position"))
        full_ids = hax.named(
            jnp.concatenate((ids.array[:1], jnp.asarray([[5]], dtype=jnp.int32)), axis=1),
            (ids.axes[0].resize(1), ids.axes[1].resize(5)),
        )
        full_logits = fixed(full_ids)

    np.testing.assert_allclose(
        np.asarray(prefill_logits.array), np.asarray(fixed_logits.array[0]), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(decode_logits.array), np.asarray(full_logits.array[:, -1]), rtol=1e-5, atol=1e-5
    )


def test_snowball_fixed_schedule_loss_casts_before_soft_cap_and_preserves_loss_dtype(monkeypatch):
    """Weighted lower-precision training loss matches an independent cast-then-cap oracle."""

    def reference_linear(x, weight):
        return jax.lax.dot_general(
            x,
            weight,
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

    def reference_loss(logits, token_ids, weights, dtype, soft_cap):
        logits = logits.astype(dtype) if dtype is not None else logits
        if soft_cap is not None:
            logits = jnp.tanh(logits / soft_cap) * soft_cap
        targets = jnp.roll(token_ids, -1, axis=-1)
        token_loss = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1)[..., 0]
        effective_weight = weights * jnp.concatenate(
            [
                jnp.ones(weights.shape[:-1] + (weights.shape[-1] - 1,), dtype=weights.dtype),
                jnp.zeros(weights.shape[:-1] + (1,), dtype=weights.dtype),
            ],
            axis=-1,
        )
        return jnp.sum(token_loss * effective_weight) / jnp.sum(effective_weight)

    monkeypatch.setattr(snowball_module, "fixed_schedule_linear", reference_linear)
    cfg = dataclasses.replace(
        _tiny_config(vocab_size=7, hidden_dim=32, intermediate_dim=32, shared_expert_intermediate_dim=32),
        projection_implementation="fixed_schedule",
    )
    ids = hax.named(
        jnp.asarray([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int32),
        ("batch", "position"),
    )
    weights = hax.named(jnp.asarray([[1.0, 0.25, 2.0, 0.0]] * 4, dtype=jnp.bfloat16), ids.axes)
    example = LmExample(tokens=ids, loss_weight=weights, attn_mask=LmAttentionMask.causal())
    soft_cap = 1.7

    with jax.set_mesh(_single_device_grug_mesh()):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(57))
        logits = model(ids)
        expected = reference_loss(logits.array, ids.array, weights.array, jnp.bfloat16, soft_cap)
        actual = model.compute_next_token_loss(example, loss_dtype=jnp.float32, logit_soft_cap=soft_cap)
        np.testing.assert_allclose(np.asarray(actual.array), np.asarray(expected), rtol=1e-2, atol=1e-2)

        def actual_model_loss(current):
            return current.compute_next_token_loss(example, loss_dtype=jnp.float32, logit_soft_cap=soft_cap).array

        def reference_model_loss(current):
            return reference_loss(current(ids).array, ids.array, weights.array, jnp.bfloat16, soft_cap)

        actual_grad = eqx.filter_grad(actual_model_loss)(model)
        reference_grad = eqx.filter_grad(reference_model_loss)(model)
        np.testing.assert_allclose(
            np.asarray(actual_grad.transformer.output_proj),
            np.asarray(reference_grad.transformer.output_proj),
            rtol=1e-2,
            atol=1e-2,
        )
        assert float(jnp.max(jnp.abs(actual_grad.transformer.output_proj))) > 0.0

        direct_logits = hax.named(logits.array.astype(jnp.bfloat16), logits.axes)
        direct_default = next_token_loss(model.Pos, model.Vocab, direct_logits, ids)
        direct_reference = reference_loss(
            logits.array.astype(jnp.bfloat16), ids.array, jnp.ones_like(logits.array[..., 0]), None, None
        )
        np.testing.assert_allclose(
            np.asarray(direct_default.array), np.asarray(direct_reference), rtol=1e-2, atol=1e-2
        )


def test_loss_helpers_keep_distinct_mixed_precision_defaults():
    """Protect the public helper defaults used by fused model and direct loss callers."""
    assert inspect.signature(maybe_fused_next_token_loss).parameters["dtype"].default is jnp.float32
    assert inspect.signature(next_token_loss).parameters["dtype"].default is None

    Pos = Axis("position", 17)
    Embed = Axis("embed", 7)
    Vocab = Axis("vocab", 31)
    embeddings = hax.named(jax.random.normal(jax.random.key(9), (17, 7), dtype=jnp.bfloat16) * 11, (Pos, Embed))
    lm_head = hax.named(jax.random.normal(jax.random.key(10), (7, 31), dtype=jnp.bfloat16) * 11, (Embed, Vocab))
    ids = hax.named(jnp.arange(17, dtype=jnp.int32) % 31, Pos)

    fused_default = maybe_fused_next_token_loss(Pos, Embed, Vocab, embeddings, lm_head, ids)
    fused_explicit = maybe_fused_next_token_loss(Pos, Embed, Vocab, embeddings, lm_head, ids, dtype=jnp.float32)
    assert fused_default.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(fused_default.array), np.asarray(fused_explicit.array))

    logits = hax.named(jax.random.normal(jax.random.key(11), (17, 31), dtype=jnp.bfloat16) * 11, (Pos, Vocab))
    direct_default = next_token_loss(Pos, Vocab, logits, ids)
    direct_explicit = next_token_loss(Pos, Vocab, logits, ids, dtype=None)
    # Leaving the input dtype unchanged still uses FP32 policy normalization.
    assert direct_default.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(direct_default.array), np.asarray(direct_explicit.array))


def test_snowball_shared_rms_norm_reference_jit_and_vjp_match_fixed_schedule_reference():
    x = jnp.linspace(-1.0, 1.0, 4 * 128 * 256, dtype=jnp.float32).reshape(4, 128, 256)
    weight = jnp.linspace(0.5, 1.5, 256, dtype=jnp.float32)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):

        def apply(value, scale):
            return RMSNorm(weight=scale, eps=1e-5, implementation="reference")(value)

        def independent(value, scale):
            value32 = value.astype(jnp.float32)
            rms = jax.lax.rsqrt(jnp.mean(jnp.square(value32), axis=-1, keepdims=True) + 1e-5)
            return (value32 * rms * scale).astype(value.dtype)

        def fixed_schedule_reference(value, scale):
            return batch_invariant_rms_norm_reference(value, scale, eps=1e-5)

        expected = independent(x, weight)
        actual = jax.jit(apply)(x, weight)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

        actual_value, actual_grad = jax.jit(
            jax.value_and_grad(lambda value, scale: jnp.sum(apply(value, scale) ** 2), argnums=(0, 1))
        )(x, weight)
        expected_reference_value, expected_grad = jax.value_and_grad(
            lambda value, scale: jnp.sum(fixed_schedule_reference(value, scale) ** 2), argnums=(0, 1)
        )(x, weight)
    expected_independent_value = jax.value_and_grad(
        lambda value, scale: jnp.sum(independent(value, scale) ** 2), argnums=(0, 1)
    )(x, weight)[0]
    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_independent_value), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_reference_value), rtol=1e-5, atol=1e-5)
    for actual_component, expected_component in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(np.asarray(actual_component), np.asarray(expected_component), rtol=1e-5, atol=1e-5)


def test_snowball_rms_norm_native_remains_default():
    cfg = SnowballConfig()
    assert cfg.rms_norm_implementation == "native"
    assert cfg.qk_rms_norm_implementation == "native"
    x = jnp.arange(2 * 256, dtype=jnp.float32).reshape(2, 256)
    weight = jnp.linspace(0.5, 1.5, 256, dtype=jnp.float32)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        norm = RMSNorm(weight=weight, eps=1e-5, implementation="native")
        expected = (x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-5) * weight).astype(x.dtype)
        np.testing.assert_array_equal(np.asarray(norm(x)), np.asarray(expected))


@pytest.mark.parametrize(
    ("rms_norm_implementation", "qk_rms_norm_implementation", "num_heads", "num_kv_heads", "head_dim"),
    [("reference", "native", 4, 2, 64), ("native", "reference", 2, 1, 128)],
)
def test_snowball_rms_norm_implementations_use_real_training_capture_and_cached_decode_paths(
    rms_norm_implementation, qk_rms_norm_implementation, num_heads, num_kv_heads, head_dim
):
    cfg = _tiny_config(
        hidden_dim=256,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="scatter",
        rms_norm_implementation=rms_norm_implementation,
        qk_rms_norm_implementation=qk_rms_norm_implementation,
    )
    ids = hax.named(jnp.asarray([[2, 7, 0, 9]], dtype=jnp.int32), ("batch", "position"))

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(43))
        training_logits = model(ids)
        capture = model.capture_route_captures(ids)
        replay_logits = model(ids, route_captures=capture)

        state_dict = model.to_state_dict()
        assert state_dict["model.embed_norm.weight"].shape == (256,)
        assert state_dict["model.layers.0.input_layernorm.weight"].shape == (256,)
        assert state_dict["model.layers.0.post_attention_layernorm.weight"].shape == (256,)
        assert state_dict["model.norm.weight"].shape == (256,)
        assert "rms_norm_implementation" not in cfg.to_hf_config(cfg.vocab_size).to_dict()

        cache = model.initial_cache(PageTableSpec(num_pages=4, page_size=2, max_seqs=1), dtype=jnp.float32)
        prefill_ids = hax.named(ids.array[0], "position")
        prefill_positions = hax.named(jnp.arange(4, dtype=jnp.int32), "position")
        prefill_info = PageBatchInfo(
            slot_ids=hax.named(jnp.asarray([0], dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray([[0, 1]], dtype=jnp.int32), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray([4], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.asarray([0, 4], dtype=jnp.int32), "seq"),
            num_seqs=jnp.asarray(1, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.arange(4, dtype=jnp.int32), "position"),
            page_size=2,
        )
        prefill_logits, cache, decode_capture = model.decode_with_route_captures(
            prefill_ids, cache, prefill_info, prefill_positions
        )
        continuation_ids = hax.named(jnp.asarray([11], dtype=jnp.int32), "position")
        continuation_positions = hax.named(jnp.asarray([4], dtype=jnp.int32), "position")
        continuation_info = dataclasses.replace(
            prefill_info,
            page_indices=hax.named(jnp.asarray([[0, 1, 2]], dtype=jnp.int32), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray([5], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.asarray([0, 1], dtype=jnp.int32), "seq"),
            new_token_dests=hax.named(jnp.asarray([4], dtype=jnp.int32), "position"),
        )
        continuation_logits, _, continuation_capture = model.decode_with_route_captures(
            continuation_ids, cache, continuation_info, continuation_positions
        )

    np.testing.assert_allclose(
        np.asarray(training_logits.array), np.asarray(replay_logits.array), rtol=1e-5, atol=1e-5
    )
    assert isinstance(decode_capture, SnowballDecodeRouteCaptureBatch)
    assert isinstance(continuation_capture, SnowballDecodeRouteCaptureBatch)
    assert prefill_logits.array.shape == (4, cfg.vocab_size)
    assert continuation_logits.array.shape == (1, cfg.vocab_size)


def test_snowball_qk_rms_reference_matches_native_projection_across_row_chunks_and_vjp():
    native_cfg = _tiny_config(
        hidden_dim=256,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=128,
        qk_rms_norm_implementation="native",
    )
    reference_cfg = dataclasses.replace(native_cfg, qk_rms_norm_implementation="reference")
    x = jnp.linspace(-1.0, 1.0, 4 * 257 * 256, dtype=jnp.float32).reshape(4, 257, 256)

    mesh = compact_grug_mesh(expert_axis_size=1)
    input_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None))
    with jax.set_mesh(mesh):
        x = jax.device_put(x, input_sharding)
        native = SnowballAttention.init(native_cfg, key=jax.random.key(91))
        reference = SnowballAttention.init(reference_cfg, key=jax.random.key(91))
        native_q, native_k, native_v = jax.jit(lambda value: native._project_qkv(value, disable_rope=True))(x)
        reference_q, reference_k, reference_v = jax.jit(
            lambda value: reference._project_qkv(value, disable_rope=True)
        )(x)
        reference_grad = jax.jit(jax.grad(lambda value: jnp.sum(reference._project_qkv(value, disable_rope=True)[0])))(
            x
        )

        def independent_rms(value):
            weight = jnp.ones((128,), dtype=jnp.float32)

            def batch_reference(batch):
                flat_batch = batch.reshape(-1, 128)
                return jnp.concatenate(
                    tuple(
                        batch_invariant_rms_norm_reference(flat_batch[start : start + 512], weight, eps=1e-6)
                        for start in range(0, flat_batch.shape[0], 512)
                    ),
                    axis=0,
                ).reshape(batch.shape)

            # The accepted custom VJP uses this fixed-schedule reference backward. Applying the
            # existing oracle over each local row block keeps this independent of Snowball's
            # layout/chunk helper while preserving the accepted fixed-schedule backward.
            return jax.vmap(lambda batch: batch_reference(batch) * native_cfg.qk_mult)(value)

        rms_oracle = jax.shard_map(
            independent_rms,
            mesh=mesh,
            in_specs=P(("replica_dcn", "data", "expert"), None, "model", None),
            out_specs=P(("replica_dcn", "data", "expert"), None, "model", None),
            check_vma=False,
        )

        if jax.device_count() == 1:

            def oracle_local(value, projection_weight):
                projected = jnp.einsum("bsh,hd->bsd", value, projection_weight).reshape(value.shape[0], 257, 2, 128)
                return independent_rms(projected)

            oracle = jax.shard_map(
                oracle_local,
                mesh=mesh,
                in_specs=(input_sharding.spec, P(None, None)),
                out_specs=P(("replica_dcn", "data", "expert"), None, "model", None),
                check_vma=False,
            )
            oracle_weight = jax.device_put(native.w_q, NamedSharding(mesh, P(None, None)))
        else:

            def oracle(value):
                projected = jnp.einsum("bsh,hd->bsd", value, native.w_q).reshape(value.shape[0], 257, 2, 128)
                return rms_oracle(projected)

            oracle_weight = None

        if oracle_weight is None:
            expected_q = jax.jit(oracle)(x)
            expected_grad = jax.jit(jax.grad(lambda value: jnp.sum(oracle(value))))(x)
        else:
            expected_q = jax.jit(oracle)(x, oracle_weight)
            expected_grad = jax.jit(jax.grad(lambda value: jnp.sum(oracle(value, oracle_weight))))(x)

    np.testing.assert_array_equal(np.asarray(reference_q), np.asarray(native_q))
    np.testing.assert_array_equal(np.asarray(reference_k), np.asarray(native_k))
    np.testing.assert_array_equal(np.asarray(reference_v), np.asarray(native_v))
    np.testing.assert_allclose(np.asarray(reference_q), np.asarray(expected_q), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(reference_grad), np.asarray(expected_grad), rtol=1e-5, atol=1e-5)


@pytest.mark.timeout(120)
def test_snowball_qk_rms_reference_restores_caller_layout_for_bfloat16_cache_pack(monkeypatch):
    monkeypatch.delenv("JAX_NUM_CPU_DEVICES", raising=False)
    run_on_cpu_devices(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import haliax as hax
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from haliax import Axis
        from levanter.inference.page_table import PageBatchInfo, PageTableSpec
        from levanter.layers.kv_cache import KvPageCache
        from levanter.models.snowball import SnowballAttention, SnowballConfig

        axes = ("replica_dcn", "data", "context", "expert", "model")
        mesh = Mesh(
            np.asarray(jax.devices()).reshape((1, 4, 1, 1, 1)),
            axes,
            axis_types=(AxisType.Explicit,) * len(axes),
        )
        caller_spec = P(("replica_dcn", "data"), "expert", None)
        caller_sharding = NamedSharding(mesh, caller_spec)
        config = SnowballConfig(
            hidden_dim=128,
            num_heads=1,
            num_kv_heads=1,
            head_dim=128,
            qk_rms_norm_implementation="reference",
            attention_implementation="reference",
        )
        host_x = jnp.linspace(-0.5, 0.5, 4 * 1 * 128, dtype=jnp.float32).reshape(4, 1, 128)
        with jax.set_mesh(mesh):
            attention = SnowballAttention.init(config, key=jax.random.key(19))
            x = jax.device_put(host_x, caller_sharding)
            q, k, v = eqx.filter_jit(
                lambda module, value: module._project_qkv(value, disable_rope=True)
            )(attention, x)
            expected_projection_spec = P(("replica_dcn", "data"), None, None, "model")
            assert q.sharding.spec == expected_projection_spec, q.sharding.spec
            assert k.sharding.spec == v.sharding.spec == q.sharding.spec, (k.sharding.spec, v.sharding.spec)

            _, input_grad = eqx.filter_jit(
                jax.value_and_grad(lambda value: jnp.sum(attention._project_qkv(value, disable_rope=True)[0]))
            )(x)
            assert input_grad.sharding.spec == caller_spec
            assert bool(jnp.isfinite(input_grad).all())

            cache = KvPageCache.init(
                PageTableSpec(num_pages=4, page_size=1, max_seqs=4),
                Axis("kv_head", 1),
                Axis("head_size", 128),
                dtype=jnp.bfloat16,
            )
            info = PageBatchInfo(
                slot_ids=hax.named(jnp.arange(4, dtype=jnp.int32), "seq"),
                page_indices=hax.named(jnp.arange(4, dtype=jnp.int32)[:, None], ("seq", "page")),
                seq_lens=hax.named(jnp.ones(4, dtype=jnp.int32), "seq"),
                cu_q_lens=hax.named(jnp.arange(5, dtype=jnp.int32), "seq"),
                num_seqs=jnp.asarray(4, dtype=jnp.int32),
                new_token_dests=hax.named(jnp.arange(4, dtype=jnp.int32), "position"),
                page_size=1,
            )
            cache_keys = hax.named(k[:, 0], ("position", "kv_head", "head_size"))
            cache_values = hax.named(v[:, 0], ("position", "kv_head", "head_size"))
            updated = eqx.filter_jit(
                lambda state, batch, keys, values: state.update(batch, keys, values)
            )(cache, info, cache_keys, cache_values)

        expected = np.concatenate((np.asarray(k[:, 0]), np.asarray(v[:, 0])), axis=1).astype(jnp.bfloat16)
        np.testing.assert_array_equal(
            np.asarray(updated.kv_pages.array)[:, 0].view(np.uint16), expected.view(np.uint16)
        )
        assert updated.kv_pages.array.dtype == jnp.bfloat16
        assert bool(jnp.isfinite(updated.kv_pages.array).all())
        """,
        device_count=4,
    )


@pytest.mark.skipif(jax.default_backend() != "tpu" or jax.device_count() != 4, reason="requires four-device TPUv4")
def test_snowball_pallas_rms_norm_explicit_train_decode_layouts_and_context_rejection():
    train_mesh = compact_grug_mesh(expert_axis_size=1, context_axis_size=1)
    train_spec = P(("replica_dcn", "data", "expert"), None, None)
    weight_spec = P(None)
    with jax.set_mesh(train_mesh):
        x_train = jax.device_put(jnp.ones((4, 256, 256), dtype=jnp.float32), NamedSharding(train_mesh, train_spec))
        weight = jax.device_put(jnp.linspace(0.5, 1.5, 256, dtype=jnp.float32), NamedSharding(train_mesh, weight_spec))

        def apply(value, scale):
            return RMSNorm(weight=scale, eps=1e-5, implementation="pallas_tpu")(value)

        train_output = apply(x_train, weight)
        train_value, train_gradient = jax.jit(
            jax.value_and_grad(lambda value, scale: jnp.sum(apply(value, scale)), argnums=0)
        )(x_train, weight)
        train_hlo = str(jax.jit(apply).lower(x_train, weight).compiler_ir()).lower()

        x_decode = jax.device_put(jnp.ones((32, 1, 256), dtype=jnp.float32), NamedSharding(train_mesh, train_spec))
        decode_output = apply(x_decode, weight)
        decode_value, decode_gradient = jax.jit(
            jax.value_and_grad(lambda value, scale: jnp.sum(apply(value, scale)), argnums=0)
        )(x_decode, weight)

        def independent(value):
            variance = jnp.mean(jnp.square(value), axis=-1, keepdims=True)
            return value * jax.lax.rsqrt(variance + 1e-5) * weight

        expected_train = independent(x_train)
        expected_decode = independent(x_decode)

    assert train_output.shape == x_train.shape
    assert decode_output.shape == x_decode.shape
    assert train_value.shape == () and decode_value.shape == ()
    assert train_gradient.shape == x_train.shape and decode_gradient.shape == x_decode.shape
    np.testing.assert_allclose(np.asarray(train_output), np.asarray(expected_train), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(decode_output), np.asarray(expected_decode), rtol=1e-5, atol=1e-5)
    assert isinstance(train_output.sharding, NamedSharding)
    assert isinstance(decode_output.sharding, NamedSharding)
    assert train_output.sharding.spec == x_train.sharding.spec
    assert decode_output.sharding.spec == x_decode.sharding.spec
    assert "all-gather" not in train_hlo and "all_gather" not in train_hlo

    context_mesh = compact_grug_mesh(expert_axis_size=1, context_axis_size=2)
    with jax.set_mesh(context_mesh):
        context_x = jax.device_put(
            jnp.ones((4, 128, 256), dtype=jnp.float32),
            NamedSharding(context_mesh, P(("replica_dcn", "data", "expert"), "context", None)),
        )
        context_weight = jax.device_put(jnp.ones((256,), dtype=jnp.float32), NamedSharding(context_mesh, P(None)))
        context_norm = RMSNorm(weight=context_weight, eps=1e-5, implementation="pallas_tpu")
        with pytest.raises(ValueError, match="later leading axis"):
            context_norm(context_x)


@pytest.mark.timeout(120)
def test_snowball_rms_norm_reference_preserves_explicit_row_sharding_over_512_rows():
    run_on_cpu_devices(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import equinox as eqx
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.models.snowball import RMSNorm

        axes = ("replica_dcn", "data", "context", "expert", "model")
        mesh = Mesh(
            np.asarray(jax.devices()).reshape((1, 4, 1, 1, 1)),
            axes,
            axis_types=(AxisType.Explicit,) * len(axes),
        )
        input_spec = P(("replica_dcn", "data", "expert"), None, None)
        with jax.set_mesh(mesh):
            host_x = jnp.linspace(-1.0, 1.0, 4 * 256 * 128, dtype=jnp.float32).reshape(4, 256, 128)
            x = jax.device_put(host_x, NamedSharding(mesh, input_spec))
            weight = jnp.linspace(0.5, 1.5, 128, dtype=jnp.float32)
            norm = RMSNorm(weight=weight, eps=1e-5, implementation="reference")

            def independent(value, scale):
                value32 = value.astype(jnp.float32)
                rms = jax.lax.rsqrt(jnp.mean(jnp.square(value32), axis=-1, keepdims=True) + 1e-5)
                return (value32 * rms * scale).astype(value.dtype)

            actual = eqx.filter_jit(norm)(x)
            expected = independent(x, weight)
            _, gradient = eqx.filter_jit(jax.value_and_grad(lambda value: jnp.sum(norm(value))))(x)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
        assert actual.sharding.spec == input_spec
        assert gradient.sharding.spec == input_spec
        assert bool(jnp.isfinite(gradient).all())
        """,
        device_count=4,
    )


def test_snowball_hf_config_does_not_initialize_jax_backend():
    script = textwrap.dedent(
        """
        from jax._src import xla_bridge
        from levanter.models.snowball import SnowballConfig

        cfg = SnowballConfig()
        SnowballConfig.from_hf_config(cfg.to_hf_config(cfg.vocab_size))
        assert not xla_bridge.backends_are_initialized()
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


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


@pytest.mark.parametrize("individual_experts", [False, True])
@pytest.mark.parametrize("prefix", [None, "policy"])
@pytest.mark.parametrize("packed_experts", [False, True])
def test_snowball_state_dict_roundtrip_is_exact(individual_experts, prefix, packed_experts):
    cfg = _tiny_config()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        src = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(1))
        dst = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(2))
        if packed_experts:
            blocks = tuple(layer.with_packed_experts() for layer in dst.transformer.blocks)
            dst = eqx.tree_at(lambda model: model.transformer.blocks, dst, blocks)
        canonical = src.to_state_dict(prefix=prefix)
        sd = {}
        for name, weight in canonical.items():
            if individual_experts and ".mlp.experts." in name:
                stem, projection = name.split(".mlp.experts.")
                expert_sharding = NamedSharding(weight.sharding.mesh, P(*weight.sharding.spec[1:]))
                sd.update(
                    {
                        f"{stem}.mlp.experts.{i}.{projection}": weight.at[i].get(out_sharding=expert_sharding)
                        for i in range(cfg.num_experts)
                    }
                )
            else:
                sd[name] = weight
        dst = hax.named_jit(lambda m, state: m.from_state_dict(state, prefix=prefix))(dst, sd)
        assert all((layer.mlp.expert_mlp.w_up is None) == packed_experts for layer in dst.transformer.blocks)
        for name, weight in dst.to_state_dict(prefix=prefix).items():
            np.testing.assert_array_equal(np.asarray(weight), np.asarray(canonical[name]), err_msg=name)

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


@pytest.mark.multidevice
@pytest.mark.parametrize("source_dtype", [jnp.float32, jnp.bfloat16])
def test_snowball_public_import_consumes_owned_buffers_preserving_fp32_values_and_sharding(
    tmp_path, monkeypatch, source_dtype
):
    with use_test_devices(4) as devices:
        mesh = Mesh(
            np.asarray(devices).reshape(1, 2, 2, 1, 1),
            ("replica_dcn", "data", "expert", "context", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        tokenizer = Tokenizer(
            models.WordLevel(vocab={"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 64)}}, unk_token="<pad>")
        )
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>")
        config = _tiny_config(vocab_size=64, num_layers=2)
        converter = config.hf_checkpoint_converter().replaced(tokenizer=fast)
        with jax.set_mesh(mesh):
            reference = SnowballLMHeadModel.init(Axis("vocab", 64), config, key=random.key(51))
            expected = {
                key: np.asarray(value.astype(source_dtype).astype(jnp.float32))
                for key, value in reference.to_state_dict().items()
            }
            converter.save_pretrained(
                reference, str(tmp_path), dtype=source_dtype, save_tokenizer=False, save_reference_code=False
            )
            # Compare the public import's full layout with direct native decoding
            # of real, independently read HF tensors on the same staging mesh.
            direct = reference.from_state_dict(converter.load_state_dict(str(tmp_path), dtype=jnp.float32))
            jax.block_until_ready(direct)
            expected_specs = [leaf.sharding.spec for leaf in jax.tree.leaves(direct) if eqx.is_array(leaf)]
            owned_inputs = {}
            load_weights = HFCheckpointConverter.load_state_dict

            def capture_weight_io(self, *args, **kwargs):
                tensors = load_weights(self, *args, **kwargs)
                owned_inputs.update(tensors)
                return tensors

            monkeypatch.setattr(HFCheckpointConverter, "load_state_dict", capture_weight_io)
            restored = converter.load_pretrained(
                SnowballLMHeadModel, ref=str(tmp_path), config=config, axis_mapping={}, dtype=jnp.float32
            )
            jax.block_until_ready(restored)
            restored_sd = restored.to_state_dict()
            assert restored_sd.keys() == expected.keys()
            for key, value in restored_sd.items():
                assert value.dtype == jnp.float32
                np.testing.assert_array_equal(np.asarray(value), expected[key])
            assert [leaf.sharding.spec for leaf in jax.tree.leaves(restored) if eqx.is_array(leaf)] == expected_specs
            assert any(array.is_deleted() for array in owned_inputs.values()), "import retained every owned HF buffer"


@pytest.mark.multidevice
def test_snowball_import_resharding_bounds_collective_scratch():
    """Reproduce the import layout redistribution on the production data4/expert8 geometry."""
    with use_test_devices(32) as devices:
        if devices[0].platform != "cpu":
            pytest.skip("Compiler memory regression uses the CPU SPMD executable")
        mesh = Mesh(
            np.asarray(devices).reshape(1, 4, 8, 1, 1),
            ("replica_dcn", "data", "expert", "context", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        config = _tiny_config(
            vocab_size=256,
            hidden_dim=64,
            intermediate_dim=64,
            shared_expert_intermediate_dim=64,
            num_experts=32,
            num_layers=4,
            head_dim=16,
        )
        with jax.set_mesh(mesh):
            reference = SnowballLMHeadModel.init(Axis("vocab", 256), config, key=random.key(52))
            canonical = {key: np.asarray(value).copy() for key, value in reference.to_state_dict().items()}
            template = eqx.filter_eval_shape(SnowballLMHeadModel.init, Axis("vocab", 256), config, key=random.key(0))

            def import_weights(template, weights):
                return from_torch_compatible_state_dict(template, weights)

            convert = hax.named_jit(import_weights, axis_resources={}, donate_args=(False, True))
            staged = {
                key: jax.device_put(value, best_effort_sharding(value.shape, mesh=mesh))
                for key, value in canonical.items()
            }
            original_shape = eqx.filter_eval_shape(import_weights, template, staged)
            original_executable = convert.lower(template, staged).compile()
            original_memory = original_executable.memory_analysis()
            _reshard_state_dict_for_import(staged, original_shape, prefix=None)
            aligned_executable = convert.lower(template, staged).compile()
            aligned_memory = aligned_executable.memory_analysis()
            # These are compiler buffer sizes, not model-size estimates or a GPU-fit assertion.
            assert aligned_memory.temp_size_in_bytes < original_memory.temp_size_in_bytes / 2
            assert "all-gather(" in original_executable.as_text()
            assert "all-gather(" not in aligned_executable.as_text()
            restored = convert(template, staged)
            jax.block_until_ready(restored)
            for key, value in restored.to_state_dict().items():
                assert value.dtype == jnp.float32
                np.testing.assert_array_equal(np.asarray(value), canonical[key])
            assert [leaf.sharding.spec for leaf in jax.tree.leaves(restored) if eqx.is_array(leaf)] == [
                leaf.sharding.spec
                for leaf in jax.tree.leaves(original_shape)
                if isinstance(leaf, jax.ShapeDtypeStruct)
            ]


def test_snowball_export_preserves_banked_weights_and_overwrites_existing_output(tmp_path):
    cfg = _tiny_config(num_layers=2, num_experts=4)
    converter = cfg.hf_checkpoint_converter()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(4))
        expected = to_torch_compatible_state_dict(model)
        options = dict(save_tokenizer=False, save_reference_code=False, max_shard_size=4096, dtype=jnp.bfloat16)
        converter.save_pretrained(model, str(tmp_path), **options)
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        tensors = {}
        for filename in set(index["weight_map"].values()):
            shard = load_file(tmp_path / filename)
            assert all(index["weight_map"][name] == filename for name in shard)
            assert not tensors.keys() & shard.keys()
            tensors.update(shard)
        assert tensors.keys() == expected.keys() == index["weight_map"].keys()
        for name, value in expected.items():
            wanted = np.asarray(jnp.asarray(value, dtype=jnp.bfloat16))
            actual = tensors[name]
            assert actual.shape == wanted.shape and actual.dtype == wanted.dtype, name
            assert actual.tobytes() == wanted.tobytes(), name
        assert index["metadata"]["total_size"] == sum(value.nbytes for value in tensors.values())
        assert json.loads((tmp_path / "config.json").read_text()) == json.loads(
            json.dumps(cfg.to_hf_config(cfg.vocab_size).to_dict())
        )

        # Ordinary converter callers can intentionally overwrite a completed destination.
        filename = next(iter(index["weight_map"].values()))
        original = (tmp_path / filename).read_bytes()
        (tmp_path / filename).write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        converter.save_pretrained(model, str(tmp_path), **options)
        assert (tmp_path / filename).read_bytes() == original


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


def test_snowball_moe_capacity_is_a_runtime_setting():
    """The capacity reaches every expert MLP, is not exported to HF metadata, and survives an HF reload."""
    cfg = _tiny_config(moe_capacity_factor=4.0)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(3))
    assert {block.mlp.expert_mlp.capacity_factor for block in model.transformer.blocks} == {4.0}
    hf = cfg.to_hf_config(cfg.vocab_size).to_dict()
    assert "moe_capacity_factor" not in hf and "capacity_factor" not in hf
    reloaded = SnowballConfig.from_hf_config(cfg.to_hf_config(cfg.vocab_size))
    assert preserve_hf_runtime_settings(reloaded, cfg).moe_capacity_factor == 4.0


def test_snowball_fp32_activations_is_a_runtime_setting():
    """The runtime flag stays out of HF metadata and rejects the BF16-only fused backend."""
    cfg = _tiny_config(fp32_activations=True)
    hf = cfg.to_hf_config(cfg.vocab_size).to_dict()
    assert "fp32_activations" not in hf
    with pytest.raises(ValueError, match="fp32_activations"):
        _tiny_config(fp32_activations=True, moe_implementation="sonic_cute")


def _bf16(key, shape, scale):
    return (scale * jax.random.normal(key, shape)).astype(jnp.bfloat16)


def _walk_eqns(jaxpr, path: tuple = ()):
    """Yield (enclosing primitive names, eqn) for every eqn, walking nested jaxprs (pjit, custom_jvp, scan, remat,
    cond, shard_map)."""
    for eqn in jaxpr.eqns:
        yield path, eqn
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else (value,):
                inner = sub.jaxpr if isinstance(sub, jax_core.ClosedJaxpr) else sub
                if isinstance(inner, jax_core.Jaxpr):
                    yield from _walk_eqns(inner, (*path, eqn.primitive.name))


def test_snowball_fp32_activations_round_once():
    """On, GatedNorm and the shared MLP equal their activations computed in FP32 by hand and cast to BF16 once; off,
    they equal the BF16 activations, with no FP32 cast in the program."""
    k = iter(jax.random.split(jax.random.key(0), 8))
    x = jnp.tile(_bf16(next(k), (3, 5, 32), 2.0), (jax.device_count(), 1, 1))
    w_down, w_up = _bf16(next(k), (32, 16), 0.5), _bf16(next(k), (16, 32), 0.5)
    w_gate, w_up2, w_down2 = (
        _bf16(next(k), (32, 24), 0.5),
        _bf16(next(k), (32, 24), 0.5),
        _bf16(next(k), (24, 32), 0.5),
    )

    def by_hand(fp32):
        def act(fn, h):
            return fn(h.astype(jnp.float32)).astype(h.dtype) if fp32 else fn(h)

        hidden = act(jax.nn.silu, jnp.einsum("...d,dr->...r", x, w_down))
        gated = x * act(jax.nn.sigmoid, jnp.einsum("...r,rd->...d", hidden, w_up))
        flat = x.reshape(-1, 32)
        mlp = jnp.einsum("tm,md->td", act(jax.nn.silu, flat @ w_gate) * (flat @ w_up2), w_down2)
        return gated, mlp.reshape(x.shape)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        outputs = {}
        for flag in (False, True):
            norm = GatedNorm(w_down=w_down, w_up=w_up, implementation="native", fp32_activations=flag)
            mlp = DenseMLP(
                w_gate=w_gate,
                w_up=w_up2,
                w_down=w_down2,
                projection_implementation="native",
                fp32_activations=flag,
            )
            outputs[flag] = (norm(x), mlp(x))
            for got, want in zip(outputs[flag], by_hand(flag)):
                assert got.dtype == jnp.bfloat16
                assert np.array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
            program = [eqn for _, eqn in _walk_eqns(jax.make_jaxpr(lambda v: norm(v))(x).jaxpr)]
            assert any(e.primitive.name == "convert_element_type" for e in program) == flag
    assert not np.array_equal(np.asarray(outputs[False][0], np.float32), np.asarray(outputs[True][0], np.float32))


def _logistic_sites(jaxpr) -> collections.Counter:
    """Count ``logistic`` eqns (sigmoid; SiLU is ``x * logistic(x)``) by (enclosing primitives, input dtype)."""
    return collections.Counter(
        (path, eqn.invars[0].aval.dtype.name) for path, eqn in _walk_eqns(jaxpr) if eqn.primitive.name == "logistic"
    )


def test_snowball_fp32_activations_cover_every_sigmoid():
    """Traced BF16 forward: with the flag on, every sigmoid and SiLU takes an FP32 input -- the embed/final and block
    GatedNorms, the attention head gate (both ``cond`` branches), the shared MLP and the routed experts
    (``shard_map``). Off, all of them take BF16 and only the router combine (always FP32) does not."""
    sites = {}
    for flag in (False, True):
        cfg = _tiny_config(fp32_activations=flag)
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
            model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(3))
            model = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if a.dtype == jnp.float32 else a, model)
            ids = _device_batched_ids(cfg.vocab_size, 8)
            jaxpr = jax.make_jaxpr(lambda m, i: m(i).array)(model, ids)
        found = _logistic_sites(jaxpr.jaxpr)
        sites[flag] = {
            "top": sum(n for (path, _), n in found.items() if "scan" not in path),
            "head_gate": sum(n for (path, _), n in found.items() if "cond" in path),
            "experts": sum(n for (path, _), n in found.items() if "shard_map" in path),
            "fp32": sum(n for (_, dtype), n in found.items() if dtype == "float32"),
            "total": sum(found.values()),
            "dtypes_by_path": found,
        }
    on, off = sites[True], sites[False]
    # Embed + final GatedNorm (2 SiLU, 2 sigmoid); one scanned block: 2 GatedNorms (4), the head gate in both
    # branches (2), the shared MLP (1), the routed experts (1) and the router combine (1).
    assert (off["top"], off["head_gate"], off["experts"], off["total"]) == (4, 2, 1, 13)
    assert {k: on[k] for k in ("top", "head_gate", "experts", "total")} == {
        k: off[k] for k in ("top", "head_gate", "experts", "total")
    }
    assert on["fp32"] == on["total"]
    assert off["fp32"] == 1, "flag off: only the router combine sigmoid is FP32"
    router = [(path, dtype) for (path, dtype) in off["dtypes_by_path"] if dtype == "float32"]
    assert router and all("scan" in path and "cond" not in path and "shard_map" not in path for path, _ in router)


def test_snowball_forward_shapes_and_finite():
    cfg = _tiny_config()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(3))
        ids = _device_batched_ids(cfg.vocab_size, 8)
        logits = hax.named_jit(lambda m, x: m(x))(model, ids)
    assert logits.axes[-1].name == "vocab" and logits.axes[-1].size == cfg.vocab_size
    assert bool(jnp.all(jnp.isfinite(logits.array)))


def test_snowball_route_observer_is_output_identical():
    cfg = _tiny_config()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(3))
        ids = _device_batched_ids(cfg.vocab_size, 8)
        plain = hax.named_jit(lambda m, x: m.activations(x))(model, ids)
        observed, routes = hax.named_jit(lambda m, x: m.activations_with_routes(x))(model, ids)

        # Layer 0's routes, recomputed by hand from the block's own pieces.
        tf = model.transformer
        tokens = ids.array
        hidden = tf.embed_gated_norm(tf.embed_norm(tf.token_embed.at[tokens].get(out_sharding=_activation_spec())))
        block = tf.blocks[0]
        short_mask = GrugAttentionMask(is_causal=True, sliding_window=cfg.sliding_window)
        x = hidden + block.attn(block.attn_gated_norm(block.rms_attn(hidden)), short_mask, disable_rope=False)
        mlp_in = block.mlp_gated_norm(block.rms_mlp(x))
        logits = jnp.einsum("bsd,de->bse", mlp_in, block.mlp.router).astype(jnp.float32) + block.mlp.router_bias
        expected = jax.lax.top_k(logits, cfg.num_experts_per_token + 1)[1][..., :-1]

    routes, expected = np.asarray(jax.device_get(routes)), np.asarray(jax.device_get(expected))
    assert np.array_equal(np.asarray(jax.device_get(plain.array)), np.asarray(jax.device_get(observed.array)))
    assert routes.shape == (cfg.num_layers, tokens.shape[0], tokens.shape[1], cfg.num_experts_per_token)
    assert ((routes >= 0) & (routes < cfg.num_experts)).all()
    np.testing.assert_array_equal(routes[0], expected)


def test_fp32_router_dot_preserves_a_route_lost_to_bfloat16_rounding():
    cfg = _tiny_config(router_dot_fp32=True, num_experts=4, num_experts_per_token=1)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballMoEMLP.init(cfg, key=jax.random.key(23))
        # BF16 rounds 1 + 1/256 to 1 before bias is added. Native FP32
        # router bias then makes expert 1 win; the FP32 dot must retain expert 0.
        router = jnp.zeros((cfg.hidden_dim, cfg.num_experts), jnp.float32)
        router = router.at[0, :2].set(1).at[1, 0].set(1 / 256)
        model = replace(model, router=router, router_bias=jnp.array([0, 0.001, -10, -10], jnp.float32))
        compute = with_field_dtype_overrides(
            jmp.get_policy("p=f32,c=bf16"), {"router_bias": jnp.dtype("float32")}
        ).cast_to_compute(model)
        inputs = jnp.zeros((jax.device_count(), 1, cfg.hidden_dim), jnp.bfloat16).at[:, 0, :2].set(1)
        # Keep primitive boundaries explicit: XLA may fuse the BF16 dot's
        # following FP32 conversion into a CPU dot with an FP32 accumulator.
        routes = lambda value: value.with_routes(inputs)[1]
        high = routes(compute)
        low = routes(replace(compute, cfg=replace(cfg, router_dot_fp32=False)))
    np.testing.assert_array_equal(high, np.zeros((jax.device_count(), 1, 1), dtype=np.int32))
    np.testing.assert_array_equal(low, np.ones((jax.device_count(), 1, 1), dtype=np.int32))


def test_snowball_moe_capture_replay_preserves_padding_and_gradients():
    cfg = _tiny_config(num_layers=1, moe_implementation="scatter")
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        layer = SnowballMoEMLP.init(cfg, key=jax.random.key(17))
        x = jax.random.normal(jax.random.key(18), (2, 4, cfg.hidden_dim))
        token_valid = jnp.asarray([True, True, False, True, True, False, True, True])
        capture = layer.capture_routes(x, token_valid=token_valid)
        reference_logits = np.einsum("td,de->te", np.asarray(x).reshape(-1, cfg.hidden_dim), np.asarray(layer.router))
        reference_logits += np.asarray(layer.router_bias)
        reference_order = np.argsort(-reference_logits, axis=-1, kind="stable")[:, : cfg.num_experts_per_token + 1]
        perturbed = eqx.tree_at(lambda m: m.router, layer, layer.router.at[:, 0].add(10.0))
        replayed = perturbed(x, token_valid=token_valid, route_capture=capture)
        fresh = perturbed(x, token_valid=token_valid)
        replay_grad = eqx.filter_grad(lambda m: jnp.sum(m(x, token_valid=token_valid, route_capture=capture)))(
            perturbed
        )

    assert not np.allclose(np.asarray(replayed), np.asarray(fresh))
    # The routed branch excludes padding; the always-on shared expert remains part of the residual.
    assert np.all(np.isfinite(np.asarray(replayed).reshape(-1, cfg.hidden_dim)[~np.asarray(token_valid)]))
    assert float(jnp.max(jnp.abs(replay_grad.router))) > 0.0
    assert float(jnp.max(jnp.abs(replay_grad.expert_mlp.w_gate))) > 0.0
    assert isinstance(capture, PreparedMoERouteCapture)
    assert capture.selected_experts.shape == (8, cfg.num_experts_per_token)
    assert capture.qb_alpha.shape == (8, 1)
    assert np.asarray(capture.qb_alpha).dtype == np.float32
    assert np.all(np.isfinite(np.asarray(capture.qb_alpha)))
    np.testing.assert_array_equal(
        np.asarray(capture.selected_experts), reference_order[:, : cfg.num_experts_per_token]
    )
    np.testing.assert_array_equal(np.asarray(capture.token_valid), np.asarray(token_valid))


def test_snowball_moe_capture_routes_reports_exact_qb_threshold_for_representable_logits():
    cfg = _tiny_config(num_layers=1, num_experts=4, num_experts_per_token=2, moe_implementation="scatter")
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        layer = SnowballMoEMLP.init(cfg, key=jax.random.key(19))
        x = (
            jnp.zeros((1, 4, cfg.hidden_dim), dtype=jnp.float32)
            .at[:, :, 0]
            .set(jnp.asarray([1, 2, 4, 8], dtype=jnp.float32))
        )
        router = jnp.zeros_like(layer.router).at[0, :4].set(jnp.asarray([1, 2, 3, 4], dtype=jnp.float32))
        router_bias = jnp.asarray([0.5, -0.5, 0.25, -0.25], dtype=jnp.float32)
        layer = eqx.tree_at(lambda m: m.router, layer, router)
        layer = eqx.tree_at(lambda m: m.router_bias, layer, router_bias)
        token_valid = jnp.asarray([True, True, False, True])
        capture = layer.capture_routes(x, token_valid=token_valid)

    reference_logits = np.einsum("td,de->te", np.asarray(x).reshape(-1, cfg.hidden_dim), np.asarray(router))
    reference_logits += np.asarray(router_bias)
    reference_order = np.argsort(-reference_logits, axis=-1, kind="stable")[:, : cfg.num_experts_per_token + 1]
    np.testing.assert_array_equal(
        np.asarray(capture.selected_experts), reference_order[:, : cfg.num_experts_per_token]
    )
    np.testing.assert_allclose(
        np.asarray(capture.qb_alpha[:, 0]), reference_logits[np.arange(4), reference_order[:, -1]], rtol=1e-7, atol=0
    )
    np.testing.assert_array_equal(np.asarray(capture.token_valid), np.asarray(token_valid))


def test_snowball_transformer_replay_jit_and_grad_use_actual_packed_validity():
    cfg = _tiny_config(num_layers=2, moe_implementation="scatter")
    batch, seq = 2, 6
    ids = jnp.arange(batch * seq, dtype=jnp.int32).reshape(batch, seq) % cfg.vocab_size
    segments = jnp.asarray([[0, 0, -1, 1, 1, -1], [2, 2, 3, 3, -1, -1]], dtype=jnp.int32)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballTransformer.init(cfg, key=jax.random.key(23))
        mask = GrugAttentionMask(segment_ids=(segments, segments))
        ordinary = model(ids, mask)
        captures = model.capture_route_captures(ids, mask)

        def loss(current_model, segment_ids):
            mask = GrugAttentionMask(segment_ids=(segment_ids, segment_ids))
            hidden = current_model(ids, mask, route_captures=captures)
            return jnp.mean(hidden * hidden)

        value, gradients = eqx.filter_jit(eqx.filter_value_and_grad(loss))(model, segments)
        replayed = model(ids, mask, route_captures=captures)

    np.testing.assert_array_equal(np.asarray(ordinary), np.asarray(replayed))
    assert bool(jnp.isfinite(value))
    assert float(jnp.max(jnp.abs(gradients.blocks[0].mlp.router))) > 0.0
    assert float(jnp.max(jnp.abs(gradients.blocks[0].mlp.expert_mlp.w_gate))) > 0.0


def test_snowball_public_capture_replay_binds_tokens_and_preserves_logits():
    cfg = _tiny_config(num_layers=2, moe_implementation="scatter")
    ids_a = hax.named(jnp.asarray([[2, 7, 0, 9, 3, 0], [4, 5, 6, 0, 8, 0]], dtype=jnp.int32), ("batch", "position"))
    ids_b = hax.named(jnp.asarray([[11, 12, 0, 13, 14, 0], [15, 16, 17, 0, 18, 0]], dtype=jnp.int32), ids_a.axes)
    q_segment_ids = hax.named(jnp.asarray([[0, 0, -1, 1, 1, -1], [2, 2, 3, -1, 3, -1]], dtype=jnp.int32), ids_a.axes)
    kv_segment_ids = hax.named(jnp.asarray([[0, 0, -1, 1, 1, -1], [2, 2, 3, -1, 4, -1]], dtype=jnp.int32), ids_a.axes)
    mask = LmAttentionMask.causal().with_segment_ids(q_segment_ids, kv_segment_ids)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(31))
        ordinary = model(ids_a, mask)
        prepared = model.capture_route_captures(ids_a, mask)
        assert isinstance(prepared, SnowballRouteCaptureBatch)
        replayed = model(ids_a, mask, route_captures=prepared)
        changed_block = eqx.tree_at(
            lambda block: block.mlp.router,
            model.transformer.blocks[0],
            model.transformer.blocks[0].mlp.router.at[:, 0].add(0.01),
        )
        changed = eqx.tree_at(lambda current: current.transformer.blocks[0], model, changed_block)
        changed_replay = changed(ids_a, mask, route_captures=prepared)
        changed_grad = eqx.filter_grad(lambda current: jnp.sum(current(ids_a, mask, route_captures=prepared).array))(
            changed
        )

    np.testing.assert_array_equal(np.asarray(ordinary.array), np.asarray(replayed.array))
    assert not np.array_equal(np.asarray(changed_replay.array), np.asarray(ordinary.array))
    assert float(jnp.max(jnp.abs(changed_grad.transformer.blocks[0].mlp.router))) > 0.0
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        with pytest.raises(TypeError, match="bound SnowballRouteCaptureBatch"):
            model(ids_a, mask, route_captures=prepared.route_captures)
        with pytest.raises(ValueError, match="different token IDs"):
            model(ids_b, mask, route_captures=prepared)
        changed_kv_array = kv_segment_ids.array.at[1, 4].set(3)
        changed_kv_segment_ids = hax.named(changed_kv_array, kv_segment_ids.axes)
        changed_kv_mask = LmAttentionMask.causal().with_segment_ids(q_segment_ids, changed_kv_segment_ids)
        with pytest.raises(ValueError, match="different packed segment IDs"):
            model(ids_a, changed_kv_mask, route_captures=prepared)


def test_snowball_next_token_loss_replays_routes_with_masks_and_gradients():
    cfg = SnowballConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=48,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=12,
        max_seq_len=16,
        sliding_window=4,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="scatter",
        rms_norm_implementation="native",
    )
    compute_axis_mapping = {
        "batch": ("replica_dcn", "data", "expert"),
        "position": None,
        "vocab": "model",
    }
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        batch = Axis("batch", 2)
        position = Axis("position", 8)
        ids = hax.named(
            jnp.asarray([[3, 5, 7, 11, 13, 17, 0, 0], [19, 23, 29, 31, 37, 0, 0, 0]], dtype=jnp.int32),
            (batch, position),
        )
        segment_ids = hax.named(
            jnp.asarray([[0, 0, 0, 0, 0, 0, -1, -1], [1, 1, 1, 1, 1, -1, -1, -1]], dtype=jnp.int32),
            (batch, position),
        )
        loss_weight = hax.named(
            jnp.asarray(
                [[1.0, 0.5, 0.25, 0.75, 1.0, 0.9, 0.0, 0.0], [1.0, 0.2, 0.8, 0.4, 0.6, 0.0, 0.0, 0.0]],
                dtype=jnp.float32,
            ),
            (batch, position),
        )
        example = LmExample(
            tokens=ids,
            loss_weight=loss_weight,
            attn_mask=LmAttentionMask.causal().with_segment_ids(segment_ids),
        )
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(33))
        with hax.axis_mapping(compute_axis_mapping):
            sharded_example = hax.auto_sharded(example)
            captures = model.capture_route_captures(sharded_example.tokens, sharded_example.attn_mask)
            changed_block = eqx.tree_at(
                lambda block: block.mlp.router,
                model.transformer.blocks[0],
                model.transformer.blocks[0].mlp.router.at[:, 0].add(0.01),
            )
            changed = eqx.tree_at(lambda current: current.transformer.blocks[0], model, changed_block)
            uncaptured_logits = changed(sharded_example.tokens, sharded_example.attn_mask)
            replay_logits = changed(sharded_example.tokens, sharded_example.attn_mask, route_captures=captures)
            replay_loss = changed.compute_next_token_loss(sharded_example, reduction=None, route_captures=captures)
            replay_mean = changed.compute_next_token_loss(sharded_example, route_captures=captures)

            def reference_loss(logits):
                logits = np.asarray(logits)
                target = np.roll(np.asarray(sharded_example.tokens.array), -1, axis=1)
                weights = np.asarray(sharded_example.loss_weight.array).copy()
                weights[:, -1] = 0
                log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
                token_loss = -np.take_along_axis(log_probs, target[..., None], axis=-1)[..., 0]
                return jnp.asarray(token_loss * weights)

            expected_unreduced = reference_loss(replay_logits.array)
            expected_mean = jnp.sum(expected_unreduced) / jnp.sum(sharded_example.loss_weight.array.at[:, -1].set(0))

            def loss_for_grad(current):
                return current.compute_next_token_loss(sharded_example, route_captures=captures).array

            _, gradients = eqx.filter_value_and_grad(loss_for_grad)(changed)

            changed_tokens = hax.named(ids.array.at[0, 0].set(11), ids.axes)
            mismatched_example = hax.auto_sharded(dataclasses.replace(example, tokens=changed_tokens))
            with pytest.raises(ValueError, match="different token IDs"):
                changed.compute_next_token_loss(mismatched_example, route_captures=captures)

    assert not np.array_equal(np.asarray(uncaptured_logits.array), np.asarray(replay_logits.array))
    np.testing.assert_allclose(np.asarray(replay_loss.array), np.asarray(expected_unreduced), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(replay_mean.array), np.asarray(expected_mean), rtol=1e-5, atol=1e-5)
    assert float(jnp.max(jnp.abs(gradients.transformer.blocks[0].mlp.router))) > 0.0


def test_snowball_public_capture_replay_jit_binds_dynamic_tokens_and_segments():
    cfg = _tiny_config(num_layers=1, moe_implementation="scatter")
    ids = hax.named(jnp.asarray([[2, 7, 0, 9, 3, 0], [4, 5, 6, 0, 8, 0]], dtype=jnp.int32), ("batch", "position"))
    q_segment_ids = hax.named(jnp.asarray([[0, 0, -1, 1, 1, -1], [2, 2, 3, -1, 3, -1]], dtype=jnp.int32), ids.axes)
    kv_segment_ids = hax.named(jnp.asarray([[0, 0, -1, 1, 1, -1], [2, 2, 3, -1, 4, -1]], dtype=jnp.int32), ids.axes)
    mask = LmAttentionMask.causal().with_segment_ids(q_segment_ids, kv_segment_ids)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(32))
        prepared = model.capture_route_captures(ids, mask)
        ordinary = model(ids, mask)

        def loss(current_model, token_array, q_array, kv_array, capture):
            dynamic_ids = hax.named(token_array, ids.axes)
            dynamic_q = hax.named(q_array, ids.axes)
            dynamic_kv = hax.named(kv_array, ids.axes)
            dynamic_mask = LmAttentionMask.causal().with_segment_ids(dynamic_q, dynamic_kv)
            return jnp.sum(current_model(dynamic_ids, dynamic_mask, route_captures=capture).array)

        replay_value, gradients = eqx.filter_jit(eqx.filter_value_and_grad(loss))(
            model, ids.array, q_segment_ids.array, kv_segment_ids.array, prepared
        )

    np.testing.assert_allclose(np.asarray(replay_value), np.asarray(jnp.sum(ordinary.array)))
    assert float(jnp.max(jnp.abs(gradients.transformer.blocks[0].mlp.router))) > 0.0


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


@pytest.mark.multidevice
def test_snowball_load_path_multidevice_sharding():
    """The load-path forward must survive a data-sharded mesh (regression for the 67B router_bias).

    ``g()``-loaded leaves (norm weights, router_bias) inherit the sharding of the incoming state
    dict, and a safetensors load auto-shards ``[E]``/``[D]`` tensors over ``data`` when the size
    divides the axis. On a single device this is invisible; with multiple devices,
    ``router_logits + router_bias`` was illegally sharded. Force-shards the state dict like
    safetensors on a data-parallel mesh to reproduce the condition.
    """
    with use_test_devices(4) as devices:
        # Four data shards are enough to reproduce the sharded router_bias placement (E=16).
        cfg = SnowballConfig(
            vocab_size=128,
            hidden_dim=64,
            intermediate_dim=64,
            shared_expert_intermediate_dim=64,
            num_experts=16,
            num_experts_per_token=4,
            num_layers=3,
            num_heads=8,
            num_kv_heads=4,
            head_dim=16,
            max_seq_len=32,
            sliding_window=4,
            qk_mult=1.37,
            layer_norm_eps=1e-5,
            initializer_std=0.02,
            attention_implementation="reference",
        )
        Vocab = Axis("vocab", cfg.vocab_size)
        mesh = Mesh(
            np.asarray(devices).reshape(1, 4, 1, 1, 1),
            ("replica_dcn", "data", "expert", "context", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        Batch = Axis("batch", len(devices))
        Pos = Axis("position", 8)
        ids = hax.named(
            (jnp.arange(Batch.size * Pos.size, dtype=jnp.int32) % cfg.vocab_size).reshape(Batch.size, Pos.size),
            (Batch, Pos),
        )

        def like_safetensors(v):
            # Auto-shard the leading axis on data when it divides 4, else replicate (mimics the
            # placement of freshly-read safetensors that broke the 67B).
            v = jnp.asarray(v)
            spec = P("data") if v.ndim >= 1 and v.shape[0] % len(devices) == 0 else P()
            return jax.device_put(v, NamedSharding(mesh, spec))

        with set_mesh(mesh):
            src = SnowballLMHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(1))
            sd = {k: like_safetensors(v) for k, v in to_torch_compatible_state_dict(src).items()}
            ref = np.asarray(hax.named_jit(lambda m, x: m(x))(src, ids).array)
            template = eqx.filter_eval_shape(SnowballLMHeadModel.init, Vocab, cfg, key=jax.random.PRNGKey(0))
            loaded = hax.named_jit(lambda t, s: from_torch_compatible_state_dict(t, s))(template, sd)
            got = np.asarray(hax.named_jit(lambda m, x: m(x))(loaded, ids).array)
        assert np.array_equal(ref, got), "data-sharded load-path logits differ from the reference"


def test_snowball_hf_config_loading_allows_distributed_initialization(tmp_path):
    """HF config loading must leave the JAX backend uninitialized for distributed setup."""
    cfg = _tiny_config()
    hf = cfg.to_hf_config(cfg.vocab_size)
    (tmp_path / "config.json").write_text(__import__("json").dumps(hf.to_dict()))

    script = textwrap.dedent(
        """
        import socket
        import sys
        import jax
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
        from levanter.models.snowball import SnowballConfig
        SnowballConfig.from_hf_config(cfg)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        jax.distributed.initialize(coordinator_address=f"127.0.0.1:{port}", num_processes=1, process_id=0)
        assert jax.distributed.is_initialized()
        jax.distributed.shutdown()
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


@pytest.mark.timeout(120)
@pytest.mark.multidevice
def test_snowball_context_parallel_values_and_gradients_match_data_parallel():
    with use_test_devices(8) as devices:
        cfg = SnowballConfig(
            vocab_size=32,
            hidden_dim=16,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            max_seq_len=8,
            sliding_window=4,
            attention_implementation="reference",
            moe_implementation="ring",
        )
        ids = hax.named(
            jnp.arange(32, dtype=jnp.int32).reshape(4, 8),
            (Axis("batch", 4), Axis("position", 8)),
        )
        axes = ("replica_dcn", "data", "expert", "context", "model")

        def run(shape):
            mesh = Mesh(
                np.asarray(devices[: int(np.prod(shape))]).reshape(shape),
                axes,
                axis_types=(AxisType.Explicit,) * len(axes),
            )
            with jax.set_mesh(mesh):
                model = SnowballLMHeadModel.init(Axis("vocab", 32), cfg, key=jax.random.key(0))
                activations = eqx.filter_jit(lambda m: m.activations(ids).array)(model)
                value, grad = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.mean(m(ids).array)))(model)
            block = grad.transformer.blocks[0]
            return activations.sharding.spec[1], (
                np.asarray(value),
                np.asarray(grad.transformer.token_embed),
                np.asarray(block.attn.w_q),
                np.asarray(block.mlp.expert_mlp.w_gate),
                np.asarray(block.shared.w_gate),
                np.asarray(grad.transformer.output_proj),
            )

        data_axis, reference = run((1, 2, 2, 1, 1))
        assert data_axis is None
        # Four context shards make the flattened token sharding larger than the batch.
        for shape in [(1, 1, 2, 2, 1), (1, 1, 2, 4, 1)]:
            context_axis, sharded = run(shape)
            assert context_axis == "context"
            for expected, actual in zip(reference, sharded, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.timeout(120)
@pytest.mark.multidevice
def test_snowball_hf_init_trains_with_context_and_expert_parallelism():
    with use_test_devices(4):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            hf_path = root / "hf"
            hf_path.mkdir()
            tokenizer = Tokenizer(
                models.WordLevel(
                    vocab={"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 64)}},
                    unk_token="<pad>",
                )
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>")
            fast.save_pretrained(hf_path)
            model_cfg = SnowballConfig(
                vocab_size=64,
                hidden_dim=32,
                intermediate_dim=32,
                shared_expert_intermediate_dim=32,
                num_experts=8,
                num_experts_per_token=2,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                max_seq_len=8,
                sliding_window=4,
                attention_implementation="reference",
                tokenizer=str(hf_path),
            )
            mesh = MeshConfig(
                axes={"data": -1, "replica": 1, "model": 1, "context": 2, "expert": 2},
                compute_mapping={
                    "batch": ["replica_dcn", "data", "expert"],
                    "position": "context",
                    "vocab": "model",
                },
            )
            trainer_cfg = TrainerConfig(
                id="local-snowball-sft",
                mesh=mesh,
                use_explicit_mesh_axes=True,
                train_batch_size=max(8, jax.device_count()),
                per_device_parallelism=1,
                num_train_steps=1,
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=str(root)),
                checkpointer=CheckpointerConfig(base_path=str(root / "ckpts")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
                log_jaxprs=False,
                log_xla_hlo=False,
            )
            with set_mesh(trainer_cfg.device_mesh):
                model = SnowballLMHeadModel.init(Axis("vocab", 64), model_cfg, key=random.key(0))
                model_cfg.hf_checkpoint_converter().replaced(tokenizer=fast).save_pretrained(
                    model,
                    str(hf_path),
                    save_tokenizer=False,
                    save_reference_code=False,
                )
            import_hf(
                ImportHfConfig(
                    hf_checkpoint=str(hf_path),
                    output_path=str(root / "native-checkpoint"),
                    model=model_cfg,
                    use_hf_model_config=False,
                    tokenizer=str(hf_path),
                    dtype="bfloat16",
                    subpath="model",
                    emit_padded_tokenizer=True,
                )
            )
            with set_mesh(trainer_cfg.device_mesh):
                template = eqx.filter_eval_shape(
                    SnowballLMHeadModel.init, Axis("vocab", 64), model_cfg, key=random.key(0)
                )
                restored = load_checkpoint(
                    template,
                    str(root / "native-checkpoint"),
                    subpath="model",
                    mesh=trainer_cfg.device_mesh,
                )
                expert_spec = restored.transformer.blocks[0].mlp.expert_mlp.w_gate.sharding.spec
                assert expert_spec == P("expert", ("data", "context"), "model"), expert_spec
                optimizer = AdamConfig(learning_rate=5e-5).build(1)
                state = named_jit(
                    lambda model: TrainerState.init(optimizer, model, key=random.key(1)),
                    axis_resources=trainer_cfg.parameter_axis_mapping,
                    out_axis_resources=trainer_cfg.parameter_axis_mapping,
                )(restored)
                expert_shape = restored.transformer.blocks[0].mlp.expert_mlp.w_gate.shape
                moment_specs = [
                    leaf.sharding.spec
                    for leaf in jax.tree.leaves(state.opt_state)
                    if isinstance(leaf, jax.Array) and leaf.shape == expert_shape
                ]
                assert moment_specs and all(
                    "expert" in spec and ("data", "context") in spec for spec in moment_specs
                ), moment_specs
            with local_cpu_mesh():
                examples = [GrugLmExample.causal(jnp.arange(8, dtype=jnp.int32) + 2) for _ in range(16)]
            data = LmDataConfig(
                components={
                    "direct": DirectDatasetComponent(
                        datasets={"train": ListAsyncDataset(examples), "validation": ListAsyncDataset(examples)}
                    )
                },
                tokenizer=str(hf_path),
            )
            main(
                TrainLmConfig(
                    data=data,
                    model=model_cfg,
                    initialize_from_hf=str(hf_path),
                    optimizer=AdamConfig(learning_rate=5e-5),
                    trainer=trainer_cfg,
                )
            )
            with (root / "eval_results.json").open() as handle:
                metrics = json.load(handle)
            assert math.isfinite(metrics["train/loss"])
            assert math.isfinite(metrics["eval/loss"])
            native_trainer = replace(
                trainer_cfg,
                id="local-snowball-native-sft",
                tracker=JsonFileTrackerConfig(output_path=str(root / "native")),
                checkpointer=CheckpointerConfig(base_path=str(root / "native-ckpts")),
            )
            main(
                TrainLmConfig(
                    data=data,
                    model=model_cfg,
                    initialize_model_from_checkpoint_path=str(root / "native-checkpoint"),
                    optimizer=AdamConfig(learning_rate=5e-5),
                    trainer=native_trainer,
                )
            )
            with (root / "native" / "eval_results.json").open() as handle:
                native_metrics = json.load(handle)
            assert math.isfinite(native_metrics["train/loss"])
            assert math.isfinite(native_metrics["eval/loss"])


@pytest.mark.timeout(120)
@pytest.mark.multidevice
def test_hf_config_reload_keeps_fp32_router_settings(tmp_path):
    with use_test_devices(4):
        hf_path = tmp_path / "hf"
        tokenizer = Tokenizer(
            models.WordLevel(vocab={"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 64)}}, unk_token="<pad>")
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>")
        fast.save_pretrained(hf_path)
        config = _tiny_config(
            vocab_size=64,
            hidden_dim=32,
            intermediate_dim=32,
            shared_expert_intermediate_dim=32,
            num_layers=1,
            head_dim=8,
            max_seq_len=128,
            router_dot_fp32=True,
            tokenizer=str(hf_path),
        )
        # HF metadata chooses the platform kernel; test reload separately from
        # context sharding, which has its unchanged reference-backend test above.
        trainer = TrainerConfig(
            id="hf-config-override-regression",
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": 1, "context": 1, "expert": 2},
                compute_mapping={"batch": ["replica_dcn", "data", "expert"], "vocab": "model"},
            ),
            field_dtype_overrides={"router_bias": jnp.dtype("float32")},
            use_explicit_mesh_axes=True,
            train_batch_size=max(8, jax.device_count()),
            per_device_parallelism=1,
            num_train_steps=1,
            max_eval_batches=1,
            require_accelerator=False,
            tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
            checkpointer=CheckpointerConfig(base_path=str(tmp_path / "ckpts")),
            distributed=DistributedConfig(initialize_jax_distributed=False),
            log_jaxprs=False,
            log_xla_hlo=False,
        )
        with set_mesh(trainer.device_mesh):
            model = SnowballLMHeadModel.init(Axis("vocab", 64), config, key=random.key(0))
            config.hf_checkpoint_converter().replaced(tokenizer=fast).save_pretrained(
                model,
                str(hf_path),
                save_tokenizer=False,
                save_reference_code=False,
            )
        with local_cpu_mesh():
            examples = [GrugLmExample.causal(jnp.arange(128, dtype=jnp.int32) % 62 + 2) for _ in range(16)]
        main(
            TrainLmConfig(
                data=LmDataConfig(
                    tokenizer=str(hf_path),
                    components={
                        "direct": DirectDatasetComponent(
                            datasets={"train": ListAsyncDataset(examples), "validation": ListAsyncDataset(examples)},
                        )
                    },
                ),
                model=config,
                initialize_from_hf=str(hf_path),
                use_hf_model_config=True,
                trainer=trainer,
                optimizer=AdamConfig(learning_rate=5e-5),
            )
        )
        metrics = json.loads((tmp_path / "eval_results.json").read_text())
        assert math.isfinite(metrics["train/loss"])
        assert math.isfinite(metrics["eval/loss"])


def test_fp32_router_training_rejects_missing_fp32_bias_before_loading(tmp_path):
    config = TrainLmConfig(
        model=_tiny_config(router_dot_fp32=True),
        data=LmDataConfig(tokenizer=str(tmp_path)),
    )
    with pytest.raises(ValueError, match="field_dtype_overrides to map router_bias to float32"):
        main(config)


@pytest.mark.parametrize("separate_key_axis", [False, True])
def test_snowball_packed_conversations_are_isolated(separate_key_axis):
    cfg = _tiny_config()
    ids = _device_batched_ids(cfg.vocab_size, 8)
    segments = hax.named(jnp.broadcast_to(jnp.array([0, 0, 0, 0, 1, 1, 1, 1]), ids.array.shape), ids.axes)
    key_segments = segments.rename({"position": "key_position"}) if separate_key_axis else segments
    mask = AttentionMask.causal().with_segment_ids(segments, key_segments)
    changed = hax.named(ids.array.at[:, :4].set((ids.array[:, :4] + 13) % cfg.vocab_size), ids.axes)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(9))
        forward = eqx.filter_jit(lambda m, tokens: m.activations(tokens, mask).array)
        before = np.array(forward(model, ids))
        after = np.array(forward(model, changed))
        observe = eqx.filter_jit(lambda m, tokens: m.activations_with_routes(tokens, mask)[0].array)
        np.testing.assert_array_equal(observe(model, ids), before)
        np.testing.assert_array_equal(observe(model, changed), after)
    assert not np.allclose(before[:, :4], after[:, :4])
    np.testing.assert_allclose(before[:, 4:], after[:, 4:], atol=1e-5, rtol=1e-5)


def test_snowball_explicit_positions_match_masked_gaps():
    """Token positions survive compacting gaps; ignoring pos_ids changes short-layer attention."""
    cfg = _tiny_config(num_layers=2, sliding_window=16, moe_capacity_factor=8)
    compact = _device_batched_ids(cfg.vocab_size, 4)
    Batch = compact.axes[0]
    ExpandedPos = Axis("position", 8)
    positions = hax.named(jnp.broadcast_to(jnp.arange(0, 8, 2), compact.array.shape), compact.axes)
    expanded = hax.named(
        jnp.zeros((Batch.size, 8), dtype=jnp.int32).at[:, ::2].set(compact.array), (Batch, ExpandedPos)
    )
    segments = hax.named(jnp.broadcast_to(jnp.arange(8) % 2, expanded.array.shape), expanded.axes)
    mask = AttentionMask.causal().with_segment_ids(segments)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(15))
        forward = eqx.filter_jit(lambda m, ids, attn, pos: m.activations(ids, attn, pos_ids=pos).array)
        expected = np.asarray(forward(model, expanded, mask, None))[:, ::2]
        actual = np.asarray(forward(model, compact, None, positions))
        ignored = np.asarray(forward(model, compact, None, None))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert np.max(np.abs(actual - ignored)) > 1e-5


def test_snowball_fp32_attention_projection_and_gate_match_reference():
    """Match selective FP32 XSA and gate, leaving projection GEMMs in BF16."""
    cfg = _tiny_config(hidden_dim=8, num_heads=1, num_kv_heads=1, head_dim=8, head_gate_fp32=True, xsa_fp32=True)
    batch_size = jax.device_count()
    values = jnp.broadcast_to(
        jnp.array(
            [
                [
                    [0.25, -0.5, 1.25, -1.5, 0.75, 0.125, -0.875, 1.75],
                    [1.5, 0.375, -0.25, 0.75, -1.125, 2.25, 0.5, -0.625],
                ]
            ],
            dtype=jnp.bfloat16,
        ),
        (batch_size, 2, 8),
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        attention = SnowballAttention.init(cfg, key=jax.random.key(3))
        attention = eqx.tree_at(
            lambda m: (m.w_q, m.w_k, m.w_v, m.w_o, m.attn_gate),
            attention,
            (
                jnp.zeros((8, 8), dtype=jnp.bfloat16),
                jnp.zeros((8, 8), dtype=jnp.bfloat16),
                jnp.eye(8, dtype=jnp.bfloat16),
                jnp.eye(8, dtype=jnp.bfloat16),
                jnp.arange(8, dtype=jnp.bfloat16)[:, None] / 13,
            ),
        )
        actual = np.asarray(
            eqx.filter_jit(lambda m, x: m(x, GrugAttentionMask.causal(), disable_rope=True))(attention, values),
            dtype=np.float32,
        )
    raw = np.asarray(values, dtype=np.float32)
    # Zero Q/K make the causal softmax exactly 1 and 1/2; the oracle needs no
    # shared attention, normalization, position or softmax implementation.
    context = np.stack([raw[:, 0], (raw[:, 0] + raw[:, 1]) / 2], axis=1)
    dot = (context * raw).sum(axis=-1, keepdims=True)
    denominator = (raw * raw).sum(axis=-1, keepdims=True) + 1e-6
    projected = np.asarray(jnp.asarray(context - dot / denominator * raw, dtype=jnp.bfloat16), dtype=np.float32)
    logits = np.asarray(
        jnp.asarray(raw @ np.asarray(attention.attn_gate, dtype=np.float32), dtype=jnp.bfloat16), dtype=np.float32
    )
    gate = np.asarray(jnp.asarray(2 / (1 + np.exp(-logits)), dtype=jnp.bfloat16), dtype=np.float32)
    expected = np.asarray(jnp.asarray(projected * gate, dtype=jnp.bfloat16), dtype=np.float32)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.timeout(180)
@pytest.mark.multidevice
@pytest.mark.parametrize(
    "implementation,sequence_length,head_dim",
    [
        ("reference", 48, 12),
        # 48 forces three blocks of 16, so xla_flash cannot fall back to reference attention.
        ("xla_flash", 48, 12),
        pytest.param("tpu_splash", 512, 64, marks=pytest.mark.tpu),
        ("gpu_fa4_cute", 1024, 64),
    ],
)
def test_snowball_packed_context_parallel_hidden_states_and_loss_are_isolated(
    implementation, sequence_length, head_dim
):
    if implementation == "tpu_splash" and jax.default_backend() != "tpu":
        pytest.skip("Splash requires TPU hardware")
    if implementation == "gpu_fa4_cute" and jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe requires GPU hardware")
    with use_test_devices(4) as devices:
        cfg = _tiny_config(
            num_layers=2,  # One sliding-window layer and one full-causal layer.
            head_dim=head_dim,
            max_seq_len=sequence_length,
            sliding_window=sequence_length // 2,
            attention_implementation=implementation,
            moe_implementation="ring",
            # Eliminate capacity drops, which couple tokens independently of attention.
            moe_capacity_factor=4,
        )
        Batch, Pos = Axis("batch", 4), Axis("position", sequence_length)
        ids = hax.named(jnp.arange(Batch.size * Pos.size).reshape(Batch.size, Pos.size) % cfg.vocab_size, (Batch, Pos))
        # Different boundaries per row, away from CP shard and kernel tile boundaries.
        boundaries = jnp.array([sequence_length // 3 + i for i in range(Batch.size)])[:, None]
        later = jnp.arange(Pos.size)[None, :] >= boundaries
        segments = hax.named(later.astype(jnp.int32), ids.axes)
        packed = AttentionMask.causal().with_segment_ids(segments, segments.rename({"position": "key_position"}))
        single = AttentionMask.causal().with_segment_ids(hax.zeros(ids.axes, dtype=jnp.int32))
        changed = hax.named(jnp.where(later, ids.array, (ids.array + 13) % cfg.vocab_size), ids.axes)
        # Score only next-token predictions wholly inside the second document.
        weights = hax.named((later & (jnp.arange(Pos.size)[None, :] < Pos.size - 1)).astype(jnp.float32), ids.axes)
        for context in (1, 4):
            mesh = Mesh(
                np.asarray(devices[:context]).reshape(1, 1, 1, context, 1),
                ("replica_dcn", "data", "expert", "context", "model"),
                axis_types=(AxisType.Explicit,) * 5,
            )
            with jax.set_mesh(mesh):
                model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(9))
                if implementation in ("tpu_splash", "gpu_fa4_cute"):
                    model = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16").cast_to_compute(model)

                # BF16 MoE scatters can vary between identical GPU runs. Compile deterministically
                # so their rounding noise cannot masquerade as a cross-document dependency.
                @eqx.filter_jit(compiler_options={"xla_gpu_exclude_nondeterministic_ops": True})
                def forward(model, tokens, mask):
                    hidden = model.activations(tokens, mask).array
                    loss = model.compute_next_token_loss(LmExample(tokens, weights, mask), reduction=None)
                    return hidden, loss.array

                before = tuple(np.asarray(x) for x in forward(model, ids, packed))
                repeated = forward(model, ids, packed)
                for expected, actual in zip(before, repeated, strict=True):
                    np.testing.assert_array_equal(expected, actual, err_msg="Identical inputs must be deterministic")
                after = tuple(np.asarray(x) for x in forward(model, changed, packed))
                assert not np.allclose(before[0][~np.asarray(later)], after[0][~np.asarray(later)])
                np.testing.assert_array_equal(before[0][np.asarray(later)], after[0][np.asarray(later)])
                np.testing.assert_array_equal(before[1], after[1])
                assert np.all(before[1][np.asarray(weights.array, dtype=bool)] > 0)

                # A single document must preserve the pre-packing causal path bit for bit.
                unpacked = forward(model, ids, None)
                one_document = forward(model, ids, single)
                for expected, actual in zip(unpacked, one_document, strict=True):
                    np.testing.assert_array_equal(expected, actual)


@pytest.mark.timeout(120)
@pytest.mark.multidevice
def test_snowball_capacity_preserves_imbalanced_outputs_and_gradients():
    with use_test_devices(2):

        def run(expert_axis, capacity):
            with jax.set_mesh(compact_grug_mesh(expert_axis_size=expert_axis)):
                cfg = SnowballConfig(
                    hidden_dim=8,
                    intermediate_dim=16,
                    num_experts=4,
                    num_experts_per_token=2,
                    initializer_std=0.2,
                    moe_implementation="ring",
                    moe_capacity_factor=capacity,
                )
                model = SnowballMoEMLP.init(cfg, key=jax.random.key(19))
                # All tokens choose experts on the same destination shard.
                model = eqx.tree_at(lambda m: m.router_bias, model, jnp.array([10.0, 9.0, -10.0, -10.0]))
                inputs = jnp.arange(256, dtype=jnp.float32).reshape(4, 8, 8) / 256
                outputs = eqx.filter_jit(lambda m: m(inputs))(model)
                _, grad = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(inputs) ** 2)))(model)
                return [
                    np.asarray(outputs),
                    np.asarray(grad.router),
                    np.asarray(grad.expert_mlp.w_gate),
                    np.asarray(grad.expert_mlp.w_up),
                    np.asarray(grad.expert_mlp.w_down),
                ]

        reference = run(1, 2)
        preserved = run(2, 2)
        dropped = run(2, 1)
        for actual, expected in zip(preserved, reference, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert not np.allclose(dropped[0], reference[0], rtol=1e-5, atol=1e-5)
        assert not np.allclose(dropped[2], reference[2], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sliding_window", [2, 4])
@jax.default_matmul_precision("highest")
def test_snowball_paged_decode_matches_full_forward(sliding_window):
    """Mixed chunked prefill/decode preserves per-request positions and June attention."""
    cfg = _tiny_config(sliding_window=sliding_window)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(17))
        # Exercise the learned head gates, whose initializer is identically zero.
        blocks = tuple(
            eqx.tree_at(
                lambda b: b.attn.attn_gate, block, jax.random.normal(jax.random.key(i), block.attn.attn_gate.shape)
            )
            for i, block in enumerate(model.transformer.blocks)
        )
        model = eqx.tree_at(lambda m: m.transformer.blocks, model, blocks)
        assert_mixed_paged_decode_matches_forward(model)


def test_snowball_cached_decode_routes_reassemble_for_teacher_forced_replay():
    """Capture routes in the actual decode MLP and bind them to the same forward layout."""
    cfg = _tiny_config(num_layers=2, moe_implementation="scatter", max_seq_len=16)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(41))
        cache = model.initial_cache(PageTableSpec(num_pages=8, page_size=2, max_seqs=1), dtype=jnp.float32)
        prefill_ids = hax.named(jnp.asarray([2, 7, 0, 9], dtype=jnp.int32), "position")
        prefill_positions = hax.named(jnp.arange(4, dtype=jnp.int32), "position")
        prefill_info = PageBatchInfo(
            slot_ids=hax.named(jnp.asarray([0], dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray([[0, 1]], dtype=jnp.int32), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray([4], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.asarray([0, 4], dtype=jnp.int32), "seq"),
            num_seqs=jnp.asarray(1, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.asarray([0, 1, 2, 3], dtype=jnp.int32), "position"),
            page_size=2,
        )
        legacy_cache = model.initial_cache(PageTableSpec(num_pages=8, page_size=2, max_seqs=1), dtype=jnp.float32)
        legacy_logits, legacy_cache = model.decode(prefill_ids, legacy_cache, prefill_info, prefill_positions)
        prefill_logits, cache, prefill_capture = model.decode_with_route_captures(
            prefill_ids, cache, prefill_info, prefill_positions
        )
        legacy_prefill_pages = legacy_cache.kv_pages.array
        capture_prefill_pages = cache.kv_pages.array
        assert isinstance(prefill_capture, SnowballDecodeRouteCaptureBatch)
        assert len(prefill_capture.route_captures) == cfg.num_layers
        assert all(c.selected_experts.shape == (4, cfg.num_experts_per_token) for c in prefill_capture.route_captures)
        teacher_ids = hax.named(prefill_ids.array[None, :], ("batch", "position"))
        teacher_positions = jnp.arange(4, dtype=jnp.int32)[None, :]
        teacher_valid = jnp.ones((4,), dtype=jnp.bool_)
        replay_capture = prefill_capture.as_teacher_forced(
            teacher_ids,
            teacher_positions,
            teacher_valid,
            batch_info=prefill_info,
        )
        replayed = model(teacher_ids, route_captures=replay_capture)
        changed_block = eqx.tree_at(
            lambda block: block.mlp.router,
            model.transformer.blocks[0],
            model.transformer.blocks[0].mlp.router.at[:, 0].add(0.01),
        )
        changed = eqx.tree_at(lambda current: current.transformer.blocks[0], model, changed_block)
        changed_replay = changed(teacher_ids, route_captures=replay_capture)
        changed_grad = eqx.filter_grad(
            lambda current: jnp.sum(current(teacher_ids, route_captures=replay_capture).array)
        )(changed)

        continuation_ids = hax.named(jnp.asarray([11], dtype=jnp.int32), "position")
        continuation_positions = hax.named(jnp.asarray([4], dtype=jnp.int32), "position")
        continuation_info = PageBatchInfo(
            slot_ids=hax.named(jnp.asarray([0], dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray([[0, 1, 2]], dtype=jnp.int32), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray([5], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.asarray([0, 1], dtype=jnp.int32), "seq"),
            num_seqs=jnp.asarray(1, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.asarray([4], dtype=jnp.int32), "position"),
            page_size=2,
        )
        legacy_continuation_logits, legacy_cache = model.decode(
            continuation_ids, legacy_cache, continuation_info, continuation_positions
        )
        continuation_logits, cache, continuation_capture = model.decode_with_route_captures(
            continuation_ids, cache, continuation_info, continuation_positions
        )
        legacy_continuation_pages = legacy_cache.kv_pages.array
        capture_continuation_pages = cache.kv_pages.array
        padded_ids = hax.named(jnp.asarray([12, 0], dtype=jnp.int32), "position")
        padded_positions = hax.named(jnp.asarray([5, -1], dtype=jnp.int32), "position")
        padded_info = PageBatchInfo(
            slot_ids=hax.named(jnp.asarray([0], dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray([[0, 1, 2]], dtype=jnp.int32), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray([6], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.asarray([0, 1], dtype=jnp.int32), "seq"),
            num_seqs=jnp.asarray(1, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.asarray([5, -1], dtype=jnp.int32), "position"),
            page_size=2,
        )
        legacy_padded_logits, legacy_cache = model.decode(padded_ids, legacy_cache, padded_info, padded_positions)
        padded_logits, cache, padded_capture = model.decode_with_route_captures(
            padded_ids, cache, padded_info, padded_positions
        )
        legacy_padded_pages = legacy_cache.kv_pages.array
        capture_padded_pages = cache.kv_pages.array
        full_capture = SnowballDecodeRouteCaptureBatch.concatenate(
            prefill_capture, continuation_capture, padded_capture
        )
        full_teacher_ids = hax.named(
            jnp.concatenate((prefill_ids.array, continuation_ids.array, padded_ids.array[:1]))[None, :],
            ("batch", "position"),
        )
        full_teacher_positions = jnp.arange(6, dtype=jnp.int32)[None, :]
        full_replay_capture = full_capture.as_teacher_forced(
            full_teacher_ids,
            full_teacher_positions,
            jnp.ones((6,), dtype=jnp.bool_),
            batch_info=padded_info,
        )
        full_replayed = model(full_teacher_ids, route_captures=full_replay_capture)
        full_ordinary = model(full_teacher_ids)

    np.testing.assert_array_equal(np.asarray(prefill_logits.array), np.asarray(legacy_logits.array))
    np.testing.assert_array_equal(capture_prefill_pages, legacy_prefill_pages)
    np.testing.assert_array_equal(np.asarray(legacy_continuation_logits.array), np.asarray(continuation_logits.array))
    np.testing.assert_array_equal(capture_continuation_pages, legacy_continuation_pages)
    np.testing.assert_array_equal(np.asarray(legacy_padded_logits.array), np.asarray(padded_logits.array))
    np.testing.assert_array_equal(capture_padded_pages, legacy_padded_pages)
    np.testing.assert_allclose(np.asarray(prefill_logits.array), np.asarray(replayed.array[0]), rtol=1e-4, atol=1e-4)
    assert not np.array_equal(np.asarray(changed_replay.array), np.asarray(replayed.array))
    assert float(jnp.max(jnp.abs(changed_grad.transformer.blocks[0].mlp.router))) > 0.0
    assert bool(jnp.all(prefill_capture.token_valid))
    assert bool(jnp.all(continuation_capture.token_valid))
    assert bool(jnp.array_equal(padded_capture.token_valid, jnp.asarray([True, False])))
    np.testing.assert_allclose(
        np.asarray(full_replayed.array[0]),
        np.asarray(full_ordinary.array[0]),
        rtol=1e-4,
        atol=1e-4,
    )
    with pytest.raises(ValueError, match="position IDs"):
        prefill_capture.as_teacher_forced(
            teacher_ids,
            teacher_positions.at[0, 3].set(8),
            teacher_valid,
            batch_info=prefill_info,
        )
    with pytest.raises(ValueError, match="token IDs"):
        prefill_capture.as_teacher_forced(
            hax.named(teacher_ids.array.at[0, 1].set(13), teacher_ids.axes),
            teacher_positions,
            teacher_valid,
            batch_info=prefill_info,
        )
    with pytest.raises(ValueError, match="contiguous"):
        continuation_capture.as_teacher_forced(
            continuation_ids,
            continuation_positions.array,
            jnp.ones((1,), dtype=jnp.bool_),
            batch_info=continuation_info,
        )
    with pytest.raises(ValueError, match="serving segment"):
        prefill_capture.as_teacher_forced(
            teacher_ids,
            teacher_positions,
            teacher_valid,
            segment_ids=(teacher_ids.array, teacher_ids.array),
            batch_info=prefill_info,
        )
    bad_metadata = [
        (dataclasses.replace(continuation_info, cu_q_lens=hax.named(jnp.asarray([0, 0]), "seq")), "cu_q_lens"),
        (
            dataclasses.replace(
                continuation_info, new_token_dests=hax.named(jnp.asarray([3], dtype=jnp.int32), "position")
            ),
            "destinations",
        ),
        (
            dataclasses.replace(
                continuation_info, page_indices=hax.named(jnp.asarray([[0, 1, 3]], dtype=jnp.int32), ("seq", "page"))
            ),
            "destinations",
        ),
        (
            dataclasses.replace(continuation_info, seq_lens=hax.named(jnp.asarray([6]), "seq")),
            "positions",
        ),
        (dataclasses.replace(continuation_info, num_seqs=jnp.asarray(2, dtype=jnp.int32)), "exactly one"),
    ]
    for altered_info, message in bad_metadata:
        altered_capture = continuation_capture._replace(batch_info=altered_info)
        with pytest.raises(ValueError, match=message):
            SnowballDecodeRouteCaptureBatch.concatenate(prefill_capture, altered_capture)
    altered_positions = continuation_capture._replace(position_ids=jnp.asarray([5], dtype=jnp.int32))
    with pytest.raises(ValueError, match="positions"):
        SnowballDecodeRouteCaptureBatch.concatenate(prefill_capture, altered_positions)
    altered_validity = continuation_capture._replace(token_valid=jnp.asarray([False]))
    with pytest.raises(ValueError, match="cu_q_lens"):
        SnowballDecodeRouteCaptureBatch.concatenate(prefill_capture, altered_validity)
    altered_info = dataclasses.replace(prefill_info, page_size=4)
    with pytest.raises(ValueError, match="page size"):
        prefill_capture.as_teacher_forced(teacher_ids, teacher_positions, teacher_valid, batch_info=altered_info)


@pytest.mark.timeout(180)
def test_snowball_paged_decode_sharding_and_expert_skew_match_unsharded():
    run_on_cpu_devices(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import haliax as hax
        from haliax import Axis
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.inference.page_table import PageBatchInfo, PageTableSpec
        from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

        cfg = SnowballConfig(
            vocab_size=32, hidden_dim=16, intermediate_dim=16,
            shared_expert_intermediate_dim=16, num_experts=4,
            num_experts_per_token=1, num_layers=2, num_heads=2,
            num_kv_heads=2, head_dim=8, max_seq_len=8, sliding_window=2,
            initializer_std=0.2,
            attention_implementation="reference", moe_implementation="ring",
        )
        ids = hax.named(jnp.array([1, 2, 3, 4, 5, 6, 0, 0], dtype=jnp.int32), "position")
        positions = hax.named(jnp.array([0, 1, 2, 0, 1, 2, -1, -1], dtype=jnp.int32), "position")
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.array([0, 1], dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.array([[0, 1], [2, 3]], dtype=jnp.int32), ("seq", "page")),
            seq_lens=hax.named(jnp.array([3, 3], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.array([0, 3, 6], dtype=jnp.int32), "seq"),
            num_seqs=jnp.array(2, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.array([0, 1, 2, 4, 5, 6, -1, -1], dtype=jnp.int32), "position"),
            page_size=2,
        )
        results = []
        for expert_size, model_size in ((1, 1), (2, 1), (1, 2)):
            with jax.set_mesh(compact_grug_mesh(expert_axis_size=expert_size, model_axis_size=model_size)):
                model = SnowballLMHeadModel.init(Axis("vocab", 32), cfg, key=jax.random.key(0))
                # Every token selects expert 0. Balanced training capacity drops half the
                # assignments on two expert ranks; serving must retain all six real tokens.
                blocks = tuple(eqx.tree_at(
                    lambda b: b.mlp.router_bias, block,
                    jnp.array([100., 0., 0., 0.]),
                ) for block in model.transformer.blocks)
                model = eqx.tree_at(lambda m: m.transformer.blocks, model, blocks)
                cache = model.initial_cache(PageTableSpec(4, 2, max_seqs=2), dtype=jnp.float32)
                logits, _ = eqx.filter_jit(lambda m, c: m.decode(ids, c, info, positions))(model, cache)
                results.append(np.asarray(logits.array)[:6])
        for actual in results[1:]:
            np.testing.assert_allclose(results[0], actual, rtol=1e-4, atol=1e-4)
        """,
        device_count=2,
    )


@pytest.mark.parametrize("model_size", [1, 2], ids=["model1", "model2"])
@pytest.mark.parametrize("update_dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_kv_cache_explicit_update_preserves_sharding_and_slots(model_size, update_dtype):
    if jax.device_count() != 4:
        pytest.skip("requires exactly four devices")
    mesh = compact_grug_mesh(model_axis_size=model_size)
    cache_sharding = NamedSharding(mesh, P(None, None, "model", None))
    update_sharding = NamedSharding(mesh, P(None, "model", None))
    heads = model_size
    with jax.set_mesh(mesh):
        cache = KvPageCache.init(PageTableSpec(4, 8, max_seqs=4), Axis("kv_head", heads), Axis("head_size", 128))
        initial = np.full((4, 8, 2 * heads, 128), -7, dtype=np.float32)
        cache = KvPageCache(hax.named(jax.device_put(initial, cache_sharding), cache.kv_pages.axes))
        # The fifth token is padding: its valid destination must remain untouched.
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.arange(4, dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.arange(4, dtype=jnp.int32)[:, None], ("seq", "page")),
            seq_lens=hax.named(jnp.asarray([1, 4, 8, 3], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.arange(5, dtype=jnp.int32), "seq"),
            num_seqs=jnp.asarray(4, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.asarray([0, 11, 23, 26, 31], dtype=jnp.int32), "position"),
            page_size=8,
        )
        # Integer values are exactly representable in both update dtypes and the FP32 cache.
        keys = (np.arange(5 * heads * 128).reshape(5, heads, 128) % 31).astype(np.float32)
        values = -keys - 1
        axes = (Axis("position", 5), Axis("kv_head", heads), Axis("head_size", 128))
        new_k = hax.named(jax.device_put(jnp.asarray(keys, dtype=update_dtype), update_sharding), axes)
        new_v = hax.named(jax.device_put(jnp.asarray(values, dtype=update_dtype), update_sharding), axes)
        updated = eqx.filter_jit(lambda state, batch, k, v: state.update(batch, k, v))(cache, info, new_k, new_v)
    expected = initial.copy()
    for token, (page, slot) in enumerate(((0, 0), (1, 3), (2, 7), (3, 2))):
        expected[page, slot, 0::2] = keys[token]
        expected[page, slot, 1::2] = values[token]
    assert updated.kv_pages.array.shape == initial.shape
    assert updated.kv_pages.array.dtype == jnp.float32
    assert updated.kv_pages.array.sharding == cache_sharding
    np.testing.assert_array_equal(np.asarray(updated.kv_pages.array), expected)


def test_snowball_four_device_decode_preserves_explicit_cache_sharding():
    if jax.device_count() != 4:
        pytest.skip("requires exactly four devices")
    mesh = compact_grug_mesh()
    cache_sharding = NamedSharding(mesh, P(None, None, "model", None))
    with jax.set_mesh(mesh):
        cfg = _tiny_config(
            hidden_dim=16,
            num_heads=2,
            num_kv_heads=1,
            head_dim=128,
            projection_implementation="native",
            rms_norm_implementation="reference",
        )
        weight_sharding = NamedSharding(mesh, P(None, None))
        attention = SnowballAttention(
            w_q=jax.device_put(np.full((16, 256), 0.01, dtype=np.float32), weight_sharding),
            w_k=jax.device_put(np.full((16, 128), 0.02, dtype=np.float32), weight_sharding),
            w_v=jax.device_put(np.arange(16 * 128, dtype=np.float32).reshape(16, 128) / 2048, weight_sharding),
            w_o=jax.device_put(np.full((256, 16), 0.01, dtype=np.float32), weight_sharding),
            attn_gate=jax.device_put(np.zeros((16, 2), dtype=np.float32), weight_sharding),
            cfg=cfg,
        )
        cache = KvPageCache.init(PageTableSpec(4, 8, max_seqs=4), Axis("kv_head", 1), Axis("head_size", 128))
        cache = KvPageCache(hax.named(jax.device_put(cache.kv_pages.array, cache_sharding), cache.kv_pages.axes))
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.arange(4, dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.arange(4, dtype=jnp.int32)[:, None], ("seq", "page")),
            seq_lens=hax.named(jnp.ones(4, dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.arange(5, dtype=jnp.int32), "seq"),
            num_seqs=jnp.asarray(4, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.asarray([0, 8, 16, 24], dtype=jnp.int32), "position"),
            page_size=8,
        )
        x = jax.device_put(
            jnp.arange(64, dtype=jnp.float32).reshape(4, 1, 16) / 64,
            NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None)),
        )
        positions = jnp.zeros(4, dtype=jnp.int32)
        _, keys, values = attention._project_qkv(x, disable_rope=False, position_ids=positions[:, None])
        output, updated = eqx.filter_jit(
            lambda attn, state, value: attn.decode(value, state, info, positions, use_long=False)
        )(attention, cache, x)
    expected = np.zeros((4, 8, 2, 128), dtype=np.float32)
    expected[:, 0, 0] = np.asarray(keys)[:, 0, 0]
    expected[:, 0, 1] = np.asarray(values)[:, 0, 0]
    assert updated.kv_pages.array.sharding == cache_sharding
    assert updated.kv_pages.array.shape == expected.shape
    assert updated.kv_pages.array.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(updated.kv_pages.array), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(updated.kv_pages.array)[:, 1:], expected[:, 1:])
    assert output.shape == x.shape
    assert bool(jnp.isfinite(output).all())


@pytest.mark.timeout(120)
def test_snowball_fixed_attention_project_output_matches_caller_layout_on_explicit_mesh(monkeypatch):
    """The fixed output contraction can feed the caller residual on an explicit mesh."""

    monkeypatch.delenv("JAX_NUM_CPU_DEVICES", raising=False)
    run_on_cpu_devices(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from levanter.models import snowball as snowball_module
        from levanter.models.snowball import SnowballAttention, SnowballConfig

        def reference_linear(x, weight):
            # This is only the terminal contraction substitute on CPU. It leaves
            # the input layout unchanged and does not prescribe an output layout.
            return jax.lax.dot_general(
                x,
                weight,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

        snowball_module.fixed_schedule_linear = reference_linear

        mesh = Mesh(
            np.asarray(jax.devices()).reshape((1, 4, 1, 1, 1)),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        gate_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None))
        attention_sharding = NamedSharding(mesh, P(("replica_dcn", "data"), "expert", None, None))
        config = SnowballConfig(
            hidden_dim=128,
            num_heads=2,
            num_kv_heads=1,
            head_dim=128,
            max_seq_len=256,
            attention_implementation="reference",
            projection_implementation="fixed_schedule",
        )
        x_host = jnp.linspace(-0.5, 0.5, 4 * 256 * 128, dtype=jnp.float32).reshape(4, 256, 128)
        attention_output_host = jnp.linspace(-0.25, 0.25, 4 * 256 * 2 * 128, dtype=jnp.float32).reshape(
            4, 256, 2, 128
        )
        value_host = jnp.linspace(-0.2, 0.2, 4 * 256 * 1 * 128, dtype=jnp.float32).reshape(4, 256, 1, 128)
        with jax.set_mesh(mesh):
            attention = SnowballAttention.init(config, key=jax.random.key(19))
            attention = eqx.tree_at(
                lambda module: module.attn_gate,
                attention,
                jax.device_put(jnp.full((128, 2), 0.01, dtype=jnp.float32), NamedSharding(mesh, P(None, None))),
            )
            x = jax.device_put(x_host, gate_sharding)
            attention_output = jax.device_put(attention_output_host, attention_sharding)
            value = jax.device_put(value_host, attention_sharding)

            project = eqx.filter_jit(
                lambda value_x: attention._project_output(value_x, attention_output, value)
            )
            actual = project(x)
            residual = eqx.filter_jit(
                lambda value_x: value_x + attention._project_output(value_x, attention_output, value)
            )(x)

            aligned_value = np.broadcast_to(np.asarray(value_host), (4, 256, 2, 128))
            attention_output_np = np.asarray(attention_output_host)
            dot = np.sum(attention_output_np * aligned_value, axis=-1, keepdims=True)
            norm = np.sum(aligned_value * aligned_value, axis=-1, keepdims=True)
            xsa = attention_output_np - dot / (norm + 1e-6) * aligned_value
            gate = 2 / (1 + np.exp(-np.asarray(x_host) @ np.full((128, 2), 0.01, dtype=np.float32)))
            expected = np.einsum(
                "bsh,hd->bsd",
                (gate[..., None] * xsa).reshape(4, 256, 256),
                np.asarray(attention.w_o),
            )
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.asarray(residual), np.asarray(x_host) + expected, rtol=1e-5, atol=1e-5)
            assert actual.shape == (4, 256, 128)
            assert actual.sharding.spec == x.sharding.spec
            assert residual.sharding.spec == x.sharding.spec
            assert bool(jnp.isfinite(residual).all())

            _, input_gradient = eqx.filter_jit(
                jax.value_and_grad(lambda value_x: jnp.mean(project(value_x)))
            )(x)
            assert input_gradient.sharding.spec == x.sharding.spec
            assert bool(jnp.isfinite(input_gradient).all())
            assert float(jnp.max(jnp.abs(input_gradient))) > 0
        """,
        device_count=4,
    )
