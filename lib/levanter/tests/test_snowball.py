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
from levanter.testing.cpu_devices import run_on_cpu_devices


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
def test_snowball_context_parallel_values_and_gradients_match_data_parallel():
    run_on_cpu_devices(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import haliax as hax
        from haliax import Axis
        from jax.sharding import AxisType, Mesh
        from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

        cfg = SnowballConfig(
            vocab_size=32, hidden_dim=16, intermediate_dim=16,
            shared_expert_intermediate_dim=16, num_experts=4,
            num_experts_per_token=2, num_layers=2, num_heads=2,
            num_kv_heads=1, head_dim=8, max_seq_len=8, sliding_window=4,
            attention_implementation="reference", moe_implementation="ring",
        )
        ids = hax.named(
            jnp.arange(32, dtype=jnp.int32).reshape(4, 8),
            (Axis("batch", 4), Axis("position", 8)),
        )
        axes = ("replica_dcn", "data", "context", "expert", "model")

        def run(shape):
            mesh = Mesh(
                np.asarray(jax.devices()).reshape(shape), axes,
                axis_types=(AxisType.Explicit,) * len(axes),
            )
            with jax.set_mesh(mesh):
                model = SnowballLMHeadModel.init(Axis("vocab", 32), cfg, key=jax.random.key(0))
                activations = eqx.filter_jit(lambda m: m.activations(ids).array)(model)
                value, grad = eqx.filter_jit(
                    eqx.filter_value_and_grad(lambda m: jnp.mean(m(ids).array))
                )(model)
            block = grad.transformer.blocks[0]
            return activations.sharding.spec[1], (
                np.asarray(value), np.asarray(grad.transformer.token_embed),
                np.asarray(block.attn.w_q), np.asarray(block.mlp.expert_mlp.w_gate),
                np.asarray(block.shared.w_gate), np.asarray(grad.transformer.output_proj),
            )

        data_axis, reference = run((1, 2, 1, 2, 1))
        context_axis, sharded = run((1, 1, 2, 2, 1))
        assert data_axis is None
        assert context_axis == "context"
        for expected, actual in zip(reference, sharded, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        """,
        device_count=4,
    )


@pytest.mark.timeout(120)
def test_snowball_hf_init_trains_with_context_and_expert_parallelism():
    run_on_cpu_devices(
        """
        import json
        import math
        import tempfile
        from dataclasses import replace
        from pathlib import Path

        import jax.numpy as jnp
        import jax.random as random
        import jax
        from jax import P
        import equinox as eqx
        from haliax import Axis, named_jit
        from haliax.partitioning import set_mesh
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        from levanter.checkpoint import CheckpointerConfig, load_checkpoint
        from levanter.data.dataset import ListAsyncDataset
        from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
        from levanter.data.text.examples import GrugLmExample
        from levanter.distributed import DistributedConfig
        from levanter.main.train_lm import TrainLmConfig, main
        from levanter.main.export_hf_to_lm import ImportHfConfig, main as import_hf
        from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
        from levanter.optim.config import AdamConfig
        from levanter.tracker.json_file import JsonFileTrackerConfig
        from levanter.trainer import TrainerConfig
        from levanter.trainer_state import TrainerState
        from levanter.utils.mesh import MeshConfig

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            hf_path = root / "hf"
            hf_path.mkdir()
            tokenizer = Tokenizer(models.WordLevel(
                vocab={"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 64)}},
                unk_token="<pad>",
            ))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>")
            fast.save_pretrained(hf_path)
            model_cfg = SnowballConfig(
                vocab_size=64, hidden_dim=32, intermediate_dim=32, shared_expert_intermediate_dim=32,
                num_experts=8, num_experts_per_token=2, num_layers=1, num_heads=4, num_kv_heads=2,
                head_dim=8, max_seq_len=8, sliding_window=4, attention_implementation="reference",
                tokenizer=str(hf_path),
            )
            mesh = MeshConfig(
                axes={"data": 1, "replica": 1, "model": 1, "context": 2, "expert": -1},
                compute_mapping={
                    "batch": ["replica_dcn", "data", "expert"],
                    "position": "context",
                    "vocab": "model",
                },
            )
            trainer_cfg = TrainerConfig(
                id="local-snowball-sft", mesh=mesh, use_explicit_mesh_axes=True,
                train_batch_size=8, num_train_steps=1, max_eval_batches=0,
                tracker=JsonFileTrackerConfig(output_path=str(root)),
                checkpointer=CheckpointerConfig(base_path=str(root / "ckpts")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
                log_jaxprs=False, log_xla_hlo=False,
            )
            with set_mesh(trainer_cfg.device_mesh):
                model = SnowballLMHeadModel.init(Axis("vocab", 64), model_cfg, key=random.key(0))
                model_cfg.hf_checkpoint_converter().replaced(tokenizer=fast).save_pretrained(
                    model, str(hf_path), save_tokenizer=False, save_reference_code=False,
                )
            import_hf(ImportHfConfig(
                hf_checkpoint=str(hf_path), output_path=str(root / "native-checkpoint"),
                model=model_cfg, use_hf_model_config=False, tokenizer=str(hf_path),
                dtype="bfloat16", subpath="model", emit_padded_tokenizer=True,
            ))
            with set_mesh(trainer_cfg.device_mesh):
                template = eqx.filter_eval_shape(SnowballLMHeadModel.init, Axis("vocab", 64), model_cfg, key=random.key(0))
                restored = load_checkpoint(
                    template, str(root / "native-checkpoint"), subpath="model", mesh=trainer_cfg.device_mesh,
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
                    leaf.sharding.spec for leaf in jax.tree.leaves(state.opt_state)
                    if isinstance(leaf, jax.Array) and leaf.shape == expert_shape
                ]
                assert moment_specs and all(
                    "expert" in spec and ("data", "context") in spec for spec in moment_specs
                ), moment_specs
            examples = [GrugLmExample.causal(jnp.arange(8, dtype=jnp.int32) + 2) for _ in range(16)]
            data = LmDataConfig(
                components={"direct": DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})},
                tokenizer=str(hf_path),
            )
            main(TrainLmConfig(
                data=data, model=model_cfg, initialize_from_hf=str(hf_path),
                optimizer=AdamConfig(learning_rate=5e-5), trainer=trainer_cfg,
            ))
            with (root / "eval_results.json").open() as handle:
                metrics = json.load(handle)
            assert math.isfinite(metrics["train/loss"])
            native_trainer = replace(
                trainer_cfg, id="local-snowball-native-sft",
                tracker=JsonFileTrackerConfig(output_path=str(root / "native")),
                checkpointer=CheckpointerConfig(base_path=str(root / "native-ckpts")),
            )
            main(TrainLmConfig(
                data=data, model=model_cfg, initialize_model_from_checkpoint_path=str(root / "native-checkpoint"),
                optimizer=AdamConfig(learning_rate=5e-5), trainer=native_trainer,
            ))
            with (root / "native" / "eval_results.json").open() as handle:
                native_metrics = json.load(handle)
            assert math.isfinite(native_metrics["train/loss"])
        """,
        device_count=8,
    )
