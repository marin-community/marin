# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for grug variants under experiments/grug/*.

These checks are intentionally variant-discovered: if a subdirectory contains
`model.py` and/or `train.py`, it is expected to satisfy the corresponding
lowering and training contracts.
"""

import dataclasses
import importlib
import json
import logging
import os
import subprocess
import sys
import textwrap
import uuid
from io import StringIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.grug.attention._fa4_cute as fa4_cute
import numpy as np
import optax
import pytest
from fray.cluster import ResourceConfig
from jax._src import config as jax_config
from jax.sharding import use_abstract_mesh
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DatasetComponent, DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import _compact_grug_mesh_shape
from levanter.schedule import BatchSchedule
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.ferries import canary_ferry
from experiments.llama import llama3_tokenizer

_TOKENIZED_CACHE = f"{TokenizedCache.__module__}.{TokenizedCache.__qualname__}"


def _discover_grug_variants_with_file(filename: str) -> list[str]:
    grug_dir = Path(__file__).resolve().parents[1] / "experiments" / "grug"
    variants: list[str] = []
    found_any = False
    for child in sorted(grug_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if (child / filename).is_file():
            found_any = True
            if _variant_has_noverify(child):
                continue
            variants.append(child.name)
    if not variants and not found_any:
        raise AssertionError(f"No grug variants with {filename} found under {grug_dir}")
    return variants


def _variant_module_name(variant: str, module: str) -> str:
    return f"experiments.grug.{variant}.{module}"


def test_compact_grug_mesh_shape_allows_expert_axis_to_span_processes():
    assert _compact_grug_mesh_shape(
        process_count=32,
        local_device_count=4,
        expert_axis_size=16,
        replica_axis_size=4,
        model_axis_size=1,
    ) == (4, 2, 16, 1)


def test_compact_grug_mesh_shape_keeps_expert_axis_at_size_one():
    """Standardized contract: compact_grug_mesh always carries the expert axis.

    The data-loader and model code reference "expert" unconditionally; we keep the axis at
    size 1 instead of dropping it so size-1 cases (e.g. the GPU canary) don't fall through
    a separate "axis absent" code path. See #6252 for the bug this contract prevents.
    """
    assert _compact_grug_mesh_shape(
        process_count=1,
        local_device_count=4,
        expert_axis_size=1,
        replica_axis_size=1,
        model_axis_size=1,
    ) == (1, 4, 1, 1)


def _variant_has_noverify(variant_dir: Path) -> bool:
    train_file = variant_dir / "train.py"
    if not train_file.is_file():
        return False
    return "# GRUG NOVERIFY" in train_file.read_text(encoding="utf-8")


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def _discover_grug_variants_with_model_and_train() -> list[str]:
    model_variants = set(_discover_grug_variants_with_file("model.py"))
    train_variants = set(_discover_grug_variants_with_file("train.py"))
    variants = sorted(model_variants & train_variants)
    if not variants and model_variants and train_variants:
        return []
    if not variants:
        raise AssertionError("No grug variants with both model.py and train.py found")
    return variants


def _small_model_config(model_config_cls, *, vocab_size: int, seq_len: int):
    base_kwargs = {
        "vocab_size": vocab_size,
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "num_kv_heads": 2,
        "max_seq_len": seq_len,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "shared_expert_intermediate_dim": 64,
    }
    field_names = {field.name for field in dataclasses.fields(model_config_cls)}
    kwargs = {k: v for k, v in base_kwargs.items() if k in field_names}
    return model_config_cls(**kwargs)


def _single_device_grug_mesh() -> jax.sharding.Mesh:
    axis_names = ("replica_dcn", "data", "expert", "model")
    axis_types = tuple(jax.sharding.AxisType.Explicit for _ in axis_names)
    devices = np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1))
    return jax.sharding.Mesh(devices, axis_names, axis_types=axis_types)


def test_grug_moe_layer_masks_preserve_thd_segment_metadata():
    model_module = importlib.import_module("experiments.grug.moe.model")
    mask = GrugAttentionMask.causal().with_segment_ids(
        jnp.array([[0, 0, 1, 1, -1, -1]], dtype=jnp.int32),
        max_segments=3,
    )

    short_mask, long_mask = model_module._layer_attention_masks(mask, sliding_window=12)

    assert short_mask.thd_segment_metadata is mask.thd_segment_metadata
    assert long_mask.thd_segment_metadata is mask.thd_segment_metadata
    assert short_mask.segment_ids is mask.segment_ids
    assert long_mask.segment_ids is mask.segment_ids


def test_grug_moe_router_metrics_omit_unmeasured_capacity_rates():
    model_module = importlib.import_module("experiments.grug.moe.model")
    router_metrics = {
        "routing_entropy_per_layer": jnp.zeros((2,), dtype=jnp.float32),
        "routing_counts_per_layer": jnp.full((2, 4), 8, dtype=jnp.float32),
        "load_balancing_loss_per_layer": jnp.zeros((2,), dtype=jnp.float32),
        "router_z_loss_per_layer": jnp.zeros((2,), dtype=jnp.float32),
        "capacity_overflow_per_layer": jnp.zeros((2,), dtype=jnp.int32),
    }

    unreported = model_module._summarize_router_metrics(router_metrics)
    reported = model_module._summarize_router_metrics(router_metrics, report_capacity_overflow=True)
    unreported_rate_keys = {key for key in unreported if "capacity_overflow_rate" in key}
    reported_rate_keys = {key for key in reported if "capacity_overflow_rate" in key}

    assert unreported_rate_keys == set()
    assert reported_rate_keys == {
        "train/router/capacity_overflow_rate_mean",
        "train/router/layer_0/capacity_overflow_rate",
        "train/router/layer_1/capacity_overflow_rate",
    }


def test_scale_report_drops_controls_real_ring_drop_count():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    script = """
        import os

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh

        from experiments.grug.moe import launch_cw_scale
        from experiments.grug.moe import model as model_module

        mesh = Mesh(
            np.asarray(jax.devices()).reshape((1, 1, 8, 1)),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        os.environ.update(
            SCALE_HIDDEN_DIM="128",
            SCALE_NUM_LAYERS="1",
            SCALE_NUM_EXPERTS="8",
            SCALE_TOP_K="2",
            SCALE_SEQ_LEN="8",
        )

        def capacity_overflow(*, report_drops: bool):
            if report_drops:
                os.environ["SCALE_REPORT_DROPS"] = "1"
            else:
                os.environ.pop("SCALE_REPORT_DROPS", None)
            config = launch_cw_scale.build_scale_model()
            with jax.set_mesh(mesh):
                moe = model_module.MoEMLP.init(config, key=jax.random.key(0))
                # Equal logits route every token to the same two experts, exceeding
                # ring's capacity while leaving the backend to compute the count.
                x = jnp.zeros((8, 1, config.hidden_dim), dtype=jnp.float32)
                _, router_stats = moe(x)
            return router_stats["capacity_overflow"]

        unreported = capacity_overflow(report_drops=False)
        reported = capacity_overflow(report_drops=True)
        assert unreported.dtype == jnp.int32, unreported.dtype
        assert reported.dtype == jnp.int32, reported.dtype
        assert int(unreported) == 0, unreported
        assert int(reported) == 12, reported
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_grug_moe_scale_launcher_rejects_unknown_optimizer(monkeypatch):
    launch_module = importlib.import_module("experiments.grug.moe.launch_cw_scale")
    monkeypatch.setenv("SCALE_OPTIMIZER", "muno")

    with pytest.raises(ValueError, match="SCALE_OPTIMIZER"):
        launch_module.build_scale_checkpoint(version="dev")


def test_grug_moe_xsa_forward_lowers_with_gpu_fa4_thd_gqa_sharding():
    if jax.default_backend() != "gpu":
        pytest.skip("gpu_fa4_thd requires the JAX GPU backend")

    model_module = importlib.import_module("experiments.grug.moe.model")
    mesh, _ = model_module.debug_mesh_and_token_pspec(num_devices=8)
    cfg = model_module.GrugModelConfig(
        vocab_size=128,
        hidden_dim=512,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=8,
        num_experts=4,
        num_experts_per_token=2,
        attention_implementation="gpu_fa4_thd",
    )

    def forward():
        attn = model_module.CausalSelfAttention.init(cfg, key=jax.random.PRNGKey(0))
        x = jax.sharding.reshard(
            jnp.zeros((8, 16, cfg.hidden_dim), dtype=jnp.bfloat16),
            jax.sharding.PartitionSpec(("replica_dcn", "data", "expert"), None, None),
        )
        segment_ids = jnp.zeros((8, 16), dtype=jnp.int32)
        mask = GrugAttentionMask.causal().with_segment_ids(segment_ids, max_segments=1)
        return attn(x, mask)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_shape = eqx.filter_eval_shape(forward)

    assert out_shape.shape == (8, 16, cfg.hidden_dim)


def test_grug_moe_embedding_lookup_hlo_has_no_collectives():
    script = textwrap.dedent(
        """
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

        import jax
        import numpy as np
        from jax.sharding import NamedSharding, PartitionSpec as P

        from experiments.grug.moe.model import _embedding_gather
        from levanter.grug.sharding import Pembed_vocab, compact_grug_mesh

        assert jax.device_count() == 4
        mesh = compact_grug_mesh(replica_axis_size=4)
        table_sharding = NamedSharding(mesh, Pembed_vocab)
        token_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        host_table = np.arange(64 * 8, dtype=np.float32).reshape(64, 8)
        host_token_ids = np.arange(16, dtype=np.int32).reshape(4, 4)
        table = jax.device_put(host_table, table_sharding)
        token_ids = jax.device_put(host_token_ids, token_sharding)

        with jax.set_mesh(mesh):
            compiled = jax.jit(
                _embedding_gather,
                in_shardings=(table_sharding, token_sharding),
            ).lower(table, token_ids).compile()
            actual = np.asarray(compiled(table, token_ids))

        np.testing.assert_array_equal(actual, host_table[host_token_ids])
        print(compiled.as_text())
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    hlo = result.stdout.lower()
    assert "num_partitions=4" in hlo
    collective_opcodes = ("all-to-all", "all-gather", "all-reduce", "collective-permute", "reduce-scatter")
    found_collectives = [opcode for opcode in collective_opcodes if opcode in hlo]
    assert not found_collectives, f"embedding lookup HLO contains collectives {found_collectives}\n{result.stdout}"


def _seed_cache_records(step, prefix: str) -> None:
    """Write the minimal record a built ``TokenizedCache`` dep would leave, so the run-time
    ``mixture`` can read each dataset's tokenizer/format offline (mirrors a real run, where the
    datasets materialize first as build dependencies)."""
    for dep in step.deps:
        if dep.artifact_type is TokenizedCache:
            write_record(
                ArtifactRecord(
                    name=dep.name,
                    version=dep.version,
                    output_path=dep.path(prefix),
                    result_type=_TOKENIZED_CACHE,
                    config={"tokenizer": llama3_tokenizer, "format": {"text_key": "text"}},
                )
            )


def test_coreweave_thd_canary_uses_fixed_shape_training_segments(monkeypatch, tmp_path):
    monkeypatch.setenv("CANARY_ACCELERATOR", "gpu")
    monkeypatch.setenv("CANARY_ATTENTION_IMPLEMENTATION", "gpu_fa4_thd")
    monkeypatch.setenv("CANARY_TRACKER", "json_logger")
    monkeypatch.setenv("RUN_ID", "test-thd")
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    # build() reads the env at call time, so set it above before resolving the config.
    step = canary_ferry.build()
    _seed_cache_records(step, str(tmp_path))
    data = materialized_config(step, str(tmp_path)).data

    components = list(data.components.values())
    assert components
    assert all(isinstance(component, DatasetComponent) for component in components)
    assert {component.pack for component in components} == {1}


@pytest.mark.parametrize(
    "variant",
    _discover_grug_variants_with_model_and_train(),
)
def test_grug_variant_one_step_contract_lowers_with_default_ctor(variant: str):
    train_module = importlib.import_module(_variant_module_name(variant, "train"))
    model_module = importlib.import_module(_variant_module_name(variant, "model"))
    model_config_cls = model_module.GrugModelConfig
    make_train_step = train_module._make_train_step
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError(f"{_variant_module_name(variant, 'model')} must define debug_mesh_and_token_pspec")

    cfg = model_config_cls(vocab_size=1024)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = mesh_fn(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((32, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((32, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_state_shape, out_metrics_shape, out_watch_shape = eqx.filter_eval_shape(one_step)

    assert out_state_shape.step.shape == ()
    assert "train/loss" in out_metrics_shape
    assert out_metrics_shape["train/loss"].shape == ()
    assert out_watch_shape is None


def test_grug_moe_variant_threads_moe_implementation_to_kernel():
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    make_train_step = train_module._make_train_step
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError("experiments.grug.moe.model must define debug_mesh_and_token_pspec")

    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    cfg = dataclasses.replace(cfg, moe_implementation="ragged_all_to_all")
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = mesh_fn(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((8, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((8, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        closed_jaxpr, _, _ = eqx.filter_make_jaxpr(one_step)()

    assert "ragged_all_to_all" in str(closed_jaxpr)


def test_grug_moe_data_loaders_build_against_single_expert_mesh():
    """Regression: build_train_loader / build_tagged_evaluator must work when the
    compact mesh's expert axis has size 1 (canary configuration).

    See https://github.com/marin-community/marin/issues/6252 — canary configurations
    always have expert_axis_size == 1. Under the standardized
    ``(replica_dcn, data, expert, model)`` contract the "expert" axis is kept at length 1
    instead of being dropped, so the data-loader pspec can name it unconditionally.
    """
    train_module = importlib.import_module("experiments.grug.moe.train")
    compact_grug_mesh = importlib.import_module("levanter.grug.sharding").compact_grug_mesh

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    assert mesh.shape.get("expert") == 1, "fixture must reproduce the canary single-expert layout"

    dataset = ListAsyncDataset(
        [
            GrugLmExample(
                tokens=jnp.zeros((4,), dtype=jnp.int32),
                loss_weight=jnp.ones((4,), dtype=jnp.float32),
                attn_mask=GrugAttentionMask.causal(),
            )
        ]
    )
    batch_schedule = BatchSchedule(max(1, len(jax.devices())))

    # This used to raise: "Resource axis: expert ... is not found in mesh: (..., model)".
    loader = train_module.build_train_loader(dataset, batch_schedule=batch_schedule, mesh=mesh)
    assert loader is not None


def test_grug_moe_model_init_against_single_expert_mesh():
    """Regression: MoEMLP.init must build when the compact mesh's expert axis has size 1.

    See https://github.com/marin-community/marin/issues/6252 — canary configurations
    have expert_axis_size == 1. Under the standardized
    ``(replica_dcn, data, expert, model)`` contract the "expert" axis is kept at length 1,
    so MoEMLP.init reads ``mesh.shape["expert"] == 1`` rather than hitting an
    "axis absent" branch.
    """
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    compact_grug_mesh = importlib.import_module("levanter.grug.sharding").compact_grug_mesh

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    assert mesh.shape.get("expert") == 1, "fixture must reproduce the canary single-expert layout"

    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=1024, seq_len=4)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")

    def build():
        return train_module.initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh.abstract_mesh):
        state_shape = eqx.filter_eval_shape(build)

    assert state_shape.params is not None


def test_grug_moe_scanned_and_unscanned_forward_values_match():
    model_module = importlib.import_module("experiments.grug.moe.model")
    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=128, seq_len=4)
    cfg = dataclasses.replace(cfg, num_layers=5, disable_long_rope=True)
    key = jax.random.PRNGKey(0)
    tokens = jnp.arange(4, dtype=jnp.int32).reshape(1, 4)
    mesh = _single_device_grug_mesh()
    forward = eqx.filter_jit(lambda model, token_ids: model(token_ids))

    with jax.set_mesh(mesh):
        scanned = model_module.Transformer.init(
            dataclasses.replace(cfg, use_array_stacked_blocks=True),
            key=key,
        )
        unscanned = model_module.Transformer.init(
            dataclasses.replace(cfg, use_array_stacked_blocks=False),
            key=key,
        )
        scanned_hidden, scanned_metrics = forward(scanned, tokens)
        unscanned_hidden, unscanned_metrics = forward(unscanned, tokens)
        scanned_state_dict = scanned.to_state_dict()
        unscanned_state_dict = unscanned.to_state_dict()

    np.testing.assert_allclose(scanned_hidden, unscanned_hidden, rtol=1e-5, atol=1e-5)
    assert scanned_metrics.keys() == unscanned_metrics.keys()
    for name in scanned_metrics:
        np.testing.assert_allclose(scanned_metrics[name], unscanned_metrics[name], rtol=1e-5, atol=1e-5)

    assert scanned_state_dict.keys() == unscanned_state_dict.keys()
    for name in scanned_state_dict:
        np.testing.assert_allclose(scanned_state_dict[name], unscanned_state_dict[name], rtol=1e-5, atol=1e-5)


def test_grug_moe_scale_launcher_builds_one_homogeneous_scan(monkeypatch):
    launch_module = importlib.import_module("experiments.grug.moe.launch_cw_scale")
    model_module = importlib.import_module("experiments.grug.moe.model")
    monkeypatch.setattr(launch_module, "VOCAB_SIZE", 128)
    monkeypatch.setattr(launch_module, "HEAD_DIM", 16)
    monkeypatch.setenv("SCALE_HIDDEN_DIM", "32")
    monkeypatch.setenv("SCALE_NUM_LAYERS", "5")
    monkeypatch.setenv("SCALE_NUM_EXPERTS", "4")
    monkeypatch.setenv("SCALE_TOP_K", "2")
    monkeypatch.setenv("SCALE_SEQ_LEN", "4")
    monkeypatch.setenv("SCALE_SCAN_LAYERS", "1")

    cfg = launch_module.build_scale_model()
    tokens = jnp.arange(4, dtype=jnp.int32).reshape(1, 4)
    mesh = _single_device_grug_mesh()
    with jax.set_mesh(mesh):
        model = model_module.Transformer.init(cfg, key=jax.random.PRNGKey(0))
        closed_jaxpr, _, _ = eqx.filter_make_jaxpr(model)(tokens)
        lowered = eqx.filter_jit(model).lower(tokens)

    scan_equations = [equation for equation in closed_jaxpr.jaxpr.eqns if equation.primitive.name == "scan"]
    assert len(scan_equations) == 1
    assert scan_equations[0].params["length"] == cfg.num_layers
    assert scan_equations[0].params["unroll"] == 1
    stablehlo = lowered.as_text()
    assert stablehlo.count("stablehlo.while") == 1
    with pytest.raises(ValueError, match="requires disable_pko=True"):
        model_module.Transformer.init(dataclasses.replace(cfg, disable_pko=False), key=jax.random.PRNGKey(0))


def test_grug_moe_fa4_bounds_are_loop_invariant_in_scanned_graph(monkeypatch):
    def consume_precomputed_bounds(q, k, v, mask):
        del k, v
        assert mask.fa4_bounds is not None
        lower_bounds, valid = mask.fa4_bounds
        return q + lower_bounds[..., None, None].astype(q.dtype) * 0 + valid[..., None, None].astype(q.dtype) * 0

    model_module = importlib.import_module("experiments.grug.moe.model")
    monkeypatch.setattr(fa4_cute, "gpu_fa4_cute_attention", consume_precomputed_bounds)
    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=128, seq_len=8)
    cfg = dataclasses.replace(
        cfg,
        num_layers=5,
        use_array_stacked_blocks=True,
        attention_implementation="gpu_fa4_cute",
    )
    tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
    mesh = _single_device_grug_mesh()

    with jax.set_mesh(mesh):
        model = model_module.Transformer.init(cfg, key=jax.random.PRNGKey(0))
        closed_jaxpr, _, _ = eqx.filter_make_jaxpr(model)(tokens)
        stablehlo = eqx.filter_jit(model).lower(tokens).as_text()

    layer_scan = next(
        equation
        for equation in closed_jaxpr.jaxpr.eqns
        if equation.primitive.name == "scan" and equation.params["length"] == cfg.num_layers
    )
    scan_body = layer_scan.params["jaxpr"]
    long_bounds, short_bounds, valid = scan_body.invars[:3]
    layer_uses_long_mask = scan_body.invars[-1]
    select_bounds, rematerialized_layer = scan_body.eqns

    assert [var.aval.shape for var in (long_bounds, short_bounds, valid)] == [(1, 8)] * 3
    assert [var.aval.dtype for var in (long_bounds, short_bounds, valid)] == [
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.bool_),
    ]
    assert select_bounds.primitive.name == "jit"
    assert [equation.primitive.name for equation in select_bounds.params["jaxpr"].eqns] == ["select_n"]
    assert select_bounds.invars == [layer_uses_long_mask, long_bounds, short_bounds]
    assert rematerialized_layer.primitive.name == "remat2"
    assert rematerialized_layer.invars[-3:] == [select_bounds.outvars[0], valid, layer_uses_long_mask]
    assert stablehlo.count("stablehlo.while") == 1
    assert "stablehlo.case" not in stablehlo


def test_grug_moe_scan_layers_one_step_lowers():
    train_module = importlib.import_module("experiments.grug.moe.train")
    model_module = importlib.import_module("experiments.grug.moe.model")
    cfg = _small_model_config(model_module.GrugModelConfig, vocab_size=128, seq_len=4)
    cfg = dataclasses.replace(cfg, use_array_stacked_blocks=True)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    train_step = train_module._make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    mesh, token_pspec = model_module.debug_mesh_and_token_pspec(num_devices=4)
    batch = GrugLmExample(
        tokens=jnp.zeros((8, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((8, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    def one_step():
        sharded_batch = dataclasses.replace(
            batch,
            tokens=jax.sharding.reshard(batch.tokens, token_pspec),
            loss_weight=jax.sharding.reshard(batch.loss_weight, token_pspec),
        )
        state = train_module.initial_state(
            cfg,
            optimizer=optimizer,
            mp=mp,
            key=jax.random.PRNGKey(0),
            ema_beta=None,
        )
        return train_step(state, sharded_batch, compute_watch=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        out_state_shape, out_metrics_shape, out_watch_shape = eqx.filter_eval_shape(one_step)

    assert out_state_shape.step.shape == ()
    assert out_metrics_shape["train/loss"].shape == ()
    assert out_watch_shape is None


@pytest.mark.parametrize(
    "variant",
    _discover_grug_variants_with_model_and_train(),
)
def test_grug_variant_initial_state_only_stores_ema_when_enabled(variant: str):
    train_module = importlib.import_module(_variant_module_name(variant, "train"))
    model_module = importlib.import_module(_variant_module_name(variant, "model"))
    model_config_cls = model_module.GrugModelConfig
    initial_state = train_module.initial_state
    mesh_fn = getattr(model_module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError(f"{_variant_module_name(variant, 'model')} must define debug_mesh_and_token_pspec")

    cfg = model_config_cls(vocab_size=1024)
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    mesh, _ = mesh_fn(num_devices=4)

    def init_state_shape(*, ema_beta: float | None):
        def build():
            return initial_state(cfg, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=ema_beta)

        with _reset_abstract_mesh(), use_abstract_mesh(mesh):
            return eqx.filter_eval_shape(build)

    no_ema_state_shape = init_state_shape(ema_beta=None)
    assert no_ema_state_shape.ema_params is None

    with_ema_state_shape = init_state_shape(ema_beta=0.999)
    assert with_ema_state_shape.ema_params is not None


def test_grug_base_run_emits_expected_metrics_with_json_tracker(tmp_path: Path):
    train_module = importlib.import_module("experiments.grug.base.train")
    model_module = importlib.import_module("experiments.grug.base.model")

    vocab_size = 128
    seq_len = 32
    examples = []
    for i in range(8):
        tokens = (jnp.arange(seq_len, dtype=jnp.int32) + i) % vocab_size
        examples.append(GrugLmExample.causal(tokens))
    eval_examples = [GrugLmExample.causal((jnp.arange(seq_len, dtype=jnp.int32) + 100) % vocab_size)]

    train_dataset = ListAsyncDataset(examples)
    eval_dataset = ListAsyncDataset(eval_examples)
    data_config = LmDataConfig(
        components={"direct": DirectDatasetComponent(datasets={"train": train_dataset, "validation": eval_dataset})},
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    logger_name = f"test_grug_json_tracker_base_{uuid.uuid4().hex}"
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        variant_tmp = tmp_path / "base"
        variant_tmp.mkdir(parents=True, exist_ok=True)
        trainer_config = TrainerConfig(
            id="test-grug-base-metrics",
            num_train_steps=1,
            train_batch_size=max(1, len(jax.devices())),
            tracker=JsonLoggerConfig(logger_name=logger_name),
            require_accelerator=False,
            use_explicit_mesh_axes=True,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            log_dir=variant_tmp / "logs",
            checkpointer=CheckpointerConfig(base_path=str(variant_tmp / "checkpoints")),
        )

        run_cfg = train_module.GrugRunConfig(
            model=_small_model_config(model_module.GrugModelConfig, vocab_size=vocab_size, seq_len=seq_len),
            data=data_config,
            resources=ResourceConfig.with_cpu(),
            trainer=train_module.GrugTrainerConfig(
                trainer=trainer_config,
                log_every=1,
                backward_flow=train_module.BackwardFlowConfig(interval=0),
            ),
            eval=train_module.GrugEvalConfig(
                eval_batch_size=1,
                steps_per_eval=1,
                max_eval_batches=1,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            ),
        )
        train_module.run_grug(run_cfg)
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    finish_records = [record for record in records if record.get("event") == "finish"]
    assert len(finish_records) == 1
    summary = finish_records[0]["summary"]

    required_keys = [
        "train/loss",
        "global_step",
        "throughput/duration",
        "throughput/hook_time",
        "throughput/loading_time",
        "throughput/total_tokens",
        "throughput/examples_per_second",
        "throughput/tokens_per_second",
        "throughput/flops_per_example_analytic",
        "eval/loss",
        "eval/loading_time",
        "eval/total_time",
    ]
    for key in required_keys:
        assert key in summary
