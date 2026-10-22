# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight-analysis and grafting tests on a tiny real Grug export.

Fixtures come from ``grugmoe_inference_state_dict`` on a two-layer, four-expert
Grug model, so names, the stacked ``[E, out, in]`` expert layout and shapes are
the export's own. ``B`` is that model with noise, rounded to bf16 (control C8);
checkpoints are ``B`` plus fp32 perturbations, written as sharded safetensors.
"""

import itertools
import json
import math
from pathlib import Path

import fsspec
import jax
import ml_dtypes
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh
from safetensors import safe_open

from experiments.grug.moe.model import GrugModelConfig, Transformer, grugmoe_inference_state_dict
from experiments.post_training.weight_analysis.checkpoint_io import (
    REQUIRED_SERVING_FILES,
    SafetensorsCheckpoint,
    TensorSpec,
    write_sharded_checkpoint,
)
from experiments.post_training.weight_analysis.components import (
    GRAFT_GROUPS,
    Component,
    component_of,
    hybrid_plan,
)
from experiments.post_training.weight_analysis.delta_stats import (
    compare_deltas,
    delta_blocks,
    paired_blocks,
    subspace_overlap,
    summarize_deltas,
)
from experiments.post_training.weight_analysis.expert_metrics import (
    additivity,
    expert_gini,
    normalize_by_seed_ceiling,
    seed_ceiling,
)
from experiments.post_training.weight_analysis.graft import graft_checkpoint

EXPERTS = 4
LAYERS = 2
SHARD_BYTES = 16 * 1024  # several shards for the tiny model
EXPERT_GATE = "model.layers.{layer}.mlp.experts.gate_proj.weight"
EVERY_COMPONENT = frozenset(Component)


@pytest.fixture(scope="module")
def grug_base() -> dict[str, np.ndarray]:
    config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=EXPERTS,
        num_experts_per_token=2,
        num_layers=LAYERS,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(config, key=jax.random.PRNGKey(0))
        exported = {
            name: np.asarray(value, dtype=np.float32) for name, value in grugmoe_inference_state_dict(model).items()
        }
    rng = np.random.default_rng(0)
    # Grug zero-initializes the router bias and attention gate; a trained B has moved them.
    return {
        name: (value + 0.02 * rng.standard_normal(value.shape)).astype(ml_dtypes.bfloat16).astype(np.float32)
        for name, value in exported.items()
    }


def zero_delta(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.zeros_like(value) for name, value in base.items()}


def write(tensors: dict[str, np.ndarray], path: str) -> str:
    names = sorted(tensors)
    plan = {name: TensorSpec(tensors[name].shape, tensors[name].dtype) for name in names}
    write_sharded_checkpoint(((name, tensors[name]) for name in names), path, plan=plan, shard_size_bytes=SHARD_BYTES)
    return path


def write_pair(tmp_path: Path, base: dict[str, np.ndarray], deltas: dict[str, dict[str, np.ndarray]]) -> dict[str, str]:
    """Write B and one checkpoint ``B + delta`` per entry; returns paths by label."""
    paths = {"base": write(base, str(tmp_path / "base"))}
    for label, delta in deltas.items():
        paths[label] = write({name: base[name] + delta[name] for name in base}, str(tmp_path / label))
    return paths


def random_delta(base: dict[str, np.ndarray], seed: int, scale: float = 1e-3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: (scale * rng.standard_normal(value.shape)).astype(np.float32) for name, value in base.items()}


def write_serving_files(directory: Path) -> str:
    directory.mkdir()
    for name in REQUIRED_SERVING_FILES:
        (directory / name).write_text(json.dumps({"file": name}))
    return str(directory)


def read_all(path: str) -> dict[str, np.ndarray]:
    checkpoint = SafetensorsCheckpoint(path)
    return {name: checkpoint.read(name) for name in checkpoint.names()}


# --- component groups on the real export --------------------------------------------


def test_every_grug_export_name_maps_to_its_component(grug_base):
    components = {name: component_of(name) for name in grug_base}
    assert set(components.values()) == EVERY_COMPONENT
    assert components["model.layers.0.mlp.router.bias"] is Component.SELECTION_BIAS
    assert components["model.layers.0.mlp.router.weight"] is Component.ROUTER
    assert components["model.layers.0.self_attn.attn_gate.weight"] is Component.NORMS_AND_GATES
    assert components["lm_head.weight"] is Component.LM_HEAD
    assert grug_base[EXPERT_GATE.format(layer=0)].shape[0] == EXPERTS
    assert len(hybrid_plan()) == 2 * len(GRAFT_GROUPS) == 14


def test_unclassified_tensor_name_raises():
    with pytest.raises(ValueError, match="matches no Grug component"):
        component_of("model.layers.0.mlp.router.expert_bias")


# --- checkpoint I/O ---------------------------------------------------------------------


def test_sharded_writer_and_reader_agree_with_safe_open(tmp_path, grug_base):
    path = write(grug_base, str(tmp_path / "ckpt"))
    index = json.loads((tmp_path / "ckpt" / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) > 1
    checkpoint = SafetensorsCheckpoint(path)
    for shard in shards:
        with safe_open(str(tmp_path / "ckpt" / shard), framework="numpy") as handle:
            for name in handle.keys():
                np.testing.assert_array_equal(checkpoint.read(name), handle.get_tensor(name))
    stacked = grug_base[EXPERT_GATE.format(layer=1)]
    for expert in range(EXPERTS):
        np.testing.assert_array_equal(checkpoint.read_expert(EXPERT_GATE.format(layer=1), expert), stacked[expert])


def test_bf16_base_upcasts_to_exact_deltas(tmp_path, grug_base):
    delta = random_delta(grug_base, seed=1)
    checkpoint = write({name: grug_base[name] + delta[name] for name in grug_base}, str(tmp_path / "ckpt"))
    base = write({name: value.astype(ml_dtypes.bfloat16) for name, value in grug_base.items()}, str(tmp_path / "base"))
    for block in delta_blocks(checkpoint, base):
        expected = grug_base[block.name] + delta[block.name] - grug_base[block.name]
        if block.expert is not None:
            expected = expected[block.expert[1]]
        np.testing.assert_array_equal(block.array, expected)


def test_c8_violations_raise(tmp_path, grug_base):
    delta = random_delta(grug_base, seed=2)
    checkpoint = write({name: grug_base[name] + delta[name] for name in grug_base}, str(tmp_path / "ckpt"))
    not_upcast = write({name: grug_base[name] + delta[name] for name in grug_base}, str(tmp_path / "not_upcast"))
    with pytest.raises(ValueError, match="not the fp32 upcast"):
        list(paired_blocks(checkpoint, not_upcast))

    base = write(grug_base, str(tmp_path / "base"))
    rounded = {name: (grug_base[name] + delta[name]).astype(ml_dtypes.bfloat16).astype(np.float32) for name in grug_base}
    bf16_export = write(rounded, str(tmp_path / "bf16_export"))
    with pytest.raises(ValueError, match="bf16 export"):
        list(paired_blocks(bf16_export, base))


def test_mismatched_checkpoints_raise(tmp_path, grug_base):
    base = write(grug_base, str(tmp_path / "base"))
    missing = write(
        {name: value for name, value in grug_base.items() if name != "lm_head.weight"}, str(tmp_path / "missing")
    )
    with pytest.raises(ValueError, match="different tensors"):
        list(paired_blocks(missing, base))
    reshaped = dict(grug_base)
    reshaped["lm_head.weight"] = np.zeros((33, 16), dtype=np.float32)
    reshaped_path = write(reshaped, str(tmp_path / "reshaped"))
    with pytest.raises(ValueError, match="shape"):
        list(paired_blocks(reshaped_path, base))


# --- delta summaries and expert Gini ------------------------------------------------------


def test_summary_and_gini_are_per_layer_expert(tmp_path, grug_base):
    delta = zero_delta(grug_base)
    delta[EXPERT_GATE.format(layer=0)][:] = 0.5  # every layer-0 expert moves equally
    delta[EXPERT_GATE.format(layer=1)][2] = 1.0  # only expert 2 of layer 1 moves
    delta["model.layers.0.self_attn.q_proj.weight"][0, 0] = 0.25
    paths = write_pair(tmp_path, grug_base, {"ckpt": delta})
    summary = summarize_deltas(paired_blocks(paths["ckpt"], paths["base"]))

    expert_size = grug_base[EXPERT_GATE.format(layer=0)][0].size
    norms = {key: value.delta_norm for key, value in summary.experts.items()}
    assert set(norms) == set(itertools.product(range(LAYERS), range(EXPERTS)))
    assert norms[(1, 2)] == pytest.approx(math.sqrt(expert_size), rel=1e-6)
    assert norms[(1, 0)] == 0.0
    assert norms[(0, 0)] == pytest.approx(0.5 * math.sqrt(expert_size), rel=1e-6)
    assert summary.experts[(1, 2)].changed_fraction == pytest.approx(1 / 3)  # gate of gate/up/down

    attention = summary.components[Component.ATTENTION]
    attention_base = sum(
        float(np.sum(grug_base[name].astype(np.float64) ** 2))
        for name in grug_base
        if component_of(name) is Component.ATTENTION
    )
    assert attention.relative_norm == pytest.approx(0.25 / math.sqrt(attention_base), rel=1e-5)
    assert attention.changed == 1

    gini = expert_gini(summary)
    values = list(norms.values())
    mean_difference = sum(abs(a - b) for a in values for b in values) / (2 * len(values) ** 2 * np.mean(values))
    assert gini.per_layer[0] == pytest.approx(0.0, abs=1e-12)
    assert gini.per_layer[1] == pytest.approx(0.75)
    assert gini.pooled == pytest.approx(mean_difference)


# --- task conflict, additivity, seed ceiling ----------------------------------------------


def test_compare_deltas_keys_conflict_by_layer_and_expert(tmp_path, grug_base):
    delta_a = zero_delta(grug_base)
    delta_b = zero_delta(grug_base)
    delta_a[EXPERT_GATE.format(layer=0)][1] = 1.0
    delta_b[EXPERT_GATE.format(layer=0)][1] = 2.0  # aligned in layer 0
    delta_a[EXPERT_GATE.format(layer=1)][1] = 1.0
    delta_b[EXPERT_GATE.format(layer=1)][1] = -1.0  # opposed in layer 1
    delta_b[EXPERT_GATE.format(layer=1)][3] = 1.0  # moved by B only
    paths = write_pair(tmp_path, grug_base, {"a": delta_a, "b": delta_b})
    comparison = compare_deltas(delta_blocks(paths["a"], paths["base"]), delta_blocks(paths["b"], paths["base"]))

    assert comparison.experts[(0, 1)].cosine == pytest.approx(1.0)
    assert comparison.experts[(0, 1)].sign_agreement == 1.0
    assert comparison.experts[(1, 1)].cosine == pytest.approx(-1.0)
    assert comparison.experts[(1, 1)].sign_agreement == 0.0
    assert comparison.experts[(1, 3)].changed_left == 0
    assert comparison.experts[(1, 3)].changed_right == grug_base[EXPERT_GATE.format(layer=1)][3].size
    assert not comparison.experts[(1, 3)].geometry.defined
    with pytest.raises(ValueError, match="sign agreement is undefined"):
        _ = comparison.experts[(1, 3)].sign_agreement
    routed = comparison.components[Component.ROUTED_EXPERTS]
    assert routed.cosine == pytest.approx(1.0 / math.sqrt(12.0))  # dot s, norms² 2s and 6s
    assert routed.sign_agreement == 0.5


def test_additivity_is_the_design_cosine_and_norm_ratio(tmp_path, grug_base):
    delta_a = random_delta(grug_base, seed=3)
    delta_b = random_delta(grug_base, seed=4)
    joint = {name: delta_a[name] + delta_b[name] for name in grug_base}
    for name in grug_base:
        if component_of(name) is Component.LM_HEAD:
            joint[name] = 2.0 * joint[name]
    paths = write_pair(tmp_path, grug_base, {"a": delta_a, "b": delta_b, "joint": joint})
    report = additivity(*(delta_blocks(paths[label], paths["base"]) for label in ("joint", "a", "b")))

    assert report.components[Component.ATTENTION].cosine == pytest.approx(1.0, abs=1e-6)
    assert report.components[Component.ATTENTION].norm_ratio == pytest.approx(1.0, rel=1e-5)
    assert report.components[Component.LM_HEAD].cosine == pytest.approx(1.0, abs=1e-6)
    assert report.components[Component.LM_HEAD].norm_ratio == pytest.approx(2.0, rel=1e-5)
    assert report.experts[(1, 3)].norm_ratio == pytest.approx(1.0, rel=1e-5)
    assert 1.0 < report.overall.norm_ratio < 2.0


def test_seed_ceiling_means_seed_pair_cosines_and_flags_near_zero(tmp_path, grug_base):
    shared = random_delta(grug_base, seed=5)
    seeds = {f"seed{index}": random_delta(grug_base, seed=10 + index) for index in range(3)}
    for noise in seeds.values():
        for name in grug_base:
            # Attention shares a direction across seeds; everything else is seed noise.
            if component_of(name) is Component.ATTENTION:
                noise[name] = noise[name] + 3.0 * shared[name]
    paths = write_pair(tmp_path, grug_base, seeds)
    seed_streams = [delta_blocks(paths[label], paths["base"]) for label in seeds]
    ceilings = seed_ceiling(seed_streams, near_zero=0.3)

    def attention_vector(label: str) -> np.ndarray:
        names = sorted(name for name in grug_base if component_of(name) is Component.ATTENTION)
        return np.concatenate([seeds[label][name].ravel() for name in names]).astype(np.float64)

    expected = [
        float(
            attention_vector(a)
            @ attention_vector(b)
            / np.linalg.norm(attention_vector(a))
            / np.linalg.norm(attention_vector(b))
        )
        for a, b in itertools.combinations(seeds, 2)
    ]
    attention = ceilings.components[Component.ATTENTION]
    assert len(attention.pair_cosines) == 3
    assert attention.pair_cosines == pytest.approx(tuple(expected), rel=1e-5)
    assert attention.cosine == pytest.approx(sum(expected) / 3, rel=1e-5)
    assert not attention.near_zero
    assert ceilings.components[Component.LM_HEAD].near_zero
    assert ceilings.experts[(0, 0)].near_zero

    method_cosine = ceilings.map(lambda ceiling: 0.5)
    normalized = normalize_by_seed_ceiling(method_cosine, ceilings)
    entry = normalized.components[Component.ATTENTION]
    assert entry.cosine == 0.5
    assert entry.seed_cosine == pytest.approx(attention.cosine)
    assert entry.ratio == pytest.approx(0.5 / attention.cosine)
    assert normalized.components[Component.LM_HEAD].near_zero_ceiling


# --- subspace overlap ------------------------------------------------------------------------


def _reference_overlap(delta: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """(projected energy, delta energy) from a full SVD of the reference."""
    _, singular, vh = np.linalg.svd(reference.astype(np.float64), full_matrices=False)
    cumulative = np.cumsum(singular**2)
    k = int(np.argmax(cumulative >= cumulative[-1] / 2)) + 1
    projected = delta.astype(np.float64) @ vh[:k].T
    return float(np.sum(projected**2)), float(np.sum(delta.astype(np.float64) ** 2))


def test_subspace_overlap_matches_svd_reference_per_expert_and_energy_weighted(tmp_path, grug_base):
    # A rank-1 reference per matrix gives k = 1, so a delta that contains it sits far above both nulls.
    rng = np.random.default_rng(7)
    reference = {}
    for name, value in grug_base.items():
        if value.ndim == 1:
            reference[name] = (1e-3 * rng.standard_normal(value.shape)).astype(np.float32)
            continue
        left = rng.standard_normal((*value.shape[:-1], 1))
        right = rng.standard_normal((*value.shape[:-2], 1, value.shape[-1]))
        reference[name] = (1e-3 * left * right).astype(np.float32)
    delta = random_delta(grug_base, seed=6, scale=1e-4)
    for name in grug_base:
        delta[name] = delta[name] + reference[name]
    paths = write_pair(tmp_path, grug_base, {"delta": delta, "reference": reference})
    report = subspace_overlap(
        delta_blocks(paths["delta"], paths["base"]), delta_blocks(paths["reference"], paths["base"]), seed=0
    )

    # Recompute from the written tensors, so fp32 rounding of B + Δ - B matches.
    stored = read_all(paths["delta"])
    stored_reference = read_all(paths["reference"])
    base = read_all(paths["base"])

    def matrices(name: str) -> list[tuple[np.ndarray, np.ndarray]]:
        d, r = stored[name] - base[name], stored_reference[name] - base[name]
        if d.ndim == 3:
            return list(zip(d, r, strict=True))
        return [(d, r)] if d.ndim == 2 else []

    expert_totals = [
        _reference_overlap(*matrices(f"model.layers.1.mlp.experts.{proj}_proj.weight")[2])
        for proj in ("gate", "up", "down")
    ]
    expert = report.experts[(1, 2)]
    assert expert.matrices == 3
    assert expert.overlap == pytest.approx(sum(t[0] for t in expert_totals) / sum(t[1] for t in expert_totals), rel=1e-6)

    pairs = [pair for name in grug_base for pair in matrices(name)]
    totals = [_reference_overlap(*pair) for pair in pairs]
    assert report.overall.matrices == len(pairs)
    assert report.overall.overlap == pytest.approx(sum(t[0] for t in totals) / sum(t[1] for t in totals), rel=1e-6)
    routed = report.components[Component.ROUTED_EXPERTS]
    assert routed.matrices == LAYERS * 3 * EXPERTS
    assert routed.overlap > 2 * routed.random_control
    assert routed.overlap > 2 * routed.permuted_control


# --- grafting ---------------------------------------------------------------------------------


def test_graft_all_groups_reproduces_the_fp32_checkpoint_exactly(tmp_path, grug_base):
    delta = random_delta(grug_base, seed=8)
    paths = write_pair(tmp_path, grug_base, {"ckpt": delta})
    serving = write_serving_files(tmp_path / "release")
    hybrid = str(tmp_path / "hybrid")
    index = graft_checkpoint(
        paths["base"], paths["ckpt"], EVERY_COMPONENT, hybrid, serving_files_dir=serving, shard_size_bytes=SHARD_BYTES
    )

    assert len(set(index["weight_map"].values())) > 1
    checkpoint = read_all(paths["ckpt"])
    grafted = read_all(hybrid)
    assert set(grafted) == set(checkpoint)
    for name, value in checkpoint.items():
        assert grafted[name].dtype == np.float32
        np.testing.assert_array_equal(grafted[name], value, err_msg=name)
    for name in REQUIRED_SERVING_FILES:
        assert json.loads((tmp_path / "hybrid" / name).read_text()) == {"file": name}


def test_graft_one_group_takes_the_checkpoint_only_inside_it(tmp_path, grug_base):
    delta = random_delta(grug_base, seed=9)
    paths = write_pair(tmp_path, grug_base, {"ckpt": delta})
    paths["bf16_base"] = write({n: v.astype(ml_dtypes.bfloat16) for n, v in grug_base.items()}, str(tmp_path / "b16"))
    serving = write_serving_files(tmp_path / "release")
    hybrid = str(tmp_path / "hybrid")
    graft_checkpoint(
        paths["bf16_base"],
        paths["ckpt"],
        hybrid_plan()["only_selection_bias"],
        hybrid,
        serving_files_dir=serving,
        shard_size_bytes=SHARD_BYTES,
    )
    checkpoint = read_all(paths["ckpt"])
    for name, value in read_all(hybrid).items():
        expected = checkpoint[name] if component_of(name) is Component.SELECTION_BIAS else grug_base[name]
        np.testing.assert_array_equal(value, expected, err_msg=name)


def test_graft_refuses_unclassified_tensors_and_mismatched_keys(tmp_path, grug_base):
    serving = write_serving_files(tmp_path / "release")
    base = write(grug_base, str(tmp_path / "base"))
    extra = dict(grug_base)
    extra["model.layers.0.mlp.router.expert_bias"] = np.zeros(EXPERTS, dtype=np.float32)
    extra_base = write(extra, str(tmp_path / "extra_base"))
    extra_ckpt = write(extra, str(tmp_path / "extra_ckpt"))
    with pytest.raises(ValueError, match="matches no Grug component"):
        graft_checkpoint(
            extra_base,
            extra_ckpt,
            EVERY_COMPONENT,
            str(tmp_path / "out1"),
            serving_files_dir=serving,
            shard_size_bytes=SHARD_BYTES,
        )
    with pytest.raises(ValueError, match="different tensors"):
        graft_checkpoint(
            base,
            extra_ckpt,
            EVERY_COMPONENT,
            str(tmp_path / "out2"),
            serving_files_dir=serving,
            shard_size_bytes=SHARD_BYTES,
        )
    assert not (tmp_path / "out1").exists() and not (tmp_path / "out2").exists()


def test_graft_streams_through_a_remote_filesystem(tmp_path, grug_base):
    delta = random_delta(grug_base, seed=11)
    base = write(grug_base, "memory://wp9/base")
    checkpoint = write({name: grug_base[name] + delta[name] for name in grug_base}, "memory://wp9/ckpt")
    memory = fsspec.filesystem("memory")
    for name in REQUIRED_SERVING_FILES:
        memory.pipe(f"memory://wp9/release/{name}", b"{}")
    graft_checkpoint(
        base,
        checkpoint,
        EVERY_COMPONENT,
        "memory://wp9/hybrid",
        serving_files_dir="memory://wp9/release",
        shard_size_bytes=SHARD_BYTES,
    )
    expected = read_all(checkpoint)
    for name, value in read_all("memory://wp9/hybrid").items():
        np.testing.assert_array_equal(value, expected[name], err_msg=name)
    assert memory.exists("memory://wp9/hybrid/config.json")
