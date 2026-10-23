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
    Keyed,
    component_of,
    hybrid_plan,
)
from experiments.post_training.weight_analysis.delta_stats import (
    DeltaComparison,
    DeltaSummary,
    PairGeometry,
    SubspaceOverlap,
    block_comparison,
    block_summary,
    compare_deltas,
    delta_blocks,
    matrix_overlaps,
    paired_blocks,
    record_block,
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
from experiments.post_training.weight_analysis.reproduce_9920 import (
    TheirResults,
    group_9920,
    parse_model_specs,
    run_pass,
    summary_rows,
)

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


def test_fp32_selection_bias_in_b_is_not_a_c8_violation(tmp_path, grug_base):
    """Releases keep Grug's router bias in fp32, so B's bias need not be bf16-exact."""
    base = dict(grug_base)
    bias = "model.layers.0.mlp.router.bias"
    base[bias] = base[bias] + np.float32(1e-5)
    assert not np.array_equal(base[bias], base[bias].astype(ml_dtypes.bfloat16).astype(np.float32))
    delta = random_delta(base, seed=3)
    checkpoint = write({name: base[name] + delta[name] for name in base}, str(tmp_path / "ckpt"))
    base_path = write(base, str(tmp_path / "base"))
    pairs = list(paired_blocks(checkpoint, base_path))
    np.testing.assert_array_equal(next(b.array for b, _ in pairs if b.name == bias), base[bias])
    hybrid = str(tmp_path / "hybrid")
    serving = write_serving_files(tmp_path / "release")
    graft_checkpoint(
        base_path,
        checkpoint,
        frozenset({Component.ATTENTION}),
        hybrid,
        serving_files_dir=serving,
        shard_size_bytes=SHARD_BYTES,
    )
    np.testing.assert_array_equal(read_all(hybrid)[bias], base[bias])


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


# --- the #9920 cross-check script ---------------------------------------------------------


def theirs_per_tensor(base: dict[str, np.ndarray], model: dict[str, np.ndarray]) -> dict[str, dict]:
    """#9920-shaped per-tensor entries, recomputed independently from the raw arrays."""
    entries = {}
    for name in sorted(base):
        base_array = base[name].astype(np.float32)
        delta = model[name].astype(np.float32) - base_array
        entries[name] = {
            "n": float(base[name].size),
            "b2": float(np.sum(base_array.astype(np.float64) ** 2)),
            "d2": float(np.sum(delta.astype(np.float64) ** 2)),
            "dmax": float(np.max(np.abs(delta))),
            "changed": float(np.count_nonzero(delta)),
            "group": group_9920(name),
        }
    return entries


def theirs_summary(base: dict[str, np.ndarray], model: dict[str, np.ndarray]) -> dict[str, dict]:
    """#9920-shaped per-group summaries, recomputed independently from the raw arrays."""
    groups = {name: group_9920(name) for name in base}
    summed: dict[str, list[float]] = {"ALL": [0.0, 0.0, 0, 0]}
    for name in sorted(base):
        base_array = base[name].astype(np.float32)
        delta = model[name].astype(np.float32) - base_array
        entry = summed.setdefault(groups[name], [0.0, 0.0, 0, 0])
        for slot, value in (
            (0, float(np.sum(base_array.astype(np.float64) ** 2))),
            (1, float(np.sum(delta.astype(np.float64) ** 2))),
            (2, int(np.count_nonzero(delta))),
            (3, base[name].size),
        ):
            entry[slot] += value
            summed["ALL"][slot] += value
    return {
        group: {
            "n": float(values[3]),
            "rel_frobenius": (values[1] / values[0]) ** 0.5,
            "frac_changed": values[2] / values[3],
        }
        for group, values in summed.items()
    }


@pytest.fixture(scope="module")
def tiny_pair(tmp_path_factory, grug_base):
    """B, two bf16 models (E and a stand-in opd75), and fabricated #9920 result JSONs.

    The comparison model uses the published ``opd75`` label so the published-row
    path runs; its fabricated numbers are not expected to match the table.
    """
    groups = {name: group_9920(name) for name in grug_base}
    tmp = tmp_path_factory.mktemp("reproduce9920")
    expert = {
        name: (
            (value + 2e-3 * np.random.default_rng(22).standard_normal(value.shape))
            .astype(ml_dtypes.bfloat16)
            .astype(np.float32)
        )
        for name, value in grug_base.items()
    }
    model = {
        name: (
            (value + 3e-3 * np.random.default_rng(21).standard_normal(value.shape))
            .astype(ml_dtypes.bfloat16)
            .astype(np.float32)
        )
        for name, value in grug_base.items()
    }
    paths = {
        "base": write({name: value.astype(ml_dtypes.bfloat16) for name, value in grug_base.items()}, str(tmp / "base")),
        "E": write({name: value.astype(ml_dtypes.bfloat16) for name, value in expert.items()}, str(tmp / "expert")),
        "opd75": write({name: value.astype(ml_dtypes.bfloat16) for name, value in model.items()}, str(tmp / "opd75")),
    }
    results = tmp / "results"
    results.mkdir()

    def cosines_vs_expert(arrays: dict[str, np.ndarray]) -> dict[str, float]:
        """The comparison model's cosine against E's delta, per #9920 group, in fp64."""
        result = {}
        for group in ("ALL", *sorted(set(groups.values()))):
            names = [name for name in sorted(grug_base) if group == "ALL" or groups[name] == group]
            left = np.concatenate(
                [(arrays[name].astype(np.float64) - grug_base[name].astype(np.float64)).ravel() for name in names]
            )
            right = np.concatenate(
                [(expert[name].astype(np.float64) - grug_base[name].astype(np.float64)).ravel() for name in names]
            )
            result[group] = float(left @ right / np.linalg.norm(left) / np.linalg.norm(right))
        return result

    entries = {
        "E": {
            "base": "hf://tiny/base",
            "model": paths["E"],
            "refs": [],
            "summary": theirs_summary(grug_base, expert),
            "per_tensor": theirs_per_tensor(grug_base, expert),
        },
        "opd75": {
            "base": "hf://tiny/base",
            "model": paths["opd75"],
            "refs": [paths["E"]],
            "summary": theirs_summary(grug_base, model),
            "per_tensor": theirs_per_tensor(grug_base, model),
        },
    }
    for label, entry in entries.items():
        (results / f"wdelta-{label}.json").write_text(json.dumps(entry))
    all_summaries = {
        "E": {"refs": [], "summary": {}},
        "opd75": {
            "refs": [paths["E"]],
            "summary": {group: {"cos_vs_ref0": value} for group, value in cosines_vs_expert(model).items()},
        },
    }
    (results / "wdelta-all-summaries.json").write_text(json.dumps(all_summaries))
    return {"base": paths["base"], "E": paths["E"], "opd75": paths["opd75"], "results": str(results)}


def load_their(pair: dict, models: dict[str, str]) -> dict[str, TheirResults]:
    return {label: TheirResults.load(label, pair["results"], path) for label, path in models.items()}


def test_reproduce_9920_matches_fabricated_results_on_a_tiny_pair(tiny_pair):
    models = {"E": tiny_pair["E"], "opd75": tiny_pair["opd75"]}
    their = load_their(tiny_pair, models)
    rows, failures, reports = run_pass(
        tiny_pair["base"], models, "E", their, tensor_filter=None, full_stage=True, seed=0, workers=2
    )
    assert failures == []
    summary, summary_failures = summary_rows(models, "E", their, reports)
    assert summary_failures == []
    for row in [*rows, *summary]:
        if row.metric.startswith("published"):
            continue  # the published rows report which convention reproduces #9920; none must
        assert row.within, (row.metric, row.component, row.model, row.ours, row.theirs)
    # The reference model has no comparison rows; the comparison model has the published rows for
    # every convention and component.
    published = [row for row in summary if row.metric.startswith("published")]
    assert {row.model for row in published} == {"opd75"}
    assert {row.metric for row in published} == {
        "published_sign_agreement",
        "published_overlap",
        "published_overlap_random",
        "published_overlap_multiple",
        "published_overlap_stacked",
        "published_overlap_stacked_random",
        "published_overlap_stacked_multiple",
    }
    assert {row.component for row in published if row.metric.endswith("_multiple")} == set(
        ("attention", "expert_mlp", "shared_expert", "router", "lm_head")
    )


def test_reproduce_9920_overlap_conventions_count_matrices_per_expert(tiny_pair):
    models = {"E": tiny_pair["E"], "opd75": tiny_pair["opd75"]}
    their = load_their(tiny_pair, models)
    _rows, _failures, reports = run_pass(
        tiny_pair["base"], models, "E", their, tensor_filter=None, full_stage=True, seed=0, workers=2
    )
    report = reports["opd75"]
    # One matrix per expert slice against one matrix per fused expert tensor.
    assert report.overlap_shares("matrix", "expert_mlp").matrices == LAYERS * 3 * EXPERTS
    assert report.overlap_shares("stacked", "expert_mlp").matrices == LAYERS * 3
    # The stacked matrix carries exactly its experts' delta energy.
    assert report.overlap_shares("stacked", "expert_mlp").total == pytest.approx(
        report.overlap_shares("matrix", "expert_mlp").total, rel=1e-9
    )
    for kind in ("matrix", "stacked"):
        overall = report.overlap_shares(kind, "ALL")
        overlap, null = overall.pooled("weighted")
        assert 0.0 < overlap < 1.0
        assert 0.0 < null < 1.0
        assert 0.0 < overall.pooled("mean")[0] < 1.0


def test_reproduce_9920_tensor_filter_restricts_the_compared_tensors(tiny_pair):
    models = {"E": tiny_pair["E"]}
    their = load_their(tiny_pair, models)
    rows, failures, reports = run_pass(
        tiny_pair["base"], models, "E", their, tensor_filter={"lm_head.weight"}, full_stage=False, seed=0, workers=2
    )
    assert failures == []
    assert {row.component for row in rows} == {"lm_head.weight"}
    assert set(reports["E"].per_tensor) == {"lm_head.weight"}
    with pytest.raises(ValueError, match="names no tensor"):
        run_pass(
            tiny_pair["base"], models, "E", their, tensor_filter={"not.a.tensor"}, full_stage=False, seed=0, workers=2
        )


def test_reproduce_9920_flags_a_corrupted_result_json(tiny_pair):
    models = {"E": tiny_pair["E"]}
    their = load_their(tiny_pair, models)
    their["E"].per_tensor["lm_head.weight"]["d2"] *= 2.0
    rows, failures, _reports = run_pass(
        tiny_pair["base"], models, "E", their, tensor_filter={"lm_head.weight"}, full_stage=False, seed=0, workers=2
    )
    assert failures and all("d2" in failure for failure in failures)
    d2_rows = [row for row in rows if row.metric == "d2"]
    assert len(d2_rows) == 1 and not d2_rows[0].within


def test_reproduce_9920_rejects_a_result_json_that_disagrees_with_the_grouping(tmp_path, tiny_pair):
    entry = json.loads((Path(tiny_pair["results"]) / "wdelta-E.json").read_text())
    entry["per_tensor"]["lm_head.weight"]["group"] = "embeddings"
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "wdelta-E.json").write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="grouping disagrees"):
        TheirResults.load("E", str(broken), entry["model"])


def test_group_9920_places_every_export_name_like_the_comment(grug_base):
    placements = {name: group_9920(name) for name in grug_base}
    assert placements["model.layers.0.self_attn.attn_gate.weight"] == "attention"
    assert placements["model.layers.0.mlp.router.bias"] == "router"
    assert placements["model.layers.0.mlp.router.weight"] == "router"
    assert placements["model.layers.0.attn_gated_norm.up_proj.weight"] == "attention"
    assert placements["model.embed_norm.weight"] == "embeddings"
    assert placements["model.final_gated_norm.down_proj.weight"] == "norms"
    assert set(placements.values()) == {
        "attention",
        "expert_mlp",
        "shared_expert",
        "router",
        "embeddings",
        "norms",
        "lm_head",
    }
    with pytest.raises(ValueError, match="no #9920 group"):
        group_9920("model.layers.0.mlp.router.expert_bias")


def test_parse_model_specs_rejects_malformed_and_duplicate_labels():
    assert parse_model_specs(["E=/a", "m1=/b"]) == {"E": "/a", "m1": "/b"}
    with pytest.raises(ValueError, match="label=path"):
        parse_model_specs(["E/a"])
    with pytest.raises(ValueError, match="twice"):
        parse_model_specs(["E=/a", "E=/b"])


def test_factored_block_functions_match_the_streaming_functions(tmp_path, grug_base):
    """The per-block functions the cross-check script calls must reproduce WP9's aggregates."""
    delta = random_delta(grug_base, seed=13)
    other = random_delta(grug_base, seed=14)
    paths = write_pair(tmp_path, grug_base, {"ckpt": delta, "other": other})

    folded = Keyed(overall=DeltaSummary())
    for base_block, checkpoint_block in paired_blocks(paths["ckpt"], paths["base"]):
        block = block_summary(base_block, checkpoint_block)
        for target in folded.targets(base_block, DeltaSummary):
            target.merge(block)
    expected = summarize_deltas(paired_blocks(paths["ckpt"], paths["base"]))
    assert folded.overall.delta_sq == pytest.approx(expected.overall.delta_sq, rel=1e-12)
    assert folded.overall.base_sq == pytest.approx(expected.overall.base_sq, rel=1e-12)
    assert folded.overall.changed == expected.overall.changed
    assert folded.experts[(0, 0)].delta_sq == pytest.approx(expected.experts[(0, 0)].delta_sq, rel=1e-12)

    def fold_comparisons(left_stream, right_stream):
        comparisons = Keyed(overall=DeltaComparison(PairGeometry()))
        for left, right in zip(left_stream, right_stream, strict=True):
            block = block_comparison(left, right)
            for target in comparisons.targets(left, lambda: DeltaComparison(PairGeometry())):
                target.merge(block)
        return comparisons

    direct = compare_deltas(delta_blocks(paths["ckpt"], paths["base"]), delta_blocks(paths["other"], paths["base"]))
    manual = fold_comparisons(delta_blocks(paths["ckpt"], paths["base"]), delta_blocks(paths["other"], paths["base"]))
    assert manual.overall.cosine == pytest.approx(direct.overall.cosine, rel=1e-12)
    assert manual.overall.sign_agreement == direct.overall.sign_agreement
    assert manual.components[Component.LM_HEAD].cosine == pytest.approx(
        direct.components[Component.LM_HEAD].cosine, rel=1e-12
    )

    records = list(
        matrix_overlaps(delta_blocks(paths["ckpt"], paths["base"]), delta_blocks(paths["other"], paths["base"]), seed=5)
    )
    aggregated = subspace_overlap(
        delta_blocks(paths["ckpt"], paths["base"]), delta_blocks(paths["other"], paths["base"]), seed=5
    )
    manual_overlap = Keyed(overall=SubspaceOverlap())
    for record in records:
        for target in manual_overlap.targets(record_block(record), SubspaceOverlap):
            if record.overlap.zero_reference:
                target.zero_reference_matrices += 1
                continue
            target.delta_energy += record.overlap.total
            target.inside += record.overlap.projected
            target.inside_random += record.overlap.random
            target.inside_permuted += record.overlap.permuted
            target.matrices += 1
    assert manual_overlap.overall.overlap == pytest.approx(aggregated.overall.overlap, rel=1e-12)
    assert manual_overlap.overall.random_control == pytest.approx(aggregated.overall.random_control, rel=1e-12)
    assert manual_overlap.overall.matrices == aggregated.overall.matrices
