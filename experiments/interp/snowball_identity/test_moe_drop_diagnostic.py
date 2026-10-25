# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import subprocess
import sys

import jax
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballLMHeadModel

from experiments.interp.snowball_identity import moe_drop_diagnostic as diag
from experiments.interp.snowball_identity.levanter_positions import PAD_ID
from experiments.interp.snowball_identity.test_levanter_positions import CONFIG, SEQUENCES

# One process, eight CPU devices: data 2 x expert 4, Snowball's batch axes. The backend reports its own drops.
RING_CHILD = """
import json, sys
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import PartitionSpec as P, reshard
from levanter.grug.grug_moe import MoEExpertMlp, MoEExpertMlpPspecs
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import _BATCH_AXES, _FSDP_AXES

E, K, D, B, S = 16, 4, 8, 8, 6
implementation, factor = sys.argv[3], float(sys.argv[4])
mesh = compact_grug_mesh(expert_axis_size=4)
assert jax.device_count() == 8 and dict(mesh.shape)["expert"] == 4
rng = np.random.default_rng(0)
# Skewed routes: expert 0 (shard 0) is picked far more than average, as identical pad tokens do.
routes = np.stack([rng.choice(E, K, replace=False, p=np.r_[0.4, np.full(E - 1, 0.6 / (E - 1))]) for _ in range(B * S)])
with jax.set_mesh(mesh):
    mlp = MoEExpertMlp.init(num_experts=E, hidden_dim=D, intermediate_dim=4, initializer_std=0.1,
        key=jax.random.key(0), implementation=implementation, capacity_factor=factor,
        pspecs=MoEExpertMlpPspecs(hidden=_FSDP_AXES))
    x = reshard(jnp.ones((B * S, D), jnp.float32), P(_BATCH_AXES))
    ids = reshard(jnp.asarray(routes, jnp.int32), P(_BATCH_AXES))
    w = reshard(jnp.full((B * S, K), 0.25, jnp.float32), P(_BATCH_AXES))
    _, counts = mlp(x, ids, w, mesh=jax.sharding.get_abstract_mesh(), report_capacity_overflow=True)
print(json.dumps({"dropped": int(counts.dropped), "routes": routes.reshape(B, S, K).tolist()}))
"""


@pytest.mark.timeout(600)
@pytest.mark.parametrize("implementation", ["ring", "ring_gather_combine"])
@pytest.mark.parametrize("factor", [1.0, 4.0])
def test_ring_drops_matches_the_backend_count(implementation, factor):
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    done = subprocess.run(
        [sys.executable, "-c", RING_CHILD, "", "", implementation, str(factor)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    expected = diag.ring_drops(np.asarray(result["routes"]), 16, 4, factor)
    assert result["dropped"] == expected
    assert (expected > 0) if factor == 1.0 else (expected == 0)


def test_ring_drops_is_zero_without_an_expert_axis():
    routes = np.zeros((8, 4, 4), dtype=np.int64)
    assert diag.ring_drops(routes, 16, 1, 1.0) == 0
    assert diag.ring_drops(routes, 16, 4, 1.0) > 0


def test_with_capacity_changes_only_the_capacity_field():
    config = dataclasses.replace(CONFIG, num_layers=2)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(3))
    changed, change = diag.with_capacity(model, 4.0)
    assert {b.mlp.expert_mlp.capacity_factor for b in changed.transformer.blocks} == {4.0}
    assert change["old"] == [1.0] and change["new"] == 4.0
    assert jax.tree.structure(changed) != jax.tree.structure(model)  # static fields are part of the treedef
    for ours, theirs in zip(jax.tree.leaves(changed), jax.tree.leaves(model), strict=True):
        assert ours is theirs
    json.dumps(change)


def test_gate_batches_follow_positions_records():
    """The drop pass must see the grids ``positions_records`` scores: sorted by length, short steps filled with
    copies of their first sequence, right-padded with PAD_ID to the multiple."""
    grids = list(diag.gate_batches(SEQUENCES, 2, 4))
    assert [g.shape for g in grids] == [(2, 12), (2, 20)]
    assert grids[0][0].tolist() == list(SEQUENCES[2].tokens) + [PAD_ID] * 7
    assert grids[0][1].tolist() == list(SEQUENCES[0].tokens) + [PAD_ID] * 2
    assert grids[1][0].tolist() == list(SEQUENCES[1].tokens) == grids[1][1].tolist()


def test_top_margins():
    logits = np.array([0.0, 3.0, 1.0, 2.0, 2.5, -1.0])
    found = diag.top_margins(logits, np.zeros(6), k=2)
    assert found["ids"] == [1, 4, 3] and found["margin_k_vs_next"] == pytest.approx(0.5)


def test_compare_dumps_skips_the_bias_entry_and_reports_each_op():
    rng = np.random.default_rng(1)
    names = ["embed"] + [f"{op}/{layer}" for layer in range(diag.DUMP_LAYERS) for op in diag.OPS]

    def dumps(noise: float) -> dict:
        record = {name: rng.normal(size=(6, 8 if not name.startswith("router") else 16)) for name in names}
        record |= {f"routes/{layer}": np.tile(np.arange(4), (6, 1)) for layer in range(diag.DUMP_LAYERS)}
        return {"check|a": {k: v + noise if k.startswith("attn_out/0") else v for k, v in record.items()}}

    reference = dumps(0.0)
    ours = {"check|a": {k: v.copy() for k, v in reference["check|a"].items()}}
    ours["check|a"]["attn_out/0"] = ours["check|a"]["attn_out/0"] + 0.5
    reference["_bias"] = {layer: np.zeros(16) for layer in range(diag.DUMP_LAYERS)}
    found = diag.compare_dumps(ours, reference, reference["_bias"])
    assert found["ops"]["embed"]["max_abs"] == 0 and found["ops"]["L0/attn_norm"]["max_abs"] == 0
    assert found["ops"]["L0/attn_out"]["max_abs"] == pytest.approx(0.5)
    assert found["routes"]["L0"] == {"positions": 6, "set_agree": 6, "ordered_agree": 6}
    assert set(found["layer0_position4"]) == {"check|a"}
