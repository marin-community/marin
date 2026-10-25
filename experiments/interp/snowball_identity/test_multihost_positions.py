# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The positions writer scores a batch that spans processes, as the gate does on four H100 hosts.

Four single-device CPU processes form one global mesh, replica_dcn 2 x context 2, so both the batch and the position
axes span processes (on the H100 job the batch is 8 on replica_dcn x data x expert and the position is on context
across hosts). The records must equal the single-process records of the same model.
"""

import json
import os
import socket
import subprocess
import sys

import jax
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballLMHeadModel

from experiments.interp.snowball_identity.levanter_positions import positions_records
from experiments.interp.snowball_identity.test_levanter_positions import CONFIG, SEQUENCES

CHILD = """
import json, sys
import jax
jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(f"localhost:{sys.argv[2]}", num_processes=4, process_id=int(sys.argv[1]))
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballLMHeadModel
from experiments.interp.snowball_identity.levanter_positions import positions_records
from experiments.interp.snowball_identity.test_levanter_positions import CONFIG, SEQUENCES
from experiments.interp.snowball_identity.test_multihost_positions import summary

mesh = compact_grug_mesh(replica_axis_size=2, context_axis_size=2)
assert dict(mesh.shape)["replica_dcn"] == 2 and dict(mesh.shape)["context"] == 2
# Each process holds one device: both sharded axes cross processes.
assert jax.local_device_count() == 1
with jax.set_mesh(mesh):
    model = SnowballLMHeadModel.init(Axis("vocab", CONFIG.vocab_size), CONFIG, key=jax.random.key(3))
    records = positions_records(model, SEQUENCES, batch=2, pad_multiple=4, pad_id=63, routes=True)
print(json.dumps(summary(records)))
"""


TRAIN_CHILD = """
import json, sys
import jax
jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(f"localhost:{sys.argv[2]}", num_processes=4, process_id=int(sys.argv[1]))
import equinox as eqx
import haliax as hax
import jax.numpy as jnp
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig


class TinyModel(eqx.Module):
    weight: jax.Array


def loss(model, example, *, key=None):
    return jnp.square(model.weight - jnp.mean(example.tokens.array)), {}


config = TrainerConfig(
    id="loss-item", tracker=(), require_accelerator=False, mesh=MeshConfig(), train_batch_size=4,
    per_device_parallelism=1, load_checkpoint=False,
)
with Trainer(config, AdamConfig(learning_rate=1e-2, warmup=0).build(1), loss, add_default_hooks=False) as trainer:
    state = trainer.initial_state(jax.random.PRNGKey(0), model=TinyModel(jnp.asarray(0.5, dtype=jnp.float32)))
    axes = (hax.Axis("batch", 4), hax.Axis("position", 2))
    example = LmExample(hax.named(jnp.ones((4, 2), dtype=jnp.int32), axes), hax.ones(axes))
    info = trainer.train_step(state, example)
    print(json.dumps({"loss_type": type(info.loss).__name__, "loss": info.loss, "step": int(info.state.step)}))
"""


def run_children(child: str, processes: int) -> list[dict]:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = str(s.getsockname()[1])
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", child, str(rank), port], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for rank in range(processes)
    ]
    outputs = [proc.communicate(timeout=600) for proc in procs]
    for proc, (_, err) in zip(procs, outputs, strict=True):
        assert proc.returncode == 0, err.decode()[-3000:]
    return [json.loads(out.decode().strip().splitlines()[-1]) for out, _ in outputs]


def summary(records: dict) -> dict:
    return {
        key: {
            "actual": [None if np.isnan(v) else float(v) for v in record["actual"].tolist()],
            "top_ids": record["top_ids"][:, :5].tolist(),
            "top_lp": [[float(v) for v in row] for row in record["top_lp"][1:, :5].tolist()],
            "routes": record["routes"].tolist(),
        }
        for key, record in records.items()
    }


# Four JAX processes each compile the tiny model; well over the 60 s default.
@pytest.mark.timeout(600)
def test_positions_records_when_batch_and_position_span_processes():
    gathered = run_children(CHILD, 4)
    assert all(result == gathered[0] for result in gathered)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", CONFIG.vocab_size), CONFIG, key=jax.random.key(3))
        expected = summary(positions_records(model, SEQUENCES, batch=1, pad_multiple=1, pad_id=63, routes=True))
    assert gathered[0].keys() == expected.keys()
    for key, record in expected.items():
        actual = gathered[0][key]
        assert actual["top_ids"] == record["top_ids"] and actual["routes"] == record["routes"], key
        np.testing.assert_allclose(
            np.array(actual["actual"], dtype=float), np.array(record["actual"], dtype=float), atol=1e-5
        )
        np.testing.assert_allclose(actual["top_lp"], record["top_lp"], atol=1e-5)


@pytest.mark.timeout(600)
def test_train_step_returns_a_host_float_loss_across_processes():
    """Levanter's train_step takes ``.item()`` of a replicated global scalar; the LR0 and SFT loops rely on it."""
    results = run_children(TRAIN_CHILD, 4)
    assert all(result == results[0] for result in results)
    assert results[0]["loss_type"] == "float" and results[0]["step"] == 1
    assert np.isclose(results[0]["loss"], 0.25)
