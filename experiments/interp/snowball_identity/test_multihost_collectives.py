# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The gate, positions and LR0 witness collectives work across two CPU processes in a global mesh, as in the trainer.

Under ``jax.set_mesh`` a ``jnp`` scalar is a global, non-fully-addressable array, and an untiled
``process_allgather`` of it raises. One process alone cannot show this: every array is addressable there.
"""

import json
import os
import socket
import subprocess
import sys

CHILD = """
import json, sys, types
import jax
jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(f"localhost:{sys.argv[2]}", num_processes=2, process_id=int(sys.argv[1]))
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec, reshard
from experiments.interp.snowball_identity.multihost import gather_global
from experiments.interp.snowball_identity.precision_gate import any_rank_failed, rank_zero_count
from experiments.post_training.identity_context_distill.precision_probe import adam_gradient_witness

rank = jax.process_index()
mesh = Mesh(np.array(jax.devices()), ("data",))
with jax.set_mesh(mesh):
    assert not jnp.int32(0).is_fully_addressable
    mu = jax.device_put(jnp.arange(8.0) - 7.0, NamedSharding(mesh, PartitionSpec("data")))
    state = types.SimpleNamespace(opt_state=optax.ScaleByAdamState(count=jnp.int32(1), mu={"w": mu}, nu={"w": mu}))
    replicated = jax.jit(lambda x: reshard(x * 2, PartitionSpec()))(mu)
    assert not replicated.is_fully_addressable
    print(json.dumps({
        "rank": rank,
        "one_failed": any_rank_failed(rank == 1),
        "none_failed": any_rank_failed(False),
        "count": rank_zero_count(3 if rank == 0 else 0),
        "witness": adam_gradient_witness(state),
        "scores": gather_global(replicated).tolist(),
        "sharded": gather_global(mu).tolist(),
    }))
"""


SHARDED = [float(i - 7) for i in range(8)]
SCORES = [2 * v for v in SHARDED]


def test_gate_and_witness_collectives_under_a_two_process_global_mesh():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = str(s.getsockname()[1])
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    children = [
        subprocess.Popen(
            [sys.executable, "-c", CHILD, str(rank), port], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for rank in range(2)
    ]
    outputs = [child.communicate(timeout=300) for child in children]
    for child, (_, err) in zip(children, outputs, strict=True):
        assert child.returncode == 0, err.decode()[-3000:]
    results = sorted((json.loads(out.decode().strip().splitlines()[-1]) for out, _ in outputs), key=lambda r: r["rank"])
    for rank, result in enumerate(results):
        expected = {"rank": rank, "one_failed": True, "none_failed": False, "count": 3, "witness": 7.0}
        assert result == {**expected, "scores": SCORES, "sharded": SHARDED}
