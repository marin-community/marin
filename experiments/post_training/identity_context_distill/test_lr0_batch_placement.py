# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The LR0 probe's host-built batch reaches the fused-loss ``train_step`` as the data loader would place it.

lr0-r6 passed its gate, then every rank raised in the fused cross-entropy ``shard_map``: the synthetic batch was a host
(replicated) array, ``P(None, None)``, while the trainer's explicit mesh expects ``P((replica_dcn, data, expert),
context)``. Four CPU processes with four devices each stand in for the H100 job: each process is one slice, and
replica_dcn, data, expert and context are all 2, with context across processes as on the four-host job.
"""

import json
import os
import socket
import subprocess
import sys

import pytest

CHILD = """
import json, sys
import jax
jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(f"localhost:{sys.argv[2]}", num_processes=4, process_id=int(sys.argv[1]))
import haliax as hax
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh
from levanter import trainer as levanter_trainer
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballLMHeadModel
from levanter.optim.config import AdamConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.interp.snowball_identity.test_levanter_positions import CONFIG


def per_process_slices(*, ici_axes, dcn_axes, axis_types=None, **_):
    # CPU devices carry no slice_index; treat each process as one slice, as each H100 host is on the job.
    names = list(ici_axes) + [name for name in dcn_axes if name not in ici_axes]
    dcn = [dcn_axes.get(name, 1) for name in names]
    ici = [ici_axes.get(name, 1) for name in names]
    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    grid = np.array(devices, dtype=object).reshape(*dcn, *ici)
    grid = grid.transpose([i for pair in zip(range(len(names)), range(len(names), 2 * len(names))) for i in pair])
    return Mesh(grid.reshape([d * i for d, i in zip(dcn, ici)]), tuple(names), axis_types=axis_types)


TrainerConfig.num_slices = property(lambda self: jax.process_count())
levanter_trainer.create_mesh_from_axis_specs = per_process_slices

mesh = MeshConfig(
    axes={"data": 2, "expert": 2, "replica": 1, "model": 1},
    dcn_axes={"replica_dcn": 2, "context": 2},
    compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
    param_mapping={"embed": ["data", "context"]},
)
config = TrainerConfig(
    id="lr0-placement", tracker=(), require_accelerator=False, mesh=mesh, use_explicit_mesh_axes=True,
    mp=jmp.get_policy("p=f32,c=bfloat16,o=f32"), train_batch_size=8, per_device_parallelism=1, load_checkpoint=False,
)


def loss(model, example, *, key=None):
    return model.compute_next_token_loss(example, key=key, logsumexp_weight=1e-4)


with Trainer(
    config, AdamConfig(learning_rate=0.0, warmup=0, weight_decay=0.0).build(1), loss, add_default_hooks=False
) as trainer:
    shape = dict(trainer.device_mesh.shape)
    assert all(shape[name] == 2 for name in ("replica_dcn", "data", "expert", "context")), shape
    model = SnowballLMHeadModel.init(hax.Axis("vocab", CONFIG.vocab_size), CONFIG, key=jax.random.key(3))
    state = trainer.initial_state(jax.random.PRNGKey(0), model=model)
    axes = (hax.Axis("batch", 8), hax.Axis("position", 16))
    tokens = (np.arange(8 * 16, dtype=np.int32).reshape(8, 16) * 7) % CONFIG.vocab_size
    weight = np.zeros((8, 16), dtype=np.float32)
    weight[:, 4:12] = 1.0
    # The probe's synthetic batch: host numpy, identical on every process.
    batch = LmExample(hax.named(tokens, axes), hax.named(weight, axes))
    if sys.argv[3] == "placed":
        from experiments.post_training.identity_context_distill.precision_probe import place_like_data_loader

        batch = place_like_data_loader(batch, trainer)
        spec = batch.tokens.array.sharding.spec
    else:
        spec = None
    info = trainer.train_step(state, batch)
    print(json.dumps({"spec": str(spec), "loss": float(info.loss), "step": int(info.state.step)}))
"""


def run(mode: str) -> list[tuple[int, str, str]]:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = str(s.getsockname()[1])
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", CHILD, str(rank), port, mode], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for rank in range(4)
    ]
    outputs = [proc.communicate(timeout=600) for proc in procs]
    return [(proc.returncode, out.decode(), err.decode()) for proc, (out, err) in zip(procs, outputs, strict=True)]


# Four JAX processes each compile a tiny Snowball train step; well over the 60 s default.
@pytest.mark.timeout(900)
def test_host_batch_fails_the_fused_loss_shard_map_as_on_lr0_r6():
    results = run("host")
    assert all(code != 0 for code, _, _ in results)
    assert any("in_specs passed to shard_map" in err and "P(None, None)" in err for _, _, err in results)


@pytest.mark.timeout(900)
def test_placed_batch_completes_one_fused_loss_train_step():
    results = run("placed")
    for code, _, err in results:
        assert code == 0, err[-3000:]
    reports = [json.loads(out.strip().splitlines()[-1]) for _, out, _ in results]
    assert all(report == reports[0] for report in reports)
    assert reports[0]["spec"] == "P(('replica_dcn', 'data', 'expert'), 'context')"
    assert reports[0]["step"] == 1 and reports[0]["loss"] > 0
