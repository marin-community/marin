# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Two train steps of tiny fast_track models on CPU; prints the losses (repr) for a before/after comparison.
#
#   XLA_FLAGS=--xla_force_host_platform_device_count=2 python scratchpad/clean/loss2.py
import dataclasses
import hashlib
import sys
import types

import jax
import jax.numpy as jnp
import jmp
import numpy as np

_quack = types.ModuleType("levanter.grug._moe.quack_symmetric_cute")
_quack.quack_symmetric_gemm = lambda x: jnp.einsum("...ik,...jk->...ij", x, x)
sys.modules["levanter.grug._moe.quack_symmetric_cute"] = _quack

from haliax.partitioning import set_mesh  # noqa: E402
from levanter.callbacks.watch import WatchConfig  # noqa: E402
from levanter.data.text.examples import GrugLmExample  # noqa: E402
from levanter.grug.attention import AttentionMask  # noqa: E402
from levanter.grug.sharding import compact_grug_mesh  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402

from experiments.grug.fast_track import belay_ckpt1, launch  # noqa: E402
from experiments.grug.fast_track import train as ft_train  # noqa: E402
from experiments.grug.fast_track.heuristic import MoeHeuristic  # noqa: E402
from experiments.grug.fast_track.model import GrugModelConfig  # noqa: E402

SEQ, VOCAB, BATCH = 256, 128, 4

TINY = dict(
    vocab_size=VOCAB,
    hidden_dim=64,
    intermediate_dim=32,
    shared_expert_intermediate_dim=32,
    num_experts=8,
    num_experts_per_token=2,
    latent_dim=32,
    num_layers=6,
    num_heads=2,
    num_kv_heads=1,
    local_kv_heads=1,
    global_kv_heads=1,
    head_dim=32,
    max_seq_len=SEQ,
    sliding_window=128,
    global_every=4,
    initializer_std=0.5 / 8,
    rel_extent=128,
    mla_kv_latent_dim=32,
)


def _train_step_fn(built):
    """``_build_train_step``'s train step: it returned ``(step, diagnostic_watch_step)`` before the cleanup."""
    return built[0] if isinstance(built, tuple) else built


def tiny_configs() -> dict[str, tuple[GrugModelConfig, dict[str, str]]]:
    base = GrugModelConfig(**TINY)
    kma = launch._kma_model(base)
    kma = dataclasses.replace(kma, attn_res_num_blocks=3, rel_extent=128)
    settings = dict(s.split("=", 1) for s in belay_ckpt1.MODEL_SETTINGS)
    settings.update(num_experts="8", num_experts_per_token="2", num_shared_experts="1")
    # XLA:CPU cannot run ragged-all-to-all; the HLO hashes cover the ragged path.
    settings["moe_implementation"] = "fixed_pooled_wave_all_to_all"
    belay = launch._apply_settings(kma, settings)
    opt = dict(s.split("=", 1) for s in belay_ckpt1.OPTIMIZER_SETTINGS)
    return {"baseline": (base, {}), "belay": (belay, opt), "kma": (kma, {})}


def two_losses(model_cfg: GrugModelConfig, opt_settings: dict[str, str]) -> list[float]:
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    optimizer_cfg = MoeHeuristic().build_optimizer_config(
        num_train_steps=10, batch_size=BATCH, hidden_dim=model_cfg.hidden_dim, seq_len=SEQ
    )
    optimizer_cfg = launch._apply_settings(optimizer_cfg, opt_settings)
    trainer = TrainerConfig(
        num_train_steps=10, train_batch_size=BATCH, mp=mp, watch=WatchConfig(interval=1), use_explicit_mesh_axes=True
    )
    config = ft_train.GrugRunConfig(
        model=model_cfg,
        data=None,
        resources=None,
        optimizer=optimizer_cfg,
        trainer=ft_train.GrugTrainerConfig(trainer=trainer, expert_axis_size=2, replica_axis_size=1),
    )
    optimizer = optimizer_cfg.build(10)
    step_fn = _train_step_fn(ft_train._build_train_step(config, optimizer))
    mesh = compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)
    rng = np.random.default_rng(0)
    losses = []
    with set_mesh(mesh):
        state = jax.jit(lambda k: ft_train.initial_state(model_cfg, optimizer=optimizer, mp=mp, key=k))(
            jax.random.PRNGKey(0)
        )
        leaves, treedef = jax.tree.flatten(state)
        step = ft_train._flat_train_step(step_fn, treedef)
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("replica_dcn", "data", "expert"), None))
        for _ in range(2):
            tokens = jax.device_put(jnp.asarray(rng.integers(0, VOCAB, (BATCH, SEQ)), jnp.int32), sharding)
            seg = jax.device_put(
                jnp.asarray(np.repeat(np.arange(2), SEQ // 2)[None].repeat(BATCH, 0), jnp.int32), sharding
            )
            weight = jax.device_put(jnp.ones((BATCH, SEQ), jnp.float32), sharding)
            batch = GrugLmExample(
                tokens=tokens, loss_weight=weight, attn_mask=AttentionMask.causal().with_segment_ids(seg)
            )
            leaves, metrics, _ = step(leaves, batch)
            losses.append(float(metrics["train/loss"]))
        digest = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in leaves)).hexdigest()[:16]
    return [*losses, digest]


def main() -> None:
    names = sys.argv[1:] or ["baseline", "kma", "belay"]
    configs = tiny_configs()
    for name in names:
        cfg, opt = configs[name]
        *losses, digest = two_losses(cfg, opt)
        print(name, [repr(x) for x in losses], "state", digest, flush=True)


if __name__ == "__main__":
    main()
