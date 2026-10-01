# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short-training test for a systematic loss bias from final-seq's value-changing components.

Trains C's model smoke (EP4: d1024, 4 layers, 16 experts top-4, 2 shared experts, short convs, QB, FA4
attention, the carry-offload remat policy) with the hero's train step, MuonH heuristic and mixed precision,
from one initialization per seed on a fixed synthetic Markov-chain token stream (learnable, so the loss
moves). Per seed it runs, on identical batches:

  main (x2):  main's routed MoE module, the pre-E QuACK backward, the Pallas short conv. The two runs
              differ only by run-to-run nondeterminism (the FA4 backward), which calibrates same-code noise.
  final:      the live module (D's expert-side dS, mirror, forward order), E (SwiGLU backward in the dh
              epilogue), the Triton short conv.
  e_off:      final with the pre-E backward.
  d_only:     main with D's expert-side dS (the frozen `sonic` module).

Per run it records the training loss per step and an end-of-run held-out loss. The summary compares, per
seed, each variant's late-window mean loss with the mean of the two main runs, against the same quantity
for main_b vs main_a, and reports the across-seed mean, standard error and sign count.

Usage (GB200x4, hero env, from a final-seq checkout with this directory):
  python autoresearch/loop-260930-mfu30/b/bias_test.py [--seeds 8] [--steps 300]
"""

import argparse
import dataclasses
import importlib.util
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, "autoresearch/loop-260930-mfu30/stack")
sys.path.insert(0, "autoresearch/loop-260930-mfu30/b")
import jax
import jmp
import levanter.grug._moe.sonic_cute as sonic_cute
import levanter.grug.grug_moe as grug_moe
import model_smoke as ms
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from unfused_backward import unfused_backward

from experiments.grug.moe_hero_ep import train as hero_train
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic

HERE = pathlib.Path(__file__).resolve().parent
BATCH_AXES = ("replica_dcn", "data", "expert")
MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
Z_LOSS_WEIGHT = 1e-4
CHAIN_SEED = 1234
SUCCESSORS = 16
EVAL_BATCHES = 4
RUNS = ("main", "main", "final", "e_off", "d_only")


def _frozen(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._moe_mlp_ep_ragged_a2a_local


LIVE_LOCAL = grug_moe._moe_mlp_ep_ragged_a2a_local
FUSED_BACKWARD = sonic_cute._expert_mlp_quack_wgrad_backward
# name -> (routed MoE shard-local function, QuACK backward, short-conv implementation)
VARIANTS = {
    "main": (_frozen("control_ep_ragged_all_to_all"), unfused_backward, "pallas_gpu"),
    "final": (LIVE_LOCAL, FUSED_BACKWARD, "triton_gpu"),
    "e_off": (LIVE_LOCAL, unfused_backward, "triton_gpu"),
    "d_only": (_frozen("sonic_ep_ragged_all_to_all"), unfused_backward, "pallas_gpu"),
}


def _chain(vocab):
    """A fixed sparse Markov chain: each token has SUCCESSORS successors with Zipf weights."""
    rng = np.random.default_rng(CHAIN_SEED)
    successors = rng.integers(0, vocab, size=(vocab, SUCCESSORS))
    weights = 1.0 / np.arange(1, SUCCESSORS + 1)
    return successors, np.cumsum(weights / weights.sum())


def _batches(seed, count, batch, seq, chain):
    successors, cdf = chain
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        tokens = np.empty((batch, seq), np.int32)
        tokens[:, 0] = rng.integers(0, successors.shape[0], size=batch)
        for t in range(1, seq):
            pick = np.searchsorted(cdf, rng.random(batch))
            tokens[:, t] = successors[tokens[:, t - 1], pick]
        out.append(tokens)
    return out


def _example(tokens, sharding):
    weight = np.ones(tokens.shape, np.float32)
    weight[:, -1] = 0
    return GrugLmExample(tokens=jax.device_put(tokens, sharding), loss_weight=jax.device_put(weight, sharding))


class Variant:
    """One variant's compiled train and eval steps (traced with its module patches in place)."""

    def __init__(self, name, steps, mesh):
        local_fn, backward, sconv = VARIANTS[name]
        self.name, self.local_fn, self.backward = name, local_fn, backward
        self.cfg = dataclasses.replace(ms.CFG, sconv_implementation=sconv)
        self.mp = jmp.get_policy(MIXED_PRECISION)
        self.optimizer = (
            MoeHeuristic()
            .build_optimizer_config(
                num_train_steps=steps, batch_size=ms.BATCH, hidden_dim=self.cfg.hidden_dim, seq_len=ms.SEQ
            )
            .build(steps)
        )
        self.train_step = hero_train._make_train_step(
            self.optimizer, self.mp, z_loss_weight=Z_LOSS_WEIGHT, ema_beta=None
        )
        self.mesh = mesh

        @jax.jit
        def eval_loss(params, batch):
            compute = self.mp.cast_to_compute(params)
            return compute.next_token_loss(batch.tokens, batch.loss_weight, reduction="mean")

        self.eval_loss = eval_loss

    def run(self, key, train_batches, eval_batches):
        grug_moe._moe_mlp_ep_ragged_a2a_local = self.local_fn
        sonic_cute._expert_mlp_quack_wgrad_backward = self.backward
        try:
            state = hero_train.initial_state(self.cfg, optimizer=self.optimizer, mp=self.mp, key=key, ema_beta=None)
            losses = []
            for batch in train_batches:
                state, metrics, _ = self.train_step(state, batch)
                losses.append(metrics["train/loss"])
            losses = [float(x) for x in jax.device_get(losses)]
            params = hero_train._apply_qb_betas(state.params, state.pending_qb_betas)
            held_out = float(np.mean([float(self.eval_loss(params, b)) for b in eval_batches]))
        finally:
            grug_moe._moe_mlp_ep_ragged_a2a_local = LIVE_LOCAL
            sonic_cute._expert_mlp_quack_wgrad_backward = FUSED_BACKWARD
        return losses, held_out


def _summary(records, steps):
    late = slice(2 * steps // 3, steps)
    by_seed = {}
    for r in records:
        by_seed.setdefault(r["seed"], []).append(r)
    out = {}
    for variant in ("main_b", "final", "e_off", "d_only"):
        late_diffs, held_diffs = [], []
        for runs in by_seed.values():
            mains = [r for r in runs if r["variant"] == "main"]
            if variant == "main_b":
                ref_late = np.mean(mains[0]["losses"][late])
                late_diffs.append(float(np.mean(mains[1]["losses"][late]) - ref_late))
                held_diffs.append(float(mains[1]["held_out"] - mains[0]["held_out"]))
                continue
            ref_late = np.mean([np.mean(m["losses"][late]) for m in mains])
            ref_held = np.mean([m["held_out"] for m in mains])
            run = next(r for r in runs if r["variant"] == variant)
            late_diffs.append(float(np.mean(run["losses"][late]) - ref_late))
            held_diffs.append(float(run["held_out"] - ref_held))

        def stats(xs):
            mean = statistics.mean(xs)
            se = statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")
            return dict(
                mean=mean, se=se, t=mean / se if se else float("nan"), positive=int(sum(x > 0 for x in xs)), n=len(xs)
            )

        out[variant] = dict(late=stats(late_diffs), held_out=stats(held_diffs), late_per_seed=late_diffs)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    devices = np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1)
    mesh = Mesh(devices, ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
    sharding = NamedSharding(mesh, P(BATCH_AXES, None))
    chain = _chain(ms.CFG.vocab_size)
    records = []
    with jax.set_mesh(mesh):
        variants = {name: Variant(name, args.steps, mesh) for name in VARIANTS}
        for seed in range(args.seeds):
            raw = _batches(seed, args.steps + EVAL_BATCHES, ms.BATCH, ms.SEQ, chain)
            train_batches = [_example(t, sharding) for t in raw[: args.steps]]
            eval_batches = [_example(t, sharding) for t in raw[args.steps :]]
            for run_index, name in enumerate(RUNS):
                start = time.perf_counter()
                losses, held_out = variants[name].run(jax.random.key(seed), train_batches, eval_batches)
                record = dict(seed=seed, run=run_index, variant=name, losses=losses, held_out=held_out)
                records.append(record)
                print(
                    "RUN "
                    + json.dumps(
                        dict(
                            seed=seed,
                            run=run_index,
                            variant=name,
                            first=losses[0],
                            last=losses[-1],
                            late_mean=float(np.mean(losses[2 * args.steps // 3 :])),
                            held_out=held_out,
                            seconds=round(time.perf_counter() - start, 1),
                        )
                    ),
                    flush=True,
                )
                print("LOSSES " + json.dumps(record), flush=True)
            print("SUMMARY " + json.dumps(_summary(records, args.steps)), flush=True)


if __name__ == "__main__":
    main()
