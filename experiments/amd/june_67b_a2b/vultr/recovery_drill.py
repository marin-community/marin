# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recovery drill for the AMD June trainer on Vultr MI355X: train on the Snowball-mix flat cache with checkpoints on
``/work``, inject faults once each, and let ``recovery_drill.sbatch`` restart the job from its latest checkpoint.

Faults, each fired at most once per run (a marker file under the run's checkpoint directory records it, so a restart
does not fire it again):

* ``--damage-inputs-at STEP``: overwrite 16 token ids of that step's batch with float32 bit patterns, as the MI355X
  stray writes do. The step's input check fails and the step is skipped.
* ``--nan-at STEP``: make the optimizer moments non-finite before that step. Its update makes the parameters
  non-finite, the trainer stops before the state can be checkpointed, and the restart resumes from the last checkpoint.
* ``--abort-at STEP``: ``os.abort()`` (exit 134, as the attention kernel's check aborts) in process
  ``--abort-process`` after that step is dispatched.

Every skipped step is reported with the input arrays that failed and their first damaged values, read from the
batch, which the drill keeps for two steps. Launch one process per node (``--gpus-per-node 8``).
"""

import argparse
import collections
import dataclasses
import functools
import logging
import os
from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from rigging.log_setup import configure_logging

from experiments.amd.june_67b_a2b import rehearsal, train
from experiments.amd.june_67b_a2b.run_config import (
    add_run_arguments,
    build_model,
    global_batch_size,
    initialize_distributed,
)
from experiments.amd.june_67b_a2b.train import GrugRunConfig, GrugTrainerConfig, _run_grug_local

logger = logging.getLogger(__name__)

FLOAT_BITS = int(np.float32(0.4375).view(np.int32))
"""A float32 read as an int32 token id: out of any vocabulary's range, as the stray writes' values are."""
LR_HORIZON = 1000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", required=True, help="Directory holding the flat Snowball-mix caches.")
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--output-root", help="Checkpoints go to <output-root>/<run id>; unset, none are written.")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/drill"))
    parser.add_argument("--diagnose-numerics", action="store_true")
    parser.add_argument("--damage-inputs-at", type=int, nargs="*", default=[])
    parser.add_argument("--nan-at", type=int)
    parser.add_argument("--abort-at", type=int)
    parser.add_argument("--abort-process", type=int, default=0)
    return parser.parse_args(argv)


def _optimizer(batch_size: int, hidden_dim: int, seq_len: int):
    base = rehearsal._HEURISTIC.build_muonh_config(
        batch_size, float(LR_HORIZON * batch_size * seq_len), hidden_dim, seq_len=seq_len
    )
    return dataclasses.replace(base, schedule_num_train_steps_override=LR_HORIZON)


def build_run_config(args: argparse.Namespace) -> GrugRunConfig:
    model, default_mp = build_model(args)
    batch_size = global_batch_size(args)
    checkpointer = CheckpointerConfig(
        base_path=os.path.join(args.output_root or str(args.log_dir), args.run_id, "checkpoints"),
        temporary_base_path=os.path.join(args.output_root or str(args.log_dir), args.run_id, "checkpoints-temp"),
        append_run_id_to_base_path=False,
        save_interval=timedelta(days=1),
        keep=[{"every": args.checkpoint_every or args.steps}],
    )
    trainer = TrainerConfig(
        id=args.run_id,
        seed=rehearsal.SEED,
        train_batch_size=batch_size,
        num_train_steps=args.steps,
        mp=jmp.get_policy(args.mp or default_mp),
        tracker=JsonLoggerConfig(),
        log_dir=args.log_dir,
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        checkpointer=checkpointer,
    )
    return GrugRunConfig(
        model=model,
        data=rehearsal.build_data(
            train_cache=rehearsal.PHASE_0_CACHE, data_root=args.data_root, validation_root=args.validation_root
        ),
        resources=ResourceConfig(),
        optimizer=_optimizer(batch_size, model.hidden_dim, model.max_seq_len),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            expert_axis_size=args.expert_axis,
            replica_axis_size=args.replica_axis,
            microbatches=args.microbatches,
            zero1=args.zero1,
            save_checkpoints=args.output_root is not None,
            diagnose_numerics=args.diagnose_numerics,
            step_timeout=args.step_timeout or None,
        ),
        eval=None,
    )


class _Faults:
    """The faults still to fire in this run; a marker file per fault records that it fired."""

    def __init__(self, marker_dir: Path | None, args: argparse.Namespace):
        self.marker_dir = marker_dir
        planned = [("damage", s) for s in args.damage_inputs_at]
        planned += [("nan", args.nan_at)] if args.nan_at is not None else []
        if args.abort_at is not None and jax.process_index() == args.abort_process:
            planned.append(("abort", args.abort_at))
        # Read once at start, before any process can write a marker in this attempt.
        self.pending = {(kind, step) for kind, step in planned if not self._marker(kind, step).exists()}
        logger.info("DRILL faults pending %s", sorted(self.pending, key=lambda f: f[1]))

    def _marker(self, kind: str, step: int) -> Path:
        if self.marker_dir is None:
            return Path("/nonexistent")
        return self.marker_dir / f"fired-{kind}-{step}"

    def fire(self, kind: str, step: int) -> bool:
        if (kind, step) not in self.pending:
            return False
        self.pending.discard((kind, step))
        if self.marker_dir is not None and (jax.process_index() == 0 or kind == "abort"):
            self.marker_dir.mkdir(parents=True, exist_ok=True)
            self._marker(kind, step).touch()
        logger.warning("DRILL firing %s at step %d", kind, step)
        return True


def _damaged_arrays(batch, vocab_size: int) -> dict[str, str]:
    """The input arrays of ``batch`` outside the ranges ``train._batch_inputs_valid`` checks, with their first values."""
    host = jax.device_get(batch)
    found = {}

    def report(name, bad, values):
        if np.any(bad):
            idx = np.argwhere(bad)
            found[name] = f"n_bad {len(idx)} first {idx[0].tolist()} last {idx[-1].tolist()} values {values[bad][:4]}"

    report("tokens", (host.tokens < 0) | (host.tokens >= vocab_size), host.tokens)
    weights = host.loss_weight
    report("loss_weight", ~np.isfinite(weights) | (weights < 0) | (weights > 1), weights)
    mask = host.attn_mask
    if isinstance(mask, AttentionMask) and mask.segment_ids is not None:
        q_seg, kv_seg = mask.segment_ids
        report("segment_ids", q_seg != kv_seg, np.stack([q_seg, kv_seg], axis=-1))
    return found


def _drill_step(make_step, faults: _Faults):
    @functools.wraps(make_step)
    def make(*args, **kwargs):
        step_fn = make_step(*args, **kwargs)
        recent: collections.deque = collections.deque()
        host_step = [None]

        def drill_step(state, batch, *, compute_watch=False):
            if host_step[0] is None:
                host_step[0] = int(state.step)
            step = host_step[0]
            if faults.fire("damage", step):
                batch = dataclasses.replace(batch, tokens=batch.tokens.at[0, :16].set(FLOAT_BITS))
            if faults.fire("nan", step):
                state = dataclasses.replace(
                    state,
                    opt_state=jax.tree.map(
                        lambda x: x * jnp.nan if jnp.issubdtype(x.dtype, jnp.floating) else x, state.opt_state
                    ),
                )
            out = step_fn(state, batch, compute_watch=compute_watch)
            host_step[0] += 1
            recent.append((step, batch, out[1][train._INPUTS_VALID_KEY]))
            if len(recent) > 2:
                old_step, old_batch, valid = recent.popleft()
                if not bool(valid):
                    logger.warning(
                        "DRILL skipped step %d damaged %s", old_step, _damaged_arrays(old_batch, kwargs["vocab_size"])
                    )
            if faults.fire("abort", step):
                os.abort()
            return out

        return drill_step

    return make


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    initialize_distributed(args)
    config = build_run_config(args)
    marker_dir = Path(args.output_root) / args.run_id / "faults" if args.output_root else None
    faults = _Faults(marker_dir, args)
    train._make_train_step = _drill_step(train._make_train_step, faults)
    _run_grug_local(config)


if __name__ == "__main__":
    main()
