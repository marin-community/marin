# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token eval-loss dumper for completed nano runs.

Loads a finished run's final checkpoint, runs forward on every tagged eval
set (paloma + uncheatable_eval; ``github_cpp`` is one of the uncheatable
subsets), captures per-position cross-entropy via
``next_token_loss(reduction="none")``, and writes a parquet of
``(run, eval_set, batch_idx, position, token_id, loss, weight)`` tuples.
Token decoding / aggregation are deferred — the parquet is the raw dump.

Sample size: ``eval_batch_size=24`` x ``seq_len=4096`` = **98,304 tokens
per tagged set**, one batch per set. Across paloma's ~18 sub-corpora plus
uncheatable's ~9 that's roughly 2.6 M token-positions per run.

Usage::

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
        --no-wait --reserve v5p-8 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.nano.scripts.per_token_loss

The default sweep dumps p16 muon + p16 adamh, both using the same P16_MODEL
architecture but with their respective trained checkpoints + optimizer
shapes (needed so ``load_checkpoint`` can map the saved opt_state back).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import fsspec
import jax
import jax.numpy as jnp
import jmp
import pyarrow as pa
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from haliax import Axis
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import CheckpointerConfig
from levanter.data import DataLoader
from levanter.data.text import LmDataConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.models.lm_model import LmExample
from levanter.optim import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.nano.launch_adamh_heuristic_walk_p16 import P16_OPTIMIZER as P16_ADAMH_OPTIMIZER
from experiments.grug.nano.launch_muon_tuned_walk_p16 import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    P16_MODEL,
)
from experiments.grug.nano.launch_muon_tuned_walk_p16 import (
    P16_OPTIMIZER as P16_MUON_OPTIMIZER,
)
from experiments.grug.nano.model import NanoModelConfig
from experiments.grug.nano.train import GrugTrainState, initial_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerTokenLossConfig:
    """One run + the checkpoint to dump.

    ``optimizer`` is the same OptimizerConfig that was used during the
    original training run. We don't actually run optimizer updates, but its
    ``init`` is needed to produce a state template with the right ``opt_state``
    shape so the checkpoint restore can map saved tensors correctly. Pass
    ``num_train_steps`` matching the original run too — some optimizer
    configs bake step-count into their schedule injection state.
    """

    run_id: str  # wandb display name to log under (e.g. "may7-nano-p16-muon-pertoken")
    checkpoint_base_path: str  # gs:// path with step-N subdirs
    output_parquet: str  # gs:// or local destination
    model: NanoModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig
    num_train_steps: int = 10343
    seq_len: int = 4096
    eval_batch_size: int = 64
    max_batches_per_set: int = 1
    mp: str = "params=float32,compute=bfloat16,output=bfloat16"


def _per_token_loss_local(config: PerTokenLossConfig) -> None:
    """Body of the dump. Runs once per ``PerTokenLossConfig`` on TPU."""
    # Pick up an iris-side trainer setup so the abstract mesh + checkpointer
    # machinery is configured the same as the original run. We only need the
    # mesh + mp + checkpoint search path; everything else is unused.
    extra_kwargs = {"mesh": MeshConfig(axes={"expert": 1})} if config.model.use_moe else {}
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=config.eval_batch_size,  # unused; just for shape inference
        num_train_steps=1,
        mp=jmp.get_policy(config.mp),
        tracker=WandbConfig(project="marin", tags=["pertoken-eval"], group="pertoken-eval", name=config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=config.checkpoint_base_path,
            append_run_id_to_base_path=False,
        ),
        **extra_kwargs,
    )
    trainer.initialize()

    with trainer.use_device_mesh():
        mesh = trainer.device_mesh

        # Build a state template via the *same* optimizer config the run
        # trained with so the restore can map saved opt_state tensors. We
        # never call the optimizer's update — we only need `params` — but
        # `load_checkpoint` needs the full GrugTrainState shape to match.
        opt = config.optimizer.build(num_train_steps=config.num_train_steps)

        @jax.jit
        def _init_state(key):
            return initial_state(
                config.model,
                optimizer=opt,
                mp=trainer.mp,
                key=key,
                ema_beta=None,
            )

        state: GrugTrainState = _init_state(jax.random.PRNGKey(0))

        # Restore from the run's permanent (force=True) final checkpoint.
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=[config.checkpoint_base_path],
            load_checkpoint_setting=True,
            mesh=mesh,
            allow_partial=False,
        )
        logger.info("Restored checkpoint at step %s", int(state.step))

        # Build per-eval-set loaders. ``data.tagged_eval_sets`` returns
        # (dataset, tags) per component; we iterate one batch from each.
        pos = Axis("position", config.seq_len)
        # MoE walks expect the batch sharded across data + expert; non-MoE
        # only needs data. With expert size = 1, the two specs are equivalent
        # at runtime, but we still need to pass the right one to DataLoader.
        is_moe = bool(getattr(config.model, "use_moe", False))
        batch_axis_resource: tuple[str, ...] | str = ("data", "expert") if is_moe else "data"
        eval_array_sharding = NamedSharding(mesh, P(batch_axis_resource, None))

        tagged = config.data.tagged_eval_sets(pos)
        if not tagged:
            raise RuntimeError("No tagged eval sets — check the data config.")

        # JIT the per-batch forward with reduction="none". Fused CE returns
        # the (B, S) per-position loss directly when reduction="none".
        @jax.jit
        def per_position_loss(model, batch):
            if isinstance(batch, LmExample):
                batch = grug_lm_example_from_named(batch)
            loss = model.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="none",
                logsumexp_weight=None,
            )
            loss = jax.sharding.reshard(loss, eval_array_sharding)
            tokens = jax.sharding.reshard(batch.tokens, eval_array_sharding)
            weights = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
            return loss, tokens, weights

        compute_params = trainer.mp.cast_to_compute(state.params)
        # Apply pending QB betas if MoE (so the router uses the trained biases).
        if is_moe:
            from experiments.grug.nano.train import _apply_qb_betas

            compute_params = _apply_qb_betas(compute_params, state.pending_qb_betas)

        # Accumulate rows in Python lists; flush to parquet once at the end.
        all_runs = []
        all_evals = []
        all_batches = []
        all_positions = []
        all_token_ids = []
        all_losses = []
        all_weights = []

        for ds, tags in tagged:
            # tag is a list[str]; the canonical eval-set name is the last tag.
            eval_set = tags[-1] if tags else "unknown"
            loader = DataLoader(
                ds,
                batch_size=config.eval_batch_size,
                mesh=mesh,
                axis_resources={"__BATCH__": batch_axis_resource},
                batch_axis_name="__BATCH__",
                allow_nondivisible_batch_size=False,
            )
            it = iter(loader)
            for batch_idx in range(config.max_batches_per_set):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                loss, tokens, weights = per_position_loss(compute_params, batch)
                # Materialize to host. (B, S) each.
                loss_np = jax.device_get(loss).astype("float32").ravel()
                tokens_np = jax.device_get(tokens).astype("int32").ravel()
                weights_np = jax.device_get(weights).astype("float32").ravel()
                B, S = loss.shape[:2]
                # Within-batch row index (`row_in_batch`) is the BS-dim coordinate;
                # `position` is the seq-dim coordinate.
                row_in_batch = jnp.broadcast_to(jnp.arange(B)[:, None], (B, S)).astype("int32").ravel()
                pi = jnp.broadcast_to(jnp.arange(S)[None, :], (B, S)).astype("int32").ravel()
                rb_np = jax.device_get(row_in_batch)
                pi_np = jax.device_get(pi)
                n = loss_np.size

                all_runs.extend([config.run_id] * n)
                all_evals.extend([eval_set] * n)
                # `batch_idx` is the across-batches counter (0..max_batches_per_set);
                # `row_in_batch` is the within-batch row. Together they uniquely
                # identify a (sequence) within an eval set.
                all_batches.extend((batch_idx * B + rb_np).tolist())
                all_positions.extend(pi_np.tolist())
                all_token_ids.extend(tokens_np.tolist())
                all_losses.extend(loss_np.tolist())
                all_weights.extend(weights_np.tolist())
                logger.info(
                    "  %s batch %d: B=%d S=%d mean_loss=%.4f",
                    eval_set,
                    batch_idx,
                    B,
                    S,
                    float(loss_np.mean()),
                )

        # Write parquet.
        table = pa.Table.from_pydict(
            {
                "run_id": all_runs,
                "eval_set": all_evals,
                "batch_idx": all_batches,
                "position": all_positions,
                "token_id": all_token_ids,
                "loss": all_losses,
                "weight": all_weights,
            }
        )
        with fsspec.open(config.output_parquet, "wb") as f:
            pq.write_table(table, f)
        logger.info(
            "Wrote %d rows to %s (size on host: %.1f MB)",
            len(all_runs),
            config.output_parquet,
            table.nbytes / 1e6,
        )


def run_per_token_loss(config: PerTokenLossConfig) -> None:
    """Dispatch to TPU via Fray, mirroring how `run_grug` does it."""
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_per_token_loss_local,
        resources=config.resources,
    )


# -----------------------------------------------------------------------------
# Targets — wandb run names paired with their checkpoint paths. Pulled from
# `trainer.trainer.checkpointer.base_path` on each wandb run's config.
# -----------------------------------------------------------------------------

# Default target sweep: p16 muon and p16 adamh, both using the same P16_MODEL.
# Each writes a parquet under this step's output_path.
_RUNS = [
    ("may7-nano-muon-tuned-p16", "gs://marin-us-east5/grug/nano-muon-tuned-p16-trial-5def74/checkpoints"),
    ("may7-nano-adamh-heuristic-p16", "gs://marin-us-east5/grug/nano-adamh-heuristic-p16-trial-5def74/checkpoints"),
]


def _step_for(run_id: str, ckpt_base: str, optimizer) -> ExecutorStep:
    # batch_size=24 at seq_len=4096 gives 24*4096 = 98,304 ~ 100k tokens per
    # batch. v5p-8 has 8 devices; 24 = 3*8 keeps the batch mesh-divisible.
    # max_batches_per_set=1 means one ~100k-token sample per tagged eval set.
    return ExecutorStep(
        name=f"grug/pertoken-{run_id}",
        fn=run_per_token_loss,
        config=PerTokenLossConfig(
            run_id=f"{run_id}-pertoken",
            checkpoint_base_path=ckpt_base,
            output_parquet=this_output_path(f"{run_id}.parquet"),
            model=P16_MODEL,
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            # Constrain to us-east5 so the v5p-8 lands near the checkpoints
            # (gs://marin-us-east5/...). Cross-region reads of multi-GB
            # checkpoint shards are costly + slow.
            resources=ResourceConfig.with_tpu("v5p-8", regions=("us-east5",)),
            optimizer=optimizer,
            num_train_steps=10343,
            seq_len=4096,
            eval_batch_size=24,
            max_batches_per_set=1,
        ),
    )


_STEPS = [
    _step_for(
        "may7-nano-muon-tuned-p16",
        "gs://marin-us-east5/grug/nano-muon-tuned-p16-trial-5def74/checkpoints",
        P16_MUON_OPTIMIZER,
    ),
    _step_for(
        "may7-nano-adamh-heuristic-p16",
        "gs://marin-us-east5/grug/nano-adamh-heuristic-p16-trial-5adaad/checkpoints",
        P16_ADAMH_OPTIMIZER,
    ),
]


if __name__ == "__main__":
    executor_main(
        steps=_STEPS,
        description="Per-token eval-loss dump for p16 muon + adamh on paloma + uncheatable.",
    )
