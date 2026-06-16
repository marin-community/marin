# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token eval-loss dumper for completed MoE runs.

Loads a finished MoE run's final checkpoint, runs forward on every tagged
eval set (paloma + uncheatable), captures per-position cross-entropy via
``model.next_token_loss(reduction='none')``, and writes a parquet of
``(run_id, eval_set, batch_idx, position, token_id, loss, weight)`` tuples.

Companion to ``experiments/grug/nano/scripts/per_token_loss.py`` from branch
``moe_grugmuon``; this version is adapted to the MoE training stack
(``GrugRunConfig``, ``GrugTrainState``, MoE checkpoint shapes).

Submit on us-east5-a (so the v5p-8 lands near the GCS-stored checkpoints):

    .venv/bin/iris --cluster=marin job run \\
      --no-wait \\
      --zone us-east5-a \\
      --priority interactive \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.per_token_loss
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta

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

from experiments.defaults import _submit_train_job
from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.moe.heuristic_adamh import build_from_heuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugTrainState, initial_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerTokenLossConfig:
    """One run + the checkpoint to dump.

    ``optimizer`` is the same OptimizerConfig that was used during the
    original training run. We don't run optimizer updates; its ``init`` is
    needed to produce a state template whose ``opt_state`` shape matches the
    saved checkpoint so the restore can map tensors correctly.
    """

    run_id: str
    checkpoint_base_path: str
    output_parquet: str
    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig
    num_train_steps: int
    batch_size: int
    seq_len: int = 4096
    eval_batch_size: int = 24
    max_batches_per_set: int = 1
    mp: str = "params=float32,compute=bfloat16,output=bfloat16"


def _per_token_loss_local(config: PerTokenLossConfig) -> None:
    """Worker-side entry: build state template, restore checkpoint, dump losses."""
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=config.batch_size,
        num_train_steps=config.num_train_steps,
        mp=jmp.get_policy(config.mp),
        tracker=WandbConfig(entity="marin-community", project="marin_moe", name=config.run_id, group="per-token-loss"),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=config.checkpoint_base_path,
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[],
        ),
    )
    trainer.initialize()

    optimizer = config.optimizer.build(trainer.num_train_steps)

    with trainer.use_device_mesh():
        mesh = trainer.device_mesh

        @jax.jit
        def _init_state(model_rng):
            return initial_state(config.model, optimizer=optimizer, mp=trainer.mp, key=model_rng, ema_beta=None)

        state: GrugTrainState = _init_state(jax.random.PRNGKey(0))
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=[config.checkpoint_base_path],
            load_checkpoint_setting=True,
            mesh=mesh,
            allow_partial=False,
        )
        logger.info("Restored checkpoint at step %s from %s", int(state.step), config.checkpoint_base_path)

        pos = Axis("position", config.seq_len)
        batch_axis_resource: tuple[str, ...] = ("data", "expert")
        eval_array_sharding = NamedSharding(mesh, P(batch_axis_resource, None))

        tagged = config.data.tagged_eval_sets(pos)
        if not tagged:
            raise RuntimeError("No tagged eval sets — check the data config.")
        logger.info("Found %d tagged eval sets", len(tagged))

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
        # Apply pending QB betas so the router uses the trained biases.
        from experiments.grug.moe.train import _apply_qb_betas

        compute_params = _apply_qb_betas(compute_params, state.pending_qb_betas)

        all_runs, all_evals, all_batches, all_positions = [], [], [], []
        all_token_ids, all_losses, all_weights = [], [], []

        for ds, tags in tagged:
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
                loss_np = jax.device_get(loss).astype("float32").ravel()
                tokens_np = jax.device_get(tokens).astype("int32").ravel()
                weights_np = jax.device_get(weights).astype("float32").ravel()
                bsz, seq = loss.shape[:2]
                row_in_batch = jnp.broadcast_to(jnp.arange(bsz)[:, None], (bsz, seq)).astype("int32").ravel()
                pi = jnp.broadcast_to(jnp.arange(seq)[None, :], (bsz, seq)).astype("int32").ravel()
                rb_np = jax.device_get(row_in_batch)
                pi_np = jax.device_get(pi)
                n = loss_np.size

                all_runs.extend([config.run_id] * n)
                all_evals.extend([eval_set] * n)
                all_batches.extend((batch_idx * bsz + rb_np).tolist())
                all_positions.extend(pi_np.tolist())
                all_token_ids.extend(tokens_np.tolist())
                all_losses.extend(loss_np.tolist())
                all_weights.extend(weights_np.tolist())
                logger.info(
                    "  %s batch %d: B=%d S=%d mean_loss=%.4f",
                    eval_set,
                    batch_idx,
                    bsz,
                    seq,
                    float(loss_np.mean()),
                )

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
        logger.info("Wrote %d rows to %s (%.1f MB)", len(all_runs), config.output_parquet, table.nbytes / 1e6)


def _run_per_token_loss_on_worker(name: str, raw_launch: PerTokenLossConfig) -> None:
    _per_token_loss_local(raw_launch)


def run_per_token_loss(config: PerTokenLossConfig) -> None:
    env = dict(os.environ)
    env_vars = {"WANDB_API_KEY": env["WANDB_API_KEY"]} if "WANDB_API_KEY" in env else {}
    _submit_train_job(
        name=config.run_id,
        entrypoint_callable=_run_per_token_loss_on_worker,
        args=[config.run_id, config],
        resources=config.resources,
        env_vars=env_vars,
        wait=True,
    )


# --- Default target: the d=1024 1e19 isoflop run (PKO enabled) ---

_TARGET_RUN_ID = "grug-moe-isoflop-v1e19-d1024-v1"
_TARGET_CHECKPOINT_BASE = (
    "gs://marin-us-east5/grug/grug_moe_isoflop_v1e19/grug-moe-isoflop-v1e19-d1024-v1-19ee50/checkpoints"
)


def _build_step() -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(budget=1e19, hidden_dim=1024)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.01,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )
    return ExecutorStep(
        name=f"grug/per_token_loss/{_TARGET_RUN_ID}",
        fn=run_per_token_loss,
        config=PerTokenLossConfig(
            run_id=f"{_TARGET_RUN_ID}-pertoken",
            checkpoint_base_path=_TARGET_CHECKPOINT_BASE,
            output_parquet=this_output_path(f"{_TARGET_RUN_ID}.parquet"),
            model=model,
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            resources=ResourceConfig.with_tpu("v5p-8", regions=("us-east5",)),
            optimizer=optimizer,
            num_train_steps=num_steps,
            batch_size=batch_size,
            seq_len=4096,
            eval_batch_size=24,
            max_batches_per_set=1,
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[_build_step()],
        description=(
            f"Per-token eval-loss dump for {_TARGET_RUN_ID} on paloma + uncheatable. "
            "24-batch x 4096-seq, one batch per tagged set."
        ),
    )
