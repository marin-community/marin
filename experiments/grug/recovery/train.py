# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trainer entrypoint for the grug hang-recovery test.

Runs a real ~10B sharded grug step (so the collectives and snapshot volume are
realistic) on synthetic data, and layers on the recovery machinery:

- restore from the host snapshot at startup (warm resume after a fault),
- snapshot the full train state to host memory every N steps,
- primary hang detection via the XLA execution deadman (set in the environment
  by the supervisor) plus a fixed-timeout ``ProgressWatchdog`` fallback,
- a per-step heartbeat the supervisor's deadman watches,
- optional fault injection to exercise all of the above.

Exceptions propagate: ``levanter.recovery._child`` maps them onto the exit-code
contract, and the supervisor restarts recoverable faults. Do not swallow here.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import jmp
import optax
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.callbacks._core import ProgressEvent
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.recovery import detection, faults
from levanter.recovery.snapshot import HostSnapshot, estimate_state_nbytes, resolve_snapshot_root
from levanter.recovery.supervisor import (
    ENV_ATTEMPT,
    ENV_HEARTBEAT_PATH,
    ENV_RUN_ID,
    ENV_SNAPSHOT_NVME_BASE,
    ENV_SNAPSHOT_TMPFS_BASE,
)
from levanter.utils.jax_utils import parameter_count

from experiments.grug.base.train import GrugTrainState, _make_train_step, initial_state
from experiments.grug.recovery.config import RecoveryRunConfig

logger = logging.getLogger(__name__)

_MP_POLICY = jmp.get_policy("params=float32,compute=bfloat16,output=float32")
_BATCH_PSPEC = P(("replica_dcn", "data"), None)


def _make_synthetic_batch(config: RecoveryRunConfig, mesh) -> GrugLmExample:
    """One reusable synthetic batch, sharded over the batch axes."""
    key = jax.random.PRNGKey(config.seed + 1)
    tokens = jax.random.randint(key, (config.batch_size, config.seq_len), 0, config.model.vocab_size, dtype=jnp.int32)
    loss_weight = jnp.ones((config.batch_size, config.seq_len), dtype=jnp.float32)
    sharding = NamedSharding(mesh, _BATCH_PSPEC)
    tokens = jax.device_put(tokens, sharding)
    loss_weight = jax.device_put(loss_weight, sharding)
    return GrugLmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=AttentionMask.causal())


def _heartbeat_path() -> str:
    path = os.environ.get(ENV_HEARTBEAT_PATH)
    if path:
        return path
    return os.path.join(tempfile.gettempdir(), "levanter-recovery.heartbeat")


def recovery_train(config: RecoveryRunConfig) -> None:
    """Run the recovery training loop until ``num_steps`` (resuming if snapshotted)."""
    detection.apply_in_process_jax_config(config.detection)

    run_id = os.environ.get(ENV_RUN_ID, "grug-recovery")
    attempt = int(os.environ.get(ENV_ATTEMPT, "1"))
    heartbeat = _heartbeat_path()
    logger.info("recovery_train start: run=%s attempt=%d devices=%d", run_id, attempt, jax.device_count())

    optimizer = optax.adamw(config.learning_rate)
    train_step = _make_train_step(optimizer, _MP_POLICY, z_loss_weight=0.0, ema_beta=None)

    mesh = compact_grug_mesh(
        replica_axis_size=config.replica_axis_size,
        model_axis_size=config.model_axis_size,
    )
    with set_mesh(mesh):

        @jax.jit
        def _init() -> GrugTrainState:
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=_MP_POLICY,
                key=jax.random.PRNGKey(config.seed),
                ema_beta=None,
            )

        state = _init()
        n_params = int(parameter_count(state.params))
        state_bytes = estimate_state_nbytes(state)
        logger.info("model params=%.2fB  full state=%.1f GiB", n_params / 1e9, state_bytes / 2**30)

        root = resolve_snapshot_root(
            run_id=run_id,
            tmpfs_base=os.environ.get(ENV_SNAPSHOT_TMPFS_BASE, "/dev/shm"),
            nvme_base=os.environ.get(ENV_SNAPSHOT_NVME_BASE, tempfile.gettempdir()),
            state_nbytes=state_bytes,
        )
        snapshot = HostSnapshot(root)
        restored = snapshot.restore_into(state)
        if restored is not None:
            state, restore_result = restored
            logger.info(
                "RESUMED from host snapshot at step %d in %.2fs (attempt %d)",
                restore_result.step,
                restore_result.seconds,
                attempt,
            )
        else:
            logger.info("no snapshot found; starting from step 0")

        batch = _make_synthetic_batch(config, mesh)
        # LEVANTER_DISABLE_WATCHDOG isolates the XLA execution deadman by turning
        # off the levanter fallback watchdog for an ablation.
        if os.environ.get("LEVANTER_DISABLE_WATCHDOG"):
            watchdog = None
            logger.info("levanter fallback watchdog disabled (LEVANTER_DISABLE_WATCHDOG)")
        else:
            watchdog = detection.fallback_watchdog_config(
                config.watchdog_step_timeout_seconds,
                startup_grace_seconds=config.watchdog_startup_grace_seconds,
                diagnostic_timeout_seconds=None,
            ).create(process_index=jax.process_index(), diagnostic=None)

        step = int(state.step)
        try:
            while step < config.num_steps:
                faults.maybe_inject(step)
                if watchdog is not None:
                    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
                t0 = time.perf_counter()
                state, metrics, _ = train_step(state, batch)
                loss = float(jax.block_until_ready(metrics["train/loss"]))
                if watchdog is not None:
                    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)

                step = int(state.step)
                detection.touch_heartbeat(heartbeat, step)
                dt = time.perf_counter() - t0
                if step % config.log_every == 0:
                    logger.info("step %d loss %.4f %.2fs", step, loss, dt)

                if step % config.snapshot_every_steps == 0:
                    stats = snapshot.save(state, step=step)
                    logger.info("snapshot step %d: %.1f GiB in %.2fs", step, stats.bytes_written / 2**30, stats.seconds)
        finally:
            if watchdog is not None:
                watchdog.on_event(ProgressEvent.TRAINING_FINISHED)

        logger.info("recovery_train finished at step %d (run=%s)", step, run_id)
