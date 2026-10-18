# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay+CPLM d512, fixed B16 x 8192, on one TPU v4-32 slice."""

import argparse
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import draccus
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from fray.cluster import ResourceConfig
from jax.experimental.multihost_utils import broadcast_one_to_all, process_allgather, sync_global_devices
from jax.sharding import PartitionSpec as P
from levanter.data.text.datasets import BlockShuffleConfig, DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.distributed import DistributedConfig
from levanter.models.lm_model import LmExample
from levanter.optim.config import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.fast_track import model, qb_balancing, train
from experiments.grug.fast_track.column_sharded_embedding import column_sharded_embedding
from experiments.grug.fast_track.heuristic import MoeHeuristic
from experiments.grug.fast_track.router_lookup import unrolled_router_logits
from experiments.grug.moe import kda
from experiments.grug.moe.kda_state_tpu import state_pass
from experiments.grug.moe_cplm.belay import CopyPasteBelay

RECIPE = Path(__file__).with_name("belay_recipe.json")
DATA = Path(__file__).with_name("july_data.json")
REFERENCE_PROJECTION = model._proj


def gathered_projection(cfg, x, weight, out_sharding=None):
    return REFERENCE_PROJECTION(cfg, x, jax.sharding.reshard(weight, P(None, None)), out_sharding)


def tpu_recurrence(prep, state, mm, lead, dk, dv, *, scan_impl):
    """Run the qualified state recurrence inside the KDA local shard."""
    chunks, chunk_size = prep.kw.shape[-3:-1]

    def flatten(x):
        return x.reshape(-1, chunks, *x.shape[-2:])

    output, final = state_pass(
        flatten(prep.q_inflate),
        flatten(prep.attn),
        flatten(prep.kw),
        flatten(prep.k_cumdecay),
        flatten(prep.v_pseudo),
        prep.decay.reshape(-1, chunks, 1, dk),
        state.reshape(-1, dk, dv),
        precision=jax.lax.Precision.HIGH,
    )
    return output.reshape(*lead, chunks, chunk_size, dv), final.reshape(*lead, dk, dv)


@contextmanager
def execution():
    """Bind TPU implementations for this single-run process, including eval."""
    with ExitStack() as stack:
        for module, name, implementation in (
            (train, "Transformer", CopyPasteBelay),
            (model, "_proj", gathered_projection),
            (model, "_replicated_fp32_gather", column_sharded_embedding),
            (model, "_selected_router_logits", unrolled_router_logits),
            (kda, "_parallel_chunk_recurrence", tpu_recurrence),
            (
                qb_balancing,
                "bincount_upper_quantile",
                partial(qb_balancing.bincount_upper_quantile, histogram_method="two-stage"),
            ),
        ):
            stack.enter_context(patch.object(module, name, implementation))
        yield


def build_config(
    *,
    data_path: Path,
    output: Path,
    checkpoint_path: str,
    run_name: str,
    wandb_entity: str,
    wandb_project: str,
    calibration: bool,
    tokenizer_path: str,
) -> train.GrugRunConfig:
    """Freeze the optimizer batch and schedule; calibration stops after 400 updates."""
    source = json.loads(RECIPE.read_text())
    model_values = source["model"]
    model_values.update(
        kda_write_gate=False,
        kda_erase_gate=False,
        kda_onchip_erase_gate=False,
        attn_res_layer_backward="SAVE",
        attn_res_remat_attention=False,
        kda_save_chunk_states=False,
        kda_state_grads="XLA",
        kda_xla_scan_impl="parallel_stacked_split",
        hidden_dim=512,
        vocab_size=128256,
        max_seq_len=8192,
    )
    model_config = draccus.decode(model.GrugModelConfig, model_values)
    optimizer = source["optimizer"]
    optimizer["type"] = "grug_fast_track_muonh_v1"
    heuristic = MoeHeuristic().build_optimizer_config(num_train_steps=10980, batch_size=16, hidden_dim=512, seq_len=8192)
    for name in ("learning_rate", "adam_lr", "epsilon", "beta2"):
        optimizer[name] = getattr(heuristic, name)
    outer = source["trainer"]
    outer.update(
        expert_axis_size=1,
        replica_axis_size=1,
        grad_accum_microbatches=1,
        pipeline_depth=1,
        loader_buffer_batches=4,
        z_loss_weight=0.0,
        ema_beta=None,
        ema_last_steps=None,
        ema_blend_sweep=[],
        ema_group_sweep=False,
        save_checkpoints=not calibration,
        completion_marker_path=checkpoint_path + "/final_eval_done.json",
    )
    inner = outer["trainer"]
    inner.update(
        id=run_name,
        seed=0,
        num_train_steps=10980,
        train_batch_size=16,
        per_device_parallelism=1,
        per_device_eval_parallelism=1,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        load_checkpoint=False,
        load_checkpoint_path=None,
        log_dir=str(output / "logs"),
        tracker=[{"type": "json_logger"}],
        log_jaxprs=False,
        log_xla_hlo=False,
        progress_watchdog={"step_timeout": None, "process_timeout": None, "startup_timeout": None},
    )
    inner["jax_config"]["jax_enable_pgle"] = False
    inner["checkpointer"].update(
        base_path=checkpoint_path,
        temporary_base_path=checkpoint_path + "/temporary",
        keep=[{"every": 1000}],
        save_interval="0:10:00",
    )
    inner["checkpointer"]["write"].update(max_staged_host_bytes=2 * 1024**3, data_copy_concurrency=4)
    trainer_config = draccus.decode(train.GrugTrainerConfig, outer)
    tracker = WandbConfig(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        id=run_name,
        mode="online",
        resume="never",
        save_code=False,
        background=False,
    )
    trainer_config = replace(
        trainer_config,
        trainer=replace(
            trainer_config.trainer,
            tracker=(*trainer_config.trainer.tracker, tracker),
            jax_config=dict(inner["jax_config"]),
        ),
    )
    data_values = json.loads(data_path.read_text())
    components = {
        name: DatasetComponent(
            cache_dir=value["cache_dir"],
            tags=value.get("tags"),
            source=UrlDatasetSourceConfig(),
            format=TextLmDatasetFormat(text_key=value.get("text_key", "text")),
        )
        for name, value in data_values["data_components"].items()
    }
    data = LmDataConfig(
        tokenizer=tokenizer_path,
        components=components,
        train_weights={name: data_values["train_weights"].get(name, 0.0) for name in components},
        cache_dir=None,
        auto_build_caches=False,
        shuffle=BlockShuffleConfig(**data_values["shuffle"]),
        permutation_type="feistel",
        block_cross_document_attention=True,
        mixture_block_size=data_values["mixture_block_size"],
    )
    return train.GrugRunConfig(
        model=model_config,
        optimizer=draccus.decode(OptimizerConfig, optimizer),
        data=data,
        trainer=trainer_config,
        resources=ResourceConfig.with_tpu("v4-32"),
        eval=(
            None
            if calibration
            else train.GrugEvalConfig(
                eval_batch_size=256,
                steps_per_eval=1000,
                compute_bpb=True,
                dropless_eval=True,
                dropless_eval_moe_implementation="scatter",
            )
        ),
        tensorstore_cache_bytes=1024**3,
        stop_after_steps=400 if calibration else None,
    )


class MetricsFileHandler(logging.FileHandler):
    """Retain partial native loss logs in regional storage during training."""

    def __init__(self, path, destination):
        super().__init__(path)
        self.destination = StoragePath(destination)
        self.last_upload = 0.0

    def upload(self):
        self.flush()
        self.destination.write_bytes(Path(self.baseFilename).read_bytes())
        self.last_upload = time.monotonic()

    def emit(self, record):
        super().emit(record)
        if time.monotonic() - self.last_upload >= 60:
            self.upload()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DATA)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--logging-check", action="store_true")
    args = parser.parse_args(argv)
    run_name = args.run_name
    args.output.mkdir(parents=True, exist_ok=True)
    if args.logging_check:
        if os.environ.get("IRIS_JOB_ID"):
            raise ValueError("Run the logging check once on the remote submit host before Iris submission")
        tracker = WandbConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name + "-logging-check",
            id="logcheck-" + uuid.uuid4().hex,
            mode="online",
            resume="never",
            background=False,
        ).init(None)
        tracker.log_hyperparameters({"training_updates": 0})
        tracker.log({"preflight/ok": 1}, step=0)
        tracker.finish()
        return
    flag = "--xla_allow_excess_precision=false"
    if flag not in os.environ.get("XLA_FLAGS", "").split():
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " " + flag
    # Iris task-attempt IDs differ by host. Broadcast one fresh process-group
    # identity before constructing tracker and checkpoint paths.
    jax_settings = json.loads(RECIPE.read_text())["trainer"]["trainer"]["jax_config"]
    jax_settings["jax_enable_pgle"] = False
    for name, value in jax_settings.items():
        jax.config.update(name, value)
    DistributedConfig().initialize()
    identity = broadcast_one_to_all(np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8).copy())
    attempt = bytes(identity).hex()
    run_name = f"{args.run_name}-{attempt}"
    args.output = args.output / attempt
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_root = args.artifact_root.rstrip("/") + "/" + attempt
    identities = json.loads(Path(__file__).with_name("tokenizer_identity.json").read_text())
    for name, expected in identities.items():
        if hashlib.sha256((Path(args.tokenizer) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Tokenizer identity mismatch: {name}")
    config = build_config(
        data_path=args.data_config,
        output=args.output,
        checkpoint_path=args.checkpoint_path + "/" + attempt,
        run_name=run_name,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        calibration=args.calibration,
        tokenizer_path=args.tokenizer,
    )
    config.trainer.trainer.distributed = replace(config.trainer.trainer.distributed, initialize_jax_distributed=False)
    handlers = []
    initialize = config.trainer.trainer.initialize

    def checked_initialize():
        initialize()
        if len(jax.devices()) != 16 or any(d.device_kind != "TPU v4" for d in jax.devices()):
            raise ValueError("This baseline requires TPU v4-32 (16 physical chips)")
        if jax.process_index() == 0:
            handler = MetricsFileHandler(args.output / "native-metrics.jsonl", artifact_root + "/native-metrics.jsonl")
            handlers.append(handler)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger("levanter.json_logger").addHandler(handler)
            (args.output / "resolved-config.json").write_text(json.dumps(asdict(config), default=str, indent=2))

    config.trainer.trainer.initialize = checked_initialize
    evaluator = train.TaggedEvaluator

    def bounded_evaluator(**kwargs):
        kwargs["max_examples_per_dataset"] = 2048
        native_loss = kwargs["loss_fn"]

        def evaluate(params, batch):
            if isinstance(batch, LmExample):
                batch = grug_lm_example_from_named(batch)

            def microbatch(index):
                chunk = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, index * 16, 16, axis=0), batch)
                return native_loss(params, chunk)

            values = jax.lax.map(microbatch, jnp.arange(batch.tokens.shape[0] // 16))
            return jax.tree.map(
                lambda x: jax.lax.reshape(
                    x, (x.shape[0] * x.shape[1], *x.shape[2:]), out_sharding=P(model._BATCH_AXES, None)
                ),
                values,
            )

        kwargs["loss_fn"] = evaluate
        return evaluator(**kwargs)

    native_flat = train._flat_train_step

    def flat_step(step_fn, treedef):
        native = native_flat(step_fn, treedef)
        completed = 0
        started = None

        @partial(jax.jit, static_argnames=("loop_active", "router_tie_active"))
        def preflight(leaves, batch, loop_active=None, router_tie_active=None):
            state = jax.tree.unflatten(treedef, leaves)
            params = train._apply_qb_betas(state.params, state.pending_qb_betas)
            (loss, _), grads = train._loss_and_grads(
                params,
                batch,
                config.trainer.trainer.mp,
                None,
                state.step,
                loop_active,
                router_tie_active=router_tie_active,
            )
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))
            norms = jnp.stack([jnp.linalg.norm(getattr(grads, k)) for k in ("cp_query", "cp_key", "cp_gate", "cp_sink")])
            return loss, finite, norms

        def run(leaves, batch, **kwargs):
            nonlocal completed, started
            if completed == 0:
                if jax.process_index() == 0:
                    if wandb.run is None or wandb.run.offline or wandb.run.disabled:
                        raise RuntimeError("Online W&B must initialize before optimizer updates")
                    wandb.config.update(
                        {
                            "seed": 0,
                            "restored_checkpoint": None,
                            "process_group_id": attempt,
                            "iris_attempt_uid": os.environ.get("IRIS_ATTEMPT_UID"),
                        }
                    )
                    (args.output / "tracking.json").write_text(json.dumps({"url": wandb.run.url, "resumed": False}))
                loss, finite, norms = jax.device_get(preflight(leaves, batch, **kwargs))
                if not (bool(finite) and np.isfinite(loss) and np.all(norms > 0)):
                    raise ValueError("Full-model preflight failed before optimizer updates")
                if jax.process_index() == 0:
                    (args.output / "full-model-preflight.json").write_text(
                        json.dumps(
                            {
                                "loss": float(loss),
                                "all_gradients_finite": True,
                                "copy_gradient_norms": norms.tolist(),
                                "training_updates": 0,
                            }
                        )
                    )
                preflight.clear_cache()
            result = native(leaves, batch, **kwargs)
            completed += 1
            if args.calibration and completed in (100, 400):
                jax.block_until_ready(result)
                sync_global_devices(f"belay-window-{completed}")
                if completed == 100:
                    started = time.monotonic()
                else:
                    elapsed = float(np.asarray(process_allgather(np.array(time.monotonic() - started))).max())
                    if jax.process_index() == 0:
                        (args.output / "training-window.json").write_text(
                            json.dumps(
                                {
                                    "first_update": 100,
                                    "last_update": 399,
                                    "updates": 300,
                                    "tokens": 39321600,
                                    "max_host_seconds": elapsed,
                                    "tokens_per_second": 39321600 / elapsed,
                                }
                            )
                        )
            return result

        run.lower = native.lower
        return run

    try:
        with (
            execution(),
            patch.object(train, "TaggedEvaluator", bounded_evaluator),
            patch.object(train, "_flat_train_step", flat_step),
        ):
            train._run_grug_local(config)
    finally:
        for handler in handlers:
            handler.upload()
        if jax.process_index() == 0:
            for path in args.output.glob("*.json"):
                StoragePath(artifact_root + "/" + path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
