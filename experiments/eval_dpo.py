#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone DPO eval profiling script with reference log-prob caching.

Supports three modes:
  --mode uncached    Run full 4-forward-pass eval (baseline)
  --mode build       Build the reference log-prob cache, then run cached eval
  --mode cached      Load existing cache, run 2-forward-pass eval

Usage on Iris:
    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
        --extra marin:tpu --tpu v5p-32 --memory 256GB --disk 50GB \
        --job-name "dpo_eval_cached" \
        -e WANDB_API_KEY "${WANDB_API_KEY}" \
        -- python experiments/eval_dpo.py --mode build
"""

import argparse
import hashlib
import json
import logging
import sys
import time
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax.partitioning import ResourceAxis, round_axis_for_partitioning

import levanter
from levanter.callbacks import eval_loss_loop
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data import DataLoader
from levanter.data.dataset import AsyncDataset
from levanter.data.text import (
    DpoExample,
    PreferenceChatLmDatasetFormat,
    dataset_for_preference_format,
    preprocessor_for_preference_format,
)
from levanter.main.model_init import load_model_from_source, prepare_model_init_context
from levanter.main.train_dpo import DpoModel, _logp_sum, dpo_loss_from_logps
from levanter.metrics import Metric, ReductionType
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache
from levanter.trainer import MeshConfig, TrainerConfig
from levanter.utils.tree_utils import inference_mode

from experiments.dpo_bloom_speceval_v2 import dpo_config as bloom_dpo_config
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import DPO_EVAL_PARALLELISM

logger = logging.getLogger(__name__)

PREFERENCE_FORMAT = PreferenceChatLmDatasetFormat()
DEFAULT_WARMUP_BATCHES = 1
MAX_EVAL_BATCHES = 999  # effectively unlimited — run the full val set
WANDB_TAGS = ["dpo", "eval-profile", "standalone"]
VAL_CACHE = "gs://marin-us-central1/tokenized/bloom_speceval_v2_val_prefs_marin_tokenizer-a06ae8/validation"
VAL_NAME = "bloom_speceval_v2_val"


# ── Reference log-prob cache ──────────────────────────────────────────────────


class CachedDpoExample(eqx.Module):
    """DPO example augmented with precomputed reference log-probs. Registered as JAX pytree via eqx.Module."""

    chosen: object  # LmExample
    rejected: object  # LmExample
    logp_ref_chosen: float
    logp_ref_rejected: float


def _ref_cache_path(val_cache: str, reference_model: str, seq_len: int) -> str:
    """Deterministic GCS path for the reference log-prob sidecar cache."""
    key_data = json.dumps(
        {
            "val_cache": val_cache,
            "reference_model": reference_model,
            "seq_len": seq_len,
            "format": "dpo_reference_logprobs_v1",
        },
        sort_keys=True,
    )
    cache_hash = hashlib.sha256(key_data.encode()).hexdigest()[:8]
    # Sibling path next to the validation cache
    parent = val_cache.rsplit("/", 1)[0]
    return f"{parent}/reference_logprobs/{cache_hash}"


def _build_ref_cache(
    *,
    reference_model,
    val_dataset: AsyncDataset,
    eval_loader: DataLoader,
    compute_axis_mapping,
    mp: jmp.Policy,
    cache_dir: str,
    reference_model_path: str,
    seq_len: int,
):
    """Compute reference log-probs for the full val set and write to a TreeCache."""
    logger.info("Building reference log-prob cache at %s", cache_dir)

    @hax.named_jit(axis_resources=compute_axis_mapping)
    def ref_logprobs_fn(ref_model, batch: DpoExample):
        ref_model = mp.cast_to_compute(ref_model)
        ref_model = inference_mode(ref_model, True)
        logp_chosen = jax.lax.stop_gradient(_logp_sum(ref_model, batch.chosen, key=None))
        logp_rejected = jax.lax.stop_gradient(_logp_sum(ref_model, batch.rejected, key=None))
        return logp_chosen, logp_rejected

    # Warmup JIT
    logger.info("Warming up reference forward pass JIT...")
    warmup_iter = iter(eval_loader)
    batch = next(warmup_iter)
    ref_logprobs_fn(reference_model, batch)
    logger.info("Reference JIT warmup done")

    # Compute reference log-probs for all examples
    all_chosen = []
    all_rejected = []
    t0 = time.time()
    for i, batch in enumerate(eval_loader):
        logp_chosen, logp_rejected = ref_logprobs_fn(reference_model, batch)
        # These are NamedArrays sharded across multihost — gather to single host
        chosen_gathered = jax.experimental.multihost_utils.process_allgather(logp_chosen.array, tiled=True)
        rejected_gathered = jax.experimental.multihost_utils.process_allgather(logp_rejected.array, tiled=True)
        chosen_np = np.array(chosen_gathered)
        rejected_np = np.array(rejected_gathered)
        all_chosen.append(chosen_np)
        all_rejected.append(rejected_np)
        if (i + 1) % 10 == 0:
            logger.info("  Reference cache: %d batches computed (%.1fs)", i + 1, time.time() - t0)

    all_chosen = np.concatenate(all_chosen).astype(np.float32)
    all_rejected = np.concatenate(all_rejected).astype(np.float32)
    build_time = time.time() - t0
    logger.info("Reference log-probs computed: %d examples in %.1fs", len(all_chosen), build_time)

    # Write cache from process 0 only (all hosts have the same data after allgather)
    if jax.process_index() == 0:
        exemplar = {
            "logp_ref_chosen": np.zeros((), dtype=np.float32),
            "logp_ref_rejected": np.zeros((), dtype=np.float32),
        }
        metadata = CacheMetadata(
            preprocessor_metadata={
                "kind": "dpo_reference_logprobs_v1",
                "val_cache": VAL_CACHE,
                "reference_model": reference_model_path,
                "seq_len": seq_len,
            }
        )

        with SerialCacheWriter(cache_dir, exemplar, metadata=metadata) as writer:
            writer.write_batch(
                {
                    "logp_ref_chosen": all_chosen,
                    "logp_ref_rejected": all_rejected,
                }
            )
        logger.info("Reference cache written to %s (%d examples)", cache_dir, len(all_chosen))
    else:
        logger.info("Process %d: skipping cache write (process 0 writes)", jax.process_index())

    # Barrier so all hosts wait for process 0 to finish writing
    jax.experimental.multihost_utils.sync_global_devices("ref_cache_write")

    return all_chosen, all_rejected, build_time


def _load_ref_cache(cache_dir: str):
    """Load precomputed reference log-probs from a TreeCache into numpy arrays."""
    logger.info("Loading reference log-prob cache from %s", cache_dir)
    t0 = time.time()
    exemplar = {
        "logp_ref_chosen": np.zeros((), dtype=np.float32),
        "logp_ref_rejected": np.zeros((), dtype=np.float32),
    }
    cache = TreeCache.load(cache_dir, exemplar=exemplar, options=CacheMetadata(preprocessor_metadata={}))
    n = len(cache)
    rows = cache.get_batch_sync(slice(0, n))
    # get_batch_sync returns a list of dicts (one per example)
    chosen = np.array([row["logp_ref_chosen"] for row in rows], dtype=np.float32)
    rejected = np.array([row["logp_ref_rejected"] for row in rows], dtype=np.float32)
    logger.info("Reference cache loaded: %d examples in %.1fs", n, time.time() - t0)
    return chosen, rejected


def _maybe_reshard_policy_for_eval(*, policy_model, eval_param_layout: str, compute_axis_mapping):
    if eval_param_layout == "fsdp":
        return policy_model

    logger.info("Resharding policy model for eval with %s parameter layout", eval_param_layout)
    t0 = time.time()
    policy_model = hax.shard_with_axis_mapping(policy_model, compute_axis_mapping)
    policy_model = jax.block_until_ready(policy_model)
    logger.info("Policy model resharded in %.1fs", time.time() - t0)
    return policy_model


# ── Loss functions ────────────────────────────────────────────────────────────


def _loss_fn_uncached(model: DpoModel, example: DpoExample, *, mp: jmp.Policy, beta: float):
    """Standard DPO loss: 4 forward passes."""
    model = mp.cast_to_compute(model)
    policy_model = model.policy
    reference_model = inference_mode(model.reference, True)

    with jax.named_scope("policy_chosen"):
        logp_pi_chosen = _logp_sum(policy_model, example.chosen, key=None)
    with jax.named_scope("policy_rejected"):
        logp_pi_rejected = _logp_sum(policy_model, example.rejected, key=None)
    with jax.named_scope("reference_chosen"):
        logp_ref_chosen = jax.lax.stop_gradient(_logp_sum(reference_model, example.chosen, key=None))
    with jax.named_scope("reference_rejected"):
        logp_ref_rejected = jax.lax.stop_gradient(_logp_sum(reference_model, example.rejected, key=None))

    return _compute_dpo_metrics(logp_pi_chosen, logp_pi_rejected, logp_ref_chosen, logp_ref_rejected, beta)


def _loss_fn_cached(model: DpoModel, example: CachedDpoExample, *, mp: jmp.Policy, beta: float):
    """Cached DPO loss: 2 forward passes (policy only)."""
    model = mp.cast_to_compute(model)
    policy_model = model.policy

    with jax.named_scope("policy_chosen"):
        logp_pi_chosen = _logp_sum(policy_model, example.chosen, key=None)
    with jax.named_scope("policy_rejected"):
        logp_pi_rejected = _logp_sum(policy_model, example.rejected, key=None)

    # Cached values may have a trailing (1,) dim from cache storage — squeeze and wrap
    ref_chosen = jnp.squeeze(example.logp_ref_chosen)
    ref_rejected = jnp.squeeze(example.logp_ref_rejected)
    logp_ref_chosen = hax.named(ref_chosen, logp_pi_chosen.axes)
    logp_ref_rejected = hax.named(ref_rejected, logp_pi_rejected.axes)

    return _compute_dpo_metrics(logp_pi_chosen, logp_pi_rejected, logp_ref_chosen, logp_ref_rejected, beta)


def _compute_dpo_metrics(logp_pi_chosen, logp_pi_rejected, logp_ref_chosen, logp_ref_rejected, beta):
    delta_pi = logp_pi_chosen - logp_pi_rejected
    delta_ref = logp_ref_chosen - logp_ref_rejected
    loss, metrics = dpo_loss_from_logps(delta_pi, delta_ref, beta=beta)
    chosen_reward = (logp_pi_chosen - logp_ref_chosen) * beta
    rejected_reward = (logp_pi_rejected - logp_ref_rejected) * beta
    metrics["dpo_chosen_reward"] = Metric.from_value(hax.mean(chosen_reward).scalar(), ReductionType.MEAN)
    metrics["dpo_rejected_reward"] = Metric.from_value(hax.mean(rejected_reward).scalar(), ReductionType.MEAN)
    return loss, metrics


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        tracker=levanter.tracker.wandb.WandbConfig(project="dpo", tags=WANDB_TAGS),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=DPO_EVAL_PARALLELISM["v5p-32"],
        num_train_steps=1,
        per_device_eval_parallelism=DPO_EVAL_PARALLELISM["v5p-32"],
        allow_nondivisible_batch_size=True,
        mesh=MeshConfig(
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            }
        ),
    )


def _load_validation_dataset(cache_path: str, *, tokenizer, Pos):
    processor = preprocessor_for_preference_format(PREFERENCE_FORMAT, tokenizer)
    cache = TreeCache.load(
        cache_path,
        exemplar=processor.output_exemplar,
        options=CacheMetadata(preprocessor_metadata=processor.metadata),
    )
    logger.info("Validation cache loaded from %s (%d examples)", cache_path, len(cache))
    return dataset_for_preference_format(PREFERENCE_FORMAT, Pos, cache)


def _log_results(label, loss, metrics, eval_time, warmup_time, max_eval_batches):
    num_batches = metrics.get("timing/num_batches", 0.0)
    load_time = metrics.get("timing/load_time", 0.0)
    loss_time = metrics.get("timing/loss_time", 0.0)

    logger.info("%s", "=" * 60)
    logger.info("DPO EVAL RESULTS (%s)", label)
    logger.info("%s", "=" * 60)
    logger.info("Loss:              %.4f", loss)
    logger.info("Total eval time:   %.1fs (%.1fmin)", eval_time, eval_time / 60)
    logger.info("Warmup time:       %.1fs", warmup_time)
    logger.info("Max eval batches:  %d", max_eval_batches)
    logger.info("Batches run:       %.0f", num_batches)
    logger.info("Data load time:    %.1fs (%.1f%%)", load_time, load_time / max(eval_time, 0.01) * 100)
    logger.info("Loss compute time: %.1fs (%.1f%%)", loss_time, loss_time / max(eval_time, 0.01) * 100)
    if num_batches > 0:
        logger.info("Avg time/batch:    %.2fs", eval_time / num_batches)
        logger.info("Avg load/batch:    %.3fs", load_time / num_batches)
        logger.info("Avg loss/batch:    %.2fs", loss_time / num_batches)
    logger.info("%s", "=" * 60)
    for key, value in sorted(metrics.items()):
        if not key.startswith("timing/"):
            logger.info("  %s: %.4f", key, value)
    logger.info("%s", "=" * 60)
    sys.stdout.flush()
    sys.stderr.flush()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DPO eval with optional reference log-prob caching")
    parser.add_argument(
        "--mode",
        choices=["uncached", "build", "cached"],
        default="build",
        help="uncached=4 fwd passes, build=compute cache+run cached eval, cached=load cache+run",
    )
    parser.add_argument(
        "--eval_param_layout",
        choices=["fsdp", "replicated"],
        default="fsdp",
        help="Parameter layout for the eval step. replicated is only supported for cached eval modes.",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=DEFAULT_WARMUP_BATCHES,
        help="Number of eval batches to run before timed evaluation for JIT warmup.",
    )
    parser.add_argument("--max_eval_batches", type=int, default=MAX_EVAL_BATCHES)
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()

    if args.warmup_batches < 0:
        raise ValueError(f"--warmup_batches must be non-negative, got {args.warmup_batches}")
    if args.eval_param_layout == "replicated" and args.mode == "uncached":
        raise ValueError("--eval_param_layout=replicated is only supported for cached eval modes (build or cached).")

    max_eval_batches = args.max_eval_batches
    warmup_batches = args.warmup_batches

    trainer_config = _build_trainer_config()
    levanter.initialize(trainer_config)

    tokenizer_name = bloom_dpo_config.tokenizer or marin_tokenizer
    # Trained DPO policy model (step 849) on GCS
    model_name = "gs://marin-us-central1/checkpoints/dpo/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_-7_seed2-947c5d/hf/step-849"
    # Reference model — must match what DPO training used
    reference_model_path = "marin-community/marin-8b-instruct"
    sequence_length = bloom_dpo_config.train_seq_len or bloom_dpo_config.max_seq_len
    beta = bloom_dpo_config.beta

    tokenizer = load_tokenizer(tokenizer_name)
    model_context = prepare_model_init_context(
        llama_8b,
        tokenizer=tokenizer,
        initialize_from_hf=model_name,
        use_hf_model_config=False,
    )
    model_config = model_context.model

    parameter_axis_mapping = trainer_config.parameter_axis_mapping
    compute_axis_mapping = trainer_config.compute_axis_mapping

    ref_cache_dir = _ref_cache_path(VAL_CACHE, reference_model_path, sequence_length)
    logger.info("Reference cache path: %s", ref_cache_dir)
    logger.info("Mode: %s", args.mode)
    logger.info("Eval parameter layout: %s", args.eval_param_layout)

    with trainer_config.use_device_mesh():
        Pos = model_config.max_Pos.resize(sequence_length)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), parameter_axis_mapping)

        key = jax.random.PRNGKey(bloom_dpo_config.seed)
        policy_key, reference_key = jax.random.split(key)

        # ── Load policy model (always needed) ─────────────────────────
        logger.info("Loading policy model from %s", model_name)
        t0 = time.time()
        policy_model = load_model_from_source(
            context=model_context,
            Vocab=Vocab,
            model_key=policy_key,
            parameter_axis_mapping=parameter_axis_mapping,
            compute_dtype=trainer_config.mp.compute_dtype,
            cast_to_param=trainer_config.mp.cast_to_param,
            hf_ref=model_name,
        )
        logger.info("Policy model loaded in %.1fs", time.time() - t0)

        # ── Load reference model (needed for uncached and build modes) ─
        reference_model = None
        if args.mode in ("uncached", "build"):
            logger.info("Loading reference model from %s", reference_model_path)
            t0 = time.time()
            reference_model = load_model_from_source(
                context=model_context,
                Vocab=Vocab,
                model_key=reference_key,
                parameter_axis_mapping=parameter_axis_mapping,
                compute_dtype=trainer_config.mp.compute_dtype,
                cast_to_param=trainer_config.mp.cast_to_param,
                hf_ref=reference_model_path,
            )
            logger.info("Reference model loaded in %.1fs", time.time() - t0)

        # ── Load val dataset ──────────────────────────────────────────
        val_dataset = _load_validation_dataset(VAL_CACHE, tokenizer=tokenizer, Pos=Pos)
        eval_loader = DataLoader(
            val_dataset,
            batch_size=trainer_config.EvalBatch,
            mesh=trainer_config.device_mesh,
            axis_resources=compute_axis_mapping,
            max_buffered_batches=128,
            prefetch_size=32,
            allow_nondivisible_batch_size=True,
        )

        # ── Build reference cache if needed ───────────────────────────
        ref_chosen_np, ref_rejected_np = None, None
        cache_build_time = 0.0

        if args.mode == "build":
            ref_chosen_np, ref_rejected_np, cache_build_time = _build_ref_cache(
                reference_model=reference_model,
                val_dataset=val_dataset,
                eval_loader=eval_loader,
                compute_axis_mapping=compute_axis_mapping,
                mp=trainer_config.mp,
                cache_dir=ref_cache_dir,
                reference_model_path=reference_model_path,
                seq_len=sequence_length,
            )
        elif args.mode == "cached":
            ref_chosen_np, ref_rejected_np = _load_ref_cache(ref_cache_dir)

        if args.mode != "uncached":
            reference_model = None
            policy_model = _maybe_reshard_policy_for_eval(
                policy_model=policy_model,
                eval_param_layout=args.eval_param_layout,
                compute_axis_mapping=compute_axis_mapping,
            )

        dpo_model = DpoModel(policy=policy_model, reference=reference_model)

        # ── Set up eval loss function ─────────────────────────────────
        if args.mode == "uncached":

            @hax.named_jit(axis_resources=compute_axis_mapping)
            def eval_loss_fn(model: DpoModel, batch: DpoExample):
                return _loss_fn_uncached(model, batch, mp=trainer_config.mp, beta=beta)

            label = "UNCACHED (4 forward passes)"
        else:

            @hax.named_jit(axis_resources=compute_axis_mapping)
            def eval_loss_fn(model: DpoModel, batch):
                return _loss_fn_cached(model, batch, mp=trainer_config.mp, beta=beta)

            label = "CACHED (2 forward passes)"

            # Build a new dataset that carries cached values as numpy scalars per example.
            # The DataLoader will batch them and the JIT function wraps them in NamedArrays.
            class CachedRefDataset(AsyncDataset):
                def __init__(self, base_dataset, ref_chosen, ref_rejected):
                    self._base = base_dataset
                    self._ref_chosen = ref_chosen
                    self._ref_rejected = ref_rejected

                async def async_len(self):
                    return await self._base.async_len()

                def is_finite(self):
                    return self._base.is_finite()

                async def getitem_async(self, index):
                    example = await self._base.getitem_async(index)
                    return CachedDpoExample(
                        chosen=example.chosen,
                        rejected=example.rejected,
                        logp_ref_chosen=self._ref_chosen[index],
                        logp_ref_rejected=self._ref_rejected[index],
                    )

                async def get_batch(self, indices):
                    examples = await self._base.get_batch(indices)
                    return [
                        CachedDpoExample(
                            chosen=ex.chosen,
                            rejected=ex.rejected,
                            logp_ref_chosen=self._ref_chosen[idx],
                            logp_ref_rejected=self._ref_rejected[idx],
                        )
                        for idx, ex in zip(indices, examples, strict=True)
                    ]

            cached_dataset = CachedRefDataset(val_dataset, ref_chosen_np, ref_rejected_np)
            eval_loader = DataLoader(
                cached_dataset,
                batch_size=trainer_config.EvalBatch,
                mesh=trainer_config.device_mesh,
                axis_resources=compute_axis_mapping,
                max_buffered_batches=128,
                prefetch_size=32,
                allow_nondivisible_batch_size=True,
            )

        # ── Warmup ────────────────────────────────────────────────────
        if warmup_batches > 0:
            logger.info("Warmup: %d batches for JIT compilation (%s)", warmup_batches, label)
            t0 = time.time()
            warmup_iter = iter(eval_loader)
            for i in range(warmup_batches):
                batch = next(warmup_iter)
                batch_loss, _ = eval_loss_fn(dpo_model, batch)
                logger.info("Warmup %d/%d: loss=%.4f", i + 1, warmup_batches, batch_loss.item())
            warmup_time = time.time() - t0
            logger.info("Warmup done in %.1fs", warmup_time)
        else:
            warmup_time = 0.0
            logger.info("Warmup skipped")

        # ── Profiled eval ─────────────────────────────────────────────
        if args.profile:
            profile_dir = f"logs/{trainer_config.id}/dpo_eval_profile"
            logger.info("Starting profiler -> %s", profile_dir)
            jax.profiler.start_trace(profile_dir, create_perfetto_link=False, create_perfetto_trace=True)

        t0 = time.time()
        loss, metrics = eval_loss_loop(
            eval_loss_fn,
            dpo_model,
            eval_loader,
            max_batches=max_eval_batches,
            name=VAL_NAME,
        )
        eval_time = time.time() - t0

        if args.profile:
            logger.info("Stopping profiler...")
            jax.profiler.stop_trace()

        # ── Log results IMMEDIATELY ───────────────────────────────────
        _log_results(label, loss, metrics, eval_time, warmup_time, max_eval_batches)
        if cache_build_time > 0:
            logger.info("Cache build time:  %.1fs (one-time cost)", cache_build_time)

        # ── Best-effort W&B upload ────────────────────────────────────
        try:
            prefix = "eval_cached" if args.mode != "uncached" else "eval_uncached"
            levanter.tracker.log(
                {
                    f"{prefix}/loss": loss,
                    **{f"{prefix}/{k}": v for k, v in metrics.items()},
                },
                step=0,
            )
            if args.profile:
                levanter.tracker.current_tracker().log_artifact(
                    f"logs/{trainer_config.id}/dpo_eval_profile",
                    type="jax_profile",
                )
        except Exception:
            logger.warning("Failed to log to W&B", exc_info=True)

        try:
            levanter.tracker.current_tracker().finish()
        except Exception:
            logger.warning("Failed to finish tracker", exc_info=True)


if __name__ == "__main__":
    main()
