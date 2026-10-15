# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the 67B-A2B step-156k checkpoint on Paloma and UncheatableEval exactly as lm-eval 0.4.12 does.

The open-weight baselines in issue #8824 used lm-eval's ``loglikelihood_rolling`` through
the ``local-completions`` backend with ``max_length=4096``. This reproduces that recipe
for a grug checkpoint:

- Each document is scored independently; nothing is packed across documents.
- Tokens come from the HF tokenizer with ``add_special_tokens=False``; the prefix token
  is BOS (the Marin tokenizer has one), so the first document token is scored.
- Windows follow ``get_rolling_token_windows`` + ``make_disjoint_window`` with
  ``max_seq_len = max_length - 2`` and ``context_len = 1``: disjoint 4,094-token
  prediction windows, each conditioned on the single preceding token, except the final
  window, which keeps a full 4,094-token left context.
- Paloma uses the test split: lm-eval prefers a task's test split over its validation split.
- Per-subset BPB = -sum(loglik) / sum(utf8 bytes of doc text) / ln 2; macro BPB is the
  unweighted mean over the subsets of each suite.

Submit (target is ``step156k`` or ``skew8``)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_67b_a2b_step156k_paloma_lm_eval_rolling skew8
"""

import dataclasses
import gzip
import json
import logging
import math
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import wandb
from fray.cluster import ResourceConfig
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import load_checkpoint
from levanter.grug.attention import AttentionMask
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, InputName, this_output_path, versioned
from transformers import AutoTokenizer

from experiments.evals.exp1600_uncheatable_evals import ACTIVE_DATASETS, ALL_UNCHEATABLE_EVAL_DATASETS, uncheatable_eval
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.eval_67b_a2b_step156k_paloma_seq4k import CHECKPOINT_PATH, CHECKPOINT_STEP, MODEL_CONFIG
from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.marin_models import marin_tokenizer
from experiments.paloma import PALOMA_DATASETS_TO_DIR, paloma

logger = logging.getLogger(__name__)

# lm-eval's API backend stores max_length - 1 and rolls with max_length - 1 again.
_LM_EVAL_MAX_LENGTH: int = 4096
_WINDOW_LEN: int = _LM_EVAL_MAX_LENGTH - 2
# Row lengths for bucketing windows; each window needs len(context) + len(continuation) slots.
_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096)
_TOKENS_PER_BATCH: int = 1 << 20
# Fields that must equal the training run's recorded model config (seq_len legitimately differs).
_CHECKED_MODEL_FIELDS: tuple[str, ...] = (
    "qk_mult",
    "hidden_dim",
    "num_layers",
    "num_experts",
    "num_experts_per_token",
    "sliding_window",
    "disable_long_rope",
    "disable_pko",
)


@dataclass(frozen=True)
class RollingEvalConfig:
    run_id: str
    output_path: str
    resources: ResourceConfig
    model: GrugModelConfig
    checkpoint_path: str
    checkpoint_step: int
    tokenizer_id: str
    window_len: int
    mp: str
    suite: str
    dataset_paths: dict[str, InputName | str]


def rolling_windows(tokens: list[int], prefix_token: int, max_seq_len: int) -> Iterator[tuple[list[int], list[int]]]:
    """lm-eval ``get_rolling_token_windows(context_len=1)`` followed by ``make_disjoint_window``."""
    if not tokens:
        return
    first = min(max_seq_len, len(tokens))
    yield [prefix_token], tokens[:first]
    predicted = first
    while predicted < len(tokens):
        pred_len = min(len(tokens) - predicted, max_seq_len)
        end = predicted + pred_len
        window = tokens[max(0, end - max_seq_len - 1) : end - 1]
        yield window[: len(window) - (pred_len - 1)], tokens[end - pred_len : end]
        predicted += pred_len


def _read_docs(pattern: str) -> list[str]:
    fs, _ = fsspec.core.url_to_fs(pattern)
    paths = sorted(fs.glob(pattern))
    assert paths, f"no files match {pattern}"
    texts: list[str] = []
    for path in paths:
        with fs.open(path, "rb") as raw, gzip.GzipFile(fileobj=raw) as f:
            for line in f:
                texts.append(str(json.loads(line)["text"]))
    return texts


def _check_model_matches_training_run(model: GrugModelConfig, checkpoint_path: str) -> None:
    """Fail fast if the eval model disagrees with the config the checkpoint was trained with."""
    run_root = checkpoint_path.split("/checkpoints/")[0]
    with fsspec.open(f"{run_root}/.executor_info", "r") as f:
        trained = json.load(f)["config"]["model"]
    for field in _CHECKED_MODEL_FIELDS:
        assert (
            getattr(model, field) == trained[field]
        ), f"{field}: eval uses {getattr(model, field)!r} but {run_root} trained with {trained[field]!r}"
    logger.info("Model config matches %s: %s", run_root, {f: trained[f] for f in _CHECKED_MODEL_FIELDS})


def _run_eval_local(config: RollingEvalConfig) -> None:
    _check_model_matches_training_run(config.model, config.checkpoint_path)
    trainer = TrainerConfig(
        id=config.run_id,
        train_batch_size=_BUCKETS[0],
        num_train_steps=1,
        profiler=ProfilerConfig(),
        mp=jmp.get_policy(config.mp),
        tracker=WandbConfig(
            project="marin_moe",
            name=config.run_id,
            tags=[
                "moe",
                "67b_a2b",
                "checkpoint_eval",
                "lm_eval_rolling",
                config.suite,
                f"qk{config.model.qk_mult:g}",
                "issue8824_compare",
            ],
            group="june-tpu-67b-a2b-evals",
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        load_checkpoint=False,
    )
    trainer.initialize()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    prefix_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    # (subset index, row tokens, first scored position, number of scored tokens) per window.
    subsets = sorted(config.dataset_paths)
    windows: list[tuple[int, list[int], int, int]] = []
    subset_bytes = np.zeros(len(subsets), dtype=np.float64)
    subset_docs = np.zeros(len(subsets), dtype=np.int64)
    subset_tokens = np.zeros(len(subsets), dtype=np.int64)
    for si, name in enumerate(subsets):
        texts = _read_docs(str(config.dataset_paths[name]))
        encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False).input_ids
        for text, ids in zip(texts, encoded, strict=True):
            subset_bytes[si] += len(text.encode("utf-8"))
            subset_docs[si] += 1
            subset_tokens[si] += len(ids)
            for context, continuation in rolling_windows(ids, prefix_token, config.window_len):
                windows.append((si, context + continuation, len(context) - 1, len(continuation)))
        logger.info("[%s] docs=%d tokens=%d bytes=%d", name, subset_docs[si], subset_tokens[si], subset_bytes[si])
    logger.info("Built %d windows over %d docs", len(windows), int(subset_docs.sum()))

    subset_nll = np.zeros(len(subsets), dtype=np.float64)
    subset_scored = np.zeros(len(subsets), dtype=np.int64)
    with trainer.use_device_mesh():
        mesh = trainer.device_mesh

        @jax.jit
        def _init(key):
            return trainer.mp.cast_to_param(Transformer.init(config.model, key=key))

        t_load = time.perf_counter()
        model = load_checkpoint(
            _init(jax.random.PRNGKey(0)),
            config.checkpoint_path,
            subpath="params",
            axis_mapping=None,
            mesh=mesh,
            allow_partial=False,
        )
        model = trainer.mp.cast_to_compute(model)
        logger.info("Restored params in %.1fs", time.perf_counter() - t_load)

        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        replicated = NamedSharding(mesh, P())

        @eqx.filter_jit
        def _row_nll(model: Transformer, tokens, weight):
            tokens = reshard(tokens, batch_sharding)
            weight = reshard(weight, batch_sharding)
            seg = jnp.zeros(tokens.shape, dtype=jnp.int32)
            mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=(seg, seg))
            per_pos = model.next_token_loss(tokens, weight, mask=mask, reduction="none")
            weight = reshard(weight, jax.typeof(per_pos).sharding)
            return reshard(jnp.sum(per_pos * weight, axis=1), replicated)

        bucket_of = np.array([next(b for b in _BUCKETS if len(w[1]) <= b) for w in windows])
        t_eval = time.perf_counter()
        for bucket in _BUCKETS:
            idx = np.flatnonzero(bucket_of == bucket)
            batch_size = _TOKENS_PER_BATCH // bucket
            logger.info("bucket=%d windows=%d batch_size=%d", bucket, len(idx), batch_size)
            for start in range(0, len(idx), batch_size):
                chunk = idx[start : start + batch_size]
                tokens = np.zeros((batch_size, bucket), dtype=np.int32)
                weight = np.zeros((batch_size, bucket), dtype=np.float32)
                for row, wi in enumerate(chunk):
                    _, row_tokens, first, count = windows[wi]
                    tokens[row, : len(row_tokens)] = row_tokens
                    # Position p predicts token p + 1.
                    weight[row, first : first + count] = 1.0
                nll = np.asarray(jax.device_get(_row_nll(model, tokens, weight)))[: len(chunk)]
                for row, wi in enumerate(chunk):
                    si = windows[wi][0]
                    subset_nll[si] += nll[row]
                    subset_scored[si] += windows[wi][3]
        logger.info("Scored all windows in %.1fs", time.perf_counter() - t_eval)

    assert np.array_equal(subset_scored, subset_tokens), "every document token must be scored exactly once"
    log: dict[str, float] = {}
    bpbs = []
    for si, name in enumerate(subsets):
        bpb = subset_nll[si] / subset_bytes[si] / math.log(2)
        bpbs.append(bpb)
        log[f"lm_eval_rolling/{config.suite}/{name}/bpb"] = bpb
        log[f"lm_eval_rolling/{config.suite}/{name}/docs"] = float(subset_docs[si])
        log[f"lm_eval_rolling/{config.suite}/{name}/tokens"] = float(subset_tokens[si])
        logger.info("[%s] bpb=%.5f", name, bpb)
    macro_key = f"lm_eval_rolling/{config.suite}/macro_bpb"
    log[macro_key] = float(np.mean(bpbs))
    log[f"lm_eval_rolling/{config.suite}/docs"] = float(subset_docs.sum())
    logger.info("%s macro_bpb=%.5f over %d docs", config.suite, log[macro_key], int(subset_docs.sum()))
    wandb.log(log, step=config.checkpoint_step)
    wandb.finish()


def run_rolling_eval(config: RollingEvalConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_eval_local,
        resources=config.resources,
    )


@dataclass(frozen=True)
class EvalTarget:
    tag: str
    model: GrugModelConfig
    checkpoint_path: str
    checkpoint_step: int


STEP156K = EvalTarget("step156k", MODEL_CONFIG, CHECKPOINT_PATH, CHECKPOINT_STEP)
# 262K context extension from step 156k with 8x long-document skew (#8813). Same architecture, qk_mult=1.75.
CTXEXT_QK175_SKEW8 = EvalTarget(
    "ctxext_qk175_skew8_step157k",
    dataclasses.replace(MODEL_CONFIG, qk_mult=1.75),
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew8-c06695/"
    "checkpoints/step-157000/",
    157_000,
)

# lm-eval scores a task's test split when it has one, so the #8824 baselines used Paloma test, not val.
PALOMA_TEST = {name: paloma.cd(f"{part}/test/test*.jsonl.gz") for name, part in PALOMA_DATASETS_TO_DIR.items()}
# The 7 legacy UncheatableEval subsets (Sept 1-14 2025 dumps, 7,052 docs), the suite Qwen3.5 and Hero used in #8824.
UNCHEATABLE_LEGACY = {name: uncheatable_eval.cd(ALL_UNCHEATABLE_EVAL_DATASETS[name]) for name in ACTIVE_DATASETS}


def rolling_eval_steps(target: EvalTarget) -> list[ExecutorStep]:
    steps = []
    for suite, paths in (("paloma", PALOMA_TEST), ("uncheatable_eval", UNCHEATABLE_LEGACY)):
        run_id = f"moe_67b_a2b_d2560_{target.tag}_{suite}_lm_eval_rolling4096"
        steps.append(
            ExecutorStep(
                name=f"grug/{run_id}",
                fn=run_rolling_eval,
                config=RollingEvalConfig(
                    run_id=run_id,
                    output_path=this_output_path(),
                    resources=versioned(ResourceConfig.with_tpu("v4-64")),
                    model=versioned(target.model),
                    checkpoint_path=versioned(target.checkpoint_path),
                    checkpoint_step=target.checkpoint_step,
                    tokenizer_id=versioned(marin_tokenizer),
                    window_len=versioned(_WINDOW_LEN),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    suite=suite,
                    dataset_paths=paths,
                ),
            )
        )
    return steps


if __name__ == "__main__":
    targets = {"step156k": STEP156K, "skew8": CTXEXT_QK175_SKEW8}
    target = targets[sys.argv.pop(1)]
    executor_main(
        steps=rolling_eval_steps(target),
        description=(
            f"67B-A2B {target.tag} (qk_mult={target.model.qk_mult}) Paloma (test split) + UncheatableEval BPB "
            f"with lm-eval 0.4.12 loglikelihood_rolling semantics (max_length={_LM_EVAL_MAX_LENGTH}, "
            "per-document, BOS prefix). v4-64 us-central2."
        ),
    )
