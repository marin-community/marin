# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score recorded lm-eval multiple-choice requests with a 67B-A2B grug checkpoint.

Stage 2 of ``lm_eval_mcq_record_replay.py``. Each task's ``{task}.npz`` holds the exact token rows lm-eval
0.4.12's ``local-completions`` model would send (BOS-prefixed, left-truncated to 4096) and each row's context
length. For every row this writes the summed log-probability of tokens ``ctxlen:``, which is what the
completions backend sums from the server's echoed prompt logprobs. Rows are scored independently (no packing).

``is_greedy`` is written as False: computing it needs full-vocabulary logits, and none of the recorded tasks
use it (their metrics are ``acc``/``acc_norm``; the replay stage asserts this).

Submit (target is ``step156k`` or ``skew8``)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_67b_a2b_lm_eval_mcq step156k
"""

import io
import json
import logging
import sys
import time
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
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.eval_67b_a2b_step156k_paloma_lm_eval_rolling import (
    CTXEXT_QK175_SKEW8,
    STEP156K,
    EvalTarget,
    check_model_matches_training_run,
)
from experiments.grug.moe.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)

# Written by ``lm_eval_mcq_record_replay.py record`` (lm-eval 0.4.12, Marin tokenizer, max_length 4096).
REQUESTS_DIR = "gs://marin-us-central2/scratch/larry/lm_eval_mcq_4096/requests"
TASKS: tuple[str, ...] = ("mmlu", "leaderboard_mmlu_pro", "arc_challenge", "hellaswag", "winogrande", "piqa")
MAX_LENGTH = 4096
_BUCKETS: tuple[int, ...] = tuple(1 << e for e in range(7, MAX_LENGTH.bit_length()))
_TOKENS_PER_BATCH: int = 1 << 20


@dataclass(frozen=True)
class McqEvalConfig:
    run_id: str
    output_path: str
    resources: ResourceConfig
    model: GrugModelConfig
    checkpoint_path: str
    checkpoint_step: int
    requests_dir: str
    tasks: tuple[str, ...]
    mp: str


def _load_requests(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fsspec.open(path, "rb") as f:
        data = np.load(io.BytesIO(f.read()))
        return data["tokens"], data["offsets"], data["ctxlens"]


def _run_eval_local(config: McqEvalConfig) -> None:
    check_model_matches_training_run(config.model, config.checkpoint_path)
    trainer = TrainerConfig(
        id=config.run_id,
        train_batch_size=_TOKENS_PER_BATCH // _BUCKETS[-1],
        num_train_steps=1,
        profiler=ProfilerConfig(),
        mp=jmp.get_policy(config.mp),
        tracker=WandbConfig(
            project="marin_moe",
            name=config.run_id,
            tags=["moe", "67b_a2b", "checkpoint_eval", "lm_eval_mcq", f"qk{config.model.qk_mult:g}"],
            group="june-tpu-67b-a2b-evals",
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        load_checkpoint=False,
    )
    trainer.initialize()

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

        for task in config.tasks:
            tokens, offsets, ctxlens = _load_requests(f"{config.requests_dir}/{task}.npz")
            with fsspec.open(f"{config.requests_dir}/{task}.json", "r") as f:
                digest = json.load(f)["digest"]
            lengths = np.diff(offsets)
            assert lengths.max() <= MAX_LENGTH and (ctxlens > 0).all() and (lengths > ctxlens).all()
            bucket_of = np.searchsorted(np.asarray(_BUCKETS), lengths)
            nll = np.zeros(len(lengths), dtype=np.float64)
            t_task = time.perf_counter()
            for bi, bucket in enumerate(_BUCKETS):
                idx = np.flatnonzero(bucket_of == bi)
                batch_size = _TOKENS_PER_BATCH // bucket
                for start in range(0, len(idx), batch_size):
                    chunk = idx[start : start + batch_size]
                    batch_tokens = np.zeros((batch_size, bucket), dtype=np.int32)
                    weight = np.zeros((batch_size, bucket), dtype=np.float32)
                    for row, ri in enumerate(chunk):
                        length = lengths[ri]
                        batch_tokens[row, :length] = tokens[offsets[ri] : offsets[ri + 1]]
                        # Position p predicts token p + 1, so continuation tokens ctxlen.. come from ctxlen - 1..
                        weight[row, ctxlens[ri] - 1 : length - 1] = 1.0
                    out = np.asarray(jax.device_get(_row_nll(model, batch_tokens, weight)))
                    nll[chunk] = out[: len(chunk)]
            logger.info("[%s] scored %d rows in %.1fs", task, len(lengths), time.perf_counter() - t_task)
            if jax.process_index() == 0:
                buf = io.BytesIO()
                np.savez(buf, logprob=-nll, greedy=np.zeros(len(nll), dtype=bool), digest=np.array(digest))
                with fsspec.open(f"{config.output_path}/{task}.npz", "wb") as f:
                    f.write(buf.getvalue())
            wandb.log({f"lm_eval_mcq/{task}/rows": len(lengths), f"lm_eval_mcq/{task}/tokens": int(offsets[-1])})
    logger.info("Wrote scores to %s", config.output_path)
    wandb.finish()


def run_mcq_eval(config: McqEvalConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_eval_local,
        resources=config.resources,
    )


def mcq_eval_step(target: EvalTarget) -> ExecutorStep:
    run_id = f"moe_67b_a2b_d2560_{target.tag}_lm_eval_mcq{MAX_LENGTH}"
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_mcq_eval,
        config=McqEvalConfig(
            run_id=run_id,
            output_path=this_output_path(),
            resources=versioned(ResourceConfig.with_tpu("v4-64")),
            model=versioned(target.model),
            checkpoint_path=versioned(target.checkpoint_path),
            checkpoint_step=target.checkpoint_step,
            requests_dir=versioned(REQUESTS_DIR),
            tasks=versioned(TASKS),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        ),
    )


if __name__ == "__main__":
    targets = {"step156k": STEP156K, "skew8": CTXEXT_QK175_SKEW8}
    target = targets[sys.argv.pop(1)]
    executor_main(
        steps=[mcq_eval_step(target)],
        description=(
            f"67B-A2B {target.tag} (qk_mult={target.model.qk_mult}) lm-eval 0.4.12 multiple-choice scoring of "
            f"recorded requests ({', '.join(TASKS)}), max_length={MAX_LENGTH}. v4-64 us-central2."
        ),
    )
