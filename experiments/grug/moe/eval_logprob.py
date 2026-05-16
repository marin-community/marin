# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal logprob lm-eval-harness on a finished grug-MoE checkpoint.

Clean-room replacement for `levanter.eval_harness` that does only what
logprob evals need:

  1. Load a grug `Transformer` from a Levanter-native checkpoint.
  2. Run lm-eval's `evaluate()` once with a `GrugLM` whose `loglikelihood`
     tokenizes the requests, runs the JIT'd forward pass loop, and returns
     real logprobs in one call. This avoids the two-pass collect-then-replay
     pattern, which broke for inlined gsm8k / humaneval tasks because
     lm-eval's two `evaluate` calls produced non-matching request sets.
  3. Hand the resulting `(loglikelihood, is_greedy)` pairs back to lm-eval
     for metric aggregation.

Eval-mode bumps `GrugModelConfig.capacity_factor` so the routed MoE doesn't
silently drop tokens at inference (training defaults to 1.0; we evaluate at
8.0).

One ExecutorStep per (model, task) so adding tasks doesn't invalidate
cached results for existing tasks.

Multi-host coordination: chief enters `evaluate()`; non-chief hosts enter
`_run_listener_loop` which mirrors chief's `broadcast_one_to_all` /
`process_allgather` calls so JAX collectives stay in lockstep. Chief sends a
`n=-1` sentinel after `evaluate()` returns to release the listeners.
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from jax.experimental import multihost_utils
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.tokenizers import load_tokenizer
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path
from marin.execution.remote import remote
from rigging.filesystem import filesystem as marin_filesystem

from experiments.exp1337_eval_suite import LOGPROB_TASKS as _TASK_SPECS
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.marin_models import marin_tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    hidden_dim: int
    budget: float
    checkpoint_subpath: str


def _task_key(task: EvalTaskConfig) -> str:
    """Output-dir / wandb-run key for an `EvalTaskConfig`."""
    return task.task_alias or f"{task.name}_{task.num_fewshot}shot"


# `_TASK_SPECS` = exp1337's `LOGPROB_TASKS`: mmlu_sl_verb (0 + 5 shot),
# logprob_gsm8k_5shot (inlined), logprob_humaneval_10shot (inlined). Extend
# the tuple locally to add more tasks; bare-continuation MCQ tasks
# (medmcqa, csqa, boolq, ...) tend to be noisy at small scale — see the
# `*_sl_verb` variants in the Helw150 fork (Helw150/lm-evaluation-harness)
# for surface-form-competition-free alternatives.


@eqx.filter_jit
def _batch_logprobs(transformer, tokens, loss_weight):
    """Sum-of-log-probs per row via grug's fused cross-entropy.

    Args:
        tokens:      ``int32 (B, S)`` — full prompt+continuation+pad.
        loss_weight: ``f32   (B, S)`` — 1.0 at positions whose label is a
                     continuation token, 0.0 elsewhere. (Recall
                     ``next_token_loss`` shifts so position ``i`` predicts
                     ``tokens[..., i+1]``.)

    Returns ``f32 (B,)`` total log-likelihood per row.
    """
    per_pos_loss = transformer.next_token_loss(tokens, loss_weight, reduction="none")
    return -jnp.sum(per_pos_loss, axis=-1)


def _lm_eval_spec(task: EvalTaskConfig) -> str | dict:
    """Spec for `get_task_dict`.

    Bare registered tasks (no inlined task_kwargs) are passed as a **string**
    so lm-eval skips the registered-task override path, which drops fields
    inherited via ``include:`` chains (e.g., ``arc_challenge``'s
    dataset_path / *_split from arc_easy). Inlined tasks (gsm8k/humaneval
    in exp1337) still need the dict form. ``num_fewshot`` and ``task_alias``
    are applied post-build via ``set_config`` so they don't re-trigger the
    drop.
    """
    if not task.task_kwargs:
        return task.name
    spec: dict = {"task": task.name}
    if task.task_alias:
        spec["task_alias"] = task.task_alias
    spec.update(task.task_kwargs)
    return spec


def _apply_num_fewshot(task_dict: dict, num_fewshot: int) -> None:
    """Walk ``task_dict`` and set ``num_fewshot`` on each leaf Task.

    Mirrors the post-build override that ``simple_evaluate`` does so we get
    consistent few-shot behavior without re-triggering the include-chain
    drop bug at build time.
    """
    for v in task_dict.values():
        if hasattr(v, "set_config"):
            v.set_config(key="num_fewshot", value=num_fewshot)
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
            _apply_num_fewshot(v[1], num_fewshot)
        elif isinstance(v, dict):
            _apply_num_fewshot(v, num_fewshot)


def _tokenize_request(
    prompt: str,
    continuation: str,
    tokenizer,
    *,
    max_seq_len: int,
    max_cont_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a single (prompt, continuation) into ``(tokens, loss_weight)`` rows.

    Returns ``(int32 (max_seq_len,), f32 (max_seq_len,))``.
    """
    pad_id = tokenizer.eos_token_id
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    cont_ids = list(tokenizer.encode(continuation, add_special_tokens=False))
    if len(cont_ids) == 0:
        raise ValueError(f"Continuation tokenized to empty string: {continuation!r}")
    if len(cont_ids) > max_cont_len:
        logger.warning(
            "Continuation has %d tokens > max_cont_len=%d; truncating from end.",
            len(cont_ids),
            max_cont_len,
        )
        cont_ids = cont_ids[:max_cont_len]

    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[-max_seq_len:]
        cont_start = max_seq_len - len(cont_ids)
    else:
        cont_start = len(prompt_ids)

    tokens = np.full((max_seq_len,), pad_id, dtype=np.int32)
    tokens[: len(full_ids)] = full_ids
    loss_weight = np.zeros((max_seq_len,), dtype=np.float32)

    # Position `cont_start + k - 1` predicts `cont_ids[k]`.
    pred_start = max(0, cont_start - 1)
    pred_end = min(max_seq_len, cont_start - 1 + len(cont_ids))
    if pred_end > pred_start:
        loss_weight[pred_start:pred_end] = 1.0
    return tokens, loss_weight


def _run_forward_pass_distributed(
    transformer,
    n: int,
    tokens: np.ndarray,
    loss_weight: np.ndarray,
    *,
    max_seq_len: int,
    batch_size: int,
    pad_id: int,
    is_chief: bool,
) -> np.ndarray | None:
    """Broadcast tokens/loss_weight across hosts, run the JIT'd forward pass loop, gather.

    Both chief and non-chief enter this with the same ``n`` (chief sets it from
    real requests; non-chief receives it via the caller's broadcast). On chief,
    ``tokens`` and ``loss_weight`` carry real data; on non-chief they're
    placeholder zeros that get overwritten by the broadcast. Returns the
    chief's per-row logprobs (``np.ndarray (n,)``) or ``None`` for non-chief.
    """
    if n == 0:
        return np.zeros(0, dtype=np.float32) if is_chief else None

    tokens_jnp = multihost_utils.broadcast_one_to_all(jnp.asarray(tokens))
    loss_weight_jnp = multihost_utils.broadcast_one_to_all(jnp.asarray(loss_weight))

    all_lps = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        real = end - start
        if real < batch_size:
            # Pad the last partial batch up to batch_size so JIT signature is stable.
            pad_tok = jnp.full((batch_size - real, max_seq_len), pad_id, dtype=jnp.int32)
            pad_lw = jnp.zeros((batch_size - real, max_seq_len), dtype=jnp.float32)
            batch_tokens = jnp.concatenate([tokens_jnp[start:end], pad_tok], axis=0)
            batch_lw = jnp.concatenate([loss_weight_jnp[start:end], pad_lw], axis=0)
        else:
            batch_tokens = tokens_jnp[start:end]
            batch_lw = loss_weight_jnp[start:end]
        sum_lp = _batch_logprobs(transformer, batch_tokens, batch_lw)
        sum_lp_full = multihost_utils.process_allgather(sum_lp, tiled=True)
        all_lps[start:end] = np.asarray(sum_lp_full)[:real]

    return all_lps if is_chief else None


def _make_grug_lm(transformer, tokenizer, *, max_seq_len, max_cont_len, batch_size, pad_id):
    """Build a `GrugLM` (lm-eval `LM`) backed by ``transformer`` for live forward passes.

    Each `loglikelihood(requests)` call:
      1. Tokenizes all requests on the chief host.
      2. Broadcasts `n`, then `tokens` and `loss_weight` to non-chief hosts.
      3. Runs the JIT'd batched forward pass with `process_allgather` per batch.
      4. Returns ``[(logprob, False), ...]`` to lm-eval for metric aggregation.

    Non-chief hosts mirror these collectives in `_run_listener_loop`.
    """
    from lm_eval.api.model import LM

    class GrugLM(LM):
        def loglikelihood(self, requests):
            n = len(requests)
            tokens = np.zeros((n, max_seq_len), dtype=np.int32)
            loss_weight = np.zeros((n, max_seq_len), dtype=np.float32)
            for i, r in enumerate(requests):
                prompt, continuation = r.args
                t, lw = _tokenize_request(
                    prompt,
                    continuation,
                    tokenizer,
                    max_seq_len=max_seq_len,
                    max_cont_len=max_cont_len,
                )
                tokens[i] = t
                loss_weight[i] = lw
            # Tell non-chief listeners how many requests are in this batch.
            multihost_utils.broadcast_one_to_all(jnp.array([n], dtype=jnp.int32))
            all_lps = _run_forward_pass_distributed(
                transformer,
                n,
                tokens,
                loss_weight,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                pad_id=pad_id,
                is_chief=True,
            )
            return [(float(lp), False) for lp in all_lps]

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("rolling logprob not supported")

        def generate_until(self, requests):
            raise NotImplementedError("generation not supported")

        @property
        def eot_token_id(self):
            return tokenizer.eos_token_id

        @property
        def max_length(self):
            return max_seq_len

        @property
        def max_gen_toks(self):
            return 0

        @property
        def batch_size(self):
            return batch_size

        @property
        def device(self):
            return "tpu"

        def tok_encode(self, s):
            return list(tokenizer.encode(s, add_special_tokens=False))

        def tok_decode(self, ids):
            return tokenizer.decode(ids)

    return GrugLM()


def _run_listener_loop(transformer, *, max_seq_len, batch_size, pad_id):
    """Non-chief listener that mirrors chief's `loglikelihood` collectives.

    Reads the broadcast `n` per chief loglikelihood call; ``n < 0`` is the
    sentinel chief sends after `evaluate()` returns, which exits the loop.
    For each `n >= 0` iteration, runs the same broadcast + forward + allgather
    sequence as `_run_forward_pass_distributed(is_chief=False)` so JAX
    collectives stay in lockstep.
    """
    while True:
        n_arr = multihost_utils.broadcast_one_to_all(jnp.array([0], dtype=jnp.int32))
        n = int(n_arr[0])
        if n < 0:
            break
        _run_forward_pass_distributed(
            transformer,
            n,
            np.zeros((n, max_seq_len), dtype=np.int32),
            np.zeros((n, max_seq_len), dtype=np.float32),
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            pad_id=pad_id,
            is_chief=False,
        )


@dataclass(frozen=True)
class GrugLogprobEvalConfig:
    grug_model_config: GrugModelConfig
    checkpoint_path: str
    output_path: str
    wandb_run_name: str
    task: EvalTaskConfig
    eval_capacity_factor: float = 8.0
    max_eval_instances: int | None = None
    batch_size: int = 8
    max_cont_len: int = 256
    wandb_tags: tuple[str, ...] = ()


def run_grug_logprob_eval(config: GrugLogprobEvalConfig) -> None:
    """Multi-host-safe single-task logprob eval.

    Layout:
      - All hosts initialize JAX/mesh and load the model (Levanter handles
        cross-host sharding).
      - **Host 0** runs `lm_eval.evaluator.evaluate` with a `GrugLM` whose
        `loglikelihood` does tokenize + broadcast + forward pass + allgather
        in a single call, returning real logprobs to lm-eval directly.
      - **Other hosts** run `_run_listener_loop`, mirroring chief's collectives.
      - After `evaluate()` returns, chief broadcasts a sentinel ``n=-1`` to
        release the listener loop, then writes `results.json`.
    """
    eval_grug = dataclasses.replace(
        config.grug_model_config,
        capacity_factor=config.eval_capacity_factor,
    )
    max_seq_len = eval_grug.max_seq_len

    # grug `Transformer.init` reshards with explicit axes; mirror the training
    # mesh so the load path matches checkpoint shardings.
    trainer_config = TrainerConfig(
        tracker=WandbConfig(
            project="marin_moe",
            name=config.wandb_run_name,
            tags=list(config.wandb_tags),
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        per_device_eval_parallelism=1,
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
    )
    trainer_config.initialize()
    is_chief = jax.process_index() == 0

    with trainer_config.use_device_mesh():
        key = jax.random.PRNGKey(0)
        with use_cpu_device():
            transformer_shape = eqx.filter_eval_shape(Transformer.init, eval_grug, key=key)
            ckpt_path = latest_checkpoint_path(str(config.checkpoint_path))
            transformer = load_checkpoint(
                transformer_shape,
                ckpt_path,
                subpath="params",
                axis_mapping=trainer_config.parameter_axis_mapping,
            )

        tokenizer = load_tokenizer(marin_tokenizer)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer = dataclasses.replace(tokenizer, _pad_id=tokenizer.eos_token_id)

        results: dict | None = None
        if is_chief:
            from lm_eval import evaluator as lm_eval_evaluator
            from lm_eval.tasks import TaskManager, get_task_dict

            task_dict = get_task_dict([_lm_eval_spec(config.task)], task_manager=TaskManager())
            _apply_num_fewshot(task_dict, config.task.num_fewshot)
            lm = _make_grug_lm(
                transformer,
                tokenizer,
                max_seq_len=max_seq_len,
                max_cont_len=config.max_cont_len,
                batch_size=config.batch_size,
                pad_id=tokenizer.eos_token_id,
            )
            results = lm_eval_evaluator.evaluate(
                lm=lm,
                task_dict=task_dict,
                limit=config.max_eval_instances,
                log_samples=False,
            )
            # Sentinel: release non-chief listeners.
            multihost_utils.broadcast_one_to_all(jnp.array([-1], dtype=jnp.int32))
        else:
            _run_listener_loop(
                transformer,
                max_seq_len=max_seq_len,
                batch_size=config.batch_size,
                pad_id=tokenizer.eos_token_id,
            )

    if is_chief and results is not None:
        results_path = os.path.join(config.output_path, "results.json")
        logger.info(f"Uploading logprob results to {results_path}")
        fs = marin_filesystem("gcs")
        with fs.open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda v: repr(v))


# ---------------------------------------------------------------------------
# Launcher: one ExecutorStep per (model, task) cross product.
# ---------------------------------------------------------------------------

_TARGET_STEPS = 2**14

# Populate with `ModelSpec(slug, hidden_dim, budget, checkpoint_subpath)` per
# trained checkpoint. The cross-product `_MODEL_SPECS x _TASK_SPECS` becomes
# one `ExecutorStep` each; cached per (model, task) so adding either
# dimension is incremental.
_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        slug="grug_moe_mix_d512",
        hidden_dim=512,
        budget=2.19e17,
        checkpoint_subpath="grug/grug_moe_mix_d512-2.19e+17-e6a48f/checkpoints",
    ),
)


def _build_eval_step(model: ModelSpec, task: EvalTaskConfig) -> ExecutorStep:
    grug_model, _, _, _ = build_from_heuristic(
        budget=model.budget,
        hidden_dim=model.hidden_dim,
        target_steps=_TARGET_STEPS,
    )
    return ExecutorStep(
        name=f"evaluation/grug_logprob/{model.slug}-{model.budget:.2e}/{_task_key(task)}",
        fn=remote(
            run_grug_logprob_eval,
            resources=ResourceConfig.with_tpu("v5p-8", zone="us-east5-a"),
            pip_dependency_groups=["eval", "tpu"],
        ),
        config=GrugLogprobEvalConfig(
            grug_model_config=grug_model,
            checkpoint_path=InputName.hardcoded(model.checkpoint_subpath),
            output_path=this_output_path(),
            wandb_run_name=f"{model.slug}_{_task_key(task)}",
            task=task,
            wandb_tags=("grug", "logprob_eval", model.slug, _task_key(task)),
        ),
    )


grug_logprob_eval_steps: list[ExecutorStep] = [
    _build_eval_step(model, task) for model in _MODEL_SPECS for task in _TASK_SPECS
]


if __name__ == "__main__":
    executor_main(steps=grug_logprob_eval_steps)
