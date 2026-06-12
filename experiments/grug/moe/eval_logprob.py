# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal logprob lm-eval-harness on a finished grug-MoE checkpoint.

Loads a grug `Transformer` from a Levanter-native checkpoint and runs
lm-eval's `evaluate()` once with a `GrugLM` whose `loglikelihood` tokenizes
the requests, runs the JIT'd forward pass loop, and returns real logprobs.
Keeping the forward pass inside `loglikelihood` puts request generation
and scoring in the same `evaluate()` call, which is required for tasks
whose few-shot sampler advances RNG state during the call (the sampler
state isn't reset between calls).

Eval-mode bumps `GrugModelConfig.capacity_factor` so the routed MoE doesn't
silently drop tokens at inference (training runs with capacity_factor=1.0;
evaluate at 8.0).

Each (model, task) is a separate `ExecutorStep`, so adding either dimension
is incremental — already-evaluated cells stay cached.

Multi-host coordination: chief enters `evaluate()`; non-chief hosts enter
`_run_listener_loop` which mirrors chief's `broadcast_one_to_all` /
`process_allgather` calls so JAX collectives stay in lockstep. Chief sends
an `n=-1` sentinel after `evaluate()` returns to release the listeners.
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
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.execution.types import this_output_path
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
    return task.task_alias or f"{task.name}_{task.num_fewshot}shot"


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
    """Build the spec entry passed to `get_task_dict`.

    Registered tasks with no `task_kwargs` are returned as bare strings;
    lm-eval loads the cached config directly. The dict form triggers
    lm-eval's override-merge path, which drops fields inherited via
    ``include:`` chains in the task's yaml. Inlined tasks need the dict
    form because the full config lives in ``task_kwargs``. ``num_fewshot``
    and ``task_alias`` are deliberately left out of the returned dict and
    applied post-build via `_apply_num_fewshot`.
    """
    if not task.task_kwargs:
        return task.name
    spec: dict = {"task": task.name}
    spec.update(task.task_kwargs)
    # Inline tasks have no include-chain to lose so it's safe to also stamp
    # the alias here; `_apply_task_field` re-stamps post-build for the
    # registered-task path.
    if task.task_alias:
        spec["task_alias"] = task.task_alias
    return spec


def _inject_loglikelihood_bpb(task_dict: dict) -> None:
    """Replace each leaf loglikelihood task's ``process_results`` with one that
    emits ``bpb``/``nll`` alongside ``perplexity``/``acc``.

    lm-eval's default ConfigurableTask.process_results for ``output_type:
    loglikelihood`` ignores the ``bpb``/``nll`` entries in ``metric_list``
    (that computation is buried in the ``multiple_choice`` branch). Without
    this patch, inline tasks like ``logprob_gsm8k_5shot`` produce empty
    ``results[name]`` dicts even though the eval ran on every sample.
    """
    import math

    NAT_TO_BIT = 1.0 / math.log(2)

    def _make_processor(task):
        def _process(doc, results):
            log_likelihood, is_greedy = results[0]
            target_text = task.doc_to_target(doc)
            n_bytes = max(1, len(str(target_text).encode("utf-8")))
            return {
                "bpb": (-log_likelihood / n_bytes) * NAT_TO_BIT,
                "nll": -log_likelihood,
                "perplexity": log_likelihood,
                "acc": int(is_greedy),
            }
        return _process

    for v in task_dict.values():
        if hasattr(v, "_config"):
            output_type = getattr(v._config, "output_type", None)
            if output_type == "loglikelihood":
                v._config.process_results = _make_processor(v)
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
            _inject_loglikelihood_bpb(v[1])
        elif isinstance(v, dict):
            _inject_loglikelihood_bpb(v)


def _apply_task_field(task_dict: dict, key: str, value) -> None:
    """Walk ``task_dict`` and set ``key`` on each leaf Task.

    Mirrors lm-eval's `simple_evaluate`, which sets `num_fewshot` after
    task construction. Setting it post-build keeps the construction-time
    override path (which drops include-chain fields — see `_lm_eval_spec`)
    out of play. Used for both `num_fewshot` (for the few-shot count) and
    `alias` (so `prepare_print_tasks` doesn't crash on `None + str` when
    aggregating inline-task results).
    """
    for v in task_dict.values():
        if hasattr(v, "set_config"):
            v.set_config(key=key, value=value)
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
            _apply_task_field(v[1], key, value)
        elif isinstance(v, dict):
            _apply_task_field(v, key, value)


def _tokenize_request(
    prompt: str,
    continuation: str,
    tokenizer,
    *,
    max_seq_len: int,
    max_cont_len: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    # Non-chief passes placeholder zeros for tokens/loss_weight; the broadcast
    # overwrites them with chief's data.
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
    """Build the chief-side `lm-eval` LM. Non-chief hosts mirror its collectives in `_run_listener_loop`."""
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
            # Paired with the listener loop's matching broadcast.
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
    eval_grug = dataclasses.replace(
        config.grug_model_config,
        capacity_factor=config.eval_capacity_factor,
    )
    max_seq_len = eval_grug.max_seq_len

    # grug `Transformer.init` reshards with explicit axes; mirror the training
    # mesh so the load path matches checkpoint shardings.
    # NoopTracker: skip wandb entirely. The eval results.json is the only artifact
    # we need, and the wandb-finalize hang on shutdown blocks iris from reaping
    # the worker for ~10-15 min per task, halving effective throughput.
    trainer_config = TrainerConfig(
        tracker=NoopConfig(),
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

            # Defensive monkey-patch: in lm-eval commit d5e3391, `prepare_print_tasks`
            # crashes with `None + str` when an inline task's alias propagates as None
            # somewhere between our spec-time injection and the aggregation read. The
            # propagation chain works locally but not in the container build; rather
            # than chase further version skew, wrap the function to fall back to the
            # task name when the read returns None.
            from lm_eval import evaluator_utils as _eu
            if not getattr(_eu, "_marin_prepare_print_tasks_patched", False):
                _orig_prep = _eu.prepare_print_tasks
                def _safe_prep(task_dict, results, task_depth=0, group_depth=0):
                    # Try the upstream aggregation but never let it block result
                    # serialisation. On failure, hand back the input `results`
                    # dict as the aggregated form — it already contains every
                    # metric/alias entry consolidate_results populated, which is
                    # what evaluate() ends up dumping to results.json as the
                    # "results" key. Falling back to an empty defaultdict (the
                    # previous workaround) silently produced empty results.json
                    # files even though the eval itself ran successfully.
                    for _name, _r in list(results.items()):
                        if isinstance(_r, dict):
                            _r["alias"] = str(_name)
                    try:
                        return _orig_prep(task_dict, results, task_depth, group_depth)
                    except (TypeError, KeyError, AttributeError) as e:
                        logger.warning("MARIN_SAFE_PREP swallowed %s: %s", type(e).__name__, e)
                        return dict(results), {}
                _eu.prepare_print_tasks = _safe_prep
                _eu._marin_prepare_print_tasks_patched = True
                # evaluator imports `prepare_print_tasks` by name at module-load,
                # so rebind the symbol it actually calls.
                lm_eval_evaluator.prepare_print_tasks = _safe_prep
                logger.warning(
                    "MARIN_PATCH installed; eu.id=%r evaluator.id=%r same=%s",
                    id(_eu.prepare_print_tasks),
                    id(lm_eval_evaluator.prepare_print_tasks),
                    _eu.prepare_print_tasks is lm_eval_evaluator.prepare_print_tasks,
                )

            task_dict = get_task_dict([_lm_eval_spec(config.task)], task_manager=TaskManager())
            _apply_task_field(task_dict, "num_fewshot", config.task.num_fewshot)
            # lm-eval's ConfigurableTask.process_results for output_type=loglikelihood
            # only returns {perplexity, acc} — `bpb`/`nll` from the metric_list never
            # land in task_output.sample_metrics, so consolidate_results writes empty
            # `results[name]` dicts. Inject a callable that computes them by hand.
            _inject_loglikelihood_bpb(task_dict)
            if config.task.task_alias:
                # lm-eval's `consolidate_results` propagates the alias via
                # `task_config["task_alias"]` (evaluator_utils.py:358), reading
                # the TaskConfig dataclass field; without this, inline tasks
                # crash on `None + str` aggregation in `prepare_print_tasks`.
                _apply_task_field(task_dict, "task_alias", config.task.task_alias)
            # Diagnostic: dump the leaf task's dump_config alias keys so we can
            # see in container logs whether the patch propagated. Also mimic
            # what get_task_list does so we can verify the TaskOutput snapshot.
            from lm_eval.evaluator_utils import get_task_list, TaskOutput
            _et = get_task_list(task_dict)
            for _to in _et:
                logger.warning(
                    "TASK_ALIAS_DEBUG2 to.task_name=%r to.task_alias=%r "
                    "to.task_config_keys=%r task_alias_in_config=%r",
                    _to.task_name, _to.task_alias,
                    sorted(list(_to.task_config.keys())) if hasattr(_to.task_config, "keys") else "NO_KEYS",
                    _to.task_config.get("task_alias") if hasattr(_to.task_config, "get") else "NO_GET",
                )
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


_TARGET_STEPS = 2**14

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
