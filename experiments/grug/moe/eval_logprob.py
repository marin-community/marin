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
import math
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
from marin.execution.remote import remote
from marin.execution.types import this_output_path
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
    and ``alias`` are deliberately left out of the returned dict and
    applied post-build via `_apply_num_fewshot`.
    """
    if not task.task_kwargs:
        return task.name
    spec: dict = {"task": task.name}
    spec.update(task.task_kwargs)
    if task.task_alias:
        spec["alias"] = task.task_alias
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
    from lm_eval.api.task import Task

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
        if isinstance(v, Task):
            if v._config.output_type == "loglikelihood":
                v._config.process_results = _make_processor(v)
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
            _inject_loglikelihood_bpb(v[1])
        elif isinstance(v, dict):
            _inject_loglikelihood_bpb(v)


def _inject_bpb_into_metric_list(task_dict: dict) -> None:
    """Ensure every leaf task's ``metric_list`` and ``_metric_fn_list``
    include ``bpb``.

    Most lm-eval task YAMLs only declare ``acc``/``acc_norm`` in their
    metric_list; for ``multiple_choice`` output_type,
    ``ConfigurableTask.process_results`` only emits ``bpb`` when
    ``"bpb" in self._metric_fn_list.keys()``. That dict is built once at
    task construction from ``metric_list``, so we patch both: append a
    metric_list entry (for downstream introspection / config dumps) AND
    insert the bpb metric_fn into ``_metric_fn_list``, which the runtime reads
    when deciding which metrics to emit.
    """
    from lm_eval.api.registry import get_aggregation, get_metric
    from lm_eval.api.task import Task

    bpb_entry = {"metric": "bpb", "aggregation": "mean", "higher_is_better": False}
    for v in task_dict.values():
        if isinstance(v, Task):
            mlist = list(v._config.metric_list or [])
            if not any(isinstance(m, dict) and m.get("metric") == "bpb" for m in mlist):
                mlist.append(bpb_entry)
                v._config.metric_list = mlist
            if "bpb" not in v._metric_fn_list:
                v._metric_fn_list["bpb"] = get_metric("bpb")
                v._aggregation_list["bpb"] = get_aggregation("mean")
                v._higher_is_better["bpb"] = False
                v._metric_fn_kwargs["bpb"] = {}
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
            _inject_bpb_into_metric_list(v[1])
        elif isinstance(v, dict):
            _inject_bpb_into_metric_list(v)


def _fill_group_bpb_from_leaves(results: dict) -> None:
    """Roll leaf ``bpb`` up into group entries that lack it.

    ``_inject_bpb_into_metric_list`` adds bpb to each leaf task, but a group's
    aggregate is built from the group YAML's ``aggregate_metric_list`` — which
    upstream configs (e.g. ``mmlu_pro``) don't list bpb in, so ``results[group]``
    has no ``bpb,none`` even when every subtask does. Compute it as a
    size-weighted mean over the leaves (matching lm-eval's ``weight_by_size``),
    so the dashboard's group-level read finds it. Best-effort by design: any
    schema surprise leaves the result untouched rather than failing the write.
    """
    res = results.get("results", {})
    subtasks = results.get("group_subtasks", {})
    nsamples = results.get("n-samples", {})
    for group, leaves in subtasks.items():
        entry = res.get(group)
        if not isinstance(entry, dict) or "bpb,none" in entry:
            continue
        num = den = 0.0
        for leaf in leaves:
            le = res.get(leaf)
            bpb = le.get("bpb,none") if isinstance(le, dict) else None
            if not isinstance(bpb, (int, float)):
                den = 0.0
                break
            weight = nsamples.get(leaf, {}).get("effective") or nsamples.get(leaf, {}).get("original") or 1
            num += bpb * weight
            den += weight
        if den > 0:
            entry["bpb,none"] = num / den


def _apply_task_field(task_dict: dict, key: str, value) -> None:
    """Walk ``task_dict`` and set ``key`` on each leaf Task.

    Mirrors lm-eval's `simple_evaluate`, which sets `num_fewshot` after
    task construction. Setting it post-build keeps the construction-time
    override path (which drops include-chain fields — see `_lm_eval_spec`)
    out of play.
    """
    from lm_eval.api.task import Task

    for v in task_dict.values():
        if isinstance(v, Task):
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
    task: EvalTaskConfig
    eval_capacity_factor: float = 8.0
    max_eval_instances: int | None = None
    batch_size: int = 8
    max_cont_len: int = 256


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

            task_dict = get_task_dict([_lm_eval_spec(config.task)], task_manager=TaskManager())
            _apply_task_field(task_dict, "num_fewshot", config.task.num_fewshot)
            # lm-eval's ConfigurableTask.process_results for output_type=loglikelihood
            # only returns {perplexity, acc} — `bpb`/`nll` from the metric_list never
            # land in task_output.sample_metrics, so consolidate_results writes empty
            # `results[name]` dicts. Inject a callable that computes them by hand.
            _inject_loglikelihood_bpb(task_dict)
            _inject_bpb_into_metric_list(task_dict)
            if config.task.task_alias:
                _apply_task_field(task_dict, "alias", config.task.task_alias)
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
        try:
            _fill_group_bpb_from_leaves(results)
        except Exception as e:
            logger.warning("group bpb fill skipped: %s", e)
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
            task=task,
        ),
    )


grug_logprob_eval_steps: list[ExecutorStep] = [
    _build_eval_step(model, task) for model in _MODEL_SPECS for task in _TASK_SPECS
]


if __name__ == "__main__":
    executor_main(steps=grug_logprob_eval_steps)
