# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Logprob lm-eval-harness for native Grug-MoE checkpoints.

Vendored from the swarm branch (tag ``swarm-branch``, commit 16ce640269) for
the mixing-via-embeddings seed panel; this is the harness that produced the
840-run swarm's per-task bpb suite under
``gs://marin-us-central2/evaluation/grug_logprob/``. Adaptations to this
branch:

- The legacy ``legacy_moe_flat`` checkpoint layout (and ``LegacyTransformer``)
  is dropped; panel checkpoints use the current layout.
- The swarm-era ``GrugModelConfig.capacity_factor`` field no longer exists, so
  the eval capacity raise is applied post-load via
  :func:`_with_eval_capacity` (MoE dispatch must not drop tokens at eval).
- The old ``ExecutorStep`` builder is dropped; drive
  :func:`run_grug_logprob_eval` via ``marin.execution.remote.remote`` (or a
  lazy ``ArtifactStep``) with explicit output paths. The swarm's exact 60-task
  set is reconstructable per task from
  ``evaluation/grug_logprob/swarm_fisher_dsp_d512_000000/<task>/.executor_info``
  → :class:`EvalTaskConfig` fields (name / num_fewshot / task_alias /
  task_kwargs).

Grug-MoE checkpoints are Levanter-native checkpoints, not HF exports. This
module loads a Grug ``Transformer`` directly, exposes a minimal lm-eval
``LM`` that implements ``loglikelihood``, and writes one ``results.json`` per
``(model, task)`` cell.
"""

from __future__ import annotations

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
from jax.experimental import multihost_utils
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.tokenizers import load_tokenizer
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from rigging.filesystem import filesystem as marin_filesystem

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_EVAL_CAPACITY_FACTOR = 8.0
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_MAX_CONT_LEN = 256
NAT_TO_BIT = 1 / math.log(2)

CSQA_SL_VERB_DOC_TO_CHOICE = (
    "{{['A. ' + choices['text'][0], 'B. ' + choices['text'][1], "
    "'C. ' + choices['text'][2], 'D. ' + choices['text'][3], "
    "'E. ' + choices['text'][4]]}}"
)
CSQA_SL_VERB_DOC_TO_TARGET = "{{choices['label'].index(answerKey)}}"
MEDMCQA_SL_VERB_DOC_TO_CHOICE = "{{['A. ' + opa, 'B. ' + opb, 'C. ' + opc, 'D. ' + opd]}}"


def _loglikelihood_target_metrics(results, target: str) -> dict[str, float]:
    """Convert one lm-eval loglikelihood result into target NLL and BPB metrics."""
    if not results:
        raise ValueError("loglikelihood results must contain one score tuple")
    logprob, _is_greedy = results[0]
    nll = -float(logprob)
    target_bytes = len(target.encode("utf-8"))
    if target_bytes <= 0:
        raise ValueError("loglikelihood target must contain at least one byte")
    return {
        "nll": nll,
        "bpb": nll * NAT_TO_BIT / target_bytes,
    }


# These process_results callables must stay module-top-level so Fray can pickle them by import path.
def process_gsm8k_logprob_results(doc, results):
    """Emit smooth GSM8K target logprob metrics for lm-eval aggregation."""
    return _loglikelihood_target_metrics(results, f" {doc['answer']}")


def process_humaneval_logprob_results(doc, results):
    """Emit smooth HumanEval target logprob metrics for lm-eval aggregation."""
    return _loglikelihood_target_metrics(results, str(doc["canonical_solution"]))


def _logprob_gsm8k_task() -> EvalTaskConfig:
    return EvalTaskConfig(
        name="logprob_gsm8k_5shot",
        num_fewshot=5,
        task_alias="logprob_gsm8k_5shot",
        task_kwargs={
            "tag": ["logprob_generative", "math_word_problems"],
            "dataset_path": "openai/gsm8k",
            "dataset_name": "main",
            "output_type": "loglikelihood",
            "training_split": "train",
            "fewshot_split": "train",
            "test_split": "test",
            "doc_to_text": "Question: {{question}}\nAnswer:",
            "doc_to_target": " {{answer}}",
            "process_results": process_gsm8k_logprob_results,
            "metric_list": [
                {"metric": "bpb", "aggregation": "mean", "higher_is_better": False},
                {"metric": "nll", "aggregation": "mean", "higher_is_better": False},
            ],
            "metadata": {"version": 1.0},
        },
    )


def _logprob_humaneval_task() -> EvalTaskConfig:
    return EvalTaskConfig(
        name="logprob_humaneval_10shot",
        num_fewshot=10,
        task_alias="logprob_humaneval_10shot",
        task_kwargs={
            "tag": ["logprob_generative", "code"],
            "dataset_path": "openai/openai_humaneval",
            "output_type": "loglikelihood",
            "test_split": "test",
            "doc_to_text": "{{prompt}}",
            "doc_to_target": "{{canonical_solution}}",
            "process_results": process_humaneval_logprob_results,
            "metric_list": [
                {"metric": "bpb", "aggregation": "mean", "higher_is_better": False},
                {"metric": "nll", "aggregation": "mean", "higher_is_better": False},
            ],
            "metadata": {"version": 1.0},
        },
    )


WSC273_PROMPTSOURCE_METRICS = (
    {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
    {"metric": "bpb", "aggregation": "mean", "higher_is_better": False},
    {"metric": "logprob", "aggregation": "mean", "higher_is_better": True},
    {"metric": "choice_logprob", "aggregation": "mean", "higher_is_better": True},
    {"metric": "choice_prob_norm", "aggregation": "mean", "higher_is_better": True},
    {"metric": "choice_logprob_norm", "aggregation": "mean", "higher_is_better": True},
)


def process_wsc273_promptsource_docs(dataset):
    """Keep the GPT-3-style WSC273 promptsource slice and apply lm-eval normalization."""
    from lm_eval.tasks.wsc273.utils import process_doc  # noqa: PLC0415 — optional eval dep

    return process_doc(dataset.filter(lambda doc: doc["template_name"] == "GPT-3 Style"))


def _wsc273_promptsource_task() -> EvalTaskConfig:
    return EvalTaskConfig(
        "wsc273_promptsource",
        0,
        task_alias="wsc273_0shot",
        task_kwargs={
            "dataset_path": "marcov/winograd_wsc_wsc273_promptsource",
            "dataset_name": "default",
            "output_type": "multiple_choice",
            "test_split": "test",
            "process_docs": process_wsc273_promptsource_docs,
            "doc_to_text": "label",
            "doc_to_target": "{% set index = pronoun_loc + pronoun | length %}{{text[index:]}}",
            "doc_to_choice": "{% set template = text[:pronoun_loc] %}{{[template+options[0], template+options[1]]}}",
            "metric_list": list(WSC273_PROMPTSOURCE_METRICS),
            "metadata": {"version": 1.0},
        },
    )


GRUG_LOGPROB_TASKS: tuple[EvalTaskConfig, ...] = (
    EvalTaskConfig("arc_challenge", 5, task_alias="arc_challenge_5shot"),
    EvalTaskConfig("arc_easy", 5, task_alias="arc_easy_5shot"),
    EvalTaskConfig("boolq", 10, task_alias="boolq_10shot"),
    EvalTaskConfig("boolq", 10, task_alias="boolq_sl_verb_10shot"),
    EvalTaskConfig("copa", 0, task_alias="copa_0shot"),
    EvalTaskConfig("commonsense_qa", 5, task_alias="csqa_5shot"),
    EvalTaskConfig(
        "commonsense_qa",
        5,
        task_alias="csqa_sl_verb_5shot",
        task_kwargs={"doc_to_choice": CSQA_SL_VERB_DOC_TO_CHOICE, "doc_to_target": CSQA_SL_VERB_DOC_TO_TARGET},
    ),
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
    EvalTaskConfig("hellaswag", 5, task_alias="hellaswag_5shot"),
    _logprob_gsm8k_task(),
    _logprob_humaneval_task(),
    EvalTaskConfig("medmcqa", 5, task_alias="medmcqa_5shot"),
    EvalTaskConfig(
        "medmcqa",
        5,
        task_alias="medmcqa_sl_verb_5shot",
        task_kwargs={"doc_to_choice": MEDMCQA_SL_VERB_DOC_TO_CHOICE},
    ),
    EvalTaskConfig("mmlu_sl", 0, task_alias="mmlu_sl_0shot"),
    EvalTaskConfig("mmlu_sl", 5, task_alias="mmlu_sl_5shot"),
    EvalTaskConfig("mmlu_sl_verb", 0, task_alias="mmlu_sl_verb_0shot"),
    EvalTaskConfig("mmlu_sl_verb", 5, task_alias="mmlu_sl_verb_5shot"),
    EvalTaskConfig("openbookqa", 0, task_alias="openbookqa_0shot"),
    EvalTaskConfig("piqa", 5, task_alias="piqa_5shot"),
    EvalTaskConfig("truthfulqa_mc1", 0, task_alias="truthfulqa_mc1_0shot"),
    EvalTaskConfig("winogrande", 5, task_alias="winogrande_5shot"),
    _wsc273_promptsource_task(),
)


def task_key(task: EvalTaskConfig) -> str:
    """Return the stable output-dir and W&B key for a task."""
    return task.task_alias or f"{task.name}_{task.num_fewshot}shot"


@eqx.filter_jit
def _batch_logprobs(transformer, tokens, loss_weight):
    per_pos_loss = transformer.next_token_loss(tokens, loss_weight, reduction="none")
    return -jnp.sum(per_pos_loss, axis=-1)


def _lm_eval_spec(task: EvalTaskConfig) -> str | dict:
    if not task.task_kwargs and not task.task_alias:
        return task.name
    spec: dict = {"task": task.name}
    if task.task_alias:
        spec["task_alias"] = task.task_alias
    if task.task_kwargs:
        spec.update(task.task_kwargs)
    return spec


def _apply_num_fewshot(task_dict: dict, num_fewshot: int) -> None:
    for value in task_dict.values():
        if hasattr(value, "set_config"):
            value.set_config(key="num_fewshot", value=num_fewshot)
            continue
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
            _apply_num_fewshot(value[1], num_fewshot)
            continue
        if isinstance(value, dict):
            _apply_num_fewshot(value, num_fewshot)


def _fill_missing_task_names(task_dict: dict) -> None:
    """Give dynamically configured lm-eval tasks stable task names before aggregation."""
    for key, value in task_dict.items():
        if hasattr(value, "config") and getattr(value.config, "task", None) is None:
            value.config.task = key
        if isinstance(value, dict):
            _fill_missing_task_names(value)


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
    if not cont_ids:
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
    if n == 0:
        return np.zeros(0, dtype=np.float32) if is_chief else None

    tokens_jnp = multihost_utils.broadcast_one_to_all(jnp.asarray(tokens))
    loss_weight_jnp = multihost_utils.broadcast_one_to_all(jnp.asarray(loss_weight))

    all_lps = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        real = end - start
        if real < batch_size:
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
    from lm_eval.api.model import LM  # noqa: PLC0415 — optional eval dep

    class GrugLM(LM):
        def loglikelihood(self, requests):
            n = len(requests)
            tokens = np.zeros((n, max_seq_len), dtype=np.int32)
            loss_weight = np.zeros((n, max_seq_len), dtype=np.float32)
            for i, request in enumerate(requests):
                prompt, continuation = request.args
                request_tokens, request_loss_weight = _tokenize_request(
                    prompt,
                    continuation,
                    tokenizer,
                    max_seq_len=max_seq_len,
                    max_cont_len=max_cont_len,
                )
                tokens[i] = request_tokens
                loss_weight[i] = request_loss_weight

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
            return [(float(logprob), False) for logprob in all_lps]

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("rolling logprob is not supported for Grug eval")

        def generate_until(self, requests):
            raise NotImplementedError("generation is not supported for Grug logprob eval")

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


def _with_eval_capacity(transformer, capacity_factor: float):
    """Raise MoE dispatch capacity on a loaded model so eval does not drop routed tokens.

    Replaces the swarm-era ``GrugModelConfig.capacity_factor`` mechanism: the
    field now lives on ``MoEExpertMlp`` as static metadata, so it is swapped
    post-load (arrays are untouched).
    """

    def _replace(module):
        if isinstance(module, MoEExpertMlp):
            return dataclasses.replace(module, capacity_factor=capacity_factor)
        return module

    return jax.tree_util.tree_map(_replace, transformer, is_leaf=lambda m: isinstance(m, MoEExpertMlp))


@dataclass(frozen=True)
class GrugLogprobEvalConfig:
    grug_model_config: GrugModelConfig
    checkpoint_path: str
    output_path: str
    wandb_run_name: str
    task: EvalTaskConfig
    eval_capacity_factor: float = DEFAULT_EVAL_CAPACITY_FACTOR
    max_eval_instances: int | None = None
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    max_cont_len: int = DEFAULT_MAX_CONT_LEN
    wandb_tags: tuple[str, ...] = ()


def run_grug_logprob_eval(config: GrugLogprobEvalConfig) -> None:
    """Run one multi-host-safe Grug logprob eval task."""
    eval_grug = config.grug_model_config
    max_seq_len = eval_grug.max_seq_len

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
            checkpoint_path = latest_checkpoint_path(str(config.checkpoint_path))
            transformer = load_checkpoint(
                transformer_shape,
                checkpoint_path,
                subpath="params",
                axis_mapping=trainer_config.parameter_axis_mapping,
            )
        transformer = _with_eval_capacity(transformer, config.eval_capacity_factor)

        tokenizer = load_tokenizer(marin_tokenizer)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer = dataclasses.replace(tokenizer, _pad_id=tokenizer.eos_token_id)

        results: dict | None = None
        if is_chief:
            from lm_eval import evaluator as lm_eval_evaluator  # noqa: PLC0415 — optional eval dep
            from lm_eval.tasks import TaskManager, get_task_dict  # noqa: PLC0415 — optional eval dep

            task_dict = get_task_dict([_lm_eval_spec(config.task)], task_manager=TaskManager())
            _fill_missing_task_names(task_dict)
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
        logger.info("Uploading Grug logprob results to %s", results_path)
        fs = marin_filesystem("gcs")
        with fs.open(results_path, "w") as handle:
            json.dump(results, handle, indent=2, default=lambda value: repr(value))
