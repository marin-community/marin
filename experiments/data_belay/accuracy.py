# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choice loss and multiple-choice accuracy of Data Belay checkpoints: `data-belay score` runs it on a GPU job.

Each task is scored cloze-style, as lm-eval-harness does: the log-likelihood of each choice after the question, summed
over the choice's tokens. ``acc`` picks the most likely choice; ``acc_norm`` divides by the choice's length in bytes
first. ``choice_bits`` is the gold choice's loss relative to the alternatives, ``-log2 softmax(ll)[gold]``, averaged
over questions: the verdict's metric for MMLU-Pro and TruthfulQA, which use the Eval Policy answer tasks' datasets,
prompts and question samples. The other tasks are an accuracy check.
"""

import dataclasses
import logging
import math
from collections.abc import Callable, Sequence

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.checkpoint import discover_checkpoint_candidates, load_checkpoint
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.data_belay.results import TaskScore, choice_path, scores_json
from experiments.datasets.eval_policy_answer_bpb import (
    TASKS_BY_NAME,
    AnswerExample,
    encode_example,
    mmlu_pro_prompt,
    sampled_rows,
    truthful_qa_prompt,
)
from experiments.grug.fast_track.launch import (
    MP_POLICY,
    SEQ_LEN,
    V16384_TOKENIZER,
    V16384_VOCAB,
    _h100_ladder_model,
    baseline_steps,
    h100_ladder_rung,
)
from experiments.grug.fast_track.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)

BATCH_ROWS = 64
LENGTH_BUCKET = 128


@dataclasses.dataclass(frozen=True)
class Question:
    prompt: str
    choices: tuple[str, ...]
    gold: int


@dataclasses.dataclass(frozen=True)
class ChoiceTask:
    name: str
    repo: str
    revision: str
    config: str | None
    split: str
    to_question: Callable[[dict], Question]


def _arc(row: dict) -> Question:
    labels = row["choices"]["label"]
    choices = tuple(f" {text}" for text in row["choices"]["text"])
    return Question(f"Question: {row['question']}\nAnswer:", choices, labels.index(row["answerKey"]))


def _sciq(row: dict) -> Question:
    choices = (row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"])
    prompt = f"{row['support'].strip()}\nQuestion: {row['question']}\nAnswer:".lstrip()
    return Question(prompt, tuple(f" {c}" for c in choices), 3)


def _openbookqa(row: dict) -> Question:
    labels = row["choices"]["label"]
    return Question(row["question_stem"], tuple(f" {t}" for t in row["choices"]["text"]), labels.index(row["answerKey"]))


def _piqa(row: dict) -> Question:
    return Question(f"Question: {row['goal']}\nAnswer:", (f" {row['sol1']}", f" {row['sol2']}"), int(row["label"]))


def _hellaswag(row: dict) -> Question:
    context = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
    return Question(context, tuple(f" {ending}" for ending in row["endings"]), int(row["label"]))


def _mmlu(row: dict) -> Question:
    return Question(f"Question: {row['question']}\nAnswer:", tuple(f" {c}" for c in row["choices"]), int(row["answer"]))


# The Eval Policy answer tasks scored by choice loss, with their answer tasks' prompts.
def _mmlu_pro(row: dict) -> Question:
    return Question(mmlu_pro_prompt(row), tuple(f" {o}" for o in row["options"]), row["answer_index"])


def _truthful_qa(row: dict) -> Question:
    # MC1: one true answer among the dataset's false ones.
    targets = row["mc1_targets"]
    return Question(truthful_qa_prompt(row), tuple(f" {c}" for c in targets["choices"]), targets["labels"].index(1))


def _eval_policy_task(name: str, to_question: Callable[[dict], Question], config: str | None = None) -> ChoiceTask:
    task = TASKS_BY_NAME[name]
    return ChoiceTask(name, task.repo, task.revision, config or task.config, task.split, to_question)


ARC = "allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"
TASKS = (
    ChoiceTask("arc_easy", *ARC, "ARC-Easy", "test", _arc),
    ChoiceTask("arc_challenge", *ARC, "ARC-Challenge", "test", _arc),
    ChoiceTask("sciq", "allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815", None, "test", _sciq),
    ChoiceTask(
        "openbookqa", "allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", "main", "test", _openbookqa
    ),
    ChoiceTask("piqa", "baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1", None, "validation", _piqa),
    ChoiceTask(
        "hellaswag", "Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", None, "validation", _hellaswag
    ),
    ChoiceTask("mmlu", "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all", "test", _mmlu),
    _eval_policy_task("mmlu_pro", _mmlu_pro),
    _eval_policy_task("truthful_qa", _truthful_qa, config="multiple_choice"),
)


def task_questions(task: ChoiceTask) -> list[Question]:
    """The task's questions, from the rows that the Eval Policy answer tasks sample."""
    return [task.to_question(row) for row in sampled_rows(task.repo, task.config, task.split, task.revision)]


def choice_rows(tokenizer: MarinTokenizer, questions: Sequence[Question]) -> list[tuple[list[int], list[float]]]:
    """One row per (question, choice): ``BOS prompt choice EOS``, weighted on the choice's tokens."""
    return [encode_example(tokenizer, AnswerExample(q.prompt, c)) for q in questions for c in q.choices]


def gold_choice_bits(scores: np.ndarray, gold: int) -> float:
    """``-log2`` of the gold choice's softmax probability over the choices, from natural-log scores."""
    return float((np.logaddexp.reduce(scores) - scores[gold]) / math.log(2))


def accuracies(questions: Sequence[Question], logprobs: Sequence[float]) -> TaskScore:
    """``acc``, ``acc_norm`` and ``choice_bits`` from each row's summed choice log-likelihood, in ``choice_rows``
    order."""
    correct = correct_norm = 0
    bits = 0.0
    i = 0
    for q in questions:
        scores = np.asarray(logprobs[i : i + len(q.choices)])
        lengths = np.asarray([len(c.encode("utf-8")) for c in q.choices])
        correct += int(np.argmax(scores) == q.gold)
        correct_norm += int(np.argmax(scores / lengths) == q.gold)
        bits += gold_choice_bits(scores, q.gold)
        i += len(q.choices)
    n = len(questions)
    return TaskScore(acc=correct / n, acc_norm=correct_norm / n, choice_bits=bits / n, questions=n)


def dense_config(rung: str) -> GrugModelConfig:
    """The fast_track dense model at ``rung``, built the way ``launch.py --dense`` builds it."""
    return dataclasses.replace(_h100_ladder_model(h100_ladder_rung(rung), dense=True), vocab_size=V16384_VOCAB)


@eqx.filter_jit
def _row_logprobs(model: Transformer, tokens: jax.Array, weights: jax.Array) -> jax.Array:
    loss = model.next_token_loss(tokens, weights, mask=AttentionMask.causal(), reduction="none")
    return -jnp.sum(loss, axis=1)


def score_rows(model: Transformer, rows: Sequence[tuple[list[int], list[float]]], pad_id: int) -> list[float]:
    """Summed choice log-likelihood of each row, batched by padded length."""
    by_length: dict[int, list[int]] = {}
    for i, (tokens, _) in enumerate(rows):
        if len(tokens) > SEQ_LEN:
            raise ValueError(f"row {i} has {len(tokens)} tokens, more than {SEQ_LEN}")
        by_length.setdefault(LENGTH_BUCKET * math.ceil(len(tokens) / LENGTH_BUCKET), []).append(i)
    out = [0.0] * len(rows)
    for length, indices in sorted(by_length.items()):
        for start in range(0, len(indices), BATCH_ROWS):
            batch = indices[start : start + BATCH_ROWS]
            tokens = np.full((BATCH_ROWS, length), pad_id, dtype=np.int32)
            weights = np.zeros((BATCH_ROWS, length), dtype=np.float32)
            for row, i in enumerate(batch):
                row_tokens, row_weights = rows[i]
                tokens[row, : len(row_tokens)] = row_tokens
                weights[row, : len(row_weights)] = row_weights
            scores = np.asarray(_row_logprobs(model, jnp.asarray(tokens), jnp.asarray(weights)))
            for row, i in enumerate(batch):
                out[i] = float(scores[row])
    return out


def final_checkpoint_path(run_output: str, step: int) -> str:
    """The complete checkpoint of a run's last step, ``step``: training saves it after its final eval."""
    for candidate in discover_checkpoint_candidates(prefix_join(run_output, "checkpoints")):
        if candidate.step == step:
            return candidate.path
    raise ValueError(f"{run_output} has no complete checkpoint of its last step, {step}, yet")


def load_model(step_dir: str, config: GrugModelConfig, mesh: jax.sharding.Mesh) -> Transformer:
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.key(0))
    params = load_checkpoint({"params": template}, step_dir, axis_mapping=None, mesh=mesh)
    logger.info("loaded %s", step_dir)
    return jmp.get_policy(MP_POLICY).cast_to_compute(params["params"])


@click.command()
@click.option("--rung", required=True, help="The runs' rung, e.g. d768.")
@click.option("--runs-root", required=True, help="The directory that holds each run's output, named by run id.")
@click.option("--choice-root", required=True, help="The directory to write each run's scores to.")
@click.argument("run_ids", nargs=-1, required=True)
def main(rung: str, runs_root: str, choice_root: str, run_ids: tuple[str, ...]) -> None:
    """Score each finished run's final checkpoint on every task, into ``results.choice_path(choice_root, run_id)``."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    steps = baseline_steps(rung, dense=True)
    checkpoints = {run_id: final_checkpoint_path(prefix_join(runs_root, run_id), steps) for run_id in run_ids}
    tokenizer = load_tokenizer(V16384_TOKENIZER)
    questions = {task.name: task_questions(task) for task in TASKS}
    rows = {name: choice_rows(tokenizer, qs) for name, qs in questions.items()}
    config = dense_config(rung)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        for run_id, step_dir in checkpoints.items():
            model = load_model(step_dir, config, mesh)
            result = {
                name: accuracies(questions[name], score_rows(model, rows[name], tokenizer.eos_token_id))
                for name in questions
            }
            logger.info("%s: %s", run_id, result)
            StoragePath(choice_path(choice_root, run_id)).write_text(scores_json(result))


if __name__ == "__main__":
    main()
