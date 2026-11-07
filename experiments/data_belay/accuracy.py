# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multiple-choice accuracy of Data Belay checkpoints, to check whether answer BPB misses a gain.

Each task is scored cloze-style, as lm-eval-harness does: the log-likelihood of each choice after the question, summed
over the choice's tokens. ``acc`` picks the most likely choice; ``acc_norm`` divides by the choice's length in bytes
first. Run it on a GPU job:

    uv run iris --cluster marin job run --enable-extra-resources --target-cluster cw-us-east-02a --gpu H100x1 \\
        --extra gpu --cpu 8 --memory 64GB --job-name <name> -- \\
        python -m experiments.data_belay.accuracy --rung d768 --output <s3 json> <run output path>...
"""

import dataclasses
import json
import logging
import math
import random
from collections.abc import Callable, Sequence

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from datasets import load_dataset
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.datasets.eval_policy_answer_bpb import AnswerExample, encode_example
from experiments.grug.fast_track import belay_tracks
from experiments.grug.fast_track.launch import (
    MP_POLICY,
    SEQ_LEN,
    V16384_TOKENIZER,
    V16384_VOCAB,
    _apply_settings,
    _h100_ladder_model,
    _parse_settings,
    h100_ladder_rung,
)
from experiments.grug.fast_track.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)

# A seeded sample of a larger task keeps one checkpoint's scoring to a few minutes.
MAX_QUESTIONS = 2000
SAMPLE_SEED = 0
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
)


def task_questions(task: ChoiceTask) -> list[Question]:
    rows = load_dataset(task.repo, task.config, split=task.split, revision=task.revision)
    indices = list(range(len(rows)))
    if len(indices) > MAX_QUESTIONS:
        indices = sorted(random.Random(SAMPLE_SEED).sample(indices, MAX_QUESTIONS))
    return [task.to_question(rows[i]) for i in indices]


def choice_rows(tokenizer: MarinTokenizer, questions: Sequence[Question]) -> list[tuple[list[int], list[float]]]:
    """One row per (question, choice): ``BOS prompt choice EOS``, weighted on the choice's tokens."""
    return [encode_example(tokenizer, AnswerExample(q.prompt, c)) for q in questions for c in q.choices]


def accuracies(questions: Sequence[Question], logprobs: Sequence[float]) -> dict[str, float]:
    """``acc`` and ``acc_norm`` from each row's summed choice log-likelihood, in ``choice_rows`` order."""
    correct = correct_norm = 0
    i = 0
    for q in questions:
        scores = np.asarray(logprobs[i : i + len(q.choices)])
        lengths = np.asarray([len(c.encode("utf-8")) for c in q.choices])
        correct += int(np.argmax(scores) == q.gold)
        correct_norm += int(np.argmax(scores / lengths) == q.gold)
        i += len(q.choices)
    return {"acc": correct / len(questions), "acc_norm": correct_norm / len(questions), "questions": len(questions)}


def vanilla_config(rung: str) -> GrugModelConfig:
    """The Belay `vanilla` Track's model at ``rung``, built the way ``launch.py`` builds it."""
    args = belay_tracks.track_args(belay_tracks.Track.VANILLA, rung)
    settings = tuple(args[i + 1] for i, arg in enumerate(args) if arg == "--model-set")
    config = dataclasses.replace(_h100_ladder_model(h100_ladder_rung(rung), dense=True), vocab_size=V16384_VOCAB)
    return _apply_settings(config, _parse_settings(settings))


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


def load_model(run_output: str, config: GrugModelConfig, mesh: jax.sharding.Mesh) -> Transformer:
    step_dir = latest_checkpoint_path(f"{run_output.rstrip('/')}/checkpoints")
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.key(0))
    params = load_checkpoint({"params": template}, step_dir, axis_mapping=None, mesh=mesh)
    logger.info("loaded %s", step_dir)
    return jmp.get_policy(MP_POLICY).cast_to_compute(params["params"])


@click.command()
@click.option("--rung", required=True, help="The checkpoints' `vanilla` rung, e.g. d768.")
@click.option("--output", required=True, help="JSON file for the results, keyed by run output path then task.")
@click.argument("run_outputs", nargs=-1, required=True)
def main(rung: str, output: str, run_outputs: tuple[str, ...]) -> None:
    """Score each run's final checkpoint on every task."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    tokenizer = load_tokenizer(V16384_TOKENIZER)
    questions = {task.name: task_questions(task) for task in TASKS}
    rows = {name: choice_rows(tokenizer, qs) for name, qs in questions.items()}
    config = vanilla_config(rung)
    results: dict[str, dict[str, dict[str, float]]] = {}
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        for run_output in run_outputs:
            model = load_model(run_output, config, mesh)
            results[run_output] = {
                name: accuracies(questions[name], score_rows(model, rows[name], tokenizer.eos_token_id))
                for name in questions
            }
            logger.info("%s: %s", run_output, json.dumps(results[run_output]))
            StoragePath(output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
