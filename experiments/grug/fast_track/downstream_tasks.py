# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multiple-choice and cloze tasks in one request format for small-model scoring.

Each example becomes one (context, continuation) pair per choice, in the cloze
formulation that OLMES uses for small models. A scorer computes the log-likelihood of
each continuation, so one pass gives accuracy, length-normalized accuracy, and the
bits per byte of the correct continuation.
"""

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

from datasets import load_dataset
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.fast_track.contracts import FAST_TRACK_ROOT

DOWNSTREAM_TASKS_VERSION = "downstream-cloze-v3"
# Large tasks are capped so one checkpoint scores in minutes. The cap is a fixed prefix
# of the dataset order, so every checkpoint sees the same examples.
MAX_EXAMPLES_PER_TASK = 2000
MMLU_MAX_EXAMPLES_PER_SUBJECT = 60


@dataclass(frozen=True)
class ChoiceRequest:
    context: str
    continuation: str


@dataclass(frozen=True)
class TaskExample:
    task: str
    example_id: str
    choices: tuple[ChoiceRequest, ...]
    label: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(line: str) -> "TaskExample":
        row = json.loads(line)
        return TaskExample(
            task=row["task"],
            example_id=row["example_id"],
            choices=tuple(ChoiceRequest(**choice) for choice in row["choices"]),
            label=row["label"],
        )


def _same_context(task: str, example_id: str, context: str, continuations: list[str], label: int) -> TaskExample:
    return TaskExample(task, example_id, tuple(ChoiceRequest(context, text) for text in continuations), label)


def _arc(config: str, task: str) -> Iterator[TaskExample]:
    for row in load_dataset("allenai/ai2_arc", config, split="test"):
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            continue
        yield _same_context(
            task,
            row["id"],
            f"Question: {row['question']}\nAnswer:",
            [f" {text}" for text in row["choices"]["text"]],
            labels.index(row["answerKey"]),
        )


def _boolq() -> Iterator[TaskExample]:
    for index, row in enumerate(load_dataset("google/boolq", split="validation")):
        yield _same_context(
            "boolq",
            str(index),
            f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:",
            [" no", " yes"],
            int(row["answer"]),
        )


def _csqa() -> Iterator[TaskExample]:
    for row in load_dataset("tau/commonsense_qa", split="validation"):
        labels = list(row["choices"]["label"])
        yield _same_context(
            "csqa",
            row["id"],
            f"Question: {row['question']}\nAnswer:",
            [f" {text}" for text in row["choices"]["text"]],
            labels.index(row["answerKey"]),
        )


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _hellaswag() -> Iterator[TaskExample]:
    for row in load_dataset("Rowan/hellaswag", split="validation"):
        context = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
        yield _same_context(
            "hellaswag",
            row["ind"] if isinstance(row["ind"], str) else str(row["ind"]),
            _hellaswag_clean(context),
            [" " + _hellaswag_clean(ending) for ending in row["endings"]],
            int(row["label"]),
        )


def _mmlu() -> Iterator[TaskExample]:
    per_subject: dict[str, int] = {}
    for index, row in enumerate(load_dataset("cais/mmlu", "all", split="test")):
        subject = row["subject"]
        if per_subject.get(subject, 0) >= MMLU_MAX_EXAMPLES_PER_SUBJECT:
            continue
        per_subject[subject] = per_subject.get(subject, 0) + 1
        yield _same_context(
            "mmlu",
            f"{subject}/{index}",
            f"Question: {row['question']}\nAnswer:",
            [f" {choice}" for choice in row["choices"]],
            int(row["answer"]),
        )


def _openbookqa() -> Iterator[TaskExample]:
    for row in load_dataset("allenai/openbookqa", "main", split="test"):
        labels = list(row["choices"]["label"])
        yield _same_context(
            "openbookqa",
            row["id"],
            f"Question: {row['question_stem']}\nAnswer:",
            [f" {text}" for text in row["choices"]["text"]],
            labels.index(row["answerKey"]),
        )


def _piqa() -> Iterator[TaskExample]:
    for index, row in enumerate(load_dataset("ybisk/piqa", revision="refs/convert/parquet", split="validation")):
        yield _same_context(
            "piqa",
            str(index),
            f"Question: {row['goal']}\nAnswer:",
            [f" {row['sol1']}", f" {row['sol2']}"],
            int(row["label"]),
        )


def _siqa() -> Iterator[TaskExample]:
    for index, row in enumerate(
        load_dataset("allenai/social_i_qa", revision="refs/convert/parquet", split="validation")
    ):
        yield _same_context(
            "siqa",
            str(index),
            f"Question: {row['context']} {row['question']}\nAnswer:",
            [f" {row['answerA']}", f" {row['answerB']}", f" {row['answerC']}"],
            int(row["label"]) - 1,
        )


def _sciq() -> Iterator[TaskExample]:
    for index, row in enumerate(load_dataset("allenai/sciq", split="test")):
        choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
        yield _same_context(
            "sciq",
            str(index),
            f"{row['support'].strip()}\nQuestion: {row['question']}\nAnswer:".lstrip(),
            [f" {choice}" for choice in choices],
            3,
        )


def _winogrande() -> Iterator[TaskExample]:
    # Partial scoring: the choice fills the blank in the context, and every choice scores
    # the same continuation after the blank.
    for index, row in enumerate(load_dataset("allenai/winogrande", "winogrande_xl", split="validation")):
        blank = row["sentence"].index("_")
        continuation = " " + row["sentence"][blank + 1 :].strip()
        prefix = row["sentence"][:blank]
        yield TaskExample(
            "winogrande",
            str(index),
            (
                ChoiceRequest(prefix + row["option1"], continuation),
                ChoiceRequest(prefix + row["option2"], continuation),
            ),
            int(row["answer"]) - 1,
        )


def _copa() -> Iterator[TaskExample]:
    for row in load_dataset("aps/super_glue", "copa", split="validation"):
        connector = " because" if row["question"] == "cause" else " therefore"
        premise = row["premise"].strip().rstrip(".")

        def lower_first(text: str) -> str:
            return text[0].lower() + text[1:]

        yield _same_context(
            "copa",
            str(row["idx"]),
            premise + connector,
            [" " + lower_first(row["choice1"]), " " + lower_first(row["choice2"])],
            int(row["label"]),
        )


def _lambada() -> Iterator[TaskExample]:
    # One choice: the score is the bits per byte of the last word and whether greedy decoding recovers it.
    for index, row in enumerate(load_dataset("EleutherAI/lambada_openai", "en", split="test")):
        context, _, last_word = row["text"].rpartition(" ")
        yield _same_context("lambada", str(index), context, [" " + last_word], 0)


def _gsm8k() -> Iterator[TaskExample]:
    # Gold-answer scoring: one choice, the reference reasoning and final answer. The calculator
    # annotations (`<<48/2=24>>`) are 14% of the answer characters and reward GSM8K-format data.
    for index, row in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
        answer = re.sub(r"<<[^>]*>>", "", row["answer"])
        yield _same_context("gsm8k", str(index), f"Question: {row['question']}\nAnswer:", [" " + answer], 0)


def _humaneval() -> Iterator[TaskExample]:
    for row in load_dataset("openai/openai_humaneval", split="test"):
        yield _same_context("humaneval", row["task_id"], row["prompt"], [row["canonical_solution"]], 0)


def _mbpp() -> Iterator[TaskExample]:
    for row in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        context = f'"""{row["text"]}\n{row["test_list"][0]}\n"""\n'
        # The gold code has `\r\n` line ends and spaces at line ends. They carry 25-56% of the bits
        # of a web-only model, so score the code with `\n` line ends, which keeps the same AST.
        code = re.sub(r"[ \t]+\n", "\n", row["code"].replace("\r\n", "\n")).rstrip()
        yield _same_context("mbpp", str(row["task_id"]), context, [code], 0)


def _math500() -> Iterator[TaskExample]:
    for row in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        yield _same_context(
            "math500", row["unique_id"], f"Problem: {row['problem']}\nSolution:", [" " + row["solution"]], 0
        )


TASK_BUILDERS: dict[str, Callable[[], Iterator[TaskExample]]] = {
    "arc_easy": lambda: _arc("ARC-Easy", "arc_easy"),
    "arc_challenge": lambda: _arc("ARC-Challenge", "arc_challenge"),
    "boolq": _boolq,
    "csqa": _csqa,
    "hellaswag": _hellaswag,
    "mmlu": _mmlu,
    "openbookqa": _openbookqa,
    "piqa": _piqa,
    "siqa": _siqa,
    "sciq": _sciq,
    "winogrande": _winogrande,
    "copa": _copa,
    "lambada": _lambada,
    "gsm8k": _gsm8k,
    "humaneval": _humaneval,
    "mbpp": _mbpp,
    "math500": _math500,
}


def task_examples(task: str) -> list[TaskExample]:
    examples = []
    for example in TASK_BUILDERS[task]():
        if task != "mmlu" and len(examples) >= MAX_EXAMPLES_PER_TASK:
            break
        examples.append(example)
    return examples


STAGED_TASKS_ROOT = f"{FAST_TRACK_ROOT}/downstream-tasks/{DOWNSTREAM_TASKS_VERSION}"


def staged_task_examples(task: str) -> list[TaskExample]:
    """Read the frozen copy of ``task`` that ``stage_tasks`` wrote."""
    with StoragePath(prefix_join(STAGED_TASKS_ROOT, f"{task}.jsonl")).open("r") as stream:
        return [TaskExample.from_json(line) for line in stream if line.strip()]


def stage_tasks(root: str) -> None:
    """Freeze every task's examples as JSONL under ``root``, so every checkpoint sees the same requests."""
    if StoragePath(root).exists():
        raise FileExistsError(f"{root} exists; bump DOWNSTREAM_TASKS_VERSION to stage new tasks")
    for task in TASK_BUILDERS:
        examples = task_examples(task)
        with StoragePath(prefix_join(root, f"{task}.jsonl")).open("w") as stream:
            for example in examples:
                stream.write(example.to_json() + "\n")
        print(f"{task}: {len(examples)} examples")


if __name__ == "__main__":
    stage_tasks(STAGED_TASKS_ROOT)
