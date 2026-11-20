# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hinted few-shot completion prompts for GSM8K and MATH-500 (Hinted Scaling Laws, arXiv 2611.02270).

The hint construction, its placement and the answer redaction follow the paper's released code,
tatsu-lab/hinted-scaling-laws at ``c0490e53ba8eae58bf5375f6b56956f0d57dad30``
(``src/hinted/benchmarks/__init__.py``, ``src/hinted/benchmarks/gsm8k_delphi.py`` and
``src/hinted/runner/__init__.py``):

- H = 0 is the plain few-shot prompt, unchanged.
- H > 0 uses the hint-block format: every exemplar shows its own reference solution degraded at the
  target's level on a ``Hint:`` line, followed by its complete solution; the target ends with its
  degraded hint and the answer prefix, so the model writes a complete solution and the scorer sees
  only that completion.
- Truncation keeps the first ``ceil(H * words)`` whitespace-delimited words and then replaces
  standalone occurrences of the final answer with ``<mask>``. Masking keeps a seeded random
  ``ceil(H * words)`` words, replaces the others with ``<mask>`` and does not redact the answer.

The paper evaluates GSM8K only; MATH-500 applies the same construction to the Minerva prompt.
"""

import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evalchemy.math import minerva_demonstration, minerva_prompt
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    compile_generation_documents,
    generation_item,
)
from taskcompendium.importers.fewshot import Document, parquet_documents, pinned_prompts
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

HINTED_SCALING_LAWS_REVISION = "c0490e53ba8eae58bf5375f6b56956f0d57dad30"
HINT_LEVELS = tuple(round(0.1 * step, 1) for step in range(1, 11))
REDACTION = "<mask>"
GSM8K_HINTED_POPULATION = 256
GSM8K_SHOTS = 5
MINERVA_SHOTS = 4
MINERVA_PROFILE = "minerva_math"
_WORD = re.compile(r"\S+")
_NUMERIC_ANSWER = re.compile(r"-?\$?[\d,]*\.?\d+")
_FINAL_ANSWER_LINE = "\nFinal Answer:"


class HintMethod(StrEnum):
    NONE = "none"
    TRUNCATION = "truncation"
    MASKING = "masking"


def _stable_int(*parts: object) -> int:
    text = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _kept_words(words: Sequence[re.Match[str]], fraction: float) -> int:
    return max(0, min(len(words), math.ceil(len(words) * fraction)))


def mask_words(text: str, fraction: float, *, seed: str) -> str:
    """Keep ``ceil(fraction * words)`` words chosen by a hash of (seed, fraction), masking the rest.

    Whitespace is preserved. Masks at different fractions are independent draws, not nested.
    """
    words = list(_WORD.finditer(text))
    keep = _kept_words(words, fraction)
    visible = set(range(len(words)))
    if len(words) > keep:
        rng = random.Random(_stable_int(seed, "mask_word", fraction))
        visible -= set(rng.sample(range(len(words)), len(words) - keep))
    parts = []
    cursor = 0
    for index, word in enumerate(words):
        parts.append(text[cursor : word.start()])
        parts.append(word.group(0) if index in visible else REDACTION)
        cursor = word.end()
    parts.append(text[cursor:])
    return "".join(parts)


def truncate_words(text: str, fraction: float) -> str:
    """Keep the source text through the last of the first ``ceil(fraction * words)`` words."""
    words = list(_WORD.finditer(text))
    keep = _kept_words(words, fraction)
    if keep == 0:
        return ""
    if keep == len(words):
        return text
    return text[: words[keep - 1].end()]


def _grouped_digits(digits: str) -> str:
    groups = []
    while len(digits) > 3:
        digits, tail = digits[:-3], digits[-3:]
        groups.insert(0, tail)
    return ",".join([digits, *groups])


def redact_answer(text: str, answer: str) -> str:
    """Replace standalone occurrences of the final answer with ``<mask>``.

    Numeric answers match their plain and comma-grouped spellings but not inside a longer number
    (18 does not match 180 or 1.18). Other answers, such as LaTeX expressions, match verbatim.
    """
    stripped = answer.strip()
    if _NUMERIC_ANSWER.fullmatch(stripped):
        plain = stripped.replace(",", "").replace("$", "")
        variants = {plain}
        sign, digits = (plain[0], plain[1:]) if plain.startswith("-") else ("", plain)
        if digits.isdigit() and len(digits) > 3:
            variants.add(sign + _grouped_digits(digits))
    else:
        variants = {stripped}
    pattern = "|".join(re.escape(variant) for variant in sorted(variants, key=len, reverse=True))
    return re.sub(rf"(?<![\d.])(?:{pattern})(?![\d.])", REDACTION, text)


def hint_text(reference: str, answer: str, method: HintMethod, level: float, *, seed: str) -> str:
    """The degraded reference shown on a ``Hint:`` line."""
    if method is HintMethod.MASKING:
        return mask_words(reference, level, seed=seed)
    if method is HintMethod.TRUNCATION:
        return redact_answer(truncate_words(reference, level), answer)
    raise ValueError(f"H > 0 requires a hint method, got {method}")


@dataclass(frozen=True)
class HintedExample:
    """One problem with the reference solution its hints are cut from."""

    problem_id: str
    question: str
    reference: str
    answer: str
    solution: str


def hint_block_prompt(
    target: HintedExample,
    exemplars: Sequence[HintedExample],
    method: HintMethod,
    level: float,
    *,
    render: Callable[[str, str, str | None], str],
) -> tuple[str, str]:
    """Return the hinted prompt and the target's hint text.

    ``render(question, hint, solution)`` formats one block; the target's block has no solution.
    Exemplar masks use the paper's seeds ``<problem>|exemplar|<index>``, the target's ``<problem>|0``.
    """
    blocks = [
        render(
            exemplar.question,
            hint_text(exemplar.reference, exemplar.answer, method, level, seed=f"{target.problem_id}|exemplar|{index}"),
            exemplar.solution,
        )
        for index, exemplar in enumerate(exemplars)
    ]
    shown = hint_text(target.reference, target.answer, method, level, seed=f"{target.problem_id}|0")
    return "".join(blocks) + render(target.question, shown, None), shown


def hinted_rows(
    base: Mapping[str, Any],
    target: HintedExample,
    exemplars: Sequence[HintedExample],
    method: HintMethod,
    *,
    render: Callable[[str, str, str | None], str],
) -> Iterator[dict[str, Any]]:
    for level in HINT_LEVELS:
        prompt, shown = hint_block_prompt(target, exemplars, method, level, render=render)
        yield {**base, "prompt": prompt, "hint_method": method.value, "hint_level": level, "hint_text": shown}


def convert_hinted(row: RawRow, context: ConversionContext, *, scorer: str, benchmark: str) -> TaskSpec:
    """One raw completion prompt with its packaged scorer; the row is kept as evidence."""
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["prompt"]),),
        package=source_package(scorer, row.data["grading"], environment=required_grader_environment(context)),
        evidence=row.data,
    )
    return task.model_copy(
        update={"tags": ("eval", "source:hinted-scaling-laws", f"benchmark:{benchmark}", "checkpoint")}
    )


def hinted_item(task: TaskSpec) -> EvaluationItem:
    """A completion item grouped by hint method and level, so the reducer reports each H."""
    evidence = source_evidence(task)
    groups = {"hint_method": evidence["hint_method"], "hint_level": f"{evidence['hint_level']:.1f}"}
    return generation_item(task).model_copy(update={"groups": groups})


# GSM8K: the first 256 test problems in lm-eval's five-shot format (Question/Answer, seed-1234 exemplars).


def gsm8k_final_answer(document: Document) -> str:
    """The comma- and dollar-free answer the paper redacts and grades against."""
    return document["answer"].rsplit("#### ", 1)[1].strip().replace(",", "").replace("$", "")


def gsm8k_reference(document: Document) -> str:
    """The rationale without its ``#### N`` line; calculator annotations are kept, as in the paper."""
    rationale, separator, _ = document["answer"].rpartition("\n#### ")
    if not separator:
        raise ValueError("GSM8K answer lacks its '#### N' line")
    return rationale


def gsm8k_example(document: Document, problem_id: str) -> HintedExample:
    return HintedExample(
        problem_id=problem_id,
        question=document["question"],
        reference=gsm8k_reference(document),
        answer=gsm8k_final_answer(document),
        solution=document["answer"],
    )


def gsm8k_block(question: str, hint: str, solution: str | None) -> str:
    if solution is None:
        return f"Question: {question}\nHint: {hint}\nAnswer:"
    return f"Question: {question}\nHint: {hint}\nAnswer: {solution}\n\n"


def gsm8k_rows(
    documents: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    method: HintMethod,
    *,
    source_row_prefix: str,
) -> Iterator[dict[str, Any]]:
    """Yield the plain prompt (``HintMethod.NONE``) or ten hinted prompts per problem."""
    template = HarnessTemplate(name="gsm8k", num_fewshot=GSM8K_SHOTS)
    plain = compile_generation_documents(documents[:GSM8K_HINTED_POPULATION], pool, template)
    for ordinal, row in enumerate(plain):
        document = row["document"]
        problem_id = str(ordinal)
        base = {
            "document": document,
            "profile": "gsm8k",
            "num_fewshot": GSM8K_SHOTS,
            "fewshot_indices": row["fewshot_indices"],
            "fewshot_seed": row["fewshot_seed"],
            "problem_id": problem_id,
            "reference_id": f"{source_row_prefix}:{ordinal}",
            "reference": gsm8k_reference(document),
            "grading": {"expected": gsm8k_final_answer(document)},
            "hinted_scaling_laws_revision": HINTED_SCALING_LAWS_REVISION,
        }
        if method is HintMethod.NONE:
            yield {**base, "prompt": row["prompt"], "hint_method": method.value, "hint_level": 0.0, "hint_text": ""}
            continue
        exemplars = [gsm8k_example(pool[index], f"train:{index}") for index in row["fewshot_indices"]]
        yield from hinted_rows(base, gsm8k_example(document, problem_id), exemplars, method, render=gsm8k_block)


def read_hinted_gsm8k(
    path: StoragePath, context: ConversionContext, *, method: HintMethod, pool_file: str, source_row_prefix: str
) -> Iterator[dict[str, Any]]:
    documents = list(parquet_documents(path))
    pool = list(parquet_documents(context.inputs["fewshot"] / pool_file))
    yield from gsm8k_rows(documents, pool, method, source_row_prefix=source_row_prefix)


# MATH-500 in the Minerva four-shot format.

_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")
_DOLLARS = re.compile(r"(?<!\\)\$\$|(?<!\\)\$")
_OPEN_DISPLAY = re.compile(r"\\\[|\\\(|\\begin\{[^}]*\}")


def _unclosed_math_start(text: str) -> int | None:
    """The offset where trailing, still-open math begins, if any."""
    starts = []
    open_delimiter: re.Match[str] | None = None
    for match in _DOLLARS.finditer(text):
        if open_delimiter is None:
            open_delimiter = match
        elif match.group(0) == open_delimiter.group(0):
            open_delimiter = None
    if open_delimiter is not None:
        starts.append(open_delimiter.start())
    for match in _OPEN_DISPLAY.finditer(text):
        opener = match.group(0)
        closer = {"\\[": "\\]", "\\(": "\\)"}.get(opener, opener.replace("\\begin", "\\end"))
        if closer not in text[match.end() :]:
            starts.append(match.start())
    return min(starts) if starts else None


def math_reference(solution: str) -> str:
    """Cut the solution before its first ``\\boxed{}`` answer and drop math the cut leaves open.

    Cutting at the first box also removes alternative solutions and alternative answer forms that
    follow it. When the whole solution is one math block ending in the box, dropping the open
    block would leave nothing, so the open block is kept.
    """
    box = _BOXED.search(solution)
    if box is None:
        raise ValueError("MATH solution lacks a boxed answer")
    head = solution[: box.start()].rstrip()
    closed = head
    while (start := _unclosed_math_start(closed)) is not None:
        closed = closed[:start]
    return closed.rstrip() or head


def minerva_example(document: Document, problem_id: str, answer: str) -> HintedExample:
    worked = document["solution"].split(_FINAL_ANSWER_LINE, 1)[0]
    return HintedExample(
        problem_id=problem_id,
        question=document["problem"],
        reference=math_reference(worked),
        answer=answer,
        solution=document["solution"],
    )


def minerva_block(problem: str, hint: str, solution: str | None) -> str:
    if solution is None:
        return f"Problem:\n{problem}\n\nHint: {hint}\n\nSolution:"
    return f"Problem:\n{problem}\n\nHint: {hint}\n\nSolution: {solution}\n\n"


def minerva_plain_prompt(document: Document, exemplars: Sequence[Document]) -> str:
    return "".join(minerva_demonstration(example) for example in exemplars) + minerva_prompt(document)


def minerva_exemplars() -> list[HintedExample]:
    """The four pinned Minerva demonstrations, answered by their 'Final Answer' line."""
    examples = []
    for index, example in enumerate(pinned_prompts()[MINERVA_PROFILE]):
        final = example["solution"].rsplit("The final answer is ", 1)[1].split(". I hope it is correct.")[0]
        examples.append(minerva_example(example, f"minerva:{index}", final.strip().strip("$")))
    return examples


def math_rows(documents: Sequence[Document], method: HintMethod, *, source_row_prefix: str) -> Iterator[dict[str, Any]]:
    """Yield four-shot Minerva prompts; H=0 uses the plain prompt.

    The longest H=1 prompt over MATH-500 is about 2,300 tokens, so every prompt leaves room for the
    1,024-token output budget in a 4,096-token context and no shots are dropped.
    """
    exemplar_documents = pinned_prompts()[MINERVA_PROFILE]
    exemplars = minerva_exemplars()
    for document in documents:
        problem_id = document["unique_id"]
        target = minerva_example(document, problem_id, document["answer"])
        base = {
            "document": document,
            "profile": MINERVA_PROFILE,
            "subject": document["subject"],
            "num_fewshot": MINERVA_SHOTS,
            "problem_id": problem_id,
            "reference_id": f"{source_row_prefix}:{problem_id}",
            "reference": target.reference,
            "grading": {"document": document},
            "hinted_scaling_laws_revision": HINTED_SCALING_LAWS_REVISION,
        }
        if method is HintMethod.NONE:
            prompt = minerva_plain_prompt(document, exemplar_documents)
            yield {**base, "prompt": prompt, "hint_method": method.value, "hint_level": 0.0, "hint_text": ""}
            continue
        yield from hinted_rows(base, target, exemplars, method, render=minerva_block)


def read_hinted_math(
    path: StoragePath, context: ConversionContext, *, method: HintMethod, source_row_prefix: str
) -> Iterator[dict[str, Any]]:
    documents = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    yield from math_rows(documents, method, source_row_prefix=source_row_prefix)
