# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic code-interpretation records for supervised PPL evals."""

import json
import posixpath
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from marin.transform.evaluation.continuation_records import render_continuation_target, render_support_and_query
from rigging.filesystem import StoragePath, open_url
from zephyr.writers import atomic_rename

CODE_INTERPRETATION_NUM_FEWSHOT = 5
CODE_INTERPRETATION_RENDERER_VERSION = "v1"
DEFAULT_CODE_INTERPRETATION_OUTPUT_FILENAME = "staged.jsonl.gz"


@dataclass(frozen=True)
class CodeInterpretationExample:
    """One static Python-like program/expression and its output."""

    example_id: str
    code: str
    target: str


@dataclass(frozen=True)
class CodeInterpretationTask:
    """A fixed code-interpretation task with five support examples."""

    key: str
    family: str
    title: str
    description: str
    support_examples: tuple[CodeInterpretationExample, ...]
    heldout_examples: tuple[CodeInterpretationExample, ...]


@dataclass(frozen=True)
class CodeInterpretationTemplate:
    """A surface form that renders supports and an unfinished held-out query."""

    key: str
    family: str
    description: str
    renderer: Callable[[CodeInterpretationExample, bool], str]


@dataclass(frozen=True)
class CodeInterpretationStagingConfig:
    """Configuration for staging one code-interpretation slice."""

    output_path: str
    task_key: str
    template_key: str
    output_filename: str = DEFAULT_CODE_INTERPRETATION_OUTPUT_FILENAME
    source_manifest: IngestionSourceManifest | None = None
    content_fingerprint: str = ""


def _examples(prefix: str, pairs: tuple[tuple[str, str], ...]) -> tuple[CodeInterpretationExample, ...]:
    return tuple(
        CodeInterpretationExample(example_id=f"{prefix}-{index:02d}", code=code, target=target)
        for index, (code, target) in enumerate(pairs, start=1)
    )


def _as_repl_input(code: str) -> str:
    prompted_lines: list[str] = []
    for index, line in enumerate(code.splitlines()):
        prompt = ">>> " if index == 0 or (line and not line.startswith(" ")) else "... "
        prompted_lines.append(f"{prompt}{line}")
    return "\n".join(prompted_lines)


def _render_python_repl(example: CodeInterpretationExample, include_target: bool) -> str:
    rendered = _as_repl_input(example.code)
    return f"{rendered}\n{example.target if include_target else ''}"


def _render_python_doctest(example: CodeInterpretationExample, include_target: bool) -> str:
    return _render_python_repl(example, include_target)


def _render_arrow_control(example: CodeInterpretationExample, include_target: bool) -> str:
    target = example.target if include_target else ""
    return f"Python:\n{example.code}\n=> {target}"


CODE_INTERPRETATION_TEMPLATES: tuple[CodeInterpretationTemplate, ...] = (
    CodeInterpretationTemplate(
        "python_repl",
        "code_transcript",
        "natural Python REPL transcript",
        _render_python_repl,
    ),
    CodeInterpretationTemplate(
        "python_doctest",
        "code_transcript",
        "Python doctest-style transcript",
        _render_python_doctest,
    ),
    CodeInterpretationTemplate(
        "arrow_control",
        "neutral_control",
        "code block with neutral arrow output",
        _render_arrow_control,
    ),
)
CODE_INTERPRETATION_TEMPLATES_BY_KEY = {template.key: template for template in CODE_INTERPRETATION_TEMPLATES}


CODE_INTERPRETATION_TASKS: tuple[CodeInterpretationTask, ...] = (
    CodeInterpretationTask(
        key="expression_strings_collections",
        family="expression_only",
        title="String and collection expression interpretation",
        description="Evaluate the final value of a small Python expression using strings, lists, dicts, and indexing.",
        support_examples=_examples(
            "expr-strings-support",
            (
                ("len('marin') + 4", "9"),
                ("'-'.join(['red', 'blue']).upper()", "RED-BLUE"),
                ("sorted(['delta', 'alpha', 'beta'])[0]", "alpha"),
                ("{'a': 2, 'b': 5}['b'] * 3", "15"),
                ("'abcdef'[1:5:2]", "bd"),
            ),
        ),
        heldout_examples=_examples(
            "expr-strings-heldout",
            (
                ("len('prompt') * 7", "42"),
                ("'-'.join(reversed(['north', 'east']))", "east-north"),
                ("{'x': [4, 8], 'y': [3]}['x'][1] + 6", "14"),
            ),
        ),
    ),
    CodeInterpretationTask(
        key="expression_arithmetic_slices",
        family="expression_only",
        title="Arithmetic and slicing expression interpretation",
        description=(
            "Evaluate the final value of a small Python expression using arithmetic, comprehensions, and slices."
        ),
        support_examples=_examples(
            "expr-arith-support",
            (
                ("sum([3, 5, 8])", "16"),
                ("sum(x * x for x in [1, 2, 5])", "30"),
                ("[n + 2 for n in [4, 7, 9]][-1]", "11"),
                ("min([12, 4, 9]) + max([1, 7, 3])", "11"),
                ("list(range(2, 9, 3))[1]", "5"),
            ),
        ),
        heldout_examples=_examples(
            "expr-arith-heldout",
            (
                ("sum(x * x for x in [2, 3, 4])", "29"),
                ("[n * 2 for n in range(3, 7)][2]", "10"),
                ("sum([10, -3, 65]) // 4", "18"),
            ),
        ),
    ),
    CodeInterpretationTask(
        key="function_definition_calls",
        family="function_definition",
        title="Function and class definition interpretation",
        description="Read simple helper definitions and evaluate the final call or expression.",
        support_examples=_examples(
            "fn-support",
            (
                (
                    "def double_then_shift(x):\n" "    value = x * 2\n" "    return value + 3\n" "double_then_shift(5)",
                    "13",
                ),
                (
                    "def join_edges(items):\n"
                    "    return items[0] + ':' + items[-1]\n"
                    "join_edges(['red', 'green', 'blue'])",
                    "red:blue",
                ),
                (
                    "def clipped_total(values, cap):\n"
                    "    total = sum(values)\n"
                    "    return min(total, cap)\n"
                    "clipped_total([4, 9, 6], 15)",
                    "15",
                ),
                (
                    "def label_size(name, count):\n"
                    "    if count >= 4:\n"
                    "        return name.upper()\n"
                    "    return name.lower()\n"
                    "label_size('Box', 3)",
                    "box",
                ),
                (
                    "class CounterBox:\n"
                    "    def __init__(self, start):\n"
                    "        self.total = start\n"
                    "    def add(self, value):\n"
                    "        self.total += value\n"
                    "        return self.total\n"
                    "box = CounterBox(4)\n"
                    "(box.add(3), box.add(5))",
                    "(7, 12)",
                ),
            ),
        ),
        heldout_examples=_examples(
            "fn-heldout",
            (
                (
                    "def edge_repeat(text):\n" "    return text[0] + text[-1] * 2\n" "edge_repeat('marin')",
                    "mnn",
                ),
                (
                    "def scale_and_tag(x, tag):\n"
                    "    y = x * 4\n"
                    "    return tag + ':' + str(y - 1)\n"
                    "scale_and_tag(6, 'job')",
                    "job:23",
                ),
                (
                    "class PairBox:\n"
                    "    def __init__(self, left, right):\n"
                    "        self.left = left\n"
                    "        self.right = right\n"
                    "    def flipped(self):\n"
                    "        return self.right + '-' + self.left\n"
                    "PairBox('north', 'east').flipped().upper()",
                    "EAST-NORTH",
                ),
            ),
        ),
    ),
)
CODE_INTERPRETATION_TASKS_BY_KEY = {task.key: task for task in CODE_INTERPRETATION_TASKS}


def code_interpretation_record(task_key: str, template_key: str, heldout_index: int) -> dict[str, Any]:
    """Return one supervised target-only record for a task/template/held-out index."""

    task = CODE_INTERPRETATION_TASKS_BY_KEY[task_key]
    template = CODE_INTERPRETATION_TEMPLATES_BY_KEY[template_key]
    heldout = task.heldout_examples[heldout_index]
    return {
        "id": f"{task.key}/{template.key}/{heldout.example_id}",
        "input": render_support_and_query(
            task=task, template=template, heldout=heldout, num_fewshot=CODE_INTERPRETATION_NUM_FEWSHOT
        ),
        "target": render_continuation_target(template=template, heldout=heldout),
        "source": "code_interpretation_static",
        "provenance": {
            "task_key": task.key,
            "task_family": task.family,
            "template_key": template.key,
            "template_family": template.family,
            "renderer_version": CODE_INTERPRETATION_RENDERER_VERSION,
            "num_fewshot": CODE_INTERPRETATION_NUM_FEWSHOT,
            "support_example_ids": [example.example_id for example in task.support_examples],
            "heldout_example_id": heldout.example_id,
            "semantic_target": heldout.target,
            "heldout_code": heldout.code,
        },
    }


def stage_code_interpretation_source(cfg: CodeInterpretationStagingConfig) -> dict[str, Any]:
    """Stage one code-interpretation task/template slice as JSONL."""

    if cfg.source_manifest is not None and cfg.content_fingerprint:
        expected = cfg.source_manifest.fingerprint()
        if cfg.content_fingerprint != expected:
            raise ValueError(
                f"content_fingerprint mismatch: config has {cfg.content_fingerprint}, source manifest has {expected}"
            )

    task = CODE_INTERPRETATION_TASKS_BY_KEY[cfg.task_key]
    if cfg.template_key not in CODE_INTERPRETATION_TEMPLATES_BY_KEY:
        raise ValueError(f"Unknown code-interpretation template: {cfg.template_key}")
    if len(task.support_examples) != CODE_INTERPRETATION_NUM_FEWSHOT:
        raise ValueError(f"{cfg.task_key} must have exactly {CODE_INTERPRETATION_NUM_FEWSHOT} support examples")

    StoragePath(cfg.output_path).mkdirs(exist_ok=True)
    out_file = posixpath.join(cfg.output_path, cfg.output_filename)
    compression = "gzip" if out_file.endswith(".gz") else None

    with atomic_rename(out_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression=compression) as outfile:
            for heldout_index in range(len(task.heldout_examples)):
                json.dump(code_interpretation_record(cfg.task_key, cfg.template_key, heldout_index), outfile)
                outfile.write("\n")

    output_size = StoragePath(out_file).size()
    result: dict[str, Any] = {
        "record_count": len(task.heldout_examples),
        "bytes_written": output_size,
        "output_file": out_file,
    }

    if cfg.source_manifest is not None:
        metadata_path = write_ingestion_metadata_json(
            manifest=cfg.source_manifest,
            materialized_output=MaterializedOutputMetadata(
                input_path="code_interpretation_static",
                output_path=cfg.output_path,
                output_file=out_file,
                record_count=len(task.heldout_examples),
                bytes_written=output_size,
                metadata={
                    "task_key": cfg.task_key,
                    "task_family": task.family,
                    "template_key": cfg.template_key,
                    "renderer_version": CODE_INTERPRETATION_RENDERER_VERSION,
                    "num_fewshot": CODE_INTERPRETATION_NUM_FEWSHOT,
                    "heldout_examples": len(task.heldout_examples),
                },
            ),
        )
        result["metadata_file"] = metadata_path

    return result
