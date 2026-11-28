# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the eval item table: one row per record of the required evals.

Each row holds the question, the answer choices and the answer of one record of
the sealed eval corpus, taken from the fields of its pinned source, so that the
matcher can compare a record with the item as the eval holds it.

lm-eval tasks follow one rule, with no per-task field names:

- question: the rendered prompt (``doc_to_text``) without its passage. The
  passage is a field that the corpus names as a passage, or a value that two docs
  of the task share, that the prompt shows and that is at least as long as every
  other field the prompt shows. A minimal-pair item without a prompt uses the
  sentence that the eval scores as right.
- choices: ``doc_to_choice(doc)`` for a multiple-choice task. When those are only
  labels ("A".."D"), the option texts come from a list field with one entry per
  label, fields named like the labels, numbered fields ("option1"..), or the
  lettered option lines of the prompt.
- answer: the option text that ``doc_to_target`` points at (an index or a
  letter), or the target text.

AA rows take the question from the first text field of the benchmark and the
answer from its answer fields; Uncheatable records and the pinned references are
free text. Every record of the manifests either gets an item or a ``reason``.

The table is built once per version (``ITEMS_VERSION``) and its directory is copied to the same path under each
region's ``MARIN_PREFIX``; the reference pipeline reads it (``prepared_items_step``). Two builds can differ (an
eval's source can reorder its choices), so every region gets the bytes of one build. Build it with the cpu and
lm_eval extras and HF_TOKEN::

    uv run iris --cluster=cw-us-east-02a job run --extra=cpu --extra=lm_eval \\
        -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.datakit.decontam.items
"""

import dataclasses
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Iterator
from itertools import pairwise
from typing import Any, NamedTuple, Protocol

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon import ITEM_SCHEMA
from marin.datakit.decon_keys import WINDOW, text_stream
from marin.datakit.decon_tables import read_table, write_table
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from experiments.datakit.decontam.eval_manifests import (
    BestEffortEvalSuite,
    CorpusGate,
    RequiredEvalManifest,
    RequiredEvalSuite,
)
from experiments.datakit.decontam.lmh_loader import (
    flatten_task_dict,
    materialize_first_nonempty_split,
    patch_hf_dataset_loading,
)
from experiments.datakit.decontam.pinned_references import (
    COVERAGE_MANIFEST,
    COVERAGE_RELATIVE,
    COVERAGE_VERSION,
    reference_catalog,
    reference_sources_sha256,
)
from experiments.datakit.decontam.prepare_eval_corpus import (
    _PASSAGE_FIELDS,
    AA_BENCHMARK_NAMES,
    AA_EVALS,
    AA_MANIFEST_RELATIVE,
    EVAL_CORPUS_VERSION,
    EVALS_RELATIVE,
    LMH_MANIFEST_RELATIVE,
    UNCHEATABLE_BENCHMARK_NAMES,
    UNCHEATABLE_MANIFEST_RELATIVE,
    UNCHEATABLE_RECORDS,
    AARecordType,
    _aa_records,
    _iter_aa_rows,
    _row_identifier,
    _uncheatable_records,
    required_lmh_tasks,
)

logger = logging.getLogger(__name__)

# Bump when the builder changes what it writes; a new corpus or reference version also names a new table.
ITEM_BUILDER_REVISION = 1
ITEMS_VERSION = f"{EVAL_CORPUS_VERSION}+{COVERAGE_VERSION}+r{ITEM_BUILDER_REVISION}"
ITEMS_RELATIVE = f"datakit/decontam/items/{ITEMS_VERSION}"
ITEMS_STEP_NAME = "datakit/decontam/items"
# Parts of the build; each loads its share of the lm-eval tasks (about 15 minutes each).
ITEM_PARTS = 4
PART_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="200g")

WINDOWS = (10, 13, 16, WINDOW)
LENGTH_BINS = (0, 10, 20, 30, 50, 70, 100, 200, 10**9)


def _stream_len(text: str) -> int:
    return len(text_stream(text))


def _is_passage(key: str) -> bool:
    name = key.lower()
    return name in _PASSAGE_FIELDS or "passage" in name or "context" in name


# Fields that hold the question itself. A value of one can repeat across items (a generic stem such as
# "Which of the following is true?"), but it is never the item's passage.
_QUESTION_FIELDS = frozenset({"prompt", "input", "problem", "instruction"})


def _is_question_field(key: str) -> bool:
    name = key.lower()
    return name in _QUESTION_FIELDS or "question" in name or "query" in name


def shared_values(docs: list[Any]) -> set[str]:
    """String values repeated across documents of a task."""
    seen: Counter[str] = Counter()
    for doc in docs:
        if isinstance(doc, dict):
            seen.update({value for value in doc.values() if isinstance(value, str)})
    return {value for value, count in seen.items() if count > 1}


def _option_fields(doc: Any, labels: list[str]) -> list[str] | None:
    """The doc fields that hold the option texts behind label choices ("A".."D"), in label order.

    In order: a list field with one entry per label; fields named like the labels ("A".."D");
    fields that share a prefix and are numbered 1..n ("option1", "mc_answer1").
    """
    if not isinstance(doc, dict):
        return None
    for value in doc.values():
        if isinstance(value, list) and len(value) == len(labels) and all(isinstance(entry, str) for entry in value):
            return list(value)
    by_name = {str(key).lower(): value for key, value in doc.items()}
    named = [by_name.get(label.lower()) for label in labels]
    if all(isinstance(value, str | int | float) for value in named):
        return [str(value) for value in named]
    for key in doc:
        name = str(key)
        if name.endswith("1"):
            prefix = name[:-1]
            numbered = [doc.get(f"{prefix}{number + 1}") for number in range(len(labels))]
            if all(isinstance(value, str | int | float) for value in numbered):
                return [str(value) for value in numbered]
    return None


_OPTION_LINE = re.compile(r"^\s*\(?([A-Z])[).:]\s+(\S.*?)\s*$", re.MULTILINE)


_BULLET_LINE = re.compile(r"^\s*[-*\u2022]\s+(\S.*?)\s*$", re.MULTILINE)
_LEADING_LABEL = re.compile(r"^\s*\(?([A-Z])[).:]\s*")


def _parsed_options(prompt: str) -> tuple[list[str], list[str]]:
    """Lettered option lines of a prompt ("(A) text", "A. text", "B: text"): their letters and texts.

    Only a run that starts at "A" and goes on letter by letter counts; the first run of two or more
    is kept, so a later line such as "A: Let's think step by step." does not replace it. Without
    lettered lines, bullet lines after an "Options:" line are the options.
    """
    letters: list[str] = []
    texts: list[str] = []
    for match in _OPTION_LINE.finditer(prompt):
        letter, text = match.group(1), match.group(2)
        if letter == chr(ord("A") + len(letters)):
            letters.append(letter)
            texts.append(text)
        elif letter == "A" and len(letters) < 2:
            letters, texts = [letter], [text]
        elif len(letters) >= 2:
            break
    if len(letters) >= 2:
        return letters, texts
    _, found, tail = prompt.partition("Options:")
    bullets = _BULLET_LINE.findall(tail) if found else []
    return ([chr(ord("A") + n) for n in range(len(bullets))], bullets) if len(bullets) >= 2 else ([], [])


def _without_labels(choices: list[str]) -> list[str]:
    """Choices written with their own letters ("(A)text", "B. text") lose the letters."""
    found = [_LEADING_LABEL.match(choice) for choice in choices]
    if all(found) and [m.group(1) for m in found] == [chr(ord("A") + n) for n in range(len(choices))]:
        return [choice[m.end() :] for choice, m in zip(choices, found, strict=True)]
    return choices


def _texts(value: Any) -> list[str]:
    if isinstance(value, dict):
        # Ids (worker_id, hit_id) are metadata, not answer text.
        return [text for key, inner in value.items() if not str(key).lower().endswith("id") for text in _texts(inner)]
    if isinstance(value, list):
        return [text for inner in value for text in _texts(inner)]
    return [str(value)] if str(value).strip() else []


def _target_text(target: Any) -> Any:
    """A structured target (DROP: number, date, spans) as its text values."""
    return "; ".join(dict.fromkeys(_texts(target))) if isinstance(target, dict) else target


def _structured_target(target: Any, doc: Any) -> Any:
    """The doc field that a target template rendered as its key names, or *target* unchanged.

    A template that joins a structured field (DROP: ``{{answer|join(',')}}``) renders the field's
    keys ("number,date,spans,worker_id,hit_id"); the answer is the field's values.
    """
    if not isinstance(target, str) or not isinstance(doc, dict):
        return target
    rendered = re.sub(r"\s", "", target)
    for value in doc.values():
        if isinstance(value, dict) and value and rendered == ",".join(map(str, value)):
            return value
    return target


class LmTask(Protocol):
    """The part of an lm-eval task that the item table reads."""

    OUTPUT_TYPE: str

    def doc_to_text(self, doc: Any) -> Any: ...

    def doc_to_target(self, doc: Any) -> Any: ...

    def doc_to_choice(self, doc: Any) -> Any: ...


class EvalRows(NamedTuple):
    """The item rows of one eval, or the error that stopped them."""

    name: str
    rows: list[dict[str, Any]]
    error: str


def item_row(
    record_id: str,
    eval_name: str,
    suite: str,
    question: str,
    choices: list[str] | None = None,
    answer: str = "",
    prompt: str = "",
    passage: str = "",
    choices_mode: str = "none",
    reason: str = "",
) -> dict[str, Any]:
    """One row of the item table (see ``marin.datakit.decon.ITEM_SCHEMA``)."""
    return {
        "id": record_id,
        "eval": eval_name,
        "suite": suite,
        "question": question,
        "choices": choices or [],
        "answer": answer,
        "prompt": prompt,
        "passage": passage,
        "choices_mode": choices_mode,
        "reason": reason,
    }


def lmh_item(leaf: str, record_id: str, task: LmTask, doc: Any, shared: set[str]) -> dict[str, Any]:
    reasons = []
    try:
        rendered = task.doc_to_text(doc)
    except Exception as error:
        rendered = ""
        reasons.append(f"doc_to_text: {error!r}")
    prompt = rendered if isinstance(rendered, str) else ""
    prompt_stream = text_stream(prompt)
    values = doc.items() if isinstance(doc, dict) else []
    pairs = [(key, value) for key, value in values if isinstance(value, str) and value.strip()]
    choices: list[str] = []
    mode = "none"
    if task.OUTPUT_TYPE == "multiple_choice":
        try:
            choices = _without_labels([str(choice) for choice in task.doc_to_choice(doc)])
            mode = "options"
        except Exception as error:
            reasons.append(f"doc_to_choice: {error!r}")
    labels = list(choices) if choices and all(_stream_len(choice) <= 2 for choice in choices) else []
    if labels:
        options = _option_fields(doc, labels)
        if options:
            choices, mode = options, "labels->fields"
        else:
            mode = "labels"
    option_set = set(choices) if labels and mode != "labels" else set()

    def id_like(value: str) -> bool:
        # Ids, numbers and one- or two-character labels are never the question.
        stream = text_stream(value)
        return not stream or stream.isdigit() or len(stream) <= 2

    def position(value: str, start: int = 0) -> int:
        return prompt_stream.find(text_stream(value), start) if text_stream(value) else -1

    candidates = [value for _, value in pairs if value not in option_set and value not in labels and not id_like(value)]
    passage_like = {
        value
        for key, value in pairs
        if (_is_passage(key) or (value in shared and not _is_question_field(key))) and value in candidates
    }
    others = [value for value in candidates if value not in passage_like and position(value) >= 0]
    longest_other = max((_stream_len(value) for value in others), default=0)
    # A passage is a shared or passage-named field that the prompt shows and that is at least as long
    # as every other field the prompt shows.
    passage = max(
        (value for value in passage_like if position(value) >= 0 and _stream_len(value) >= longest_other),
        key=_stream_len,
        default="",
    )
    fields = [value for value in candidates if value not in passage_like]
    if (labels and mode == "labels") or (not choices and prompt):
        letters, parsed = _parsed_options(prompt)
        if parsed and (not labels or len(parsed) == len(labels)):
            labels, choices, mode = letters, parsed, "parsed"
    question = prompt.replace(passage, " ").strip() if passage else prompt.strip()
    try:
        target = task.doc_to_target(doc)
    except Exception as error:
        target = ""
        reasons.append(f"doc_to_target: {error!r}")
    target = _target_text(_structured_target(target, doc))
    if isinstance(target, list):
        target = _target_text(target[0]) if target else ""
    if isinstance(target, int) and 0 <= target < len(choices):
        answer = choices[target]
    elif isinstance(target, str) and target.strip().strip("()") in labels and len(choices) == len(labels):
        answer = choices[labels.index(target.strip().strip("()"))]
    else:
        answer = str(target)
    if not question and choices:
        # A minimal-pair item (BLiMP) has no prompt; its content is the sentence that the eval scores as right.
        question = answer if answer in choices else choices[0]
    if not question:
        question = max(fields, key=_stream_len, default="")
    if not question:
        reasons.append("no question")
    return item_row(
        record_id, leaf, "lm-eval", question, choices, answer, prompt, passage, mode, "; ".join(reasons)[:300]
    )


def lmh_rows(task_names: list[str], leaves_of: dict[str, list[str]]) -> Iterator[EvalRows]:
    """The rows of each leaf of the named required lm-eval tasks."""
    from lm_eval.tasks import get_task_dict  # noqa: PLC0415  # optional dep: the lm_eval extra

    patch_hf_dataset_loading()
    for name in task_names:
        try:
            loaded = dict(flatten_task_dict(get_task_dict([name])))
        except Exception as error:
            for leaf in leaves_of[name]:
                yield EvalRows(leaf, [], f"load: {error!r}"[:300])
            continue
        for leaf in leaves_of[name]:
            task = loaded.get(leaf)
            chosen = materialize_first_nonempty_split(task) if task is not None else None
            if chosen is None:
                yield EvalRows(leaf, [], "leaf not loaded or no docs")
                continue
            split, docs = chosen
            shared = shared_values(docs)
            rows = [lmh_item(leaf, f"{leaf}-{split}-{index}", task, doc, shared) for index, doc in enumerate(docs)]
            yield EvalRows(leaf, rows, "")


def _answer_fields(fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if "answer" in field.lower()]


def _has_wrong_answers(fields: list[str]) -> bool:
    """True when the answer fields list wrong answers too, so together they are the choices (GPQA)."""
    return any("incorrect" in field.lower() for field in fields)


def aa_rows() -> Iterator[EvalRows]:
    for cfg in AA_EVALS:
        try:
            raw = list(_iter_aa_rows(cfg))
            records = _aa_records(raw, cfg)
        except Exception as error:
            yield EvalRows(cfg.name, [], f"load: {error!r}"[:300])
            continue
        rows = []
        if cfg.record_type != AARecordType.ROW:
            # Multi-part tasks (SciCode steps, tau3 tasks, Terminal-Bench tasks): the record text is the item.
            for record in records:
                rows.append(item_row(record["id"], cfg.name, "aa", record["text"], prompt=record["text"]))
            yield EvalRows(cfg.name, rows, "")
            continue
        answers = _answer_fields(cfg.text_fields)
        questions = [field for field in cfg.text_fields if field not in answers]
        kept = [(index, row) for index, row in enumerate(raw) if cfg.skip_if is None or not cfg.skip_if(row)]
        for index, row in kept:
            question = next(
                (row[field] for field in questions if isinstance(row.get(field), str) and row[field].strip()), ""
            )
            values = [str(row[field]) for field in answers if isinstance(row.get(field), str | int | float)]
            choices = values if _has_wrong_answers(answers) else []
            rows.append(
                item_row(
                    f"{cfg.subdir}-{_row_identifier(row, index)}",
                    cfg.name,
                    "aa",
                    question,
                    choices,
                    answer=values[0] if values else "",
                    choices_mode="options" if choices else "none",
                    reason="" if question else "no question field",
                )
            )
        yield EvalRows(cfg.name, rows, "")


def uncheatable_rows() -> Iterator[EvalRows]:
    for subset in UNCHEATABLE_RECORDS:
        try:
            records = _uncheatable_records(subset)
        except Exception as error:
            yield EvalRows(f"uncheatable-{subset}", [], f"load: {error!r}"[:300])
            continue
        name = f"uncheatable-{subset}"
        yield EvalRows(name, [item_row(record["id"], name, "uncheatable", record["text"]) for record in records], "")


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for the per-eval report: question length bins, items not indexable at each window, symbol-only choices."""
    lengths = [_stream_len(row["question"]) for row in rows]
    longest = [
        max([length, *(_stream_len(choice) for choice in row["choices"])])
        for row, length in zip(rows, lengths, strict=True)
    ]
    bins = Counter()
    for length in lengths:
        bins[next(f"{low}-{high}" for low, high in pairwise(LENGTH_BINS) if length < high)] += 1
    return {
        "items": len(rows),
        "with_reason": sum(bool(row["reason"]) for row in rows),
        "reasons": dict(Counter(row["reason"] for row in rows if row["reason"]).most_common(5)),
        "choices_mode": dict(Counter(row["choices_mode"] for row in rows)),
        "question_length_bins": dict(bins),
        "not_indexable": {str(window): sum(length < window for length in longest) for window in WINDOWS},
        "symbol_only_choice": sum(
            any(choice.strip() and not _stream_len(choice) for choice in row["choices"]) for row in rows
        ),
    }


def coverage_rows(manifest_path: str) -> Iterator[EvalRows]:
    """The pinned reference suite (``pinned_references``): each record's text is the item."""
    manifest = json.loads(StoragePath(manifest_path).read_text())
    parent = os.path.dirname(manifest_path)
    for entry in manifest["benchmarks"]:
        name = entry["name"]
        try:
            records = read_table(prefix_join(parent, entry["artifact"]), ["id", "text"]).to_pylist()
        except Exception as error:
            yield EvalRows(name, [], f"load: {error!r}"[:300])
            continue
        yield EvalRows(name, [item_row(record["id"], name, "coverage", record["text"]) for record in records], "")


class ItemTable(BaseModel):
    """Outcome of :func:`build_items`: one row per record of the required evals (``marin.datakit.decon.ITEM_SCHEMA``).

    Attributes:
        items_dir: Directory of ``<eval>.parquet`` item files.
        report_path: Per-eval counts (``summary``), with the sealed record count of each eval.
        counters: ``evals`` and ``items``.
    """

    items_dir: DatakitArtifactPath
    report_path: DatakitArtifactPath
    counters: dict[str, int | float]


def _part_report_path(output_path: str, part: int) -> str:
    return prefix_join(output_path, f"report/part-{part:02d}.json")


def _make_part(lmh_manifest_path: str, coverage_manifest_path: str, output_path: str, parts: int):
    """A ``map_shard`` function over part numbers: each part writes the items of its share of the required lm-eval
    tasks (part 0 also AA, Uncheatable and the pinned references) and its report."""

    def build(numbers: Iterator[int], shard: ShardInfo) -> Iterator[dict]:
        manifest = json.loads(StoragePath(lmh_manifest_path).read_text())
        leaves_of = manifest["required_tasks"]
        for part in numbers:
            sources = [lmh_rows(sorted(leaves_of)[part::parts], leaves_of)]
            if part == 0:
                sources += [aa_rows(), uncheatable_rows(), coverage_rows(coverage_manifest_path)]
            report = {}
            for source in sources:
                for result in source:
                    if result.rows:
                        table = pa.Table.from_pylist(result.rows, schema=ITEM_SCHEMA)
                        write_table(table, prefix_join(output_path, f"items/{result.name}.parquet"))
                    counts = summary(result.rows) if result.rows else {"items": 0}
                    report[result.name] = {"error": result.error, **counts}
                    logger.info("%s: %d items %s", result.name, counts["items"], result.error)
            report_path = _part_report_path(output_path, part)
            StoragePath(report_path).write_text(json.dumps(report, indent=1))
            yield {"part": part, "evals": len(report)}

    return build


def expected_records(lmh_manifest_path: str, coverage_manifest_path: str) -> dict[str, int]:
    """The sealed record count of every required eval: each leaf of the required lm-eval tasks, every AA and
    Uncheatable benchmark, and every pinned reference."""
    lmh = json.loads(StoragePath(lmh_manifest_path).read_text())
    records = {artifact["task"]: artifact["records"] for artifact in lmh["artifacts"]}
    expected = {leaf: records[leaf] for leaves in lmh["required_tasks"].values() for leaf in leaves}
    expected.update({cfg.name: cfg.expected_records for cfg in AA_EVALS})
    expected.update({f"uncheatable-{subset}": count for subset, count in UNCHEATABLE_RECORDS.items()})
    coverage = json.loads(StoragePath(coverage_manifest_path).read_text())
    expected.update({entry["name"]: entry["expected_records"] for entry in coverage["benchmarks"]})
    return expected


def build_items(
    *,
    gate: CorpusGate,
    lmh_manifest_path: str,
    coverage_manifest_path: str,
    output_path: str,
    parts: int,
    worker_resources: ResourceConfig,
) -> ItemTable:
    """Build the item table of the required evals under *output_path* once *gate* passes, and fail unless every
    required eval loaded with exactly its sealed record count."""
    gate.check()
    context = ZephyrContext(name="decon-items", resources=worker_resources, max_workers=parts)
    context.execute(
        Dataset.from_list(list(range(parts))).map_shard(
            _make_part(lmh_manifest_path, coverage_manifest_path, output_path, parts)
        ),
        map_task_resources=worker_resources,
    )
    report: dict[str, Any] = {}
    for part in range(parts):
        report.update(json.loads(StoragePath(_part_report_path(output_path, part)).read_text()))
    expected = expected_records(lmh_manifest_path, coverage_manifest_path)
    failures = [
        f"{name}: {report.get(name, {}).get('items', 0)} items, expected {count} "
        f"({report.get(name, {}).get('error') or 'count differs'})"
        for name, count in expected.items()
        if name not in report or report[name]["error"] or report[name]["items"] != count
    ]
    if failures:
        raise ValueError(f"the item table is incomplete for {len(failures)} required evals: {failures[:20]}")
    report_path = prefix_join(output_path, "report.json")
    merged = {name: {**entry, "expected_records": expected.get(name)} for name, entry in report.items()}
    StoragePath(report_path).write_text(json.dumps(merged, indent=1))
    return ItemTable(
        items_dir=prefix_join(output_path, "items"),
        report_path=report_path,
        counters={"evals": len(report), "items": sum(entry["items"] for entry in report.values())},
    )


def required_gate() -> CorpusGate:
    """The sealed corpora under the current storage prefix that the item table must cover."""
    eval_root = prefix_join(marin_prefix(), EVALS_RELATIVE)
    return CorpusGate(
        corpus_version=EVAL_CORPUS_VERSION,
        required_suites=(
            RequiredEvalSuite(prefix_join(eval_root, AA_MANIFEST_RELATIVE), AA_BENCHMARK_NAMES),
            RequiredEvalSuite(prefix_join(eval_root, UNCHEATABLE_MANIFEST_RELATIVE), UNCHEATABLE_BENCHMARK_NAMES),
        ),
        required_manifests=(
            RequiredEvalManifest(
                path=coverage_manifest_path(),
                version=COVERAGE_VERSION,
                names=tuple(source.name for source in reference_catalog()),
                source_catalog_sha256=reference_sources_sha256(reference_catalog()),
            ),
        ),
        best_effort_suite=BestEffortEvalSuite(
            prefix_join(eval_root, LMH_MANIFEST_RELATIVE), EVAL_CORPUS_VERSION, required_lmh_tasks()
        ),
    )


def coverage_manifest_path() -> str:
    return prefix_join(prefix_join(marin_prefix(), COVERAGE_RELATIVE), COVERAGE_MANIFEST)


def prepared_items_step() -> StepSpec:
    """The prepared item table of the current storage prefix, as a step that the pipeline reads but never runs."""
    return StepSpec(
        name=ITEMS_STEP_NAME,
        override_output_path=prefix_join(marin_prefix(), ITEMS_RELATIVE),
        hash_attrs={"version": ITEMS_VERSION},
    )


def main() -> None:
    configure_logging(logging.INFO)
    registered = prepared_items_step()
    lmh_manifest = prefix_join(prefix_join(marin_prefix(), EVALS_RELATIVE), LMH_MANIFEST_RELATIVE)
    build = dataclasses.replace(
        registered,
        fn=lambda output_path: build_items(
            gate=required_gate(),
            lmh_manifest_path=lmh_manifest,
            coverage_manifest_path=coverage_manifest_path(),
            output_path=output_path,
            parts=ITEM_PARTS,
            worker_resources=PART_RESOURCES,
        ),
    )
    StepRunner().run([build])


if __name__ == "__main__":
    main()
