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
answer from its answer fields; Uncheatable records are free text. Every record
of the manifests either gets an item or a ``reason``.
"""

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from itertools import pairwise
from typing import Any, NamedTuple, Protocol

import pyarrow as pa
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.encoder.tables import write_table
from experiments.datakit.decontam.encoder.text_keys import WINDOW, text_stream
from experiments.datakit.decontam.lmh_loader import (
    flatten_task_dict,
    materialize_first_nonempty_split,
    patch_hf_dataset_loading,
)
from experiments.datakit.decontam.prepare_eval_corpus import (
    _PASSAGE_FIELDS,
    AA_EVALS,
    UNCHEATABLE_RECORDS,
    AARecordType,
    _aa_records,
    _iter_aa_rows,
    _row_identifier,
    _uncheatable_records,
)

logger = logging.getLogger(__name__)

WINDOWS = (10, 13, 16, WINDOW)
LENGTH_BINS = (0, 10, 20, 30, 50, 70, 100, 200, 10**9)

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("eval", pa.string()),
        ("suite", pa.string()),
        ("question", pa.string()),
        ("choices", pa.list_(pa.string())),
        ("answer", pa.string()),
        ("prompt", pa.string()),
        ("passage", pa.string()),
        ("choices_mode", pa.string()),
        ("reason", pa.string()),
    ]
)


def _stream_len(text: str) -> int:
    return len(text_stream(text))


def _is_passage(key: str) -> bool:
    name = key.lower()
    return name in _PASSAGE_FIELDS or "passage" in name or "context" in name


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
    """One row of the item table (see ``SCHEMA``)."""
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
    passage_like = {value for key, value in pairs if (_is_passage(key) or value in shared) and value in candidates}
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


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="the sealed lm-eval manifest (lmh/_manifest.json) of the corpus"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--part", type=int, required=True, help="this part of the lm-eval task list; part 0 also does AA and Uncheatable"
    )
    parser.add_argument("--parts", type=int, required=True)
    args = parser.parse_args()

    with StoragePath(args.manifest).open("r") as source:
        manifest = json.load(source)
    expected = {artifact["task"]: artifact["records"] for artifact in manifest["artifacts"]}
    expected.update({cfg.name: cfg.expected_records for cfg in AA_EVALS})
    expected.update({f"uncheatable-{subset}": count for subset, count in UNCHEATABLE_RECORDS.items()})
    leaves_of = manifest["required_tasks"]
    names = sorted(leaves_of)[args.part :: args.parts]
    sources = [lmh_rows(names, leaves_of)]
    if args.part == 0:
        sources += [aa_rows(), uncheatable_rows()]
    report = {}
    for source in sources:
        for result in source:
            if result.rows:
                table = pa.Table.from_pylist(result.rows, schema=SCHEMA)
                write_table(table, prefix_join(args.output, f"items/{result.name}.parquet"))
            counts = summary(result.rows) if result.rows else {"items": 0}
            entry = {"error": result.error, "expected_records": expected.get(result.name), **counts}
            report[result.name] = entry
            logger.info(
                "%s: %d items (expected %s) %s", result.name, entry["items"], entry["expected_records"], result.error
            )
    StoragePath(prefix_join(args.output, f"report/part-{args.part:02d}.json")).write_text(json.dumps(report, indent=1))
    logger.info(
        "part %d done: %d evals, %d items", args.part, len(report), sum(entry["items"] for entry in report.values())
    )


if __name__ == "__main__":
    main()
