# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF-backed delimiter, whitespace, and table-format target-only PPL slices."""

from __future__ import annotations

import csv
import io
import json
import os
import posixpath
import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import fsspec
from levanter.utils import fsspec_utils
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import HfDatasetSpec

EPIC_5005 = 5005
SYNTHETIC_DELIMITER_FORMAT_ISSUE = 5618
SYNTHETIC_DELIMITER_FORMAT_HF_DATASET_ID = "marin-community/synth-delimiter-format-ppl"
SYNTHETIC_DELIMITER_FORMAT_SOURCE = "generated_synthetic_delimiter_format_ppl_v1"
SYNTHETIC_DELIMITER_FORMAT_HF_REVISION = "5d3d6dfdd1f1dd8a691099edade2f210d2f2e2e8"
SYNTHETIC_DELIMITER_FORMAT_SEED = 6505
EXAMPLES_PER_CONFIG = 1000
LOCAL_SAMPLE_EXAMPLES_PER_CONFIG = 4
RAW_SHARD_NAME = "data-00000-of-00001.jsonl.gz"


class DelimiterFormatFamily(StrEnum):
    DELIMITED_FIELDS = "delimited_fields"
    TABLE_ROWS = "table_rows"
    WHITESPACE_CONTROL = "whitespace_control"


@dataclass(frozen=True)
class DelimiterFormatPplSlice:
    family: DelimiterFormatFamily
    task_name: str
    hf_config_name: str

    @property
    def registry_key(self) -> str:
        return posixpath.join("synthetic_delimiter_format_ppl", self.family.value, self.task_name)

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_delimiter_format_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{SYNTHETIC_DELIMITER_FORMAT_ISSUE}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"seed:{SYNTHETIC_DELIMITER_FORMAT_SEED}",
            f"examples:{EXAMPLES_PER_CONFIG}",
            f"source:{SYNTHETIC_DELIMITER_FORMAT_SOURCE}",
            f"hf_revision:{SYNTHETIC_DELIMITER_FORMAT_HF_REVISION}",
            "loss:target_only",
            "format:supervised",
        )

    def to_raw_text_dataset(self, *, hf_dataset_id: str) -> RawTextEvaluationDataset:
        return supervised_text_dataset(
            HfDatasetSpec(
                id=hf_dataset_id,
                name=self.hf_config_name,
                revision=SYNTHETIC_DELIMITER_FORMAT_HF_REVISION,
            ),
            input_key="input",
            target_key="target",
            split="validation",
            tags=self.tags,
        )


def _slice(
    *,
    family: DelimiterFormatFamily,
    task_name: str,
) -> DelimiterFormatPplSlice:
    return DelimiterFormatPplSlice(
        family=family,
        task_name=task_name,
        hf_config_name=task_name,
    )


SYNTHETIC_DELIMITER_FORMAT_PPL_SLICES: tuple[DelimiterFormatPplSlice, ...] = (
    _slice(family=DelimiterFormatFamily.DELIMITED_FIELDS, task_name="tsv_next_field"),
    _slice(family=DelimiterFormatFamily.DELIMITED_FIELDS, task_name="csv_next_field"),
    _slice(family=DelimiterFormatFamily.DELIMITED_FIELDS, task_name="rare_control_delimiters"),
    _slice(family=DelimiterFormatFamily.TABLE_ROWS, task_name="pipe_rows"),
    _slice(family=DelimiterFormatFamily.TABLE_ROWS, task_name="fixed_width_rows"),
    _slice(family=DelimiterFormatFamily.TABLE_ROWS, task_name="markdown_table_rows"),
    _slice(family=DelimiterFormatFamily.WHITESPACE_CONTROL, task_name="aligned_space_columns"),
    _slice(family=DelimiterFormatFamily.WHITESPACE_CONTROL, task_name="python_indentation_or_makefile_tabs"),
)


def synthetic_delimiter_format_raw_validation_sets(
    *, hf_dataset_id: str = SYNTHETIC_DELIMITER_FORMAT_HF_DATASET_ID
) -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(hf_dataset_id=hf_dataset_id)
        for slice_ in SYNTHETIC_DELIMITER_FORMAT_PPL_SLICES
    }


def _record(
    *,
    task_name: str,
    row_index: int,
    seed: int,
    input_text: str,
    target: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "id": f"{task_name}_{row_index:05d}",
        "input": input_text,
        "target": target,
        "subset": task_name,
        "task": task_name,
        "seed": seed,
        "metadata": metadata,
    }


def _word(rng: random.Random, prefix: str) -> str:
    syllables = ("al", "br", "cy", "dox", "emi", "flux", "garn", "hel", "ion", "juno", "kest", "luma")
    return prefix + rng.choice(syllables) + str(rng.randint(10, 99))


def _csv_row(values: list[str]) -> str:
    sink = io.StringIO()
    writer = csv.writer(sink, lineterminator="")
    writer.writerow(values)
    return sink.getvalue()


def _generate_tsv_next_field(rng: random.Random, row_index: int, seed: int) -> dict[str, object]:
    columns = ("record_id", "batch", "status", "checksum")
    rows = []
    for offset in range(4):
        rows.append(
            [
                f"rec-{row_index:04d}-{offset}",
                _word(rng, "b"),
                rng.choice(("queued", "ready", "blocked", "done")),
                f"{rng.randrange(16**6):06x}",
            ]
        )
    target_row = [
        f"rec-{row_index:04d}-4",
        _word(rng, "b"),
        rng.choice(("queued", "ready", "blocked", "done")),
        f"{rng.randrange(16**6):06x}",
    ]
    input_text = "\t".join(columns) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n"
    return _record(
        task_name="tsv_next_field",
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target="\t".join(target_row) + "\n",
        metadata={"delimiter": "\\t", "columns": columns, "target_columns": columns},
    )


def _generate_csv_next_field(rng: random.Random, row_index: int, seed: int) -> dict[str, object]:
    columns = ("ticket", "owner", "note", "amount")
    notes = ("needs, review", 'quoted "ok"', "plain", "comma, inside")
    rows = []
    for i in range(3):
        rows.append(
            [
                f"T-{row_index:04d}-{i}",
                _word(rng, "u"),
                rng.choice(notes),
                f"{rng.randint(100, 999)}.{rng.randint(0, 99):02d}",
            ]
        )
    target_row = [
        f"T-{row_index:04d}-3",
        _word(rng, "u"),
        rng.choice(notes),
        f"{rng.randint(100, 999)}.{rng.randint(0, 99):02d}",
    ]
    input_text = _csv_row(list(columns)) + "\n" + "\n".join(_csv_row(row) for row in rows) + "\n"
    return _record(
        task_name="csv_next_field",
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=_csv_row(target_row) + "\n",
        metadata={"delimiter": ",", "quotechar": '"', "columns": columns, "target_columns": columns},
    )


def _generate_rare_control_delimiters(rng: random.Random, row_index: int, seed: int) -> dict[str, object]:
    delimiter = rng.choice(("\x1f", "\x1e", "\x1d"))
    delimiter_name = {"\x1f": "unit_separator", "\x1e": "record_separator", "\x1d": "group_separator"}[delimiter]
    rows = []
    for i in range(4):
        rows.append([f"k{row_index:04d}_{i}", _word(rng, "v"), f"{rng.randint(1000, 9999)}"])
    target_row = [f"k{row_index:04d}_4", _word(rng, "v"), f"{rng.randint(1000, 9999)}"]
    input_text = "\n".join(delimiter.join(row) for row in rows) + "\n"
    return _record(
        task_name="rare_control_delimiters",
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=delimiter.join(target_row) + "\n",
        metadata={"delimiter_codepoint": f"U+{ord(delimiter):04X}", "delimiter_name": delimiter_name},
    )


def _generate_pipe_rows(rng: random.Random, row_index: int, seed: int) -> dict[str, object]:
    rows = []
    for i in range(4):
        rows.append([f"node-{i}", rng.choice(("east", "west", "north", "south")), str(rng.randint(1, 9))])
    target_row = ["node-4", rng.choice(("east", "west", "north", "south")), str(rng.randint(1, 9))]
    input_text = "\n".join(" | ".join(row) for row in rows) + "\n"
    return _record(
        task_name="pipe_rows",
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=" | ".join(target_row) + "\n",
        metadata={"delimiter": "|", "space_padded": True},
    )


def _generate_fixed_width_rows(rng: random.Random, row_index: int, seed: int) -> dict[str, object]:
    widths = (8, 10, 6)

    def render(row: list[str]) -> str:
        return f"{row[0]:<{widths[0]}}{row[1]:>{widths[1]}}{row[2]:>{widths[2]}}"

    rows = [[f"A{row_index:03d}{i}", str(rng.randint(100, 9999)), f"{rng.randint(0, 99)}%"] for i in range(4)]
    target_row = [f"A{row_index:03d}4", str(rng.randint(100, 9999)), f"{rng.randint(0, 99)}%"]
    input_text = "\n".join(render(row) for row in rows) + "\n"
    return _record(
        task_name="fixed_width_rows",
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=render(target_row) + "\n",
        metadata={"widths": widths, "target_columns": (1, 2, 3)},
    )


def _generate_markdown_table_rows(rng: random.Random, row_index: int, seed: int) -> dict[str, object]:
    columns = ("key", "mode", "value")
    rows = [[f"`k{row_index}_{i}`", rng.choice(("read", "write", "sync")), str(rng.randint(10, 99))] for i in range(3)]
    target_row = [f"`k{row_index}_3`", rng.choice(("read", "write", "sync")), str(rng.randint(10, 99))]
    input_text = (
        "| "
        + " | ".join(columns)
        + " |\n"
        + "| --- | --- | --- |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in rows)
        + "\n"
    )
    return _record(
        task_name="markdown_table_rows",
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target="| " + " | ".join(target_row) + " |\n",
        metadata={"delimiter": "|", "table_format": "markdown"},
    )


def _generate_aligned_space_columns(rng: random.Random, row_index: int, seed: int) -> dict[str, object]:
    widths = (12, 9, 7)

    def render(row: list[str]) -> str:
        return f"{row[0]:<{widths[0]}}{row[1]:>{widths[1]}}{row[2]:>{widths[2]}}"

    rows = [[_word(rng, "col"), str(rng.randint(1, 999)), rng.choice(("ok", "hold", "fail"))] for _ in range(5)]
    target_row = [_word(rng, "col"), str(rng.randint(1, 999)), rng.choice(("ok", "hold", "fail"))]
    input_text = "\n".join(render(row) for row in rows) + "\n"
    return _record(
        task_name="aligned_space_columns",
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=render(target_row) + "\n",
        metadata={"widths": widths, "alignment": ("left", "right", "right")},
    )


def _generate_python_indentation_or_makefile_tabs(rng: random.Random, row_index: int, seed: int) -> dict[str, object]:
    if rng.random() < 0.5:
        indent = " " * rng.choice((4, 8))
        item = _word(rng, "item")
        input_text = (
            "def normalize_example(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += item.value\n"
            "    return total\n\n"
            f"def normalize_{row_index}(items):\n"
            f"{indent}total = 0\n"
            f"{indent}for item in items:\n"
        )
        target = f"{indent * 2}total += item.{item}\n{indent}return total\n"
        metadata = {"format": "python", "indent": len(indent), "target_requires_leading_spaces": True}
    else:
        target_name = f"build_{row_index}"
        command = rng.choice(("python -m pytest", "ruff check .", "mkdir -p dist", "touch $@"))
        second_command = rng.choice(("python -m build", "ruff format .", "cp -r assets dist/", "chmod +x $@"))
        input_text = (
            "build_example: src/build_example.py\n"
            "\tpython -m pytest\n"
            "\tpython -m build\n\n"
            f"{target_name}: src/{target_name}.py\n"
        )
        target = "\t" + command + "\n\t" + second_command + "\n"
        metadata = {"format": "makefile", "target_requires_leading_tab": True}
    return _record(
        task_name="python_indentation_or_makefile_tabs",
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata=metadata,
    )


_GENERATORS: dict[str, Callable[[random.Random, int, int], dict[str, object]]] = {
    "tsv_next_field": _generate_tsv_next_field,
    "csv_next_field": _generate_csv_next_field,
    "rare_control_delimiters": _generate_rare_control_delimiters,
    "pipe_rows": _generate_pipe_rows,
    "fixed_width_rows": _generate_fixed_width_rows,
    "markdown_table_rows": _generate_markdown_table_rows,
    "aligned_space_columns": _generate_aligned_space_columns,
    "python_indentation_or_makefile_tabs": _generate_python_indentation_or_makefile_tabs,
}


def iter_synthetic_delimiter_format_records(
    *,
    examples_per_config: int = EXAMPLES_PER_CONFIG,
    seed: int = SYNTHETIC_DELIMITER_FORMAT_SEED,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for config_index, slice_ in enumerate(SYNTHETIC_DELIMITER_FORMAT_PPL_SLICES):
        rng = random.Random(seed + config_index * 1_000_003)
        generator = _GENERATORS[slice_.task_name]
        for row_index in range(examples_per_config):
            records.append(generator(rng, row_index, seed))
    return tuple(records)


def materialize_synthetic_delimiter_format_raw(
    output_path: str,
    *,
    examples_per_config: int = EXAMPLES_PER_CONFIG,
    seed: int = SYNTHETIC_DELIMITER_FORMAT_SEED,
) -> None:
    fsspec_utils.mkdirs(output_path)
    records_by_subset: dict[str, list[dict[str, object]]] = {
        slice_.task_name: [] for slice_ in SYNTHETIC_DELIMITER_FORMAT_PPL_SLICES
    }
    for record in iter_synthetic_delimiter_format_records(examples_per_config=examples_per_config, seed=seed):
        records_by_subset[str(record["subset"])].append(record)

    for subset, records in records_by_subset.items():
        output_file = os.path.join(output_path, subset, RAW_SHARD_NAME)
        fsspec_utils.mkdirs(os.path.dirname(output_file))
        with fsspec.open(output_file, "wt", compression="gzip") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
                sink.write("\n")


synthetic_delimiter_format_raw = StepSpec(
    name=os.path.join("raw", "evals", "synthetic_delimiter_format_ppl"),
    fn=materialize_synthetic_delimiter_format_raw,
    hash_attrs={
        "source": SYNTHETIC_DELIMITER_FORMAT_SOURCE,
        "examples_per_config": EXAMPLES_PER_CONFIG,
        "seed": SYNTHETIC_DELIMITER_FORMAT_SEED,
        "schema": ("input", "target", "id", "subset", "task", "seed", "metadata"),
    },
)


def write_local_sample_jsonl(
    output_file: str,
    *,
    examples_per_config: int = LOCAL_SAMPLE_EXAMPLES_PER_CONFIG,
    seed: int = SYNTHETIC_DELIMITER_FORMAT_SEED,
) -> None:
    fsspec_utils.mkdirs(os.path.dirname(output_file))
    with fsspec.open(output_file, "wt") as sink:
        for record in iter_synthetic_delimiter_format_records(examples_per_config=examples_per_config, seed=seed):
            sink.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            sink.write("\n")


if __name__ == "__main__":
    write_local_sample_jsonl("/tmp/marin_synthetic_delimiter_format_ppl_sample.jsonl")
