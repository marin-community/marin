# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure pdf-inspector's own output text, one narrow row per document.

Router v1 could only guess at extraction damage. Its cheap route was Docling, Docling costs ~1000 ms
a page, and nothing that expensive can run *before* the decision about whether to run it -- so every
signal about garbling, repetition and token shape had to be inferred from the page's fonts and
geometry without decoding anything. ``mean_fonts_unmappable`` is a prediction that text will come
out wrong; it is not an observation that it did.

pdf-inspector removes that constraint. It costs 4.66 ms/page, it runs on every document whether or
not the document is later escalated, and its markdown is sitting there. Measuring it is free in the
only sense that matters to a router: the pass is already paid for.

:mod:`~experiments.datakit.build_pdf_source.quality.build_inspector_study` deliberately keeps no
text -- it stores agreement columns and library signals and nothing else, which is what makes it a
40 MB table over a 126 GB sample. This module adds the statistics rather than the text, for the same
reason, and joins to that table on ``source_id``.

The statistics are the ones a router can act on:

``replacement`` and ``alpha``
    Whether the glyph-to-Unicode mapping produced characters. A broken ToUnicode CMap shows up as
    replacement characters or as a collapse in the alphabetic fraction, and unlike the font-table
    signals this is what actually came out.
``repeat_line`` and ``max_line_repeats``
    Repetitive page structure, which is what sends the VLM into a decode loop. Loop repair is one of
    the failure modes the score has to predict rather than be gated on, and its cause is visible in
    the cheap route's own output.
``chars_per_source_page`` and ``markdown_chars``
    Expected output length, which is what predicts VLM truncation: a page with 12,000 characters of
    text will not fit the completion budget, and the router should know that before it spends the
    render.
``pipe_row`` and ``heading``
    How much structure pdf-inspector believes it recovered, as distinct from how much
    ``pages_with_tables`` says it found.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-inspector-output-study --extra pdf \\
        --cpu 8 --memory 24GB --disk 16GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.build_inspector_output_study
"""

import json
import logging
import re
import sys
from collections import Counter
from importlib.metadata import version

import polars as pl
from fray.types import ResourceConfig
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.quality.build_route_study import shards, storage
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import WORKER_FLAG, Worker, read_exactly

logger = logging.getLogger(__name__)

LIBRARY_VERSION = "1.17.0"
STUDY_ROOT = "s3://marin-us-east-02a/marin/data/pdf_quality"
OUTPUT_PREFIX = f"{STUDY_ROOT}/cc_focus_2026_22_inspector_output_{LIBRARY_VERSION.replace('.', '_')}"

MODULE_NAME = "experiments.datakit.build_pdf_source.quality.build_inspector_output_study"
EXTRACT_OP = "extract"

READ_COLUMNS = ("source_id", "num_pages", "pdf")

# A token this long is not a word. Long runs come from tables serialized without separators and from
# CID-mapped subsets that decode into one unbroken string, and both are extraction damage.
LONG_TOKEN_CHARS = 20
REPLACEMENT_CHAR = "�"

_TOKEN = re.compile(r"\S+")
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")

_TASK_RESOURCES = ResourceConfig(cpu=2, ram="12g", disk="8g")
_WORKER_RESOURCES = ResourceConfig(cpu=16, ram="96g", disk="64g")
# Explicit, and not the 1 GB default: the coordinator holds shard, retry and shuffle state for every
# task, and at this shard count the default is what dies at exit 137 one task short of the end.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=2, ram="16g", preemptible=False)
_MAX_WORKERS = 12
_HEARTBEAT_TIMEOUT = 30 * 60

# Every column a failed extraction leaves empty, typed rather than inferred: polars reads an
# all-null column as `Null`, and a shard where nothing failed would not concatenate against one
# where something did.
_MEASURED = (
    "replacement_ratio",
    "alpha_ratio",
    "digit_ratio",
    "space_ratio",
    "newline_ratio",
    "single_char_token_ratio",
    "mean_token_length",
    "long_token_ratio",
    "repeat_line_ratio",
    "max_line_repeats",
    "empty_page_fraction",
    "chars_per_source_page",
    "pipe_row_ratio",
    "heading_ratio",
)
OUTPUT_SCHEMA = {
    "inspector_output_error": pl.String,
    **{f"inspector_output_{name}": pl.Float64 for name in _MEASURED},
}


def measure(pages: list[str], num_pages: int) -> dict:
    """Every statistic this table carries, from one document's page markdown.

    One pass over the concatenated text for the character ratios and one over the lines for the
    structural ones. Ratios are per character or per line rather than per document so a long report
    and a flyer are on the same scale; the two quantities that are deliberately *not* normalized
    that way are ``chars_per_source_page``, which is the truncation predictor and has to be an
    absolute length, and ``max_line_repeats``, which is a count because one line repeated 400 times
    is a loop whatever the document's length.
    """
    text = "\n".join(pages)
    characters = len(text)
    if characters == 0:
        return {
            f"inspector_output_{name}": 0.0
            for name in _MEASURED
            if name not in ("empty_page_fraction", "mean_token_length")
        } | {
            "inspector_output_empty_page_fraction": 1.0,
            "inspector_output_mean_token_length": 0.0,
        }

    alpha = digits = spaces = newlines = replacements = 0
    for character in text:
        alpha += character.isalpha()
        digits += character.isdigit()
        spaces += character == " "
        newlines += character == "\n"
        replacements += character == REPLACEMENT_CHAR

    tokens = _TOKEN.findall(text)
    token_count = max(len(tokens), 1)
    lines = [line.strip() for line in text.split("\n")]
    content_lines = [line for line in lines if line]
    line_count = max(len(content_lines), 1)
    repeats = Counter(content_lines)

    return {
        "inspector_output_replacement_ratio": replacements / characters,
        "inspector_output_alpha_ratio": alpha / characters,
        "inspector_output_digit_ratio": digits / characters,
        "inspector_output_space_ratio": spaces / characters,
        "inspector_output_newline_ratio": newlines / characters,
        "inspector_output_single_char_token_ratio": sum(len(token) == 1 for token in tokens) / token_count,
        "inspector_output_mean_token_length": sum(len(token) for token in tokens) / token_count,
        "inspector_output_long_token_ratio": sum(len(token) > LONG_TOKEN_CHARS for token in tokens) / token_count,
        # Lines beyond the first occurrence of each distinct line: 0.0 for a document that never
        # repeats itself, approaching 1.0 for one that says the same thing over and over.
        "inspector_output_repeat_line_ratio": (line_count - len(repeats)) / line_count,
        "inspector_output_max_line_repeats": float(max(repeats.values(), default=0)),
        "inspector_output_empty_page_fraction": sum(not page.strip() for page in pages) / max(len(pages), 1),
        "inspector_output_chars_per_source_page": characters / max(num_pages, 1),
        "inspector_output_pipe_row_ratio": sum(bool(_PIPE_ROW.match(line)) for line in content_lines) / line_count,
        "inspector_output_heading_ratio": sum(bool(_HEADING.match(line)) for line in content_lines) / line_count,
    }


def worker_main() -> None:
    """Serve length-prefixed documents from stdin until the driver closes it.

    Process-isolated for Stage 0's reason: three unbounded-depth recursions over nested Form
    XObjects remain in the crate, and a stack overflow is a ``SIGSEGV`` rather than a catchable
    panic.
    """
    import faulthandler  # noqa: PLC0415 - only the disposable process needs a fault handler

    import pdf_inspector  # noqa: PLC0415 - the whole point is to import it out of process

    installed = version("pdf-inspector")
    if installed != LIBRARY_VERSION:
        raise RuntimeError(f"{OUTPUT_PREFIX} is the {LIBRARY_VERSION} table; pdf-inspector {installed} installed")
    faulthandler.enable()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        payload = read_exactly(stdin, json.loads(header)["size"])
        try:
            pages = [page.markdown for page in pdf_inspector.extract_pages_markdown_bytes(payload).pages]
            reply = {"pages": pages}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # PyO3 derives PanicException from BaseException.
            reply = {"error": f"{type(error).__name__}: {error}"[:500]}
        stdout.write(json.dumps(reply).encode() + b"\n")
        stdout.flush()


def study_shard(work: tuple[int, str]) -> int:
    """Emit one row per document in one sample shard, or skip it if its output already exists."""
    index, shard = work
    fs = storage()
    output = f"{OUTPUT_PREFIX}/part-{index:05d}.parquet"
    if fs.exists(output):
        return 0

    with fs.open(shard, "rb") as stream:
        table = pl.read_parquet(stream, columns=list(READ_COLUMNS))

    worker = Worker(MODULE_NAME)
    rows, failed = [], 0
    try:
        for document in table.iter_rows(named=True):
            reply = worker.call(EXTRACT_OP, document["pdf"])
            pages = reply.result.get("pages")
            row = {"source_id": document["source_id"], "inspector_output_error": reply.result.get("error")}
            if pages is None:
                failed += 1
                row.update(dict.fromkeys(OUTPUT_SCHEMA.keys() - {"inspector_output_error"}))
            else:
                row.update(measure(pages, document["num_pages"]))
            rows.append(row)
    finally:
        worker.stop()

    counters.pipeline.update_counter("inspector_output/documents", len(rows))
    counters.pipeline.update_counter("inspector_output/failed", failed)
    frame = pl.DataFrame(rows, schema_overrides=OUTPUT_SCHEMA)
    with fs.open(output, "wb") as stream:
        frame.write_parquet(stream)
    logger.info("shard %d: %d documents, %d failed", index, len(rows), failed)
    return len(rows)


def main() -> None:
    configure_logging(logging.INFO)
    installed = version("pdf-inspector")
    if installed != LIBRARY_VERSION:
        raise RuntimeError(f"{OUTPUT_PREFIX} is the {LIBRARY_VERSION} table; pdf-inspector {installed} installed")

    work = shards()
    logger.info("inspector output study: %d shards -> %s", len(work), OUTPUT_PREFIX)
    outcome = ZephyrContext(
        name="pdf-inspector-output-study",
        resources=_WORKER_RESOURCES,
        coordinator_resources=_COORDINATOR_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(Dataset.from_list(work).map(study_shard), map_task_resources=_TASK_RESOURCES)
    logger.info("done, counters %s", dict(outcome.counters))


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        main()
