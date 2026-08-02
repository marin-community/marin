# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit exact counts and row nesting in the expanded student manifests."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from build_manifest import manifest_digest
from ladder_config import EVAL_ROWS_PER_SOURCE, MANIFEST_ROOT, read_json, write_json

IDENTITY_COLUMNS = (
    "input_path",
    "input_row_group",
    "input_row_in_group",
    "raw_sha256",
    "normalized_sha256",
    "split",
    "eval_rank",
    "train_rank",
)
RESULT_FILE = Path("/tmp/luxical-expanded-manifest-audit")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def expanded_root(rung: str) -> str:
    """Return the artifact root for one expanded rung."""
    return f"{MANIFEST_ROOT}/fast-student/expanded-{rung}"


def selected_rows(table: pa.Table, rung: str) -> pa.Table:
    """Return the fixed evaluation rows and selected training rung."""
    selected = pc.or_(pc.equal(table["split"], "eval"), table[f"in_{rung}"])
    return table.filter(selected).select(IDENTITY_COLUMNS)


def input_positions(table: pa.Table) -> set[tuple[str, int, int]]:
    """Return the input positions in one source table."""
    return set(
        zip(
            table["input_path"].to_pylist(),
            table["input_row_group"].to_pylist(),
            table["input_row_in_group"].to_pylist(),
            strict=True,
        )
    )


def source_metrics(
    ten_million_url: str,
    thirty_million_url: str,
    expected_ten_million_rows: int,
    expected_thirty_million_rows: int,
) -> dict[str, int]:
    """Verify exact 10M-to-30M nesting for one source."""
    columns = [*IDENTITY_COLUMNS, "in_10m", "in_30m"]
    ten_filesystem, ten_path = fsspec.core.url_to_fs(ten_million_url)
    thirty_filesystem, thirty_path = fsspec.core.url_to_fs(thirty_million_url)
    ten = pq.read_table(
        ten_path,
        filesystem=ten_filesystem,
        columns=[column for column in columns if column != "in_30m"],
    )
    thirty = pq.read_table(thirty_path, filesystem=thirty_filesystem, columns=columns)

    ten_train = pc.equal(ten["split"], "train")
    thirty_train = pc.equal(thirty["split"], "train")
    ten_count = int(pc.sum(pc.cast(ten["in_10m"], "int64")).as_py())
    thirty_ten_count = int(pc.sum(pc.cast(thirty["in_10m"], "int64")).as_py())
    thirty_count = int(pc.sum(pc.cast(thirty["in_30m"], "int64")).as_py())
    if ten_count != expected_ten_million_rows or thirty_ten_count != expected_ten_million_rows:
        raise ValueError("The source has an incorrect 10M row count")
    if thirty_count != expected_thirty_million_rows:
        raise ValueError("The source has an incorrect 30M row count")
    if pc.any(pc.and_(thirty["in_10m"], pc.invert(thirty["in_30m"]))).as_py():
        raise ValueError("The source has a 10M row outside the 30M rung")
    if (
        pc.any(pc.and_(pc.invert(ten_train), ten["in_10m"])).as_py()
        or pc.any(pc.and_(pc.invert(thirty_train), pc.or_(thirty["in_10m"], thirty["in_30m"]))).as_py()
    ):
        raise ValueError("The source has an evaluation row in a training rung")

    ten_selected = selected_rows(ten, "10m")
    thirty_selected = selected_rows(thirty, "10m")
    if not ten_selected.equals(thirty_selected):
        raise ValueError("The exact 10M rows differ between the expanded manifests")
    if len(ten_selected) != EVAL_ROWS_PER_SOURCE + expected_ten_million_rows:
        raise ValueError("The source has an incorrect evaluation row count")
    if len(thirty) != EVAL_ROWS_PER_SOURCE + expected_thirty_million_rows:
        raise ValueError("The source has an incorrect total row count")

    train_ranks = thirty.filter(thirty_train)["train_rank"].to_pylist()
    if train_ranks != list(range(expected_thirty_million_rows)):
        raise ValueError("The source training ranks are not continuous")
    if len(input_positions(thirty)) != len(thirty):
        raise ValueError("The source input positions are not unique")

    return {
        "evaluation_rows": EVAL_ROWS_PER_SOURCE,
        "train_10m_rows": ten_count,
        "train_30m_rows": thirty_count,
        "total_30m_rows": len(thirty),
    }


def audit_expanded_manifest() -> dict[str, Any]:
    """Audit both expanded manifests and return the full report."""
    ten_manifest_url = f"{expanded_root('10m')}/manifest.json"
    thirty_manifest_url = f"{expanded_root('30m')}/manifest.json"
    ten = read_json(ten_manifest_url)
    thirty = read_json(thirty_manifest_url)
    if manifest_digest(ten) != ten["sha256"] or manifest_digest(thirty) != thirty["sha256"]:
        raise ValueError("An expanded manifest digest does not match its content")
    if ten["base_manifest_sha256"] != thirty["base_manifest_sha256"]:
        raise ValueError("The expanded manifests have different base manifests")
    if set(ten["sources"]) != set(thirty["sources"]):
        raise ValueError("The expanded manifests have different sources")
    if int(ten["training_targets"]["10m"]) != 10_000_000:
        raise ValueError("The 10M manifest has an incorrect training target")
    if int(thirty["training_targets"]["30m"]) != 30_000_000:
        raise ValueError("The 30M manifest has an incorrect training target")

    sources = {}
    for index, source in enumerate(sorted(thirty["sources"]), start=1):
        logger.info("Auditing expanded source %d/%d: %s", index, len(thirty["sources"]), source)
        ten_result = ten["sources"][source]
        thirty_result = thirty["sources"][source]
        expected_ten = int(ten_result["counts"]["train_10m"])
        if int(thirty_result["counts"]["train_10m"]) != expected_ten:
            raise ValueError(f"Source {source} has different 10M quotas")
        metrics = source_metrics(
            ten_result["output_url"],
            thirty_result["output_url"],
            expected_ten,
            int(thirty_result["counts"]["train_30m"]),
        )
        expected_input_counts = {
            result["path"]: int(result["selected_rows"]) for result in thirty_result["selected_input_files"]
        }
        filesystem, path = fsspec.core.url_to_fs(thirty_result["output_url"])
        paths = pq.read_table(path, filesystem=filesystem, columns=["input_path"])["input_path"].to_pylist()
        if Counter(paths) != expected_input_counts:
            raise ValueError(f"Source {source} has different input-file counts")
        sources[source] = metrics

    totals = {
        key: sum(metrics[key] for metrics in sources.values())
        for key in ("evaluation_rows", "train_10m_rows", "train_30m_rows", "total_30m_rows")
    }
    expected_evaluation_rows = len(sources) * EVAL_ROWS_PER_SOURCE
    expected_totals = {
        "evaluation_rows": expected_evaluation_rows,
        "train_10m_rows": 10_000_000,
        "train_30m_rows": 30_000_000,
        "total_30m_rows": expected_evaluation_rows + 30_000_000,
    }
    if totals != expected_totals:
        raise ValueError(f"The expanded manifest totals differ: {totals}")
    return {
        "ten_million_manifest_url": ten_manifest_url,
        "ten_million_manifest_sha256": ten["sha256"],
        "thirty_million_manifest_url": thirty_manifest_url,
        "thirty_million_manifest_sha256": thirty["sha256"],
        "base_manifest_sha256": thirty["base_manifest_sha256"],
        "source_count": len(sources),
        "totals": totals,
        "exact_ten_million_nesting": True,
        "passed": True,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = audit_expanded_manifest()
    report_url = f"{expanded_root('30m')}/manifest-audit.json"
    write_json(report_url, report)
    summary = {key: value for key, value in report.items() if key != "sources"} | {"audit_url": report_url}
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("FAST_STUDENT_EXPANDED_MANIFEST_AUDIT=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
