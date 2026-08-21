# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "pandas", "tqdm"]
# ///
"""Post-hoc rescore DCLM BigBench generation samples with DCLM-safe filtering.

The first full 300M DCLM Core sweep used the pinned lm-eval BigBench
generate-until template without a response filter and with a 32-token generation
cap. This made ordinary answers such as ``" 17"`` fail exact match against
``"17"``. This script corrects the already-collected sample files for the
tasks whose failure mode is response formatting, then recomputes the DCLM macro.

``bb_repeat_copy_logic_10shot`` is intentionally not repaired here: old samples
were generated with 32 tokens and can be truncated, so they require a rerun with
the fixed 128-token launcher config.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
from tqdm import tqdm


TWO_PHASE_MANY_DIR = Path("experiments/domain_phase_mix/exploratory/two_phase_many")
METRIC_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "metric_registry" / "300m_dclm_core_completion"
DEFAULT_SAMPLE_PREFIX = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_300m_dclm_core_full22_all_20260608/evaluation/lm_evaluation_harness"
)
DEFAULT_MAX_WORKERS = 32
DCLM_TOTAL_TASKS = 22
EVAL_KEY_RE = re.compile(r"(?P<eval_key>dclm300m_.+)-[0-9a-f]{6}$")
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class BigBenchGenerationTask:
    """One BigBench generation component that can be rescored from samples."""

    alias: str
    task_name: str
    sample_stem: str

    @property
    def task_dir(self) -> str:
        return f"{self.alias}_10shot"

    @property
    def metric_column(self) -> str:
        return f"lm_eval/{self.task_name}/exact_match"

    @property
    def raw_score_column(self) -> str:
        return f"lm_eval/dclm_core/{self.alias}/raw_score"

    @property
    def centered_column(self) -> str:
        return f"lm_eval/dclm_core/{self.alias}/centered_accuracy"


RESCORABLE_TASKS = (
    BigBenchGenerationTask("bb_qa_wikidata_10shot", "bigbench_qa_wikidata_generate_until", "bigbench_qa_wikidata_generate_until"),
    BigBenchGenerationTask("bb_dyck_languages_10shot", "bigbench_dyck_languages_generate_until", "bigbench_dyck_languages_generate_until"),
    BigBenchGenerationTask("bb_operators_10shot", "bigbench_operators_generate_until", "bigbench_operators_generate_until"),
    BigBenchGenerationTask("bb_cs_algorithms_10shot", "bigbench_cs_algorithms_generate_until", "bigbench_cs_algorithms_generate_until"),
)
RERUN_REQUIRED_TASKS = ("bb_repeat_copy_logic_10shot",)


def _gcs_path(path: str) -> str:
    return path.removeprefix("gs://")


def _generation_eval_key(value: Any) -> str | None:
    for part in str(value).split(";"):
        part = part.strip()
        if "_generation_" in part:
            return part
    return None


def _eval_key_from_sample_path(path: str) -> str | None:
    for part in path.split("/"):
        match = EVAL_KEY_RE.fullmatch(part)
        if match:
            return match.group("eval_key")
    return None


def _first_response(record: dict[str, Any]) -> str:
    responses = record.get("resps")
    if isinstance(responses, list) and responses:
        first_group = responses[0]
        if isinstance(first_group, list) and first_group:
            return str(first_group[0])
        return str(first_group)
    filtered = record.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0])
    return str(filtered or "")


def filtered_exact_match(record: dict[str, Any]) -> float:
    """Return lm-eval exact match after the DCLM-safe left-strip filter."""
    prediction = _first_response(record).lstrip().translate(PUNCTUATION_TABLE)
    reference = str(record.get("target", "")).translate(PUNCTUATION_TABLE)
    return float(prediction == reference)


def score_sample_records(records: list[dict[str, Any]]) -> float:
    if not records:
        return math.nan
    return float(sum(filtered_exact_match(record) for record in records) / len(records))


def _sample_glob(sample_prefix: str, task: BigBenchGenerationTask) -> str:
    prefix = _gcs_path(sample_prefix.rstrip("/"))
    return f"{prefix}/*/{task.task_dir}/**/samples_{task.sample_stem}_*.jsonl"


def _select_sample_path(paths: list[str]) -> str:
    """Return the deterministic sample file for an eval key.

    Retries can leave multiple sample JSONLs under the same eval directory. The
    filename ends in the harness write timestamp, so sorted order picks the
    latest sample file while remaining stable and auditable.
    """
    return sorted(paths)[-1]


def build_sample_index(sample_prefix: str, tasks: tuple[BigBenchGenerationTask, ...]) -> dict[tuple[str, str], list[str]]:
    """Map ``(task_alias, generation_eval_key)`` to matching sample files."""
    fs = fsspec.filesystem("gcs")
    index: dict[tuple[str, str], list[str]] = {}
    for task in tasks:
        paths = sorted(fs.glob(_sample_glob(sample_prefix, task)))
        for path in paths:
            eval_key = _eval_key_from_sample_path(path)
            if eval_key is None:
                continue
            index.setdefault((task.alias, eval_key), []).append(path)
    return index


def _read_jsonl(fs: Any, path: str) -> list[dict[str, Any]]:
    with fs.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _score_sample_path(path: str) -> tuple[str, float, int]:
    fs = fsspec.filesystem("gcs")
    records = _read_jsonl(fs, path)
    return path, score_sample_records(records), len(records)


def score_sample_paths(paths: list[str], max_workers: int) -> dict[str, tuple[float, int]]:
    """Score sample files in parallel and return ``path -> (score, sample_count)``."""
    if not paths:
        return {}
    scores: dict[str, tuple[float, int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_score_sample_path, path) for path in paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="reading sample files"):
            path, score, sample_count = future.result()
            scores[path] = (score, sample_count)
    return scores


def _dclm_centered_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if column.startswith("lm_eval/dclm_core/")
        and column.endswith("/centered_accuracy")
        and column != "lm_eval/dclm_core/centered_accuracy_macro"
    )


def recompute_dclm_macro(frame: pd.DataFrame) -> None:
    """Update DCLM macro/task-count columns in-place from per-task centered columns."""
    centered_columns = _dclm_centered_columns(frame)
    numeric = frame[centered_columns].apply(pd.to_numeric, errors="coerce")
    task_counts = numeric.notna().sum(axis=1)
    frame["lm_eval/dclm_core/task_count"] = task_counts.astype(float)
    frame["lm_eval/dclm_core/missing_task_count"] = (DCLM_TOTAL_TASKS - task_counts).astype(float)
    frame["lm_eval/dclm_core/centered_accuracy_macro"] = numeric.mean(axis=1)


def apply_rescores(
    frame: pd.DataFrame,
    sample_index: dict[tuple[str, str], list[str]],
    tasks: tuple[BigBenchGenerationTask, ...],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return corrected results and an audit table."""
    corrected = frame.copy()
    audit_rows: list[dict[str, Any]] = []
    pending_updates: list[tuple[int, BigBenchGenerationTask, str, int]] = []

    for row_index, row in tqdm(list(corrected.iterrows()), desc="indexing DCLM BigBench samples"):
        eval_key = _generation_eval_key(row.get("eval_key"))
        if eval_key is None:
            continue
        for task in tasks:
            paths = sample_index.get((task.alias, eval_key), [])
            old_score = row.get(task.metric_column)
            audit_row = {
                "row_index": row_index,
                "run_name": row.get("run_name"),
                "eval_key": eval_key,
                "task_alias": task.alias,
                "metric_column": task.metric_column,
                "old_score": old_score,
                "sample_file_count": len(paths),
                "sample_path": paths[0] if len(paths) == 1 else "",
                "status": "pending",
            }
            if not paths:
                audit_row["status"] = "missing_samples"
                audit_rows.append(audit_row)
                continue

            path = _select_sample_path(paths)
            audit_row["sample_path"] = path
            audit_row["status"] = "pending_rescore_latest_duplicate" if len(paths) > 1 else "pending_rescore"
            audit_rows.append(audit_row)
            pending_updates.append((row_index, task, path, len(audit_rows) - 1))

    score_cache = score_sample_paths(sorted({path for _, _, path, _ in pending_updates}), max_workers)
    for row_index, task, path, audit_index in pending_updates:
        new_score, sample_count = score_cache[path]
        old_score = corrected.at[row_index, task.metric_column] if task.metric_column in corrected.columns else math.nan
        corrected.at[row_index, task.metric_column] = new_score
        corrected.at[row_index, task.raw_score_column] = new_score
        corrected.at[row_index, task.centered_column] = new_score
        audit_rows[audit_index]["new_score"] = new_score
        audit_rows[audit_index]["sample_count"] = sample_count
        audit_rows[audit_index]["delta"] = new_score - float(old_score) if pd.notna(old_score) else math.nan
        audit_rows[audit_index]["status"] = (
            "rescored_latest_duplicate" if audit_rows[audit_index]["sample_file_count"] > 1 else "rescored"
        )

    recompute_dclm_macro(corrected)
    return corrected, pd.DataFrame.from_records(audit_rows)


def write_summary(corrected: pd.DataFrame, audit: pd.DataFrame, output_dir: Path, input_csv: Path) -> None:
    summary: dict[str, Any] = {
        "input_csv": str(input_csv),
        "row_count": int(len(corrected)),
        "rescorable_tasks": [task.alias for task in RESCORABLE_TASKS],
        "rerun_required_tasks": list(RERUN_REQUIRED_TASKS),
        "audit_status_counts": audit["status"].value_counts(dropna=False).to_dict() if not audit.empty else {},
        "task_summaries": {},
    }
    for task in RESCORABLE_TASKS:
        task_audit = audit[audit["task_alias"] == task.alias]
        values = pd.to_numeric(corrected.get(task.raw_score_column), errors="coerce")
        summary["task_summaries"][task.alias] = {
            "rescored_rows": int((task_audit["status"] == "rescored").sum()) if not task_audit.empty else 0,
            "missing_sample_rows": int((task_audit["status"] == "missing_samples").sum()) if not task_audit.empty else 0,
            "mean_raw_score": float(values.mean()) if values.notna().any() else math.nan,
            "min_raw_score": float(values.min()) if values.notna().any() else math.nan,
            "max_raw_score": float(values.max()) if values.notna().any() else math.nan,
            "std_raw_score": float(values.std()) if values.notna().sum() > 1 else math.nan,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "README.md").write_text(
        "# DCLM BigBench generation post-hoc rescore\n\n"
        "This directory contains a corrected 300M DCLM Core matrix for BigBench "
        "generation tasks whose old exact-match scores were zero because model "
        "responses had leading whitespace. The correction applies the intended "
        "`remove_whitespace` left-strip filter and punctuation-insensitive exact "
        "match to logged samples, then recomputes DCLM centered accuracy columns.\n\n"
        "`bb_repeat_copy_logic_10shot` is not repaired here because the logged "
        "samples were generated with `max_gen_toks=32`; that task needs a rerun "
        "with the fixed 128-token generation config.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Historical DCLM matrix to rescore. Required to avoid accidentally using stale local data.",
    )
    parser.add_argument("--sample-prefix", default=DEFAULT_SAMPLE_PREFIX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the intermediate BigBench-only rescore artifact.",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    sample_index = build_sample_index(args.sample_prefix, RESCORABLE_TASKS)
    corrected, audit = apply_rescores(frame, sample_index, RESCORABLE_TASKS, max_workers=args.max_workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = args.output_dir / "300m_dclm_core_eval_results_bigbench_rescored.csv"
    audit_path = args.output_dir / "bigbench_generation_rescore_audit.csv"
    corrected.to_csv(corrected_path, index=False)
    audit.to_csv(audit_path, index=False)
    write_summary(corrected, audit, args.output_dir, args.input_csv)
    print(f"wrote {corrected_path}")
    print(f"wrote {audit_path}")
    print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
