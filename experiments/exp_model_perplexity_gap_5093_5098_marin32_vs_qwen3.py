# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the first capped #5005 eval wave for issues #5093, #5095-#5098.

This experiment compares Marin 32B against Qwen3 32B on newly wired long-tail
PPL slices without expanding default validation sets:

- #5093 diagnostic logs: https://github.com/marin-community/marin/issues/5093
- #5095 diff/patch text: https://github.com/marin-community/marin/issues/5095
- #5096 paired paraphrase/translation robustness: https://github.com/marin-community/marin/issues/5096
- #5097 ASR/OCR noisy text: https://github.com/marin-community/marin/issues/5097
- #5098 GH Archive structured events: https://github.com/marin-community/marin/issues/5098

Run with ``--run_only <bundle-name> --max_concurrent 1`` to submit one bundle
at a time from the same script.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import posixpath
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import fsspec
from datasets import load_dataset
from fray.v2 import ResourceConfig

from experiments.evals.asr_ocr_noisy_ppl import (
    NoisyAsrOcrRawConfig,
    materialize_noisy_asr_ocr_raw,
    noisy_asr_ocr_raw_validation_sets,
)
from experiments.evals.gh_archive_structured_output import (
    GH_ARCHIVE_EVAL_END_DATE,
    GH_ARCHIVE_EVAL_END_HOUR,
    GH_ARCHIVE_EVAL_START_DATE,
    GH_ARCHIVE_EVAL_START_HOUR,
    GH_ARCHIVE_STRUCTURED_OUTPUT_SLICES,
    gh_archive_structured_output_raw_validation_sets,
)
from experiments.evals.paired_robustness_ppl import (
    PAIRED_ROBUSTNESS_SLICES,
    paired_robustness_raw_steps,
    paired_robustness_raw_validation_sets,
)
from experiments.exp5095_diff_patch_ppl import (
    DIFF_PATCH_SLICES,
    SWE_BENCH_PROVENANCE_FIELDS,
    DiffPatchMetric,
    DiffPatchSlice,
    build_commitpack_commit_message_plus_diff_eval_text,
    build_diff_patch_eval_text,
    build_swe_bench_issue_to_patch_eval_text,
)
from marin.datakit.download.gh_archive import make_gh_archive_step
from marin.evaluation.perplexity_gap import GapFinderModelConfig, RawTextEvaluationDataset, default_model_perplexity_gap
from marin.evaluation.perplexity_gap import raw_text_dataset
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
CPU_MATERIALIZE_RESOURCES = ResourceConfig.with_cpu(cpu=4, ram="16g", disk="30g", regions=["us-central1"])
MAX_DOCS_PER_DATASET = 128
MAX_DOC_BYTES = 32_768
MAX_EVAL_LENGTH = 4096
PER_DEVICE_BATCH_SIZE = 1

MARIN_32B_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-32b-base",
    checkpoint_is_hf=True,
    tokenizer="marin-community/marin-32b-base",
)

QWEN3_32B_MODEL = GapFinderModelConfig(
    checkpoint_path="Qwen/Qwen3-32B",
    checkpoint_is_hf=True,
    tokenizer="Qwen/Qwen3-32B",
)


@dataclass(frozen=True)
class UrlTextSample:
    name: str
    url: str
    output_relative_path: str
    compression: str | None
    max_rows: int
    max_bytes: int


@dataclass(frozen=True)
class DiagnosticLogUrlSampleConfig:
    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    samples: tuple[UrlTextSample, ...] = (
        UrlTextSample(
            name="ghalogs_runs",
            url="https://zenodo.org/api/records/14796970/files/runs.json.gz/content",
            output_relative_path="ghalogs/runs.jsonl.gz",
            compression="gzip",
            max_rows=64,
            max_bytes=8_000_000,
        ),
        UrlTextSample(
            name="loghub_apache_2k",
            url="https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log",
            output_relative_path="loghub/apache.jsonl.gz",
            compression=None,
            max_rows=512,
            max_bytes=2_000_000,
        ),
        UrlTextSample(
            name="loghub_linux_2k",
            url="https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log",
            output_relative_path="loghub/linux.jsonl.gz",
            compression=None,
            max_rows=512,
            max_bytes=2_000_000,
        ),
    )


@dataclass(frozen=True)
class DiffPatchRawConfig:
    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    swe_bench_max_rows: int = MAX_DOCS_PER_DATASET
    commitpack_max_rows: int = MAX_DOCS_PER_DATASET


def materialize_diagnostic_log_url_samples(config: DiagnosticLogUrlSampleConfig) -> None:
    for sample in config.samples:
        output_file = posixpath.join(config.output_path, sample.output_relative_path)
        _write_text_rows_from_url(
            url=sample.url,
            output_file=output_file,
            compression=sample.compression,
            max_rows=sample.max_rows,
            max_bytes=sample.max_bytes,
        )


def materialize_diff_patch_raw(config: DiffPatchRawConfig) -> None:
    buffers = _empty_diff_patch_buffers()

    swe_bench_rows = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", streaming=True)
    for row_index, row in enumerate(swe_bench_rows):
        if row_index >= config.swe_bench_max_rows:
            break
        _append_metric_texts(
            buffers,
            "swe_bench/issue_to_patch",
            build_swe_bench_issue_to_patch_eval_text(row),
        )
        _append_metric_texts(
            buffers,
            "swe_bench/raw_git_diff",
            build_diff_patch_eval_text(
                row,
                patch_field="patch",
                context_fields=(),
                masked_fields=SWE_BENCH_PROVENANCE_FIELDS,
            ),
        )

    commitpack_rows = load_dataset("bigcode/commitpackft", "diff", split="train", streaming=True)
    for row_index, row in enumerate(commitpack_rows):
        if row_index >= config.commitpack_max_rows:
            break
        normalized_row = _commitpack_diff_row(row)
        _append_metric_texts(
            buffers,
            "commitpack/commit_message_plus_diff",
            build_commitpack_commit_message_plus_diff_eval_text(normalized_row),
        )

    for path_stem, metrics in buffers.items():
        for metric, records in metrics.items():
            if not records:
                continue
            output_file = posixpath.join(config.output_path, f"{path_stem}_{metric.value}.jsonl.gz")
            _write_jsonl_records(output_file, records)


def diagnostic_log_sample_validation_sets(
    raw_step: ExecutorStep,
) -> dict[str, RawTextEvaluationDataset]:
    return {
        "diagnostic_logs/ghalogs/runs": raw_text_dataset(
            raw_step.cd("ghalogs/runs.jsonl.gz"),
            tags=("diagnostic_logs", "issue:5093", "source:ghalogs"),
        ),
        "diagnostic_logs/loghub/apache": raw_text_dataset(
            raw_step.cd("loghub/apache.jsonl.gz"),
            tags=("diagnostic_logs", "issue:5093", "source:loghub", "subset:apache"),
        ),
        "diagnostic_logs/loghub/linux": raw_text_dataset(
            raw_step.cd("loghub/linux.jsonl.gz"),
            tags=("diagnostic_logs", "issue:5093", "source:loghub", "subset:linux"),
        ),
    }


def diff_patch_sample_validation_sets(raw_step: ExecutorStep) -> dict[str, RawTextEvaluationDataset]:
    datasets: dict[str, RawTextEvaluationDataset] = {}
    for slice_spec in DIFF_PATCH_SLICES:
        for metric in slice_spec.metrics:
            datasets[slice_spec.dataset_key(metric)] = raw_text_dataset(
                raw_step.cd(_diff_patch_relative_path(slice_spec, metric)),
                tags=(*slice_spec.tags, f"metric:{metric.value}"),
            )
    return {posixpath.join("diff_patch", name): dataset for name, dataset in datasets.items()}


def _write_text_rows_from_url(
    *,
    url: str,
    output_file: str,
    compression: str | None,
    max_rows: int,
    max_bytes: int,
) -> None:
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}.")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}.")

    rows_written = 0
    bytes_written = 0
    with urllib.request.urlopen(url, timeout=120) as response:
        raw_stream = gzip.GzipFile(fileobj=response) if compression == "gzip" else response
        with io.TextIOWrapper(raw_stream, encoding="utf-8", errors="replace") as reader:
            with _open_jsonl_gzip_writer(output_file) as writer:
                for raw_line in reader:
                    text = raw_line.strip()
                    if not text:
                        continue
                    payload = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    payload_bytes = len(payload.encode("utf-8"))
                    if bytes_written + payload_bytes > max_bytes:
                        break
                    writer.write(payload)
                    rows_written += 1
                    bytes_written += payload_bytes
                    if rows_written >= max_rows:
                        break

    if rows_written == 0:
        raise ValueError(f"No rows were materialized from {url}.")


def _open_jsonl_gzip_writer(path: str):
    fs, fs_path = fsspec.core.url_to_fs(path)
    output_dir = os.path.dirname(fs_path)
    if output_dir:
        fs.makedirs(output_dir, exist_ok=True)
    return fsspec.open(path, mode="wt", compression="gzip")


def _write_jsonl_records(path: str, records: Sequence[Mapping[str, str]]) -> None:
    with _open_jsonl_gzip_writer(path) as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False))
            writer.write("\n")


def _empty_diff_patch_buffers() -> dict[str, dict[DiffPatchMetric, list[dict[str, str]]]]:
    buffers: dict[str, dict[DiffPatchMetric, list[dict[str, str]]]] = {}
    for slice_spec in DIFF_PATCH_SLICES:
        buffers[_diff_patch_path_stem(slice_spec)] = {metric: [] for metric in slice_spec.metrics}
    return buffers


def _append_metric_texts(
    buffers: dict[str, dict[DiffPatchMetric, list[dict[str, str]]]],
    path_stem: str,
    metric_texts: Mapping[DiffPatchMetric, str],
) -> None:
    for metric, text in metric_texts.items():
        if metric in buffers[path_stem]:
            buffers[path_stem][metric].append({"text": text})


def _diff_patch_path_stem(slice_spec: DiffPatchSlice) -> str:
    return slice_spec.relative_path.removesuffix(".jsonl.gz")


def _diff_patch_relative_path(slice_spec: DiffPatchSlice, metric: DiffPatchMetric) -> str:
    return f"{_diff_patch_path_stem(slice_spec)}_{metric.value}.jsonl.gz"


def _commitpack_diff_row(row: Mapping[str, object]) -> dict[str, object]:
    message = row.get("message") or row.get("subject")
    return {
        "diff": row.get("new_contents") or row.get("old_contents"),
        "commit_message": message,
        "commit_hash": row.get("commit"),
        "repo_name": row.get("repos"),
    }


def _report_step(
    *,
    bundle_name: str,
    datasets: dict[str, RawTextEvaluationDataset],
    issue_ids: Iterable[int],
) -> ExecutorStep:
    return default_model_perplexity_gap(
        name=f"issue5005-{bundle_name}-marin-32b-base-vs-qwen3-32b-doccap{MAX_DOCS_PER_DATASET}",
        model_a=MARIN_32B_MODEL,
        model_b=QWEN3_32B_MODEL,
        datasets=datasets,
        resource_config=RESOURCE_CONFIG,
        per_device_batch_size=PER_DEVICE_BATCH_SIZE,
        max_eval_length=MAX_EVAL_LENGTH,
        max_docs_per_dataset=MAX_DOCS_PER_DATASET,
        max_doc_bytes=MAX_DOC_BYTES,
        wandb_tags=[
            "eval=perplexity-gap",
            "epic=5005",
            f"bundle={bundle_name}",
            "model_a=marin-community/marin-32b-base",
            "model_b=Qwen/Qwen3-32B",
            "region=us-central1",
            f"max_docs_per_dataset={MAX_DOCS_PER_DATASET}",
            *(f"issue={issue_id}" for issue_id in issue_ids),
        ],
    )


DIAGNOSTIC_LOG_RAW = ExecutorStep(
    name="raw/evals/issue5005/diagnostic_logs_marin32_qwen3_sample",
    description="Materialize capped public diagnostic log samples for issue #5093.",
    fn=remote(
        materialize_diagnostic_log_url_samples,
        resources=CPU_MATERIALIZE_RESOURCES,
        pip_dependency_groups=["cpu"],
    ),
    config=DiagnosticLogUrlSampleConfig(),
)

DIFF_PATCH_RAW = ExecutorStep(
    name="raw/evals/issue5005/diff_patch_marin32_qwen3_sample",
    description="Materialize capped SWE-bench and CommitPack diff/patch eval samples for issue #5095.",
    fn=remote(materialize_diff_patch_raw, resources=CPU_MATERIALIZE_RESOURCES, pip_dependency_groups=["cpu"]),
    config=DiffPatchRawConfig(),
)

ASR_OCR_RAW = ExecutorStep(
    name="raw/evals/issue5005/asr_ocr_noisy_marin32_qwen3_sample",
    description="Materialize capped ASR/OCR noisy-clean eval samples for issue #5097.",
    fn=remote(materialize_noisy_asr_ocr_raw, resources=CPU_MATERIALIZE_RESOURCES, pip_dependency_groups=["cpu"]),
    config=NoisyAsrOcrRawConfig(max_rows_per_slice_override=MAX_DOCS_PER_DATASET),
)

PAIRED_ROBUSTNESS_CAPPED_SLICES = tuple(
    replace(slice_, max_pairs=min(slice_.max_pairs, MAX_DOCS_PER_DATASET)) for slice_ in PAIRED_ROBUSTNESS_SLICES
)
PAIRED_ROBUSTNESS_RAW_STEPS = paired_robustness_raw_steps(
    slices=PAIRED_ROBUSTNESS_CAPPED_SLICES,
    name_prefix="raw/evals/issue5005/paired_robustness_marin32_qwen3_sample",
    resources=CPU_MATERIALIZE_RESOURCES,
)

GH_ARCHIVE_RAW = make_gh_archive_step(
    name="raw/evals/issue5005/gh_archive_structured_output_marin32_qwen3_sample",
    start_date=GH_ARCHIVE_EVAL_START_DATE,
    end_date=GH_ARCHIVE_EVAL_END_DATE,
    start_hour=GH_ARCHIVE_EVAL_START_HOUR,
    end_hour=GH_ARCHIVE_EVAL_END_HOUR,
    event_types=tuple(slice_.event_type for slice_ in GH_ARCHIVE_STRUCTURED_OUTPUT_SLICES),
    max_events_per_event_type=MAX_DOCS_PER_DATASET,
)

DIAGNOSTIC_LOG_REPORT = _report_step(
    bundle_name="diagnostic-logs",
    datasets=diagnostic_log_sample_validation_sets(DIAGNOSTIC_LOG_RAW),
    issue_ids=(5093,),
)

DIFF_PATCH_REPORT = _report_step(
    bundle_name="diff-patch",
    datasets=diff_patch_sample_validation_sets(DIFF_PATCH_RAW),
    issue_ids=(5095,),
)

PAIRED_ROBUSTNESS_REPORT = _report_step(
    bundle_name="paired-robustness",
    datasets=paired_robustness_raw_validation_sets(
        slices=PAIRED_ROBUSTNESS_CAPPED_SLICES,
        raw_steps=PAIRED_ROBUSTNESS_RAW_STEPS,
    ),
    issue_ids=(5096,),
)

ASR_OCR_REPORT = _report_step(
    bundle_name="asr-ocr-noisy",
    datasets=noisy_asr_ocr_raw_validation_sets(noisy_asr_ocr_raw=ASR_OCR_RAW),
    issue_ids=(5097,),
)

GH_ARCHIVE_REPORT = _report_step(
    bundle_name="gh-archive-structured-output",
    datasets=gh_archive_structured_output_raw_validation_sets(gh_archive_raw=GH_ARCHIVE_RAW),
    issue_ids=(5098,),
)

ALL_STEPS = [
    DIAGNOSTIC_LOG_RAW,
    DIAGNOSTIC_LOG_REPORT,
    DIFF_PATCH_RAW,
    DIFF_PATCH_REPORT,
    *PAIRED_ROBUSTNESS_RAW_STEPS.values(),
    PAIRED_ROBUSTNESS_REPORT,
    ASR_OCR_RAW,
    ASR_OCR_REPORT,
    GH_ARCHIVE_RAW,
    GH_ARCHIVE_REPORT,
]


if __name__ == "__main__":
    executor_main(
        ALL_STEPS,
        description="Run capped issue #5005 long-tail perplexity-gap bundles for Marin 32B vs Qwen3 32B.",
    )
