# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a small stratified Common Crawl DOCX extraction and render a review report."""

import argparse
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from functools import partial
from statistics import median

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.download.common_crawl.plan import (
    DISCOVERY_SCHEMA,
    CommonCrawlDiscoverySummary,
    CommonCrawlIndexKind,
    CommonCrawlSource,
    common_crawl_plan_step,
    discover_index_partition,
    selected_common_crawl_record,
)
from marin.datakit.download.common_crawl.warc import common_crawl_index_partitions
from marin.datakit.download.common_crawl_docx import (
    COMMON_CRAWL_DOCX_SCHEMA,
    CommonCrawlDocxConfig,
    CommonCrawlDocxStageResult,
    DoclingDocxExtractor,
    DocxRecordSelector,
    DocxSelectionReason,
    LinguaLanguageDetector,
    evenly_spaced_sample,
    extract_common_crawl_docx,
    fetch_common_crawl_docx,
    identify_common_crawl_docx_language,
)
from marin.datakit.normalize import DedupMode, NormalizedData, normalize_step
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import prefix_join, url_to_fs
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

DEFAULT_INDEX_PARTITIONS = 6
DEFAULT_CANDIDATES_PER_REASON_PER_PARTITION = 10
DEFAULT_EXAMPLES_PER_REASON = 3


class CommonCrawlDocxSampleReport(BaseModel):
    """Paths and headline counts emitted by the sample report step."""

    markdown_path: str
    examples_path: str
    candidates: int
    extracted: int
    normalized: int


def sample_partition_records(
    index_partition: str,
    *,
    source: CommonCrawlSource,
    batch_rows: int,
    candidates_per_reason: int,
) -> Iterator[dict[str, object]]:
    """Take the first deterministic candidates from each primary selection stratum."""
    selected: Counter[DocxSelectionReason] = Counter()
    for candidate in discover_index_partition(
        index_partition,
        source=source,
        selector=DocxRecordSelector(),
        batch_rows=batch_rows,
    ):
        reason = DocxSelectionReason(str(candidate.selection.metadata["selection_reason"]))
        if selected[reason] >= candidates_per_reason:
            continue
        selected[reason] += 1
        counters.pipeline.update_counter("common_crawl/selected_records", 1)
        yield selected_common_crawl_record(candidate)
        if all(selected[reason] >= candidates_per_reason for reason in DocxSelectionReason):
            return


def discover_sample_candidates(
    output_path: str,
    config: CommonCrawlDocxConfig,
    *,
    index_partitions: int,
    candidates_per_reason: int,
) -> CommonCrawlDiscoverySummary:
    """Scan an evenly spaced partition slice and materialize a bounded candidate manifest."""
    if len(config.sources) != 1:
        raise ValueError("The bounded sample pipeline requires exactly one Common Crawl source")
    source = config.sources[0]
    all_partitions = common_crawl_index_partitions(
        source.paths_manifest_url,
        crawl_id=source.crawl_id,
        subset=source.subset,
    )
    selected_partitions = evenly_spaced_sample(all_partitions, index_partitions)
    pipeline = (
        Dataset.from_list(list(selected_partitions))
        .flat_map(
            partial(
                sample_partition_records,
                source=source,
                batch_rows=config.index_batch_rows,
                candidates_per_reason=candidates_per_reason,
            )
        )
        .write_parquet(
            prefix_join(output_path, "records/part-{shard:05d}-of-{total:05d}.parquet"),
            schema=DISCOVERY_SCHEMA,
            skip_existing=True,
        )
    )
    outcome = ZephyrContext(
        name=f"common-crawl-docx-sample-discovery-{source.crawl_id.lower()}",
        resources=ResourceConfig(cpu=1, ram="8g"),
        max_workers=min(config.max_workers, len(selected_partitions)),
    ).execute(pipeline)
    return CommonCrawlDiscoverySummary(
        manifest_path=prefix_join(output_path, "records"),
        num_sources=1,
        num_records=int(outcome.counters.get("common_crawl/selected_records", 0)),
    )


def _parquet_rows(path: str) -> list[dict[str, object]]:
    fs, resolved = url_to_fs(prefix_join(path, "*.parquet"))
    protocol = fsspec.core.split_protocol(path)[0]
    rows: list[dict[str, object]] = []
    for matched_path in sorted(fs.glob(resolved)):
        full_path = f"{protocol}://{matched_path}" if protocol else matched_path
        with fsspec.open(full_path, "rb") as stream:
            rows.extend(pq.read_table(stream).to_pylist())
    return rows


def _percentage(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100 * numerator / denominator:.1f}%"


def _excerpt(text: object, maximum_chars: int = 600) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= maximum_chars else f"{compact[:maximum_chars].rstrip()}…"


def _candidate_selection_reason(row: Mapping[str, object]) -> str:
    metadata = json.loads(str(row["selection_metadata"]))
    return str(metadata["selection_reason"])


def sample_report_markdown(
    *,
    source: CommonCrawlSource,
    candidate_rows: list[dict[str, object]],
    extracted_rows: list[dict[str, object]],
    normalized_rows: list[dict[str, object]],
    stage_counters: Mapping[str, int | float],
    examples_per_reason: int,
) -> tuple[str, list[dict[str, object]]]:
    candidate_counts = Counter(_candidate_selection_reason(row) for row in candidate_rows)
    extracted_counts = Counter(str(row["selection_reason"]) for row in extracted_rows)
    examples_by_reason: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sorted(extracted_rows, key=lambda item: (str(item["selection_reason"]), str(item["url"]))):
        reason = str(row["selection_reason"])
        if len(examples_by_reason[reason]) < examples_per_reason:
            examples_by_reason[reason].append(
                {
                    "selection_reason": reason,
                    "url": row["url"],
                    "language": row["language"],
                    "word_count": row["word_count"],
                    "table_count": row["table_count"],
                    "excerpt": _excerpt(row["text"]),
                }
            )
    examples = [example for reason in DocxSelectionReason for example in examples_by_reason[reason.value]]
    word_counts = [int(row["word_count"]) for row in extracted_rows]
    table_documents = sum(int(row["table_count"]) > 0 for row in extracted_rows)
    language_counts = Counter(str(row["language"]) for row in extracted_rows)
    routine_counters = {
        "fetched",
        "fetched_payload_bytes",
        "valid_files",
        "text_bytes",
        "words",
        "tables",
        "images",
        "documents_with_tables",
    }
    review_counters: dict[str, int | float] = {}
    for key, value in sorted(stage_counters.items()):
        if "common_crawl_docx/" not in key or not value:
            continue
        counter_name = key.rsplit("common_crawl_docx/", maxsplit=1)[1]
        if counter_name not in routine_counters:
            review_counters[key] = value

    lines = [
        f"# Common Crawl DOCX sample: {source.crawl_id}",
        "",
        "Deterministic sample from evenly spaced URL Index partitions. Each selected partition contributes "
        "up to the configured cap for each candidate's primary selection reason.",
        "",
        "## Extraction funnel",
        "",
        "| Stage | Documents | Yield from candidates |",
        "| --- | ---: | ---: |",
        f"| Candidates | {len(candidate_rows):,} | 100.0% |",
        f"| Extracted DOCX | {len(extracted_rows):,} | {_percentage(len(extracted_rows), len(candidate_rows))} |",
        f"| Normalized unique text | {len(normalized_rows):,} | "
        f"{_percentage(len(normalized_rows), len(candidate_rows))} |",
        "",
        "## Yield by primary selection reason",
        "",
        "| Reason | Candidates | Extracted | Yield |",
        "| --- | ---: | ---: | ---: |",
    ]
    for reason in DocxSelectionReason:
        candidates = candidate_counts[reason.value]
        extracted = extracted_counts[reason.value]
        lines.append(f"| `{reason.value}` | {candidates:,} | {extracted:,} | {_percentage(extracted, candidates)} |")

    lines.extend(
        [
            "",
            "## Extracted-document characteristics",
            "",
            f"- Median words: {median(word_counts):,.0f}" if word_counts else "- Median words: n/a",
            f"- Documents containing tables: {table_documents:,} ({_percentage(table_documents, len(extracted_rows))})",
            "- Languages: "
            + (", ".join(f"{language}={count}" for language, count in language_counts.most_common(10)) or "none"),
            "",
            "## Stage counters requiring review",
            "",
        ]
    )
    if review_counters:
        lines.extend(f"- `{name}`: {value:,}" for name, value in review_counters.items())
    else:
        lines.append("- None")

    lines.extend(["", "## Manual-review examples", ""])
    for example in examples:
        lines.extend(
            [
                f"### {example['selection_reason']}: {example['url']}",
                "",
                f"Language `{example['language']}`; {example['word_count']} words; {example['table_count']} tables.",
                "",
                str(example["excerpt"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n", examples


def write_sample_report(
    output_path: str,
    *,
    source: CommonCrawlSource,
    discovery_path: str,
    language_path: str,
    stage_counter_paths: tuple[tuple[str, str], ...],
    normalized_path: str,
    examples_per_reason: int,
) -> CommonCrawlDocxSampleReport:
    """Render the sample funnel and bounded review examples as Markdown and JSONL."""
    discovery = read_artifact(discovery_path, CommonCrawlDiscoverySummary)
    candidate_rows = _parquet_rows(discovery.manifest_path)
    extracted_rows = _parquet_rows(prefix_join(language_path, "data"))
    normalized = read_artifact(normalized_path, NormalizedData)
    normalized_rows = _parquet_rows(normalized.main_output_dir)
    stage_counters: dict[str, int | float] = {}
    for stage_name, stage_path in stage_counter_paths:
        result = read_artifact(stage_path, CommonCrawlDocxStageResult)
        stage_counters.update({f"{stage_name}/{key}": value for key, value in result.counters.items()})
    markdown, examples = sample_report_markdown(
        source=source,
        candidate_rows=candidate_rows,
        extracted_rows=extracted_rows,
        normalized_rows=normalized_rows,
        stage_counters=stage_counters,
        examples_per_reason=examples_per_reason,
    )
    markdown_path = prefix_join(output_path, "report.md")
    examples_path = prefix_join(output_path, "examples.jsonl")
    with fsspec.open(markdown_path, "wt") as stream:
        stream.write(markdown)
    with fsspec.open(examples_path, "wt") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    return CommonCrawlDocxSampleReport(
        markdown_path=markdown_path,
        examples_path=examples_path,
        candidates=len(candidate_rows),
        extracted=len(extracted_rows),
        normalized=len(normalized_rows),
    )


def common_crawl_docx_sample_steps(
    config: CommonCrawlDocxConfig,
    *,
    index_partitions: int = DEFAULT_INDEX_PARTITIONS,
    candidates_per_reason: int = DEFAULT_CANDIDATES_PER_REASON_PER_PARTITION,
    examples_per_reason: int = DEFAULT_EXAMPLES_PER_REASON,
) -> tuple[StepSpec, StepSpec, StepSpec, StepSpec, StepSpec, StepSpec, StepSpec]:
    """Build the sample discovery, fetch, extraction, LID, normalization, and report DAG."""
    if candidates_per_reason <= 0 or examples_per_reason <= 0:
        raise ValueError("candidate and example limits must be positive")
    if len(config.sources) != 1:
        raise ValueError("The bounded sample pipeline requires exactly one Common Crawl source")
    source = config.sources[0]
    slug = config.name.lower()
    discovery = StepSpec(
        name=f"samples/common-crawl-docx/{slug}/candidates",
        fn=remote(
            partial(
                discover_sample_candidates,
                config=config,
                index_partitions=index_partitions,
                candidates_per_reason=candidates_per_reason,
            ),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        hash_attrs={
            "crawl_id": source.crawl_id,
            "index_kind": source.index_kind,
            "paths_manifest_url": source.paths_manifest_url,
            "base_url": source.base_url,
            "subset": source.subset,
            "index_partitions": index_partitions,
            "candidates_per_reason": candidates_per_reason,
            "schema_version": 2,
        },
    )
    plan = common_crawl_plan_step(
        name=f"samples/common-crawl-docx/{slug}/plan",
        discovery=discovery,
        options=config.plan,
    )
    extractor = DoclingDocxExtractor()
    detector = LinguaLanguageDetector()
    fetch = StepSpec(
        name=f"samples/common-crawl-docx/{slug}/fetched",
        fn=remote(
            partial(fetch_common_crawl_docx, plan_output_path=plan.output_path, config=config),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[plan],
        hash_attrs={
            "maximum_warc_record_bytes": config.maximum_warc_record_bytes,
            "maximum_payload_bytes": config.maximum_payload_bytes,
            "schema_version": 1,
        },
    )
    extraction = StepSpec(
        name=f"samples/common-crawl-docx/{slug}/extracted",
        fn=remote(
            partial(
                extract_common_crawl_docx,
                fetched_input_path=fetch.output_path,
                config=config,
                extractor=extractor,
            ),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[fetch],
        hash_attrs={
            "maximum_zip_entries": config.maximum_zip_entries,
            "maximum_uncompressed_bytes": config.maximum_uncompressed_bytes,
            "extractor": extractor.version,
            "schema_version": 5,
        },
    )
    language = StepSpec(
        name=f"samples/common-crawl-docx/{slug}/language",
        fn=remote(
            partial(
                identify_common_crawl_docx_language,
                extracted_input_path=extraction.output_path,
                config=config,
                detector=detector,
            ),
            resources=ResourceConfig(cpu=1, ram="4g"),
            pip_dependency_groups=["datakit"],
        ),
        deps=[extraction],
        hash_attrs={
            "language_detector": detector.version,
            "chunk_chars": config.language_chunk_chars,
            "sample_chunks": config.language_sample_chunks,
            "minimum_alpha_bytes": config.language_minimum_alpha_bytes,
            "minimum_alpha_ratio": config.language_minimum_alpha_ratio,
            "maximum_table_alpha_bytes": config.language_maximum_table_alpha_bytes,
            "distribution_top_k": config.language_distribution_top_k,
            "minimum_score": config.language_minimum_score,
            "schema_version": 3,
        },
    )
    normalized = normalize_step(
        name=f"samples/common-crawl-docx/{slug}/normalized",
        download=language,
        relative_input_path="data",
        file_extensions=(".parquet",),
        id_field="source_id",
        dedup_mode=DedupMode.EXACT,
        output_schema=COMMON_CRAWL_DOCX_SCHEMA,
    )
    report = StepSpec(
        name=f"samples/common-crawl-docx/{slug}/report",
        fn=partial(
            write_sample_report,
            source=source,
            discovery_path=discovery.output_path,
            language_path=language.output_path,
            stage_counter_paths=(
                ("fetch", fetch.output_path),
                ("extraction", extraction.output_path),
                ("language", language.output_path),
            ),
            normalized_path=normalized.output_path,
            examples_per_reason=examples_per_reason,
        ),
        deps=[discovery, fetch, extraction, language, normalized],
        hash_attrs={"examples_per_reason": examples_per_reason, "report_version": 2},
    )
    return discovery, plan, fetch, extraction, language, normalized, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--paths-manifest-url", required=True)
    parser.add_argument("--index-partitions", type=int, default=DEFAULT_INDEX_PARTITIONS)
    parser.add_argument(
        "--candidates-per-reason-per-partition",
        type=int,
        default=DEFAULT_CANDIDATES_PER_REASON_PER_PARTITION,
    )
    parser.add_argument("--examples-per-reason", type=int, default=DEFAULT_EXAMPLES_PER_REASON)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    config = CommonCrawlDocxConfig(
        name=args.crawl_id,
        sources=(
            CommonCrawlSource(
                crawl_id=args.crawl_id,
                index_kind=CommonCrawlIndexKind.MAIN,
                paths_manifest_url=args.paths_manifest_url,
            ),
        ),
        max_workers=args.max_workers,
    )
    steps = common_crawl_docx_sample_steps(
        config,
        index_partitions=args.index_partitions,
        candidates_per_reason=args.candidates_per_reason_per_partition,
        examples_per_reason=args.examples_per_reason,
    )
    StepRunner().run([steps[-1]], dry_run=args.dry_run, max_concurrent=1)
    if not args.dry_run:
        report = read_artifact(steps[-1].output_path, CommonCrawlDocxSampleReport)
        print(f"Markdown report: {report.markdown_path}")
        print(f"Review examples: {report.examples_path}")


if __name__ == "__main__":
    main()
