# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Balanced functional-DNA documents for general language-model training.

The source combines three genomes-v5 interval datasets with two Zoonomia
projection datasets. Each interval is rendered with DNA and natural-language
region tags so a general language model can condition on the modality and
functional class. Each component receives the same cap on rendered UTF-8 text
bytes before the combined corpus is normalized and exact-deduplicated.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from fray.types import ResourceConfig
from rigging.filesystem import StoragePath
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_file

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

SOURCE_NAME = "dna/functional-regions"
SOURCE_ID_FIELD = "source_id"
# The smallest component has about 3.89 GB of sequence text. Leave enough
# headroom for every selected shard to satisfy its share of the budget.
TARGET_TEXT_BYTES_PER_DATASET = 3_750_000_000


@dataclass(frozen=True)
class DnaDatasetSpec:
    """One upstream component of the balanced DNA source."""

    name: str
    hf_dataset_id: str
    revision: str
    text_field: str
    region_type: str
    id_fields: tuple[str, ...]
    shard_globs: tuple[str, ...]
    num_download_shards: int


@dataclass(frozen=True)
class DnaShardTask:
    """One downloaded shard and its contribution to the byte budget."""

    input_path: str
    dataset: DnaDatasetSpec
    max_text_bytes: int


DNA_DATASETS = (
    DnaDatasetSpec(
        name="genomes-v5-cds",
        hf_dataset_id="marin-dna/genomes-v5-genome_set-animals-intervals-v5_255_128",
        revision="ffe3e78c99868077c65ad6568e1445d80e480794",
        text_field="seq",
        region_type="coding sequence",
        id_fields=("id",),
        shard_globs=("data/train/shard_000[0-4].jsonl.zst",),
        num_download_shards=5,
    ),
    DnaDatasetSpec(
        name="genomes-v5-promoter",
        hf_dataset_id="marin-dna/genomes-v5-genome_set-animals-intervals-v1_255_128",
        revision="d93209847b02a0c9be5c03591a0a5e56ee09c35d",
        text_field="seq",
        region_type="promoter",
        id_fields=("id",),
        shard_globs=(
            "data/train/shard_000[0-9].jsonl.zst",
            "data/train/shard_001[0-4].jsonl.zst",
        ),
        num_download_shards=15,
    ),
    DnaDatasetSpec(
        name="genomes-v5-downstream",
        hf_dataset_id="marin-dna/genomes-v5-genome_set-animals-intervals-v15_255_128",
        revision="b009afaab756937d75b8da3b1271ad8f0cec0b4d",
        text_field="seq",
        region_type="downstream",
        id_fields=("id",),
        shard_globs=(
            "data/train/shard_00[0-3][0-9].jsonl.zst",
            "data/train/shard_004[0-7].jsonl.zst",
        ),
        num_download_shards=48,
    ),
    DnaDatasetSpec(
        name="zoonomia-ccre-non-promoter",
        hf_dataset_id="marin-dna/zoonomia-v1-v3_ccre_non_promoter",
        revision="862485aa18eed53a53e693ba4c2eb45e0afc5087",
        text_field="sequence",
        region_type="candidate cis regulatory element outside promoters",
        id_fields=("query_name", "species", "t_chrom", "t_start", "t_end", "t_strand", "augmentation"),
        shard_globs=(
            "data/train/shard_000[0-9].jsonl.zst",
            "data/train/shard_0010.jsonl.zst",
        ),
        num_download_shards=11,
    ),
    DnaDatasetSpec(
        name="zoonomia-ncrna-exon",
        hf_dataset_id="marin-dna/zoonomia-v1-v3_ncrna_exon",
        revision="3e48d9ae7c604b99ccfc8bd07e391b960c1ea21a",
        text_field="sequence",
        region_type="noncoding RNA exon",
        id_fields=("query_name", "species", "t_chrom", "t_start", "t_end", "t_strand", "augmentation"),
        shard_globs=("data/train/shard_*.jsonl.zst",),
        num_download_shards=64,
    ),
)


def dna_document_prefix(region_type: str) -> str:
    """Return the conditioning prefix shared by training and DNA evaluations."""
    return f"[DNA]\n[Region: {region_type}]\n"


def dna_document_text(sequence: str, region_type: str) -> str:
    """Render one tagged DNA interval for regular language-model training."""
    return f"{dna_document_prefix(region_type)}{sequence}"


def dna_documents(task: DnaShardTask) -> Iterator[dict]:
    """Yield whole DNA records until this shard's text-byte budget is full."""
    selected_bytes = 0
    selected_rows = 0
    for row in load_file(task.input_path):
        sequence = str(row[task.dataset.text_field])
        text = dna_document_text(sequence, task.dataset.region_type)
        text_bytes = len(text.encode("utf-8"))
        if selected_bytes + text_bytes > task.max_text_bytes:
            break

        source_key = ":".join(str(row[field]) for field in task.dataset.id_fields)
        yield {
            "text": text,
            SOURCE_ID_FIELD: f"{task.dataset.hf_dataset_id}:{source_key}",
            "source": task.dataset.hf_dataset_id,
            "region_type": task.dataset.region_type,
        }
        selected_bytes += text_bytes
        selected_rows += 1

    counters.pipeline.update_counter(f"dna/{task.dataset.name}/selected_text_bytes", selected_bytes)
    counters.pipeline.update_counter(f"dna/{task.dataset.name}/selected_rows", selected_rows)


def balanced_shard_tasks(
    source_files: dict[DnaDatasetSpec, list[str]],
    target_text_bytes_per_dataset: int,
) -> list[DnaShardTask]:
    """Divide each dataset's byte budget evenly across its downloaded shards."""
    tasks: list[DnaShardTask] = []
    for dataset, files in source_files.items():
        if not files:
            raise ValueError(f"No downloaded shards found for {dataset.hf_dataset_id}")

        shard_budget, remainder = divmod(target_text_bytes_per_dataset, len(files))
        for index, path in enumerate(sorted(files)):
            tasks.append(
                DnaShardTask(
                    input_path=path,
                    dataset=dataset,
                    max_text_bytes=shard_budget + (index < remainder),
                )
            )
    return tasks


def write_balanced_dna(
    *,
    source_files: dict[DnaDatasetSpec, list[str]],
    output_path: str,
    target_text_bytes_per_dataset: int,
) -> None:
    """Write a single Parquet corpus with equal text-byte budgets per dataset."""
    tasks = balanced_shard_tasks(source_files, target_text_bytes_per_dataset)
    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(dna_documents)
        .write_parquet(
            str(StoragePath(output_path) / "data-{shard:05d}-of-{total:05d}.parquet"),
            skip_existing=True,
        )
    )
    ctx = ZephyrContext(name="balance-dna", resources=ResourceConfig(cpu=1, ram="4g"))
    ctx.execute(pipeline)


def stage_balanced_dna(
    *,
    download_paths: dict[DnaDatasetSpec, str],
    output_path: str,
    target_text_bytes_per_dataset: int,
) -> None:
    """Discover downloaded shards and materialize the balanced DNA corpus."""
    source_files: dict[DnaDatasetSpec, list[str]] = {}
    for dataset, download_path in download_paths.items():
        shard_glob = StoragePath(download_path) / "**" / "*.jsonl.zst"
        files = sorted(str(path) for path in shard_glob.glob())
        if len(files) != dataset.num_download_shards:
            raise ValueError(
                f"Expected {dataset.num_download_shards} shards for {dataset.hf_dataset_id}, found {len(files)}"
            )
        source_files[dataset] = files

    write_balanced_dna(
        source_files=source_files,
        output_path=output_path,
        target_text_bytes_per_dataset=target_text_bytes_per_dataset,
    )


def dna_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the download, byte-balance, and normalization chain."""
    downloads = tuple(
        download_hf_step(
            f"raw/{SOURCE_NAME}/{dataset.name}",
            hf_dataset_id=dataset.hf_dataset_id,
            revision=dataset.revision,
            hf_urls_glob=list(dataset.shard_globs),
        )
        for dataset in DNA_DATASETS
    )
    download_paths = {dataset: download.output_path for dataset, download in zip(DNA_DATASETS, downloads, strict=True)}
    processed = StepSpec(
        name=f"processed/{SOURCE_NAME}",
        deps=list(downloads),
        fn=lambda output_path: stage_balanced_dna(
            download_paths=download_paths,
            output_path=output_path,
            target_text_bytes_per_dataset=TARGET_TEXT_BYTES_PER_DATASET,
        ),
        hash_attrs={
            "version": "2026.07.31",
            "target_text_bytes_per_dataset": TARGET_TEXT_BYTES_PER_DATASET,
        },
    )
    normalized = normalize_step(
        name=f"normalized/{SOURCE_NAME}",
        download=processed,
        id_field=SOURCE_ID_FIELD,
        file_extensions=(".parquet",),
    )
    return (*downloads, processed, normalized)
