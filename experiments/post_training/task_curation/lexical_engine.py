# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the Datakit decontamination engine to the taskcompendium lexical stage.

The engine owns one frozen eval corpus: its prebuilt bloom and hash index, the
corpus-common global drop set, and the document-frequency thresholds that shape
each source's local drop set. ``mark_population`` builds that local drop set,
marks the population, and attributes every match through the corpus hash index
into eval IDs and families, following the reference decontamination report.
"""

import logging
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.decon import (
    FEATURE_FILTER_VERSION,
    NGramConfig,
    build_all_source_drop_sets,
    decon_to_parquet,
)
from marin.datakit.normalize import NormalizedData
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.lexical import TASK_FLAGS_SCHEMA, MarkedPopulation, TextPopulation
from zephyr.context import ZephyrContext

from experiments.datakit.decontam.config import (
    GLOBAL_DF_COMMON_MIN_ABS,
    GLOBAL_DF_COMMON_MIN_SOURCES,
    GLOBAL_DF_SAMPLE_DOCS,
    SOURCE_DF_COMMON_FRAC,
    SOURCE_DF_COMMON_MIN_ABS,
    SOURCE_DF_SAMPLE_DOCS,
)
from experiments.datakit.decontam.prepare_eval_corpus import AA_EVALS
from experiments.datakit.decontam.viewer.export_run import eval_id_to_family

logger = logging.getLogger(__name__)

NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=2)
DEFAULT_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="4g")
AA_SUBDIR_TO_NAME = {config.subdir: config.name for config in AA_EVALS}
INDEX_BATCH_ROWS = 131_072


def eval_family(eval_id: str) -> str:
    """Map one eval record ID to its benchmark family, matching the reference report."""
    for subdir, name in AA_SUBDIR_TO_NAME.items():
        if eval_id == subdir or eval_id.startswith(f"{subdir}-"):
            return name
    return eval_id_to_family(eval_id)


def _hash_to_evals(index_path: str, needed_hashes: set[int]) -> dict[int, set[str]]:
    """Read the eval hash index once, keeping only the marks' matched hashes."""
    mapping: dict[int, set[str]] = {}
    with StoragePath(index_path).open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=INDEX_BATCH_ROWS, columns=["hash", "eval_id"]):
            for feature_hash, eval_id in zip(
                batch.column("hash").to_pylist(), batch.column("eval_id").to_pylist(), strict=True
            ):
                if feature_hash in needed_hashes:
                    mapping.setdefault(feature_hash, set()).add(str(eval_id))
    return mapping


class DatakitDecontamination:
    """The Datakit bloom marker for one frozen eval corpus, as a lexical stage engine."""

    def __init__(
        self,
        *,
        bloom_dir: str,
        global_drop_set_dir: str,
        eval_corpus_version: str,
        frozen_source_drop_sets: Mapping[str, str] | None = None,
        worker_resources: ResourceConfig | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.bloom_dir = bloom_dir
        self.global_drop_set_dir = global_drop_set_dir
        self.eval_corpus_version = eval_corpus_version
        self.frozen_source_drop_sets = dict(frozen_source_drop_sets or {})
        self.worker_resources = worker_resources or DEFAULT_WORKER_RESOURCES
        self.max_workers = max_workers
        self._index_path = str(StoragePath(prefix_join(bloom_dir, "_bloom/eval_hash_index.parquet")))

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "eval_corpus_version": self.eval_corpus_version,
            "eval_bloom_uri": self.bloom_dir,
            "eval_hash_index_uri": self._index_path,
            "feature_filter_version": FEATURE_FILTER_VERSION,
            "ngram_length": NGRAM.ngram_length,
            "overlap_threshold": NGRAM.overlap_threshold,
            "min_matched_features": NGRAM.min_matched_features,
        }

    def _source_drop_set(self, population: TextPopulation, output_path: str, context: ZephyrContext) -> str:
        frozen = self.frozen_source_drop_sets.get(population.source)
        if frozen is not None:
            logger.info("Adopting the frozen source drop set for %s at %s", population.source, frozen)
            return frozen
        drops = build_all_source_drop_sets(
            sources=[(population.source, population.path)],
            prebuilt_bloom_dir=self.bloom_dir,
            output_path=output_path,
            ngram=NGRAM,
            sample_docs=SOURCE_DF_SAMPLE_DOCS,
            common_frac=SOURCE_DF_COMMON_FRAC,
            common_min_abs=SOURCE_DF_COMMON_MIN_ABS,
            global_sample_docs=GLOBAL_DF_SAMPLE_DOCS,
            global_common_min_abs=GLOBAL_DF_COMMON_MIN_ABS,
            global_common_min_sources=GLOBAL_DF_COMMON_MIN_SOURCES,
            worker_resources=self.worker_resources,
            max_workers=self.max_workers,
            zephyr_context=context,
        )
        return str(StoragePath(prefix_join(str(drops.output_dir), population.source)))

    def mark_population(
        self, population: TextPopulation, output_path: str, *, context: ZephyrContext
    ) -> MarkedPopulation:
        source_key = canonical_sha256({"source": population.source})[:20]
        local_dir = self._source_drop_set(population, prefix_join(output_path, f"df/{source_key}"), context)
        marks = decon_to_parquet(
            normalized_data=NormalizedData(main_output_dir=population.path, dup_output_dir="", counters={}),
            prebuilt_bloom_dir=self.bloom_dir,
            output_path=prefix_join(output_path, f"marks/{source_key}"),
            ngram=NGRAM,
            drop_set_dirs=[local_dir, self.global_drop_set_dir],
            flagged_sample_size=0,
            worker_resources=self.worker_resources,
            max_workers=self.max_workers,
            zephyr_context=context,
        )
        marked_dir = str(StoragePath(prefix_join(str(marks.main_output_dir), "*.parquet")))
        flags_dir = prefix_join(output_path, f"flags/{source_key}")
        count = self._attribute(marked_dir, flags_dir, str(marks.eval_hash_index_path))
        logger.info("%s: attributed %d marked tasks", population.source, count)
        return MarkedPopulation(
            flags_dir=flags_dir,
            attributes={"drop_set_uris": [local_dir, self.global_drop_set_dir]},
        )

    def _attribute(self, marked_glob: str, flags_dir: str, index_path: str) -> int:
        """Attribute every mark's matched hashes through the eval index and write typed flags."""
        needed_hashes: set[int] = set()
        marks = sorted(str(path) for path in StoragePath(marked_glob).glob())
        StoragePath(flags_dir).mkdirs()
        for path in marks:
            with StoragePath(path).open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(columns=["matched_hashes"]):
                    for hashes in batch.column(0).to_pylist():
                        needed_hashes.update(hashes)
        hash_evals = _hash_to_evals(index_path, needed_hashes) if needed_hashes else {}
        eval_groups: dict[frozenset[str], frozenset[str]] = {}
        hash_groups: dict[int, frozenset[str]] = {}
        for feature_hash, eval_ids in hash_evals.items():
            group = frozenset(eval_ids)
            hash_groups[feature_hash] = eval_groups.setdefault(group, group)
        StoragePath(flags_dir).mkdirs()
        rows = 0
        for path in marks:
            target = str(StoragePath(prefix_join(flags_dir, StoragePath(path).name)))
            with StoragePath(path).open("rb") as stream:
                writer = None
                try:
                    for batch in pq.ParquetFile(stream).iter_batches():
                        mark_columns = (
                            "id",
                            "max_overlap",
                            "contaminated",
                            "matched_hashes",
                        )
                        columns = {name: batch.column(name).to_pylist() for name in mark_columns}
                        eval_ids = [
                            sorted(set().union(*{hash_groups[h] for h in hashes if h in hash_groups}))
                            for hashes in columns["matched_hashes"]
                        ]
                        families = [sorted({eval_family(eval_id) for eval_id in ids}) for ids in eval_ids]
                        table = pa.Table.from_pylist(
                            [
                                {
                                    "id": task_id,
                                    "max_overlap": overlap,
                                    "contaminated": flag,
                                    "matched_hashes": hashes,
                                    "matched_eval_ids": ids,
                                    "matched_eval_families": names,
                                }
                                for task_id, overlap, flag, hashes, ids, names in zip(
                                    columns["id"],
                                    columns["max_overlap"],
                                    columns["contaminated"],
                                    columns["matched_hashes"],
                                    eval_ids,
                                    families,
                                    strict=True,
                                )
                            ],
                            schema=TASK_FLAGS_SCHEMA,
                        )
                        if writer is None:
                            writer = pq.ParquetWriter(target, TASK_FLAGS_SCHEMA, compression="zstd")
                        writer.write_table(table)
                        rows += batch.num_rows
                finally:
                    if writer is not None:
                        writer.close()
        return rows
