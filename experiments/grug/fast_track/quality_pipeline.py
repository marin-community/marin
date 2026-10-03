# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train an embedding head and materialize its selection from a frozen quality bundle."""

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import groupby
from tempfile import TemporaryFile
from typing import BinaryIO, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.text.datasets import LmDataConfig
from levanter.store.cache import SerialCacheWriter, TreeCache
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import register_fingerprint
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.fast_track.contracts import FrozenBaselineComponent, FrozenBaselineManifest, ResolvedTrainingBudget
from experiments.grug.fast_track.launch import FlatCacheTrainingSource
from experiments.grug.fast_track.quality import (
    LabelledEmbedding,
    PoolDocument,
    PoolRequirements,
    audit_pool,
    fit_ridge_head,
    select_top_tokens,
    selection_token_overlap,
)

PARQUET_BATCH_ROWS = 1024
CACHE_BATCH_ROWS = 128
HASH_BLOCK_BYTES = 8 * 1024 * 1024
TOKEN_CACHE_NAME = "tokens"


class PinnedFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str
    sha256: str = Field(pattern="^[0-9a-f]{64}$")


register_fingerprint(PinnedFile, lambda value: value.model_dump(mode="json"))


class QualityBundle(BaseModel):
    """A pinned join of labels, embeddings, and token references."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    tokenizer: str
    embedding_revision: str = Field(min_length=1)
    embedding_scale: float = Field(gt=0)
    label_revision: str = Field(min_length=1)
    incumbent_revision: str = Field(min_length=1)
    baseline_recipe: str = Field(min_length=1)
    sampling_method: Literal["production-weighted-hash"]
    pool_seed: int
    split_seed: int
    labels: tuple[PinnedFile, ...] = Field(min_length=1)
    pool: tuple[PinnedFile, ...] = Field(min_length=1)
    quality_bin_edges: tuple[float, ...]
    requirements: PoolRequirements


class SelectionMethod(StrEnum):
    RIDGE = "ridge"
    INCUMBENT = "incumbent"
    RANDOM = "random"


class QualityData(Artifact):
    tokenizer: str
    cache_dir: str
    requested_tokens: int
    report_path: str


@dataclass(frozen=True)
class QualitySpec:
    bundle: PinnedFile
    selection_method: SelectionMethod
    fraction: float
    regularization: float
    tie_seed: int


@dataclass(frozen=True)
class QualityConfig:
    spec: QualitySpec
    output_path: str


@contextmanager
def verified_file(file: PinnedFile) -> Iterator[BinaryIO]:
    """Verify one file and parse the same bytes without a second remote read."""
    with TemporaryFile() as local:
        digest = hashlib.sha256()
        with StoragePath(file.path).open("rb") as stream:
            while block := stream.read(HASH_BLOCK_BYTES):
                digest.update(block)
                local.write(block)
        if digest.hexdigest() != file.sha256:
            raise ValueError(f"quality input checksum differs: {file.path}")
        local.seek(0)
        yield local


def parquet_batches(files: Sequence[PinnedFile]) -> Iterator[pa.RecordBatch]:
    for file in files:
        with verified_file(file) as stream:
            yield from pq.ParquetFile(stream).iter_batches(batch_size=PARQUET_BATCH_ROWS)


def embedding_matrix(batch: pa.RecordBatch, scale: float) -> np.ndarray:
    embeddings = batch.column("embedding")
    if embeddings.null_count or embeddings.flatten().null_count:
        raise ValueError("embeddings must not contain null values")
    lengths = np.asarray(embeddings.value_lengths()) if not pa.types.is_fixed_size_list(embeddings.type) else None
    if lengths is not None and (lengths.size == 0 or np.any(lengths != lengths[0])):
        raise ValueError("embedding dimensions must be constant")
    return embeddings.flatten().to_numpy().reshape(len(batch), -1).astype(np.float64) * scale


def prepare_quality_data(config: QualityConfig) -> QualityData:
    """Validate frozen inputs, score the pool, and write the selected token cache."""
    spec = config.spec
    with verified_file(spec.bundle) as stream:
        bundle = QualityBundle.model_validate_json(stream.read())
    edges = np.asarray(bundle.quality_bin_edges, dtype=np.float64)
    bin_names = bundle.requirements.quality_bins
    if len(edges) != len(bin_names) + 1 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError("quality-bin edges must increase and match the declared bin names")
    labels: list[LabelledEmbedding] = []
    for batch in parquet_batches(bundle.labels):
        embeddings = embedding_matrix(batch, bundle.embedding_scale)
        for row, embedding in zip(batch.drop_columns(["embedding"]).to_pylist(), embeddings, strict=True):
            labels.append(
                LabelledEmbedding(
                    row["source"], row["id"], row["duplicate_group"], tuple(embedding.tolist()), float(row["label"])
                )
            )
    head, development = fit_ridge_head(labels, regularization=spec.regularization, split_seed=bundle.split_seed)
    documents: list[PoolDocument] = []
    candidate_scores: list[float] = []
    incumbent_scores: list[float] = []
    references: list[tuple[str, int, str]] = []
    for batch in parquet_batches(bundle.pool):
        rows = batch.drop_columns(["embedding"]).to_pylist()
        embeddings = embedding_matrix(batch, bundle.embedding_scale)
        candidate_scores.extend(head.scores(embeddings).tolist())
        for row in rows:
            incumbent_score = float(row["incumbent_score"])
            if not np.isfinite(incumbent_score) or not edges[0] <= incumbent_score <= edges[-1]:
                raise ValueError("incumbent score is outside the declared quality-bin range")
            bin_index = min(int(np.searchsorted(edges, incumbent_score, side="right")) - 1, len(bin_names) - 1)
            documents.append(
                PoolDocument(
                    row["source"],
                    row["id"],
                    row["duplicate_group"],
                    row["token_count"],
                    bin_names[bin_index],
                    row["content_type"],
                    row["language"],
                )
            )
            references.append((row["cache_path"], row["cache_row"], row["token_sha256"]))
            incumbent_scores.append(incumbent_score)
    audit = audit_pool(
        documents, requirements=bundle.requirements, labelled_groups={row.duplicate_group for row in labels}
    )
    incumbent = select_top_tokens(
        documents,
        incumbent_scores,
        fraction=spec.fraction,
        tie_seed=spec.tie_seed,
    )
    if spec.selection_method is SelectionMethod.INCUMBENT:
        selection = incumbent
    else:
        scores = [0.0] * len(documents) if spec.selection_method is SelectionMethod.RANDOM else candidate_scores
        selection = select_top_tokens(documents, scores, fraction=spec.fraction, tie_seed=spec.tie_seed)
    overlap = selection_token_overlap(documents, selection, incumbent)
    cache_dir = prefix_join(config.output_path, TOKEN_CACHE_NAME)
    exemplar = {"input_ids": np.zeros(0, dtype=np.int32)}
    # The loader shuffles blocks, so the cache must already interleave documents.
    selected_indices = np.random.default_rng(spec.tie_seed).permutation(sorted(selection.indices)).tolist()
    caches: dict[str, TreeCache] = {}
    with SerialCacheWriter(cache_dir, exemplar) as writer:
        for start in range(0, len(selected_indices), CACHE_BATCH_ROWS):
            batch_indices = selected_indices[start : start + CACHE_BATCH_ROWS]
            output: dict[int, dict[str, np.ndarray]] = {}
            read_indices = sorted(batch_indices, key=lambda i: references[i][:2])
            for cache_path, group in groupby(read_indices, key=lambda i: references[i][0]):
                if cache_path not in caches:
                    caches[cache_path] = TreeCache.load(cache_path, exemplar)
                indices = list(group)
                records = caches[cache_path].get_batch_sync([references[i][1] for i in indices])
                for index, record in zip(indices, records, strict=True):
                    tokens = np.asarray(record["input_ids"], dtype="<i4")
                    expected_hash = references[index][2]
                    if (
                        len(tokens) != documents[index].token_count
                        or hashlib.sha256(tokens.tobytes()).hexdigest() != expected_hash
                    ):
                        raise ValueError(
                            f"token reference differs for {documents[index].source}/{documents[index].document_id}"
                        )
                    output[index] = {"input_ids": tokens}
            writer.write_batch([output[index] for index in batch_indices])
    report_path = prefix_join(config.output_path, "selection.json")
    ids_path = prefix_join(config.output_path, "selected_ids.jsonl")
    report = {
        "bundle_sha256": spec.bundle.sha256,
        "baseline_recipe": bundle.baseline_recipe,
        "embedding_revision": bundle.embedding_revision,
        "label_revision": bundle.label_revision,
        "incumbent_revision": bundle.incumbent_revision,
        "pool_seed": bundle.pool_seed,
        "split_seed": bundle.split_seed,
        "head": asdict(head),
        "development": asdict(development),
        "pool": asdict(audit),
        "selection": {key: value for key, value in asdict(selection).items() if key != "indices"},
        "selection_method": spec.selection_method.value,
        "tie_seed": spec.tie_seed,
        "selected_ids_path": ids_path,
        "selected_documents": len(selected_indices),
        "token_overlap": overlap,
        "identical_incumbent_selection": set(selection.indices) == set(incumbent.indices),
    }
    with StoragePath(report_path).open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with StoragePath(ids_path).open("w") as stream:
        for index in selected_indices:
            row = documents[index]
            stream.write(json.dumps({"source": row.source, "id": row.document_id}) + "\n")
    return QualityData(
        tokenizer=bundle.tokenizer,
        cache_dir=cache_dir,
        requested_tokens=selection.requested_tokens,
        report_path=report_path,
    )


@dataclass(frozen=True)
class QualityTrainingSource:
    selection: ArtifactStep[QualityData]

    def dependencies(self) -> tuple[ArtifactStep, ...]:
        return (self.selection,)

    def data_config(
        self,
        *,
        ctx: StepContext,
        validation: Sequence[ArtifactStep[TokenizedCache]],
        tokenizer: str,
        budget: ResolvedTrainingBudget,
    ) -> LmDataConfig:
        if ctx.is_fingerprint:
            cache_dir = prefix_join(ctx.artifact_path(self.selection), TOKEN_CACHE_NAME)
        else:
            data = ctx.resolved(self.selection)
            if data.tokenizer != tokenizer:
                raise ValueError("quality selection tokenizer differs from the training tokenizer")
            # Do not count whole-document cutoff overshoot as pool capacity.
            if data.requested_tokens < budget.token_count:
                raise ValueError("quality selection is too small for this rung without repetition")
            cache_dir = data.cache_dir
        return FlatCacheTrainingSource(
            FrozenBaselineManifest(tokenizer, (FrozenBaselineComponent("quality", cache_dir, 1.0),))
        ).data_config(ctx=ctx, validation=validation, tokenizer=tokenizer, budget=budget)


def build_quality_data(spec: QualitySpec, *, version: str | None = None) -> ArtifactStep[QualityData]:
    identity = {
        "bundle": spec.bundle.model_dump(),
        "method": spec.selection_method.value,
        "fraction": spec.fraction,
        "regularization": spec.regularization,
        "tie_seed": spec.tie_seed,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    name = f"fast-track/quality/{digest}"
    version = resolve_version(name, version)
    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=QualityData,
        run=prepare_quality_data,
        build_config=lambda ctx: QualityConfig(spec, ctx.output_path),
    )
