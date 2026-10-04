# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze joined GLM labels and cached Harrier embeddings for head experiments."""

import hashlib
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.writers import ensure_parent_dir

from experiments.grug.fast_track.label_exclusion import LabelExclusion
from experiments.grug.fast_track.quality import (
    LabelledEmbedding,
    LabelMetrics,
    RidgeHead,
    RidgeHeadConfig,
    fit_quality_head,
)
from experiments.grug.fast_track.quality_features import HARRIER_FEATURE_IDENTITY, normalize_harrier_embeddings

GLM_LABELS_PATH = "s3://marin-us-east-02a/marin/user/rav/quality_v2/glm52_labels_88k.parquet"
GLM_HARRIER_JOIN_PATH = (
    "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_88k-x-harrier-oss-v1-0.6b-50m-text-v1"
)
GLM_ORACLE = "zai-org/GLM-5.2-FP8@ba978f7d347eaf65d22f1a86833408afdb953541"
LABEL_BATCH = "glm52_rubric_v2"
DUPLICATE_EMBEDDING_POLICY = "first_sorted_path_row"
LABEL_COLUMNS = ["source", "id", "quality", "score_normalized", "valid", "content_type", "label_batch"]
JOIN_COLUMNS = ["id", "text", "embedding", *(f"glm52_{name}" for name in LABEL_COLUMNS if name != "id")]


@dataclass(frozen=True)
class QualityLabelSpec:
    """Explicit source paths and model identities for the frozen label set."""

    labels_path: str
    joined_path: str
    oracle_identity: str
    feature_identity: dict


class FrozenQualityLabels(Artifact):
    """Compact labels, input hashes, and exclusions for all original label IDs."""

    table_path: str
    table_sha256: str
    manifest_path: str
    duplicate_report_path: str
    label_exclusion: LabelExclusion
    feature_identity: dict
    documents: int
    original_documents: int
    missing_by_source: dict[str, int]
    missing_by_content_type: dict[str, int]


class FittedRidgeQualityHead(Artifact):
    """A fitted embedding head with fixed labels, features, and development metrics."""

    head: RidgeHead
    label_exclusion: LabelExclusion
    feature_identity: dict
    split_seed: int
    training_documents: int
    development: LabelMetrics


@dataclass(frozen=True)
class _Label:
    source: str
    document_id: str
    score: float
    content_type: str


@dataclass(frozen=True)
class _JoinedLabels:
    records: dict[str, dict]
    keys: set[tuple[str, str]]
    duplicate_groups: set[str]
    duplicate_rows: list[dict]


def _label(row: dict, *, prefix: str = "") -> _Label:
    if row[prefix + "label_batch"] != LABEL_BATCH:
        raise ValueError(f"unexpected label batch for {row['id']}")
    quality = row[prefix + "quality"]
    score = row[prefix + "score_normalized"]
    if quality not in (1, 2, 3, 4, 5) or score != (quality - 1) / 4:
        raise ValueError(f"invalid GLM quality target for {row['id']}")
    # The oracle marks junk as invalid content. These are negative examples,
    # not failed label requests, and must remain in the training distribution.
    if row[prefix + "valid"] is False and quality != 1:
        raise ValueError(f"invalid-content labels must have quality 1: {row['id']}")
    return _Label(row[prefix + "source"], row["id"], score, row[prefix + "content_type"])


def _read_input(path: str, hashes: dict[str, str]) -> pq.ParquetFile:
    # Hash the same bytes that the Parquet reader consumes, not a separate read.
    contents = StoragePath(path).read_bytes()
    hashes[path] = hashlib.sha256(contents).hexdigest()
    return pq.ParquetFile(io.BytesIO(contents))


def _embedding_difference(retained: np.ndarray, duplicate: np.ndarray) -> tuple[int, float, float, bool]:
    delta = retained.astype(np.int16) - duplicate.astype(np.int16)
    max_abs_delta = int(np.abs(delta).max())
    l2_delta = float(np.linalg.norm(delta.astype(np.float64)))
    retained_float = retained.astype(np.float64)
    duplicate_float = duplicate.astype(np.float64)
    retained_norm = float(np.linalg.norm(retained_float))
    duplicate_norm = float(np.linalg.norm(duplicate_float))
    identical = not np.any(delta)
    if identical:
        cosine = 1.0
    elif retained_norm == 0 or duplicate_norm == 0:
        cosine = 0.0
    else:
        cosine = float(np.clip(np.dot(retained_float, duplicate_float) / (retained_norm * duplicate_norm), -1.0, 1.0))
    return max_abs_delta, l2_delta, cosine, identical


def _read_original_labels(path: str, hashes: dict[str, str]) -> tuple[dict[tuple[str, str], _Label], dict[str, float]]:
    originals: dict[tuple[str, str], _Label] = {}
    global_scores: dict[str, float] = {}
    for batch in _read_input(path, hashes).iter_batches(batch_size=256, columns=LABEL_COLUMNS):
        for row in batch.to_pylist():
            label = _label(row)
            key = label.source, label.document_id
            if key in originals and originals[key] != label:
                raise ValueError(f"conflicting original labels for {key}")
            if label.document_id in global_scores and global_scores[label.document_id] != label.score:
                raise ValueError(f"conflicting quality targets across sources for {label.document_id}")
            originals[key] = label
            global_scores[label.document_id] = label.score
    if not originals:
        raise ValueError("the original label table is empty")
    return originals, global_scores


def _read_joined_labels(
    joined_path: str, originals: dict[tuple[str, str], _Label], hashes: dict[str, str]
) -> _JoinedLabels:
    paths = sorted(
        str(directory / filename)
        for directory, _, filenames in StoragePath(prefix_join(joined_path, "outputs")).walk()
        for filename in filenames
        if filename.endswith(".parquet") and not filename.startswith((".", "_"))
    )
    if not paths:
        raise ValueError(f"no joined label shards at {joined_path}/outputs")

    records: dict[str, dict] = {}
    joined_keys: set[tuple[str, str]] = set()
    duplicate_groups: set[str] = set()
    duplicate_rows = []
    for path in paths:
        joined_row = 0
        for batch in _read_input(path, hashes).iter_batches(batch_size=256, columns=JOIN_COLUMNS):
            for row in batch.to_pylist():
                label = _label(row, prefix="glm52_")
                key = label.source, label.document_id
                if key not in originals or originals[key] != label:
                    raise ValueError(f"joined label differs from its original label: {key}")
                text = row["text"]
                if not isinstance(text, str) or generate_id(text) != label.document_id:
                    raise ValueError(f"joined full text does not match its normalized content ID: {key}")
                embedding = np.asarray(row["embedding"])
                if embedding.shape != (1024,) or embedding.dtype.kind not in "iu":
                    raise ValueError(f"joined Harrier embedding must contain 1024 signed int8 values: {key}")
                if np.any(embedding < -128) or np.any(embedding > 127):
                    raise ValueError(f"joined Harrier embedding is outside the int8 range: {key}")
                embedding = embedding.astype(np.int8)
                group = hashlib.sha256(text.encode("utf-8")).hexdigest()
                duplicate_groups.add(group)
                joined_keys.add(key)
                if label.document_id not in records:
                    records[label.document_id] = {
                        "source": label.source,
                        "id": label.document_id,
                        "duplicate_group": group,
                        "embedding": embedding,
                        "label": label.score,
                        "retained_path": path,
                        "retained_row": joined_row,
                        "text": text,
                    }
                else:
                    retained = records[label.document_id]
                    if retained["text"] != text:
                        raise ValueError(f"conflicting joined text for {label.document_id}")
                    if retained["label"] != label.score:
                        raise ValueError(f"conflicting joined quality targets for {label.document_id}")
                    max_abs_delta, l2_delta, cosine, identical = _embedding_difference(retained["embedding"], embedding)
                    duplicate_rows.append(
                        {
                            "id": label.document_id,
                            "retained_source": retained["source"],
                            "retained_path": retained["retained_path"],
                            "retained_row": retained["retained_row"],
                            "duplicate_source": label.source,
                            "duplicate_path": path,
                            "duplicate_row": joined_row,
                            "embedding_identical": identical,
                            "embedding_max_abs_delta_int8": max_abs_delta,
                            "embedding_l2_delta_int8": l2_delta,
                            "embedding_cosine_similarity_int8": cosine,
                        }
                    )
                joined_row += 1
    if not records:
        raise ValueError("the joined artifact has no quality labels")
    return _JoinedLabels(records, joined_keys, duplicate_groups, duplicate_rows)


def _write_frozen_quality_labels(
    spec: QualityLabelSpec,
    output_path: str,
    hashes: dict[str, str],
    originals: dict[tuple[str, str], _Label],
    global_scores: dict[str, float],
    joined: _JoinedLabels,
) -> FrozenQualityLabels:
    missing = [label for key, label in originals.items() if key not in joined.keys]
    missing_by_source = dict(Counter(label.source for label in missing))
    missing_by_type = dict(Counter(label.content_type for label in missing))
    schema = pa.schema(
        [
            ("source", pa.string()),
            ("id", pa.string()),
            ("duplicate_group", pa.string()),
            ("embedding", pa.list_(pa.int8(), 1024)),
            ("label", pa.float64()),
        ]
    )
    table_rows = [
        {name: joined.records[key][name] for name in ("source", "id", "duplicate_group", "embedding", "label")}
        for key in sorted(joined.records)
    ]
    table = pa.Table.from_pylist(table_rows, schema=schema)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    table_bytes = buffer.getvalue()
    table_hash = hashlib.sha256(table_bytes).hexdigest()
    duplicate_schema = pa.schema(
        [
            ("id", pa.string()),
            ("retained_source", pa.string()),
            ("retained_path", pa.string()),
            ("retained_row", pa.int64()),
            ("duplicate_source", pa.string()),
            ("duplicate_path", pa.string()),
            ("duplicate_row", pa.int64()),
            ("embedding_identical", pa.bool_()),
            ("embedding_max_abs_delta_int8", pa.int64()),
            ("embedding_l2_delta_int8", pa.float64()),
            ("embedding_cosine_similarity_int8", pa.float64()),
        ]
    )
    duplicate_table = pa.Table.from_pylist(joined.duplicate_rows, schema=duplicate_schema)
    duplicate_buffer = io.BytesIO()
    pq.write_table(duplicate_table, duplicate_buffer, compression="zstd")
    duplicate_report_path = prefix_join(output_path, "duplicate_embeddings.parquet")
    duplicate_report_bytes = duplicate_buffer.getvalue()
    duplicate_report_hash = hashlib.sha256(duplicate_report_bytes).hexdigest()
    duplicate_summary = {
        "policy": DUPLICATE_EMBEDDING_POLICY,
        "occurrences": len(joined.duplicate_rows),
        "affected_documents": len({row["id"] for row in joined.duplicate_rows}),
        "identical_occurrences": sum(row["embedding_identical"] for row in joined.duplicate_rows),
        "differing_occurrences": sum(not row["embedding_identical"] for row in joined.duplicate_rows),
        "max_abs_delta_int8": max((row["embedding_max_abs_delta_int8"] for row in joined.duplicate_rows), default=0),
        "max_l2_delta_int8": max((row["embedding_l2_delta_int8"] for row in joined.duplicate_rows), default=0.0),
        "minimum_cosine_similarity_int8": min(
            (row["embedding_cosine_similarity_int8"] for row in joined.duplicate_rows), default=None
        ),
        "report_path": duplicate_report_path,
    }
    revision = hashlib.sha256(
        canonical_json(
            {
                "spec": asdict(spec),
                "inputs": hashes,
                "table": table_hash,
                "duplicate_report": duplicate_report_hash,
                "duplicate_embedding_policy": DUPLICATE_EMBEDDING_POLICY,
            }
        ).encode()
    ).hexdigest()
    exclusion = LabelExclusion(
        label_revision=f"sha256:{revision}",
        duplicate_groups=frozenset(joined.duplicate_groups),
        normalized_document_ids=frozenset(global_scores),
    )
    table_path = StoragePath(prefix_join(output_path, "labels.parquet"))
    ensure_parent_dir(str(table_path))
    table_path.write_bytes(table_bytes)
    StoragePath(duplicate_report_path).write_bytes(duplicate_report_bytes)
    manifest_path = prefix_join(output_path, "labels.json")
    StoragePath(manifest_path).write_text(
        json.dumps(
            {
                "spec": asdict(spec),
                "input_sha256": hashes,
                "table_sha256": table_hash,
                "duplicate_report_sha256": duplicate_report_hash,
                "label_revision": exclusion.label_revision,
                "original_source_id_pairs": len(originals),
                "original_unique_ids": len(global_scores),
                "joined_source_id_pairs": len(joined.keys),
                "labelled_unique_documents": len(joined.records),
                "missing_by_source": missing_by_source,
                "missing_by_content_type": missing_by_type,
                "duplicate_embeddings": duplicate_summary,
            },
            sort_keys=True,
            indent=2,
        )
    )
    StoragePath(prefix_join(output_path, "label_exclusion.json")).write_text(exclusion.model_dump_json())
    return FrozenQualityLabels(
        table_path=str(table_path),
        table_sha256=table_hash,
        manifest_path=manifest_path,
        duplicate_report_path=duplicate_report_path,
        label_exclusion=exclusion,
        feature_identity=spec.feature_identity,
        documents=len(joined.records),
        original_documents=len(global_scores),
        missing_by_source=missing_by_source,
        missing_by_content_type=missing_by_type,
    )


def freeze_quality_labels(spec: QualityLabelSpec, *, output_path: str) -> FrozenQualityLabels:
    """Validate original and joined labels, then seal the content-hashed label artifact."""
    if spec.feature_identity != HARRIER_FEATURE_IDENTITY:
        raise ValueError("joined quality labels require the pinned Harrier feature identity")
    hashes: dict[str, str] = {}
    originals, global_scores = _read_original_labels(spec.labels_path, hashes)
    joined = _read_joined_labels(spec.joined_path, originals, hashes)
    return _write_frozen_quality_labels(spec, output_path, hashes, originals, global_scores, joined)


def read_labelled_embeddings(labels: FrozenQualityLabels) -> list[LabelledEmbedding]:
    """Read verified frozen labels with the same normalization as pool features."""
    contents = StoragePath(labels.table_path).read_bytes()
    if hashlib.sha256(contents).hexdigest() != labels.table_sha256:
        raise ValueError("frozen quality labels changed after preparation")
    result = []
    for batch in pq.ParquetFile(io.BytesIO(contents)).iter_batches(batch_size=256):
        rows = batch.to_pylist()
        embeddings = normalize_harrier_embeddings(np.asarray([row["embedding"] for row in rows], dtype=np.int8))
        result.extend(
            LabelledEmbedding(row["source"], row["id"], row["duplicate_group"], embedding, row["label"])
            for row, embedding in zip(rows, embeddings, strict=True)
        )
    return result


def build_quality_labels(spec: QualityLabelSpec, *, version: str | None = None) -> ArtifactStep[FrozenQualityLabels]:
    """Bind label validation and freezing as one reusable artifact dependency."""
    digest = hashlib.sha256(canonical_json(asdict(spec)).encode()).hexdigest()[:20]
    name = f"fast-track/quality-labels/{digest}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> dict:
        return {"spec": spec, "output_path": ctx.output_path}

    def run(config: dict) -> FrozenQualityLabels:
        return freeze_quality_labels(config["spec"], output_path=config["output_path"])

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=FrozenQualityLabels,
        build_config=build_config,
        run=run,
    )


@dataclass(frozen=True)
class _RidgeFitConfig:
    labels_path: str
    regularization: float
    split_seed: int


def _fit_ridge(config: _RidgeFitConfig) -> FittedRidgeQualityHead:
    labels = read_artifact(config.labels_path, FrozenQualityLabels)
    if labels.feature_identity != HARRIER_FEATURE_IDENTITY:
        raise ValueError("frozen labels use different Harrier features")
    fitted = fit_quality_head(
        read_labelled_embeddings(labels),
        head=RidgeHeadConfig(config.regularization),
        split_seed=config.split_seed,
    )
    assert isinstance(fitted.scorer, RidgeHead)
    return FittedRidgeQualityHead(
        head=fitted.scorer,
        label_exclusion=labels.label_exclusion,
        feature_identity=labels.feature_identity,
        split_seed=config.split_seed,
        training_documents=fitted.training_documents,
        development=fitted.development,
    )


def build_ridge_quality_head(
    labels: ArtifactStep[FrozenQualityLabels], *, regularization: float, split_seed: int, version: str | None = None
) -> ArtifactStep[FittedRidgeQualityHead]:
    """Fit one ridge candidate without reading audit labels for training or metrics."""
    identity = {"labels": [labels.name, labels.version], "regularization": regularization, "split_seed": split_seed}
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:20]
    name = f"fast-track/quality-head/{digest}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> _RidgeFitConfig:
        return _RidgeFitConfig(ctx.artifact_path(labels), regularization, split_seed)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=FittedRidgeQualityHead,
        deps=(labels,),
        build_config=build_config,
        run=_fit_ridge,
    )
