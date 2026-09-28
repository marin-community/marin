# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predict each document's content type from its Harrier embedding.

The per-type quality calibration routes a document through its predicted content
type. This step stores that type for every document: one Zephyr task per
normalized shard walks the shard's ids, takes the Harrier rows beside them
(:class:`shards.AlignedColumn`, which checks the ids), runs the
:mod:`domain_mlp` classifier over the L2-normalized embeddings, and writes one
output shard under the same basename in the normalized row order.

Each row carries ``id``, the argmax ``content_type``, its probability
``content_type_prob``, and the whole distribution ``content_type_probs``. The
distribution is kept because the classifier under-recalls the residual ``other``
class, and correcting that is a per-class logit shift that needs every class's
probability; recomputing it would mean re-reading the embeddings.

CPU, not GPU: the forward is ~1.3 MFLOPs a document against a 1 KB embedding
read, so the object store bounds the step rather than the matmuls.
"""

import functools
import logging
from collections.abc import Iterator
from functools import partial

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import ShardInfo

from experiments.datakit.cluster.quality.fast_transformer import domain_mlp
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    ContentTypePin,
    classifier_path,
    require_pinned_classifier,
)
from experiments.datakit.cluster.quality.fast_transformer.score_fusion import normalize_embeddings
from experiments.datakit.cluster.quality.fast_transformer.shards import (
    ShardPool,
    map_normalized_shards,
    read_aligned_column,
    rebatch,
)

logger = logging.getLogger(__name__)

CONTENT_TYPE_VERSION = 1
# Rows per normalize-and-forward block. The embedding shard is read whole as int8
# (2.75 GB for the largest, 2,682,446 documents); its float32 expansion is not, so
# the block bounds that at 268 MB.
BLOCK_ROWS = 65_536
# Four tasks per worker, like the bucket step. A task holds one int8 embedding
# shard, one float32 block and its activations.
POOL = ShardPool(
    worker=ResourceConfig(cpu=8, ram="32g", disk="8g"),
    task=ResourceConfig(cpu=2, ram="8g", disk="8g"),
    max_workers=128,
)


class ContentTypes(BaseModel):
    """Co-partitioned per-source content types.

    ``output_dir`` holds one parquet shard per normalized shard, same basename, with
    ``id``, ``content_type``, ``content_type_prob`` and ``content_type_probs``, one
    row per normalized document in the normalized shard's row order.
    """

    version: str = f"v{CONTENT_TYPE_VERSION}"
    output_dir: DatakitArtifactPath
    embedding_dir: DatakitArtifactPath
    classifier: str
    model_sha256: str
    labels: list[str]
    counters: dict[str, int | float]


def content_type_hash_attrs(pin: ContentTypePin) -> dict[str, str | int | list[str]]:
    """The identity of a content-type step, shared by its producer and its consumers."""
    return {
        "classifier": pin.name,
        "model_sha256": pin.model_sha256,
        "labels": list(pin.labels),
        "v": CONTENT_TYPE_VERSION,
    }


def content_type_schema(labels: tuple[str, ...]) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("content_type", pa.string()),
            pa.field("content_type_prob", pa.float32()),
            pa.field("content_type_probs", pa.list_(pa.float32(), len(labels))),
        ]
    )


@functools.cache
def pinned_classifier(path: str, pin: ContentTypePin) -> domain_mlp.DomainMlp:
    """The process's classifier, digest- and label-checked on first use."""
    require_pinned_classifier(pin, path)
    model, labels = domain_mlp.load(path)
    if labels != pin.labels:
        raise ValueError(f"{path} emits {labels}, but {pin.name} pins {pin.labels}")
    return model


def type_batch(model: domain_mlp.DomainMlp, labels: tuple[str, ...], ids: pa.Array, embeddings: np.ndarray):
    """One record batch of types for ``ids`` from their raw int8 embeddings."""
    probabilities = domain_mlp.predict_probabilities(model, normalize_embeddings(embeddings))
    index = probabilities.argmax(axis=1)
    return pa.RecordBatch.from_arrays(
        [
            ids,
            pa.array(np.asarray(labels)[index], type=pa.string()),
            pa.array(probabilities[np.arange(len(index)), index], type=pa.float32()),
            pa.FixedSizeListArray.from_arrays(pa.array(probabilities.ravel(), type=pa.float32()), len(labels)),
        ],
        schema=content_type_schema(labels),
    )


def _type_shard(
    batches: Iterator[pa.RecordBatch],
    shard: ShardInfo,
    side_paths: tuple[str, ...],
    *,
    model_path: str,
    pin: ContentTypePin,
) -> Iterator[pa.RecordBatch]:
    """Type one normalized shard from its embedding shard, in the normalized order."""
    model = pinned_classifier(model_path, pin)
    (embedding_path,) = side_paths
    embeddings = read_aligned_column(embedding_path, "embedding", f"shard {shard.shard_idx} ({embedding_path})")
    documents = 0
    for batch in rebatch(batches, BLOCK_ROWS):
        ids = batch.column("id")
        documents += len(ids)
        yield type_batch(model, pin.labels, ids, embeddings.take(ids.to_numpy(zero_copy_only=False)))
    embeddings.require_consumed()
    counters.pipeline.update_counter("content_type/docs_typed", documents)
    counters.pipeline.update_counter("content_type/shards", 1)


def predict_content_types(
    output_path: str,
    *,
    normalized: NormalizedData,
    embedding_dir: str,
    classifier: ContentTypePin,
    pool: ShardPool = POOL,
    zephyr_context: ZephyrContext | None = None,
) -> ContentTypes:
    """Type every shard of one normalized source; one Zephyr task per shard.

    Output shards that already exist are skipped, so a rerun after a partial failure
    types only the remainder.
    """
    outcome = map_normalized_shards(
        name="content-type",
        text_dir=normalized.main_output_dir,
        side_dirs=[embedding_dir],
        columns=["id"],
        shard_fn=partial(_type_shard, model_path=classifier_path(classifier), pin=classifier),
        output_path=output_path,
        schema=content_type_schema(classifier.labels),
        pool=pool,
        zephyr_context=zephyr_context,
    )
    return ContentTypes(
        output_dir=output_path,
        embedding_dir=embedding_dir,
        classifier=classifier.name,
        model_sha256=classifier.model_sha256,
        labels=list(classifier.labels),
        counters=dict(outcome.counters),
    )


def content_type_step(
    *,
    name: str,
    normalized: StepSpec,
    embedding: StepSpec,
    classifier: ContentTypePin,
    pool: ShardPool = POOL,
    zephyr_context: ZephyrContext | None = None,
) -> StepSpec:
    """A step that types ``normalized`` from its Harrier leaf ``embedding`` with ``classifier``.

    Its ``output_path`` is a shard directory co-partitioned with ``normalized``, which
    is what :func:`bucket.quality_step` takes as its ``content_type`` input.
    """
    return StepSpec(
        name=name,
        deps=[normalized, embedding],
        hash_attrs=content_type_hash_attrs(classifier),
        fn=lambda output_path: predict_content_types(
            output_path,
            normalized=read_artifact(normalized.output_path, NormalizedData),
            embedding_dir=embedding.output_path,
            classifier=classifier,
            pool=pool,
            zephyr_context=zephyr_context,
        ),
    )
