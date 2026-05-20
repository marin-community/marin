# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a fasttext classifier over datakit-normalized data.

Reads datakit-normalized Parquet (``id``, ``text``, ``partition_id``), runs a
fasttext ``.bin`` model over each record's text, and emits a co-partitioned
Parquet attributes dataset with the predicted label distribution.

Schema of the emitted Parquet attributes (datakit ``{id, attributes}`` convention,
consumable by :func:`marin.processing.classification.consolidate.consolidate`):

    id                       : string         — matches source document id
    partition_id             : int            — matches source partition
    attributes               : struct
        top_label            : string         — fasttext top-1 label (``__label__`` stripped)
        top_score            : float          — top-1 probability
        labels               : list[string]   — all labels in fasttext score order
        scores               : list[float]    — corresponding probabilities

If ``score_target_label`` is passed to :func:`classify_fasttext_to_parquet` or
:func:`classify_fasttext_step` (binary classifiers, e.g. dolma3-quality with
``"1"``), the ``attributes`` struct collapses to a single ``high_score: float64``
field equal to ``P(label == score_target_label)`` — the full label distribution
is redundant when there are only two classes.

The model itself is materialized once via :func:`prepare_fasttext_model_step` —
a tiny prep step that pulls the model ``.bin`` from HuggingFace and stages it
into the step's ``output_path`` on GCS. Workers in the classify step then read
from the in-region GCS path, so a 100-source fan-out doesn't hammer the HF Hub.

Output is co-partitioned with the source: one ``part-NNNNN-of-MMMMM.parquet``
per input partition, preserving the source filenames so consolidate can
sorted-merge-join without a shuffle.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from functools import cache, partial
from typing import Any

import pyarrow as pa
from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_glob
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ShardInfo, ZephyrContext, atomic_rename, counters, write_parquet_file
from zephyr.readers import load_file

logger = logging.getLogger(__name__)


# Default cap on text length sent to the classifier. The Dolma3 topic
# classifier (and most fasttext web-text classifiers) is trained on short,
# truncated documents — passing megabytes of text just slows the predict
# loop without improving the prediction. 100 KiB ~ first ~25 K tokens at 4
# chars/token, plenty for a topic decision and matches what AllenAI's own
# pipeline does upstream of this classifier.
DEFAULT_MAX_TEXT_CHARS = 100_000

# fasttext.predict rejects strings containing a literal newline, so we
# replace them with a single space before predicting. Matches the
# canonical preprocessing every fasttext-on-web-text pipeline applies.
_NEWLINE_REPLACEMENT = " "


class FastTextModel(BaseModel):
    """Artifact: a fasttext ``.bin`` model staged at a stable GCS path.

    Persisted as the prep step's ``.artifact`` so downstream classify steps
    can locate the model without re-running the download.

    Attributes:
        model_dir: Directory containing ``model.bin``. Equal to the prep
            step's ``output_path``.
        model_path: Resolved path to the ``.bin`` (``<model_dir>/model.bin``).
        hf_repo_id: Source HF repo (e.g. ``"allenai/dolma3-fasttext-weborganizer-topic-classifier"``).
        hf_filename: Filename within the HF repo that was downloaded.
        revision: HF commit / branch / tag pulled.
        size_bytes: Size of the staged ``.bin``.
    """

    version: str = "v1"
    model_dir: str
    model_path: str
    hf_repo_id: str
    hf_filename: str
    revision: str
    size_bytes: int


class FastTextAttributes(BaseModel):
    """Outcome of :func:`classify_fasttext_to_parquet`: a co-partitioned attributes dataset.

    Persisted as the step's ``.artifact`` so downstream consumers can locate
    the output without re-running the pipeline.

    Attributes:
        output_dir: Directory containing ``part-NNNNN-of-MMMMM.parquet`` files.
        num_partitions: Number of output partitions; matches the source.
        model_path: Path to the fasttext ``.bin`` used to produce these
            attributes. Recorded so consumers can confirm provenance.
        counters: Aggregated zephyr counters from the classify pipeline.
    """

    version: str = "v1"
    output_dir: str
    num_partitions: int
    model_path: str
    counters: dict[str, int]


_MODEL_FILENAME = "model.bin"
_LABEL_PREFIX = "__label__"


def prepare_fasttext_model(
    *,
    hf_repo_id: str,
    hf_filename: str,
    revision: str,
    output_path: str,
) -> FastTextModel:
    """Download a fasttext ``.bin`` from HF and stage it at ``<output_path>/model.bin``.

    Args:
        hf_repo_id: HuggingFace repo id (e.g. ``"allenai/dolma3-fasttext-weborganizer-topic-classifier"``).
        hf_filename: Filename within the repo (e.g. ``"model.bin"``).
        revision: HF commit hash / tag. Pin this — fasttext models are silently
            re-uploaded on the Hub and we want the cache key tied to bytes.
        output_path: Step output directory. ``model.bin`` is written into it.

    Returns:
        :class:`FastTextModel` artifact pointing at the staged ``.bin``.
    """
    # Import local: huggingface_hub is only needed by the prep step, not by classify workers.
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, revision=revision)
    size = os.path.getsize(local)
    target = os.path.join(output_path, _MODEL_FILENAME)
    target_fs, resolved = url_to_fs(target)
    target_fs.mkdirs(os.path.dirname(resolved), exist_ok=True)
    # atomic_rename keeps the staged path crash-safe: a half-uploaded blob never appears at `target`.
    with atomic_rename(target) as tmp:
        tmp_fs, tmp_resolved = url_to_fs(tmp)
        tmp_fs.put(local, tmp_resolved)
    logger.info("Staged %s@%s/%s → %s (%d bytes)", hf_repo_id, revision, hf_filename, target, size)
    return FastTextModel(
        model_dir=output_path,
        model_path=target,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        revision=revision,
        size_bytes=size,
    )


def _normalize_for_fasttext(text: str, max_chars: int | None) -> str:
    """Strip newlines and truncate. Matches the canonical fasttext-on-web preprocessing."""
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text.replace("\n", _NEWLINE_REPLACEMENT).replace("\r", _NEWLINE_REPLACEMENT)


def _strip_label_prefix(label: str) -> str:
    """Drop the leading ``__label__`` fasttext attaches to every class name."""
    return label.removeprefix(_LABEL_PREFIX)


_FULL_ATTRIBUTES_STRUCT = pa.struct(
    [
        pa.field("top_label", pa.string()),
        pa.field("top_score", pa.float64()),
        pa.field("labels", pa.list_(pa.string())),
        pa.field("scores", pa.list_(pa.float64())),
    ]
)

_BINARY_ATTRIBUTES_STRUCT = pa.struct(
    [
        pa.field("high_score", pa.float64()),
    ]
)


def _output_schema(score_target_label: str | None) -> pa.Schema:
    """Pick the output schema based on whether we're collapsing to a single high-score."""
    attrs = _BINARY_ATTRIBUTES_STRUCT if score_target_label is not None else _FULL_ATTRIBUTES_STRUCT
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("partition_id", pa.int64()),
            pa.field("attributes", attrs),
        ]
    )


# Per-process model cache — survives across map_shard calls under InlineRunner,
# so a worker loads the .bin exactly once regardless of how many shards it handles.
@cache
def _load_fasttext_model(model_path_str: str) -> Any:
    """Return a fasttext model loaded from a local copy of *model_path_str*.

    The .bin is streamed from GCS to a per-process tempfile on first call;
    subsequent calls in the same worker process return the cached model object.
    """
    # fasttext-wheel 0.9.2 calls ``np.array(..., copy=False)`` inside
    # ``FastText.predict``; NumPy 2.x rejects ``copy=False`` (must use
    # ``copy=None``). Patch before importing fasttext so the predict path
    # inherits the shim. Idempotent across calls in the same process.
    import numpy as np

    if not getattr(np, "_fasttext_copy_compat", False):
        _orig_np_array = np.array

        def _np_array_copy_compat(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("copy") is False:
                kwargs["copy"] = None
            return _orig_np_array(*args, **kwargs)

        np.array = _np_array_copy_compat
        np._fasttext_copy_compat = True

    # Local import so the driver (which never loads a model) doesn't pay the
    # fasttext C-extension import cost.
    import fasttext

    fs, resolved = url_to_fs(model_path_str)
    fd, local = tempfile.mkstemp(prefix="fasttext-", suffix=".bin")
    os.close(fd)
    fs.get(resolved, local)
    return fasttext.load_model(local)


def _classify_shard(
    paths: Iterator[str],
    shard: ShardInfo,
    *,
    model_path_str: str,
    output_dir: str,
    text_field: str,
    max_text_chars: int | None,
    k: int,
    threshold: float,
    score_target_label: str | None,
) -> Iterator[dict[str, Any]]:
    """Classify each input partition, write a co-partitioned output parquet."""
    model = _load_fasttext_model(model_path_str)
    output_schema = _output_schema(score_target_label)

    def _empty_attrs() -> dict[str, Any]:
        if score_target_label is not None:
            return {"high_score": 0.0}
        return {"top_label": "", "top_score": 0.0, "labels": [], "scores": []}

    def _attrs_from_prediction(stripped: list[str], probs: Any) -> dict[str, Any]:
        if score_target_label is not None:
            # Linear scan is fine: K=-1 typical, but even for full
            # binary/multi-class outputs this is a 2-24 element list.
            for label, prob in zip(stripped, probs, strict=False):
                if label == score_target_label:
                    return {"high_score": float(prob)}
            return {"high_score": 0.0}
        top_label = stripped[0] if stripped else ""
        top_score = float(probs[0]) if len(probs) > 0 else 0.0
        return {
            "top_label": top_label,
            "top_score": top_score,
            "labels": stripped,
            "scores": [float(p) for p in probs],
        }

    for input_path in paths:

        def rows_for(p: str) -> Iterator[dict[str, Any]]:
            for record in load_file(p):
                text = str(record.get(text_field, "") or "")
                pid = record.get("partition_id", shard.shard_idx)
                if not text:
                    counters.increment("classify/empty_text")
                    yield {"id": record["id"], "partition_id": pid, "attributes": _empty_attrs()}
                    continue
                cleaned = _normalize_for_fasttext(text, max_text_chars)
                labels, probs = model.predict(cleaned, k=k, threshold=threshold)
                stripped = [_strip_label_prefix(label) for label in labels]
                counters.increment("classify/predicted")
                yield {
                    "id": record["id"],
                    "partition_id": pid,
                    "attributes": _attrs_from_prediction(stripped, probs),
                }

        out_filename = os.path.basename(input_path)
        out_path = f"{output_dir.rstrip('/')}/{out_filename}"
        result = write_parquet_file(rows_for(input_path), output_path=out_path, schema=output_schema)
        yield result


def classify_fasttext_to_parquet(
    *,
    normalized_data: NormalizedData,
    model: FastTextModel,
    output_path: str,
    text_field: str = "text",
    max_text_chars: int | None = DEFAULT_MAX_TEXT_CHARS,
    k: int = -1,
    threshold: float = 0.0,
    score_target_label: str | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
) -> FastTextAttributes:
    """Classify records in *normalized_data* with the fasttext model at *model*.

    Args:
        normalized_data: Upstream :class:`NormalizedData` artifact. Reads from
            ``normalized_data.main_output_dir`` (the flat, co-partitioned
            Parquet directory produced by datakit normalize). Records must
            have ``id``, ``text``, and ``partition_id`` columns.
        model: :class:`FastTextModel` artifact pointing at a staged
            ``.bin``. The mark workers stream the bin from GCS to local
            ``/tmp`` once per worker process.
        output_path: Directory for co-partitioned Parquet attributes. One
            output file is written per input partition, preserving filenames.
        text_field: Text column name in the input records.
        max_text_chars: Truncate input text to this many UTF-8 chars before
            predict. ``None`` disables truncation.
        k: Top-K labels to keep from each prediction. ``-1`` keeps the full
            label distribution.
        threshold: Minimum score for a label to be kept.
        score_target_label: If set, collapse output ``attributes`` to a single
            ``high_score: float64`` = ``P(label == score_target_label)``. Use
            for binary classifiers; ``None`` keeps the full struct (top_label,
            top_score, labels, scores).
        worker_resources: Per-shard resource request. Defaults to 2 CPU /
            8 GB RAM, matching the decon mark step.
        max_workers: Max Zephyr workers. Defaults to Zephyr's own default.

    Returns:
        :class:`FastTextAttributes` describing the output dataset and counters.
    """
    input_path = normalized_data.main_output_dir
    files: list[str] = sorted(fsspec_glob(f"{input_path.rstrip('/')}/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {input_path}")
    num_partitions = len(files)
    logger.info(
        "classify: %s → %s, %d input partitions, model=%s",
        input_path,
        output_path,
        num_partitions,
        model.model_path,
    )

    pipeline = Dataset.from_list(files).map_shard(
        partial(
            _classify_shard,
            model_path_str=model.model_path,
            output_dir=output_path,
            text_field=text_field,
            max_text_chars=max_text_chars,
            k=k,
            threshold=threshold,
            score_target_label=score_target_label,
        )
    )

    resources = worker_resources or ResourceConfig(cpu=2, ram="8g")
    ctx_kwargs: dict[str, Any] = {"name": "classify-fasttext", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = ZephyrContext(**ctx_kwargs)
    outcome = ctx.execute(pipeline)

    return FastTextAttributes(
        output_dir=output_path,
        num_partitions=num_partitions,
        model_path=model.model_path,
        counters=dict(outcome.counters),
    )


def prepare_fasttext_model_step(
    *,
    name: str,
    hf_repo_id: str,
    hf_filename: str,
    revision: str,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """StepSpec factory for :func:`prepare_fasttext_model`.

    Args:
        name: Step name (e.g. ``"datakit/classify/_model/dolma3-weborg-topic"``).
        hf_repo_id, hf_filename, revision: Identify the exact ``.bin`` to
            stage. All three feed ``hash_attrs`` so re-pinning re-stages.
        output_path_prefix, override_output_path: StepSpec routing.
    """
    hash_attrs: dict[str, Any] = {
        "hf_repo_id": hf_repo_id,
        "hf_filename": hf_filename,
        "revision": revision,
    }
    return StepSpec(
        name=name,
        fn=lambda output_path: prepare_fasttext_model(
            hf_repo_id=hf_repo_id,
            hf_filename=hf_filename,
            revision=revision,
            output_path=output_path,
        ),
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )


def classify_fasttext_step(
    *,
    name: str,
    normalized: StepSpec,
    model_step: StepSpec,
    text_field: str = "text",
    max_text_chars: int | None = DEFAULT_MAX_TEXT_CHARS,
    k: int = -1,
    threshold: float = 0.0,
    score_target_label: str | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that classifies a normalized dataset with a fasttext model.

    Args:
        name: Step name (e.g. ``"datakit/classify/topic/fineweb"``).
        normalized: Upstream datakit normalize step whose output is the input.
        model_step: Upstream :func:`prepare_fasttext_model_step` whose output
            directory holds the ``.bin``.
        text_field: Text column name in the input records.
        max_text_chars: Truncate input text to this many chars before predict.
            ``None`` disables truncation.
        k: Top-K labels to keep. ``-1`` keeps the full distribution.
        threshold: Minimum probability for a label to be kept.
        score_target_label: Collapse the output attributes to a single
            ``high_score: float64`` equal to ``P(label == score_target_label)``.
            Use for binary classifiers (e.g. dolma3-quality, ``"1"``). ``None``
            keeps the full ``{top_label, top_score, labels, scores}`` struct.
        worker_resources, max_workers: Zephyr execution knobs.
        output_path_prefix, override_output_path: StepSpec routing.
    """
    hash_attrs: dict[str, Any] = {
        "text_field": text_field,
        "max_text_chars": max_text_chars,
        "k": k,
        "threshold": threshold,
    }
    # Only include when set so the existing full-distribution callers
    # (weborganizer topic) keep their cache key — adding a ``None`` here
    # would re-hash and force a re-classify of every already-classified source.
    if score_target_label is not None:
        hash_attrs["score_target_label"] = score_target_label
    return StepSpec(
        name=name,
        fn=lambda output_path: classify_fasttext_to_parquet(
            normalized_data=Artifact.from_path(normalized, NormalizedData),
            model=Artifact.from_path(model_step, FastTextModel),
            output_path=output_path,
            text_field=text_field,
            max_text_chars=max_text_chars,
            k=k,
            threshold=threshold,
            score_target_label=score_target_label,
            worker_resources=worker_resources,
            max_workers=max_workers,
        ),
        deps=[normalized, model_step],
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
