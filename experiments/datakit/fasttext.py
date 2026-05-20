# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Building blocks for running a fasttext classifier in a Zephyr pipeline.

Provides three composable pieces:

* :func:`prepare_fasttext_model_step` — a StepSpec that pulls a fasttext
  ``.bin`` from HuggingFace and stages it on GCS. The bin is materialized
  once; classify workers load it from the in-region GCS path so a 100-source
  fan-out doesn't hammer the HF Hub.

* :func:`get_fasttext_batch_predict` — bind the caller's classifier knobs
  (model path, max_text_chars, k, threshold, ...) and return a
  ``flat_map``-shaped callable. Plug into a pipeline as::

      .window(batch_size).flat_map(get_fasttext_batch_predict(...))

  **Caller responsibility**: pass ``stage_runner_factory=InlineRunner``
  to your :class:`ZephyrContext`. The model is loaded once per worker
  process via an ``@cache`` on :func:`_load_fasttext_model`; under the
  default ``SubprocessRunner`` every shard is a fresh process and the
  cache buys nothing.

* :func:`output_schema` — the pyarrow schema for the per-record outputs,
  for callers that want to pass an explicit schema to
  ``Dataset.write_parquet``.

Output record shape (one dict per non-empty input record, keyed by ``id``):

    Full distribution (default)        Binary (``score_target_label`` set)
    ───────────────────────────        ───────────────────────────────────
    id        : string                 id         : string
    top_label : string                 high_score : float64
    top_score : float64
    labels    : list[string]
    scores    : list[float64]

Empty-text and whitespace-only records are skipped — consolidate joins by
``id``, so absence is the right signal that the classifier had nothing to
score. Individual classifiers wire their own input discovery, output
naming, hash_attrs, and downstream artifact shape.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from functools import cache, partial
from typing import Any

import pyarrow as pa
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from zephyr import atomic_rename, counters

logger = logging.getLogger(__name__)


# Default cap on text length sent to the classifier. The Dolma3 topic
# classifier (and most fasttext web-text classifiers) is trained on short,
# truncated documents — passing megabytes of text just slows the predict
# loop without improving the prediction. 100 KiB ~ first ~25 K tokens at 4
# chars/token, plenty for a topic decision and matches what AllenAI's own
# pipeline does upstream of this classifier.
DEFAULT_MAX_TEXT_CHARS = 100_000

# Records per fasttext.predict call. Large enough to amortize Python↔C++
# overhead, small enough that a worker's peak text buffer stays bounded
# (~25 MB at the 100 KB max_text_chars cap, well within an 8 GB worker).
DEFAULT_BATCH_SIZE = 256

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


def output_schema(score_target_label: str | None) -> pa.Schema:
    """Pyarrow schema for the per-record output of :func:`get_fasttext_batch_predict`.

    Use as the ``schema=`` argument to ``Dataset.write_parquet`` so the writer
    doesn't have to infer from the first record (fragile when the first record
    happens to be missing fields).
    """
    if score_target_label is not None:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("high_score", pa.float64()),
            ]
        )
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("top_label", pa.string()),
            pa.field("top_score", pa.float64()),
            pa.field("labels", pa.list_(pa.string())),
            pa.field("scores", pa.list_(pa.float64())),
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


def _attrs_from_prediction(
    stripped: list[str],
    probs: Any,
    score_target_label: str | None,
) -> dict[str, Any]:
    """Project a single fasttext (labels, probs) pair into the output attribute fields."""
    if score_target_label is not None:
        # Linear scan: K=-1 typical, but even full binary/multi-class outputs
        # are only 2-24 elements.
        for label, prob in zip(stripped, probs, strict=False):
            if label == score_target_label:
                return {"high_score": float(prob)}
        raise ValueError(
            f"score_target_label={score_target_label!r} not in predicted labels {stripped!r}; "
            f"check the label spelling, or relax k/threshold so the target label survives."
        )
    top_label = stripped[0] if stripped else ""
    top_score = float(probs[0]) if len(probs) > 0 else 0.0
    return {
        "top_label": top_label,
        "top_score": top_score,
        "labels": stripped,
        "scores": [float(p) for p in probs],
    }


def _predict_batch(
    batch: list[dict[str, Any]],
    *,
    model_path_str: str,
    text_field: str,
    max_text_chars: int | None,
    k: int,
    threshold: float,
    score_target_label: str | None,
) -> Iterator[dict[str, Any]]:
    """One fasttext.predict call per batch; yield per-record output dicts for non-empty docs.

    Empty-text and whitespace-only records are skipped (consolidate joins by
    id, so absence is the correct signal that the classifier had nothing
    to score).
    """
    model = _load_fasttext_model(model_path_str)
    counters.increment("classify/batches_in")
    counters.increment("classify/docs_in", len(batch))

    texts: list[str] = []
    records_to_predict: list[dict[str, Any]] = []
    bytes_in = 0
    truncated = 0
    for record in batch:
        raw = str(record.get(text_field, "") or "")
        normalized = _normalize_for_fasttext(raw, max_text_chars)
        # Skip empty input AND whitespace-only post-normalization (e.g. a doc
        # whose only non-whitespace content sits past max_text_chars and gets
        # cut off, leaving just spaces after newline replacement).
        if not normalized.strip():
            continue
        bytes_in += len(raw)
        if max_text_chars is not None and len(raw) > max_text_chars:
            truncated += 1
        texts.append(normalized)
        records_to_predict.append(record)
    counters.increment("classify/bytes_in", bytes_in)
    counters.increment("classify/empty_text", len(batch) - len(texts))
    if truncated:
        counters.increment("classify/docs_truncated", truncated)

    if not texts:
        return

    labels_list, probs_list = model.predict(texts, k=k, threshold=threshold)
    for record, labels, probs in zip(records_to_predict, labels_list, probs_list, strict=True):
        stripped = [_strip_label_prefix(label) for label in labels]
        counters.increment("classify/predicted")
        yield {"id": record["id"], **_attrs_from_prediction(stripped, probs, score_target_label)}


def get_fasttext_batch_predict(
    *,
    model_path: str,
    text_field: str = "text",
    max_text_chars: int | None = DEFAULT_MAX_TEXT_CHARS,
    k: int = -1,
    threshold: float = 0.0,
    score_target_label: str | None = None,
) -> Callable[[list[dict[str, Any]]], Iterator[dict[str, Any]]]:
    """Bind classifier knobs and return a ``flat_map`` callable for batched predict.

    Usage::

        ctx = ZephyrContext(..., stage_runner_factory=InlineRunner)
        pipeline = (
            Dataset.from_list(files)
            .flat_map(load_file)
            .window(batch_size)
            .flat_map(get_fasttext_batch_predict(model_path=..., ...))
            .write_parquet(pattern, schema=output_schema(...), skip_existing=True)
        )

    Args:
        model_path: GCS path to the staged fasttext ``.bin`` (the
            ``model_path`` field of a :class:`FastTextModel` artifact produced
            by :func:`prepare_fasttext_model_step`).
        text_field: Text column name in the input records.
        max_text_chars: Truncate input text to this many UTF-8 chars before
            predict. ``None`` disables truncation.
        k: Top-K labels to keep from each prediction. ``-1`` keeps the full
            label distribution.
        threshold: Minimum probability for a label to be kept.
        score_target_label: If set, the output collapses to a single
            ``high_score: float64`` field equal to
            ``P(label == score_target_label)``. Use for binary classifiers.
            ``None`` keeps the full ``{top_label, top_score, labels, scores}``
            field set.

    Returns:
        A ``(list[dict] -> Iterator[dict])`` callable suitable for
        ``.flat_map(...)`` after ``.window(batch_size)``. See
        :func:`output_schema` for the exact per-record output shape.
    """
    return partial(
        _predict_batch,
        model_path_str=model_path,
        text_field=text_field,
        max_text_chars=max_text_chars,
        k=k,
        threshold=threshold,
        score_target_label=score_target_label,
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
