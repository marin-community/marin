# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Building blocks for running a fasttext classifier in a Zephyr pipeline.

Provides two composable pieces:

* :func:`prepare_fasttext_model_step` — a StepSpec that pulls a fasttext
  ``.bin`` from HuggingFace and stages it on GCS. The bin is materialized
  once; classify workers load it from the in-region GCS path so a 100-source
  fan-out doesn't hammer the HF Hub.

* :func:`get_fasttext_batch_predict` — bind the caller's classifier knobs
  (model path, max_text_chars, k, threshold, ...) and return a
  ``flat_map``-shaped callable that **annotates** each input record with
  one new field (name configurable via ``output_field_name``, default
  ``"fasttext_result"``). All original fields are preserved. Plug into a
  pipeline as::

      .window(batch_size).flat_map(get_fasttext_batch_predict(...))

  **Caller responsibility**: pass ``stage_runner_factory=InlineRunner``
  to your :class:`ZephyrContext`. The model is loaded once per worker
  process via an ``@cache`` on :func:`_load_fasttext_model`; under the
  default ``SubprocessRunner`` every shard is a fresh process and the
  cache buys nothing.

Annotation value shape (also what pyarrow infers for the parquet field type
under ``Dataset.write_parquet``, so callers typically don't need to pass an
explicit schema):

    score_target_label set      → float64  (``P(label == score_target_label)``)
    score_target_label is None  → struct {top_label, top_score, labels, scores}

Empty-text and whitespace-only records are skipped — consolidate joins by
``id``, so absence is the right signal that the classifier had nothing to
score. Individual classifiers wire their own input discovery, output
naming, hash_attrs, projection, and downstream artifact shape.
"""

import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from functools import cache, partial
from typing import Any

import fasttext
import numpy as np
from huggingface_hub import hf_hub_download
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import StoragePath, atomic_rename
from zephyr import counters

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
    local = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, revision=revision)
    size = os.path.getsize(local)
    target = os.path.join(output_path, _MODEL_FILENAME)
    StoragePath(target).parent.mkdirs()
    # atomic_rename keeps the staged path crash-safe: a half-uploaded blob never appears at `target`.
    with atomic_rename(target) as tmp:
        StoragePath(tmp).upload_from(local)
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
    # ``copy=None``). Patch before the first predict so the predict path
    # inherits the shim. Idempotent across calls in the same process.
    if not getattr(np, "_fasttext_copy_compat", False):
        _orig_np_array = np.array

        def _np_array_copy_compat(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("copy") is False:
                kwargs["copy"] = None
            return _orig_np_array(*args, **kwargs)

        np.array = _np_array_copy_compat
        np._fasttext_copy_compat = True

    fd, local = tempfile.mkstemp(prefix="fasttext-", suffix=".bin")
    os.close(fd)
    StoragePath(model_path_str).download_to(local)
    return fasttext.load_model(local)


def _value_from_prediction(
    stripped: list[str],
    probs: Any,
    score_target_label: str | None,
) -> Any:
    """Project a single fasttext (labels, probs) pair into the annotation value.

    Returns a float when ``score_target_label`` is set, otherwise a dict
    with the full label distribution.
    """
    if score_target_label is not None:
        for label, prob in zip(stripped, probs, strict=False):
            if label == score_target_label:
                return float(prob)
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
    output_field_name: str,
    model_load_fn: Callable[[str], Any],
) -> Iterator[dict[str, Any]]:
    """One fasttext.predict call per batch; yield each non-empty record with the prediction annotated.

    The output record is ``{**record, output_field_name: value}`` -- input
    fields are preserved, with one extra field carrying the prediction
    (float when ``score_target_label`` is set, otherwise a struct dict).

    Empty-text and whitespace-only records are skipped (consolidate joins by
    ``id``, so absence is the correct signal that the classifier had nothing
    to score).
    """
    model = model_load_fn(model_path_str)
    counters.pipeline.update_counter("classify/batches_in", 1)
    counters.pipeline.update_counter("classify/docs_in", len(batch))

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
    counters.pipeline.update_counter("classify/bytes_in", bytes_in)
    counters.pipeline.update_counter("classify/empty_text", len(batch) - len(texts))
    if truncated:
        counters.pipeline.update_counter("classify/docs_truncated", truncated)

    if not texts:
        counters.pipeline.update_counter("classify/batches_skipped_empty", 1)
        return

    # IMPORTANT: per-text predict, NOT batch predict.
    #
    # ``model.predict(list_of_texts, k=-1, threshold=0.0)`` is broken in
    # fasttext-wheel 0.9.2: it returns each text's top-label probability
    # duplicated across BOTH labels (e.g. ``[__label__0, __label__1]`` with
    # probs ``[0.97, 0.97]``) instead of the per-label softmax. The
    # corresponding single-text call ``model.predict(text, k=-1, ...)``
    # returns the correct per-label distribution (probs sum to ~1.0).
    #
    # Net effect on the LLM-quality binary classifier: every record's
    # ``P(label="1")`` collapses to ~``max(P(0), P(1)) >= 0.5`` regardless of
    # which class actually wins, destroying the score distribution
    # downstream consumers bin on. See full bench in the regression test
    # ``test_predict_batch_is_per_text``.
    for record, text in zip(records_to_predict, texts, strict=True):
        labels, probs = model.predict(text, k=k, threshold=threshold)
        stripped = [_strip_label_prefix(label) for label in labels]
        counters.pipeline.update_counter("classify/predicted", 1)
        yield {**record, output_field_name: _value_from_prediction(stripped, probs, score_target_label)}


def get_fasttext_batch_predict(
    *,
    model_path: str,
    text_field: str = "text",
    max_text_chars: int | None = DEFAULT_MAX_TEXT_CHARS,
    k: int = -1,
    threshold: float = 0.0,
    score_target_label: str | None = None,
    output_field_name: str = "fasttext_result",
    model_load_fn: Callable[[str], Any] | None = None,
) -> Callable[[list[dict[str, Any]]], Iterator[dict[str, Any]]]:
    """Bind classifier knobs and return a ``flat_map`` callable that annotates each input record.

    Usage::

        ctx = ZephyrContext(..., stage_runner_factory=InlineRunner)
        pipeline = (
            Dataset.from_list(files)
            .flat_map(load_file)
            .window(batch_size)
            .flat_map(get_fasttext_batch_predict(
                model_path=..., score_target_label="1", output_field_name="score"))
            .select("id", "score")  # drop input fields we don't want to persist
            .write_parquet(pattern, skip_existing=True)
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
        score_target_label: If set, the annotation value is the float
            ``P(label == score_target_label)``. Use for binary classifiers.
            ``None`` keeps the full ``{top_label, top_score, labels, scores}``
            struct as the annotation value.
        output_field_name: Field name added to each output record. Existing
            input fields are preserved alongside it.
        model_load_fn: Override the (cached) model loader. Defaults to
            :func:`_load_fasttext_model`, which streams the ``.bin`` from
            GCS once per worker process.

    Returns:
        A ``(list[dict] -> Iterator[dict])`` callable suitable for
        ``.flat_map(...)`` after ``.window(batch_size)``. Each output record
        is ``{**input_record, output_field_name: <prediction>}``.
    """
    return partial(
        _predict_batch,
        model_path_str=model_path,
        text_field=text_field,
        max_text_chars=max_text_chars,
        k=k,
        threshold=threshold,
        score_target_label=score_target_label,
        output_field_name=output_field_name,
        model_load_fn=model_load_fn or _load_fasttext_model,
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
