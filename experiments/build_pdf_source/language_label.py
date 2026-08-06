# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label every fuzzy-clean document with its FinePDFs-style GlotLID language bucket.

The final step of the all-routes pipeline: a 1:1 map over the fuzzy-clean corpus that appends
``language`` and ``language_score`` and drops nothing, emitting the dataset #7621 trains on. The
tagger is a deliberate port of FinePDFs' page-level averaging (``postprocessing/language.py`` +
``pipeline_utils/language.py`` in the FinePDFs repo), because the vendored per-language thresholds
in ``lid_th_values.json`` were calibrated against exactly that behavior:

* Each page is cleaned (table lines, markdown punctuation, whitespace collapse) and gated on the
  UTF-8 **byte** length and ratio of its alphabetic characters. A gated page contributes no scores
  but still counts in the averaging denominator, so mostly-empty documents average down.
* GlotLID scores each surviving page with ``k=1000``; per-language scores are summed and divided
  by the total page count, and languages averaging strictly above ``LANGUAGE_THRESHOLD`` become
  the document's candidates.
* :func:`select_bucket` walks the candidates by descending score: a top-1 ``zxx_*`` sticks, the
  first candidate strictly above its per-language threshold wins, and a document whose candidates
  all fail is bucketed ``{top}_removed`` (threshold known) or kept under its raw top label
  (threshold unknown) -- ``"unknown"`` when no page produced a candidate at all.

The thresholds transfer un-recalibrated from FinePDFs' extraction distribution. That is expected:
this corpus is OCR text like theirs, and recalibrating would need per-language labels this project
does not have. Do not tune the vendored values.

No document is dropped here -- ``{lang}_removed`` and ``"unknown"`` are labels, so the training
side decides what to keep with full information.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Protocol, cast

import fasttext
import pyarrow as pa
from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath, prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

from experiments.build_pdf_source.common import OcrModelData
from experiments.build_pdf_source.repair_ocr_all import split_pages

logger = logging.getLogger(__name__)

_COUNTER_PREFIX = "focus_crawl_pdf_lid"

GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILENAME = "model_v3.bin"
# Repo head at pin time; model_v3.bin has never been re-uploaded, so any revision holding it
# resolves to the same LFS object. The sha256 below is that object's LFS metadata hash.
GLOTLID_REVISION = "85cd6716494360367b75f642b5bc78667605d0b4"
GLOTLID_SHA256 = "a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e"

# The FinePDFs thresholds file, vendored from their repo's thresholds/th_values.json (identical
# JSON; this repo's lint adds a trailing newline).
_THRESHOLDS_PATH = Path(__file__).with_name("lid_th_values.json")
# FinePDFs floors every threshold at 0.05 when loading. The file's minimum is 0.10003, so the
# floor never fires; it is kept for load-path fidelity with the shipped pipeline.
THRESHOLD_FLOOR = 0.05
# zxx_* ("no linguistic content") gets a threshold no score can fail, so a zxx candidate is never
# re-routed to a weaker real language -- at any rank, not only top-1, which is FinePDFs' shipped
# behavior. These keys are absent from the file; the overrides add them.
_ZXX_OVERRIDES = {"zxx_Latn": -1.0, "zxx_Zzzz": -1.0, "zxx_Arab": -1.0}
# A candidate whose language has no threshold entry compares against this, which no score in [0, 1]
# exceeds -- unknown-threshold languages are unselectable mid-list and survive only as the raw
# fallback bucket when they are the top candidate.
UNSELECTABLE_THRESHOLD = 10_000.0

# A language becomes a document candidate only when its page-average score is strictly above this.
LANGUAGE_THRESHOLD = 0.01
# Page gates, applied to the *cleaned* page text: UTF-8 byte length of its alphabetic characters,
# and that against the byte length of everything that is not a space.
MIN_ALPHA_LENGTH_BYTES = 50
MIN_ALPHA_RATIO = 0.2
PREDICT_TOP_K = 1000

# FinePDFs called ``table_pattern.sub("", page_text, re.MULTILINE)``, passing the flag where
# ``count`` goes -- and re.MULTILINE == 8, so only the first 8 table lines of a page are removed.
# The thresholds were calibrated with that behavior, so the 8 is replicated deliberately.
TABLE_LINES_REMOVED = 8
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")

_MODEL_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
# Each worker holds one ~1.7 GB fasttext model in memory and a local copy on disk.
_WORKER_RESOURCES = ResourceConfig(cpu=4, ram="8g", disk="6g")
# One task per worker: prediction is sequential per shard, so multiplexing tasks buys nothing and
# would multiply the model's memory accounting.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=4, ram="8g", disk="6g")
# The fuzzy-clean corpus has ~23 shards; more workers than tasks would queue for unusable capacity.
_MAX_WORKERS = 23
# Not Zephyr's 1 GB default -- see repair_ocr_all: that default is OOM-killed at run end.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)


class LidModel(Protocol):
    """The slice of fasttext's API the tagger uses: the list path of ``predict``.

    The list path is load-bearing, not a convenience: fasttext-wheel 0.9.2's single-string
    ``predict`` crashes under numpy 2.x (``np.array(probs, copy=False)`` raises), while the list
    path builds its arrays differently and works.
    """

    def predict(self, lines: list[str], k: int) -> tuple[Sequence[Sequence[str]], Sequence[Sequence[float]]]: ...


@dataclass(frozen=True)
class PageScores:
    """Per-language page-average scores for one document, after the ``LANGUAGE_THRESHOLD`` cut."""

    averages: dict[str, float]
    gated: int


def load_thresholds() -> dict[str, float]:
    """Load the vendored per-language selection thresholds, floored and with the zxx overrides."""
    raw = json.loads(_THRESHOLDS_PATH.read_text())
    thresholds = {language: max(float(value), THRESHOLD_FLOOR) for language, value in raw.items()}
    thresholds.update(_ZXX_OVERRIDES)
    return thresholds


def _thresholds_sha256() -> str:
    return hashlib.sha256(_THRESHOLDS_PATH.read_bytes()).hexdigest()


def clean_page(page: str) -> str:
    """Strip table furniture and markdown punctuation and collapse whitespace, as FinePDFs does.

    The whitespace collapse also guarantees the result holds no newline, which the fasttext list
    predict path rejects.
    """
    page = _TABLE_LINE.sub("", page, count=TABLE_LINES_REMOVED)
    page = page.replace("|", "").replace("-", "").replace("*", "").replace("#", "")
    return _WHITESPACE.sub(" ", page)


def _alpha_byte_length(page: str) -> int:
    return len("".join(char for char in page if char.isalpha()).encode("utf-8"))


def page_scores(pages: Sequence[str], model: LidModel) -> PageScores:
    """Average per-language GlotLID scores over a document's cleaned pages.

    A page whose cleaned text is too short or insufficiently alphabetic (both measured in UTF-8
    bytes -- the gates predate this port and were calibrated on bytes, unlike ``page_offsets``,
    which are character offsets) is not scored at all, but still divides the average: the
    denominator is the total page count.
    """
    totals: defaultdict[str, float] = defaultdict(float)
    gated = 0
    for page in pages:
        alpha_length = _alpha_byte_length(page)
        non_space_length = len(page.replace(" ", "").replace("\n", "").encode("utf-8"))
        alpha_ratio = alpha_length / non_space_length if non_space_length else 0.0
        if alpha_length < MIN_ALPHA_LENGTH_BYTES or alpha_ratio < MIN_ALPHA_RATIO:
            gated += 1
            continue
        labels, scores = model.predict([page], k=PREDICT_TOP_K)
        for label, score in zip(labels[0], scores[0], strict=True):
            totals[label.removeprefix("__label__")] += float(score)

    averages = {
        language: average for language, total in totals.items() if (average := total / len(pages)) > LANGUAGE_THRESHOLD
    }
    return PageScores(averages=averages, gated=gated)


def select_bucket(averages: dict[str, float], thresholds: dict[str, float]) -> tuple[str, float]:
    """Pick the document's language bucket, re-routing sub-threshold candidates to the next best.

    A port of FinePDFs' ``SelectBestLanguage``: candidates descend by score; a top-1 ``zxx_*``
    label is taken as-is; otherwise the first candidate strictly above its threshold wins, with
    unknown-threshold languages unselectable. When nothing passes, the top candidate becomes
    ``{lang}_removed`` if its threshold is known and stays the raw label if not; with no candidates
    at all the bucket is ``"unknown"`` with score 0.0.
    """
    ranked = sorted(averages.items(), key=lambda item: item[1], reverse=True)
    for position, (language, score) in enumerate(ranked):
        if position == 0 and language.startswith("zxx_"):
            return language, score
        if score > thresholds.get(language, UNSELECTABLE_THRESHOLD):
            return language, score

    if not ranked:
        return "unknown", 0.0
    top_language, top_score = ranked[0]
    if top_language in thresholds:
        return f"{top_language}_removed", top_score
    return top_language, top_score


def label_document(row: dict, model: LidModel, thresholds: dict[str, float]) -> dict:
    """Append ``language`` and ``language_score`` to one stored document."""
    text = row["text"]
    offsets = row["page_offsets"]
    if not offsets or offsets[-1] < len(text):
        # Text longer than the recorded pages has no known upstream cause, so slicing would
        # silently mislabel; refuse instead.
        end = offsets[-1] if offsets else None
        raise ValueError(f"document {row['id']}: page_offsets end at {end} but text holds {len(text)} characters")
    if offsets[-1] > len(text):
        # normalize's whitespace-run capping shrank text on a few documents without updating
        # page_offsets. The drift equals the characters removed (tens, against ~2000-char pages),
        # so clamped slicing shifts page boundaries immaterially for page-averaged LID; exact
        # page recovery is impossible after the mutation, and this is a label-only step, so
        # degrading beats dropping. Trailing pages clamped to empty gate out but still count in
        # the averaging denominator, like any other gated page.
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/docs_stale_offsets", 1)
        offsets = [min(offset, len(text)) for offset in offsets]

    pages = [clean_page(page) for page in split_pages(text, offsets)]
    scores = page_scores(pages, model)
    bucket, best_score = select_bucket(scores.averages, thresholds)

    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/docs_labeled", 1)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/pages_seen", len(pages))
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/pages_gated", scores.gated)
    if bucket.endswith("_removed"):
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/docs_removed_bucket", 1)
    if bucket == "unknown":
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/docs_unknown", 1)

    return {**row, "language": bucket, "language_score": best_score}


@cache
def load_lid_model(model_path: str) -> LidModel:
    """Copy the staged GlotLID model to local disk and load it, once per worker process.

    fasttext only loads from a local filesystem path, and the file is ~1.7 GB, so both the copy
    and the load are cached for the life of the process. The LID stage runs under ``InlineRunner``
    precisely so this cache survives across the shards a worker handles.
    """
    local_path = os.path.join(tempfile.mkdtemp(prefix="glotlid-"), GLOTLID_FILENAME)
    StoragePath(model_path).download_to(local_path)
    # fasttext's own signature is untyped (``text``, ndarray scores); the list path satisfies the
    # protocol at runtime.
    return cast(LidModel, fasttext.load_model(local_path))


def label_batch(batch: pa.RecordBatch, model_path: str, thresholds: dict[str, float]) -> Iterator[dict]:
    """Label one Parquet row group. Pure map: no drops, no reordering."""
    model = load_lid_model(model_path)
    for row in batch.to_pylist():
        yield label_document(row, model, thresholds)


def label_output_schema(input_schema: pa.Schema) -> pa.Schema:
    """The input schema plus the two label columns this step appends."""
    return input_schema.append(pa.field("language", pa.string(), nullable=False)).append(
        pa.field("language_score", pa.float32(), nullable=False)
    )


def _sha256_of_file(path: str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_glotlid_model(output_path: str) -> OcrModelData:
    """Stage GlotLID v3 from Hugging Face into the marin prefix, refusing a hash mismatch."""
    local_path = hf_hub_download(GLOTLID_REPO, GLOTLID_FILENAME, revision=GLOTLID_REVISION)
    digest = _sha256_of_file(local_path)
    if digest != GLOTLID_SHA256:
        raise ValueError(
            f"{GLOTLID_REPO}/{GLOTLID_FILENAME}@{GLOTLID_REVISION} hashed to {digest}, expected {GLOTLID_SHA256}"
        )

    model_path = prefix_join(output_path, GLOTLID_FILENAME)
    StoragePath(model_path).upload_from(local_path)
    logger.info("Staged GlotLID v3 (%d bytes) at %s", os.path.getsize(local_path), model_path)
    return OcrModelData(model_path=model_path, revision=GLOTLID_REVISION, sha256=digest)


def label_languages(
    output_path: str, clean_output_path: str, model_output_path: str, schema: pa.Schema
) -> NormalizedData:
    """Run the LID map over every fuzzy-clean shard and write the final labeled corpus."""
    clean = read_artifact(clean_output_path, NormalizedData)
    model = read_artifact(model_output_path, OcrModelData)
    thresholds = load_thresholds()
    logger.info(
        "Labeling %s with %s (%s) under %d language thresholds",
        clean.main_output_dir,
        model.model_path,
        model.revision,
        len(thresholds),
    )

    filesystem, path = url_to_fs(clean.main_output_dir)
    shards = sorted(filesystem.unstrip_protocol(shard) for shard in filesystem.glob(f"{path}/*.parquet"))
    if not shards:
        raise RuntimeError(f"No fuzzy-clean shards under {clean.main_output_dir}")

    main_output_dir = prefix_join(output_path, "outputs/main")
    pipeline = (
        Dataset.from_list(shards)
        .load_parquet(batch_mode=True)
        .flat_map(partial(label_batch, model_path=model.model_path, thresholds=thresholds))
        .write_parquet(
            prefix_join(main_output_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=schema,
            skip_existing=True,
        )
    )
    with ZephyrContext(
        name="focus-crawl-pdf-lid",
        resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=InlineRunner,
        coordinator_resources=_COORDINATOR_RESOURCES,
    ) as pool:
        outcome = pool.execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)

    return NormalizedData(
        main_output_dir=main_output_dir,
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=dict(outcome.counters),
    )


def glotlid_model_step() -> StepSpec:
    """Build the step that stages the GlotLID v3 model, pinned by revision and content hash."""
    return StepSpec(
        name="data/datakit/model/glotlid_v3",
        hash_attrs={
            "repo": GLOTLID_REPO,
            "filename": GLOTLID_FILENAME,
            "revision": GLOTLID_REVISION,
            "sha256": GLOTLID_SHA256,
        },
        fn=remote(stage_glotlid_model, resources=_MODEL_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def language_label_step(clean: StepSpec, model: StepSpec, input_schema: pa.Schema) -> StepSpec:
    """Build the LID labeling step over the fuzzy-clean corpus -- the pipeline's final dataset."""
    return StepSpec(
        name="data/datakit/final/common_crawl_focus_2026_22_pdf_ocr_all",
        deps=[clean, model],
        hash_attrs={
            "model_sha256": GLOTLID_SHA256,
            "thresholds_sha256": _thresholds_sha256(),
            "threshold_floor": THRESHOLD_FLOOR,
            "language_threshold": LANGUAGE_THRESHOLD,
            "min_alpha_length_bytes": MIN_ALPHA_LENGTH_BYTES,
            "min_alpha_ratio": MIN_ALPHA_RATIO,
            "predict_k": PREDICT_TOP_K,
            "table_lines_removed": TABLE_LINES_REMOVED,
            "zxx_overrides": _ZXX_OVERRIDES,
            "unselectable_threshold": UNSELECTABLE_THRESHOLD,
            "schema_version": 1,
        },
        fn=remote(
            partial(
                label_languages,
                clean_output_path=clean.output_path,
                model_output_path=model.output_path,
                schema=label_output_schema(input_schema),
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
