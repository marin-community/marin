# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality-label the clean all-routes OCR corpus and drop the documents with no readable window.

Two steps. :func:`scorer_model_step` stages the pooled fast-transformer and its calibration into
the marin prefix, every file pinned by content hash so a run's scores are attributable to exact
weights. :func:`quality_label_step` maps over the clean corpus
(:mod:`~experiments.build_pdf_source.repair_ocr_all`'s final output), scores every document, and
keeps only those with at least one window at or above :data:`EDU_KEEP_THRESHOLD`.

**Windows are scored individually, not mean-pooled.** The deployed ``score_bme`` convention cuts a
document into begin/middle/end ~2,000-character windows (one window when the whole text fits in
one), but averages them -- which is right for ranking and wrong for a drop gate: a scanned book
whose front matter is OCR noise but whose body is a textbook would average its way under the floor.
This step keeps all three per-window scores as columns (``edu_begin``/``edu_middle``/``edu_end``,
equal on the short-doc path) and gates on ``edu_max``, so a document is dropped only when *no*
window shows educational content. The 2,000-character cut itself is replicated exactly -- the
calibration was fit on scores produced by it, so a different cut would invalidate the calibration.

**The 0-4 scale exists only through the calibration.** The scorer's raw output is a sigmoid in
[0, 1]; ``calib_bme.json``'s knots map it onto a quality fraction, and times four gives the
FineWeb-Edu-style 0-4 scale. ``edu_max < 1.0`` is therefore "best window calibrates below 0.25".

The output is co-partitioned 1:1 with the input: each of the 23 shards maps to one output shard
under the same basename, and filtering preserves the input's ``id`` sort, so no re-shard or re-sort
happens here. Inference is CPU JAX -- the model is 37M parameters and jits once for its fixed
512-token window -- so no GPUs are requested anywhere.
"""

import functools
import hashlib
import json
import logging
import posixpath
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath, marin_prefix, open_url, prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner
from zephyr.writers import write_parquet_file

from experiments.build_pdf_source.common import OcrModelData
from experiments.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS
from experiments.build_pdf_source.extract_ocr import _OCR_FIELDS
from experiments.datakit.cluster.quality.fast_transformer.score import MODEL_CALIB
from experiments.datakit.cluster.quality.fast_transformer.scorer import (
    CHUNK_CHARS,
    MODEL_EQX,
    MODEL_META,
    MODEL_REMAP,
    PooledScorer,
    load_pooled_scorer,
)

logger = logging.getLogger(__name__)

_COUNTER_PREFIX = "focus_crawl_pdf_quality"
_CORPUS = "common_crawl_focus_2026_22_pdf_ocr_all"

# The clean corpus, by resolved output path. Named literally rather than rebuilt through
# repair_ocr_all's specs for the reason its _PARTITION_DIRS are: reconstructing the upstream chain
# would present any drift in it as a cache miss over a corpus that is content-addressed and done.
CLEAN_CORPUS_DIR = "data/datakit/clean/common_crawl_focus_2026_22_pdf_ocr_all_6357923a"

# The pooled fast-transformer trained on the 100k oracle sample, plus the bme calibration fit on
# its scores. Pinned by hash rather than by path; regenerate the model, don't edit these.
SCORER_SOURCE_DIR = "s3://marin-us-east-02a/marin/data/pdf_quality/scorer_100k/model"
SCORER_FILE_SHA256: dict[str, str] = {
    MODEL_EQX: "e594dc684508154e9fe55150be3024411e3d46715fba8f38e6db02f5aaeca04f",
    MODEL_REMAP: "37316b250610d13b0c8b271106a7da9fe402ec5ec1d385c8551168af1880068c",
    MODEL_META: "8c457add304b8527870b5318670c62333108816eb48418e2e3ad6ff49e0ecd1f",
    MODEL_CALIB: "9f83a52f828e295f4c7c28405e03c893c2b79565b0a83c4f110f1f8eacba98dd",
}
SCORER_EQX_SHA256 = SCORER_FILE_SHA256[MODEL_EQX]

EDU_SCALE = 4.0  # the calibrated quality fraction in [0, 1], expressed on the FineWeb-Edu 0-4 scale
EDU_KEEP_THRESHOLD = 1.0  # a document lives iff its best window reaches this on the 0-4 scale
# The model's inference batch, in windows. predict() pads every chunk to a constant
# 262144-token budget -- 512 rows at the scorer's fixed T=512 -- so any smaller sub-batch
# executes the same jitted forward with the difference as padding; 512 fills it exactly.
_WINDOW_BATCH = 512
# Documents per scoring call; at least one window each, so this keeps every full
# _WINDOW_BATCH sub-batch fed even when a whole batch of documents is short.
_BATCH_DOCS = 512

QUALITY_FIELDS: tuple[pa.Field, ...] = (
    pa.field("edu_begin", pa.float32(), nullable=False),
    pa.field("edu_middle", pa.float32(), nullable=False),
    pa.field("edu_end", pa.float32(), nullable=False),
    # max of the three, and the filter key. All four are calibrated onto the 0-4 scale; a document
    # short enough for one window carries that window's score in all of them.
    pa.field("edu_max", pa.float32(), nullable=False),
)
QUALITY_SCHEMA = pa.schema([*PDF_DOCUMENT_FIELDS, *_OCR_FIELDS, *QUALITY_FIELDS])

_MODEL_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="2g")
_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
# 4g OOM'd scoring workers in prior runs; 24g holds a whole ~270 MB shard's rows plus JAX
# activations with margin. The sixteen CPUs feed jit'd CPU matmuls, not parallel tasks.
_WORKER_RESOURCES = ResourceConfig(cpu=16, ram="24g", disk="8g")
# One task per input shard; more workers than shards would queue for capacity the stage cannot use.
_MAX_WORKERS = 23
# Never Zephyr's 1 GB default: it is OOM-killed (exit 137) at run end on every stage of this
# pipeline family, after the work is already on disk.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)


@dataclass(frozen=True)
class DocumentScores:
    """One document's calibrated 0-4 window scores."""

    edu_begin: float
    edu_middle: float
    edu_end: float
    edu_max: float


def stage_scorer_model(output_path: str) -> OcrModelData:
    """Copy the pinned scorer artifacts into the step's output prefix, refusing any failed hash."""
    for filename, expected in SCORER_FILE_SHA256.items():
        payload = StoragePath(prefix_join(SCORER_SOURCE_DIR, filename)).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(f"{SCORER_SOURCE_DIR}/{filename} hashed to {digest}, expected {expected}")
        StoragePath(prefix_join(output_path, filename)).write_bytes(payload)
        logger.info("Staged %s (%d bytes) under %s", filename, len(payload), output_path)
    return OcrModelData(model_path=output_path, revision=SCORER_EQX_SHA256[:12], sha256=SCORER_EQX_SHA256)


def cut_windows(text: str) -> list[str]:
    """Cut a document into the deployed begin/middle/end scoring windows.

    Replicates ``score_bme``'s convention exactly -- one window when the text fits in
    :data:`~experiments.datakit.cluster.quality.fast_transformer.scorer.CHUNK_CHARS`, else three --
    but returns the windows instead of mean-pooling their scores, so the per-window columns and the
    ``edu_max`` gate can exist.
    """
    if len(text) <= CHUNK_CHARS:
        return [text]
    middle = len(text) // 2
    half = CHUNK_CHARS // 2
    return [text[:CHUNK_CHARS], text[middle - half : middle + half], text[-CHUNK_CHARS:]]


def calibrate(raw: np.ndarray, xk: np.ndarray, yk: np.ndarray) -> np.ndarray:
    """Map raw sigmoid scores in [0, 1] onto the 0-4 scale through the calibration knots."""
    return np.asarray(np.interp(raw, xk, yk)) * EDU_SCALE


def keep_document(scores: DocumentScores) -> bool:
    """Whether a document survives the quality gate: its best window reaches the threshold."""
    return scores.edu_max >= EDU_KEEP_THRESHOLD


def score_texts(texts: list[str], scorer: PooledScorer, xk: np.ndarray, yk: np.ndarray) -> list[DocumentScores]:
    """Score a batch of documents, one scorer call over their flattened windows.

    A short document's single window fills all three columns, so ``edu_begin == edu_middle ==
    edu_end`` marks the one-window path in the output.
    """
    windows = [cut_windows(text) for text in texts]
    flat = [window for cut in windows for window in cut]
    calibrated = calibrate(scorer.score(flat, batch_size=_WINDOW_BATCH), xk, yk)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/windows_scored", len(flat))

    scores: list[DocumentScores] = []
    cursor = 0
    for cut in windows:
        values = calibrated[cursor : cursor + len(cut)]
        cursor += len(cut)
        if len(values) == 1:
            begin = middle = end = float(values[0])
        else:
            begin, middle, end = (float(value) for value in values)
        scores.append(DocumentScores(begin, middle, end, max(begin, middle, end)))
    return scores


@functools.cache
def _load_scorer(model_dir: str) -> tuple[PooledScorer, np.ndarray, np.ndarray]:
    """Load the scorer + calibration once per worker process.

    ``load_pooled_scorer`` streams the ``.eqx`` to a local tempfile itself, which eqx
    deserialisation requires.
    """
    scorer = load_pooled_scorer(model_dir)
    with open_url(prefix_join(model_dir, MODEL_CALIB), "r") as fh:
        calib = json.loads(fh.read())
    logger.info("Loaded FT scorer + calibration from %s", model_dir)
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def score_shard(input_file: str, output_dir: str, model_dir: str) -> dict:
    """Score one input shard and write its filtered counterpart under the same basename.

    The input is sorted by ``id`` and filtering preserves order, so the output is co-partitioned
    with the input by basename and stays id-sorted without a shuffle. A shard whose every document
    is dropped still writes its (empty) output file, keeping the 1:1 partition map intact.
    """
    scorer, xk, yk = _load_scorer(model_dir)
    filesystem, path = url_to_fs(input_file)
    with filesystem.open(path, "rb") as stream:
        rows = pq.read_table(stream).to_pylist()

    kept: list[dict] = []
    for start in range(0, len(rows), _BATCH_DOCS):
        batch = rows[start : start + _BATCH_DOCS]
        scores = score_texts([row["text"] for row in batch], scorer, xk, yk)
        for row, document in zip(batch, scores, strict=True):
            if keep_document(document):
                kept.append(
                    row
                    | {
                        "edu_begin": document.edu_begin,
                        "edu_middle": document.edu_middle,
                        "edu_end": document.edu_end,
                        "edu_max": document.edu_max,
                    }
                )

    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/docs_in", len(rows))
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/docs_kept", len(kept))
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/docs_dropped", len(rows) - len(kept))
    return write_parquet_file(kept, prefix_join(output_dir, posixpath.basename(input_file)), schema=QUALITY_SCHEMA)


def quality_label(output_path: str, model_output_path: str) -> NormalizedData:
    """Score the clean corpus and write the quality-filtered dataset to ``output_path``.

    Input shards whose output basename already exists are dropped up front (each output file is
    written atomically, so its presence means fully scored) -- a re-run after a partial failure
    only scores the remainder.
    """
    model = read_artifact(model_output_path, OcrModelData)
    input_dir = prefix_join(prefix_join(marin_prefix(), CLEAN_CORPUS_DIR), "outputs/main")
    filesystem, path = url_to_fs(input_dir)
    files = sorted(filesystem.unstrip_protocol(shard) for shard in filesystem.glob(f"{path}/*.parquet"))
    if not files:
        raise RuntimeError(f"No clean shards under {input_dir}")

    main_output_dir = prefix_join(output_path, "outputs/main")
    output_filesystem, output_dir_path = url_to_fs(main_output_dir)
    done = {posixpath.basename(shard) for shard in output_filesystem.glob(f"{output_dir_path}/*.parquet")}
    pending = [file for file in files if posixpath.basename(file) not in done]
    logger.info("Scoring %d/%d shards with %s (%s)", len(pending), len(files), model.model_path, model.revision)

    aggregated: dict[str, int | float] = {}
    if pending:
        pipeline = Dataset.from_list(pending).map(
            functools.partial(score_shard, output_dir=main_output_dir, model_dir=model.model_path)
        )
        # InlineRunner keeps the per-process cached model (and its jit cache) alive across a
        # worker's shards; SubprocessRunner would re-load and re-jit per task.
        outcome = ZephyrContext(
            name="focus-crawl-pdf-quality",
            resources=_WORKER_RESOURCES,
            max_workers=_MAX_WORKERS,
            stage_runner_factory=InlineRunner,
            coordinator_resources=_COORDINATOR_RESOURCES,
        ).execute(pipeline)
        aggregated = dict(outcome.counters)

    return NormalizedData(
        main_output_dir=main_output_dir,
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=aggregated,
    )


def scorer_model_step() -> StepSpec:
    """Build the step that stages the pooled quality scorer and its calibration."""
    return StepSpec(
        name="data/datakit/model/pdf_quality_scorer",
        # The full per-file hash dict, not just the weights: the calibration and remap determine
        # every score, so refitting calib_bme.json must re-key this step (and, through the dep
        # hash, the scoring step) rather than serve a stale staged copy.
        hash_attrs={"source": SCORER_SOURCE_DIR, "file_sha256": SCORER_FILE_SHA256},
        fn=remote(stage_scorer_model, resources=_MODEL_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def quality_label_step(model: StepSpec) -> StepSpec:
    """Build the scoring + filtering step over the pinned clean corpus."""
    return StepSpec(
        name=f"data/datakit/quality/{_CORPUS}",
        deps=[model],
        hash_attrs={
            "input": CLEAN_CORPUS_DIR,
            "model_sha256": SCORER_EQX_SHA256,
            "edu_keep_threshold": EDU_KEEP_THRESHOLD,
            "schema_version": 1,
        },
        fn=remote(
            functools.partial(quality_label, model_output_path=model.output_path),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
