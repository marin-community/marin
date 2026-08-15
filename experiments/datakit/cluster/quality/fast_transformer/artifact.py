# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The quality step's output contract.

Deliberately import-light (pydantic only): the store step consumes
:class:`QualityScores` for its attribute join and must not drag the jax/equinox/
transformers stack of the scorer into its workers.
"""

import hashlib
import json

from marin.datakit.source_key import DatakitArtifactPath
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath

# Fixed score cutpoints. Calibration (calibrate.py) warps the raw score so these
# 0.2-wide buckets are quality-coherent across content types; `score.py` buckets
# with np.digitize against these edges, giving quality_bucket 0..len(BUCKET_EDGES).
BUCKET_EDGES = (0.2, 0.4, 0.6, 0.8)


DEFAULT_CALIBRATION_KEY = "default"
"""The calibration fitted over the whole labelled set, ignoring content type."""


def _interior_knots(entry: dict, where: str) -> tuple[float, ...]:
    """Return one calibration's interior knots, checked against the bucket scheme."""
    edges = tuple(float(k) for k in entry["xk"][1:-1])
    if len(edges) != len(BUCKET_EDGES):
        raise ValueError(
            f"{where} has {len(edges)} interior knots but the bucket scheme "
            f"has {len(BUCKET_EDGES)} cutpoints {BUCKET_EDGES}"
        )
    return edges


def _verified_calibration(calibration_path: str, expected_sha256: str) -> dict:
    """Load a calibration, refusing bytes the pin does not vouch for.

    The path that leads here is a constant somebody wrote down, so it is checked
    against the pin rather than trusted: scoring already refuses a model directory
    that digests to the wrong value, and reading the cutpoints is the same claim
    made on the read side.
    """
    blob = StoragePath(calibration_path).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"{calibration_path} digests to {digest}, but the pin carries {expected_sha256}; "
            "its cutpoints would bucket the corpus under a calibration nothing verified"
        )
    return json.loads(blob)


def calibration_bucket_edges(calibration_path: str, *, expected_sha256: str) -> tuple[float, ...]:
    """Return the cutpoints a calibration puts at :data:`BUCKET_EDGES`, in raw score.

    Calibration fits a monotone spline through knots, and the interior knots are
    exactly the raw scores that the spline sends to 0.2, 0.4, 0.6 and 0.8. Cutting
    raw scores there is the same partition as calibrating first and cutting at
    :data:`BUCKET_EDGES`, with one fewer pass over 18.7 billion documents. The
    outer two knots bound the spline's domain rather than dividing it.

    These are the content-type-blind cutpoints. See
    :func:`calibration_edges_by_content_type` for the per-type ones, which is what
    a corpus spanning code, math and prose actually wants.
    """
    blob = _verified_calibration(calibration_path, expected_sha256)
    return _interior_knots(blob[DEFAULT_CALIBRATION_KEY], calibration_path)


def calibration_edges_by_content_type(calibration_path: str, *, expected_sha256: str) -> dict[str, tuple[float, ...]]:
    """Return per-content-type cutpoints, keyed by type, plus ``default``.

    One score means different things in different content. The scorer is fitted
    on a labelled set spanning code, math, prose and the rest, and a single set of
    cutpoints reads that mixture through one lens: measured on this corpus, the
    content-blind edges put 36% of math documents and 24% of code documents in a
    different bucket than their own calibration does, consistently promoting them.

    ``default`` is what a document gets when its content type has no calibration
    of its own -- ``other``, and anything the classifier learns to emit later.
    Keeping it in the same mapping means the caller never has to decide what a
    missing type means.
    """
    blob = _verified_calibration(calibration_path, expected_sha256)
    edges = {DEFAULT_CALIBRATION_KEY: _interior_knots(blob[DEFAULT_CALIBRATION_KEY], calibration_path)}
    for content_type, entry in blob.get("types", {}).items():
        edges[content_type] = _interior_knots(entry, f"{calibration_path} types/{content_type}")
    return edges


class QualityScores(BaseModel):
    """Outcome of :func:`score.score_normalized`: calibrated quality scores for one source.

    Persisted as the step's ``.artifact``. Load via
    ``read_artifact(step.output_path, QualityScores)``.

    Attributes:
        main_output_dir: Directory of lean scored parquet
            (``source``/``id``/``score``/``quality_bucket``), one file per input
            shard, co-partitioned with the source ``NormalizedData`` by basename
            and row order.
        samples_output_dir: Directory of the ~``sample_pct`` systematic sample
            side output (same columns plus truncated ``text``) the stage report
            reads for spot-checks.
        model_dir: Scorer artifacts + calibration json used. Model dirs are
            immutable by convention -- the step hash covers the *path*, not the
            bytes, so retrained models must land in new dirs.
        calib_file: Calibration json name inside ``model_dir``.
        bucket_edges: Score cutpoints behind ``quality_bucket``; the store joins
            on the bucket column and records these in its own artifact.
        counters: Aggregated zephyr counters from the scoring pipeline.
    """

    version: str = "v1"
    main_output_dir: DatakitArtifactPath
    samples_output_dir: DatakitArtifactPath
    model_dir: DatakitArtifactPath
    calib_file: str
    bucket_edges: list[float]
    counters: dict[str, int | float]
