# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The quality step's output contract.

Deliberately import-light (pydantic only): the store step consumes
:class:`QualityScores` for its 5-way join and must not drag the jax/equinox/
transformers stack of the scorer into its workers.
"""

from pydantic import BaseModel

# Fixed score cutpoints. Calibration (calibrate.py) warps the raw score so these
# 0.2-wide buckets are quality-coherent across content types; `score.py` buckets
# with np.digitize against these edges, giving quality_bucket 0..len(BUCKET_EDGES).
BUCKET_EDGES = (0.2, 0.4, 0.6, 0.8)


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
    main_output_dir: str
    samples_output_dir: str
    model_dir: str
    calib_file: str
    bucket_edges: list[float]
    counters: dict[str, int | float]
