# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score every source of a clustered sample tree with a quality model.

A clustered sample (``datakit/samples/<name>/<source>/*.parquet``) is a plain tree:
no per-source ``.artifact.json``, and each source keeps its own extra columns. So
sources are discovered by listing the directories that directly hold parquet, and
each is wrapped in a :class:`~marin.datakit.normalize.NormalizedData` for the
deployed scoring path — which only reads ``id`` and ``text`` and is indifferent to
the rest of a source's schema.

One step per source rather than one over the tree, for two reasons: the scorer globs
a single directory level by design (a recursive glob over a large tree makes s3fs
pathological), and a step is cached by its ``.executor_status``, so a run over
hundreds of shards resumes instead of restarting.

Submit on iris::

    uv run iris --cluster=cw-us-east-02a job run --priority batch --cpu 2 --memory 8GB \\
        --enable-extra-resources -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.scripts.score_clustered_sample \\
            --sample-root s3://.../datakit/samples/harrier-oss-v1-0.6b-50m-text-v1 \\
            --out-prefix s3://.../user/rav/quality_v2/harrier_scored_v1 \\
            --quality-model s3://.../quality_v2/models/pooled_glm52_v1 \\
            --model-version glm52-v1
"""

import argparse
import logging
import posixpath

from marin.datakit.normalize import NormalizedData
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.score import score_normalized
from experiments.datakit.reference_pipeline import DEFAULT_SCALE, DRIVER_RESOURCES, quality_model_path

logger = logging.getLogger(__name__)


def discover_sources(sample_root: str) -> list[str]:
    """Source paths relative to ``sample_root``: every directory holding parquet."""
    root = sample_root.rstrip("/")
    shards = [str(m) for m in StoragePath(f"{root}/**/*.parquet").glob()]
    sources = sorted({posixpath.dirname(s)[len(root) + 1 :] for s in shards})
    logger.info("score_clustered_sample: %d sources over %d shards", len(sources), len(shards))
    return sources


def build_steps(
    *, sample_root: str, out_prefix: str, quality_model: str, model_version: str, pool_workers: int | None
) -> list[StepSpec]:
    root = sample_root.rstrip("/")
    out_root = out_prefix.rstrip("/")
    max_workers = pool_workers if pool_workers is not None else DEFAULT_SCALE.pool.n_workers
    return [
        StepSpec(
            name=f"{out_root.rsplit('/', 1)[-1]}/{source}",
            hash_attrs={"model_version": model_version, "v": 1},
            override_output_path=f"{out_root}/{source}",
            fn=remote(
                lambda output_path, src=source: score_normalized(
                    output_path=output_path,
                    normalized=NormalizedData(main_output_dir=f"{root}/{src}", dup_output_dir="", counters={}),
                    source=src,
                    model_dir=quality_model,
                    max_workers=max_workers,
                    worker_resources=DEFAULT_SCALE.pool.worker,
                ),
                resources=DRIVER_RESOURCES,
            ),
        )
        for source in discover_sources(sample_root)
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", required=True, help="clustered sample tree")
    parser.add_argument("--out-prefix", required=True, help="root the per-source scores are written under")
    parser.add_argument("--quality-model", default=None, help="scorer + calib dir (default: the deployed model)")
    parser.add_argument("--model-version", required=True, help="identity tag for the scorer, hashed into the step")
    parser.add_argument("--pool-workers", type=int, default=None, help="Zephyr worker count per source")
    parser.add_argument("--max-concurrent", type=int, default=16, metavar="N", help="max steps run at once")
    parser.add_argument("--dry-run", action="store_true", help="List the steps and exit.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    steps = build_steps(
        sample_root=args.sample_root,
        out_prefix=args.out_prefix,
        quality_model=args.quality_model or quality_model_path(),
        model_version=args.model_version,
        pool_workers=args.pool_workers,
    )
    if args.dry_run:
        for step in steps[:10]:
            logger.info("  %s -> %s", step.name, step.output_path)
        logger.info("score_clustered_sample: dry run, %d steps not submitted", len(steps))
        return
    StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
