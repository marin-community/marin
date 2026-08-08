# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score every source of a Datakit sample tree with a fast-transformer quality model.

Runs the reference pipeline's quality stage
(:func:`~experiments.datakit.cluster.quality.fast_transformer.score.score_normalized`)
over a standalone sample tree instead of the full source registry, so a candidate
scorer can be evaluated on a fixed corpus without re-driving the whole DAG.

One step per source, each writing ``{output_prefix}/{source}`` with the same
``outputs/main`` + ``outputs/samples`` layout the pipeline's ``datakit/quality/<source>``
step produces. Steps are cached by ``.executor_status``, so a re-run resumes.

``--quality-model-version`` is the model's identity tag, hashed in place of the
region-specific model dir exactly as the reference pipeline does, so two scorers
never collide in one output tree.

Submit on iris::

    uv run iris --cluster=cw-us-east-02a job run --priority batch --cpu 2 --memory 8GB \\
        --enable-extra-resources -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.scripts.score_sample_quality \\
            --source-prefix s3://marin-us-east-02a/marin/user/rav/sample_10pct_91269634_50M \\
            --output-prefix s3://marin-us-east-02a/marin/user/rav/sample_10pct_91269634_50M_quality \\
            --quality-model-version pooled-junkgate2
"""

import argparse
import logging

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.score import score_normalized
from experiments.datakit.reference_pipeline import DEFAULT_SCALE, DRIVER_RESOURCES, quality_model_path, sample_sources

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True, help="sample tree whose sources are scored")
    parser.add_argument("--output-prefix", required=True, help="root the per-source scores are written under")
    parser.add_argument("--quality-model", default=None, help="scorer + calib dir (default: the deployed model)")
    parser.add_argument(
        "--quality-model-version",
        required=True,
        help="Stable identity tag for --quality-model (e.g. 'pooled-junkgate2'), hashed in its place.",
    )
    parser.add_argument("--sources", help="Comma-separated source names (default: every source in --source-prefix).")
    parser.add_argument("--pool-workers", type=int, default=None, help="Zephyr worker count (override scale)")
    parser.add_argument("--max-concurrent", type=int, default=16, metavar="N", help="max steps StepRunner runs at once")
    parser.add_argument("--dry-run", action="store_true", help="List the steps and exit without scoring.")
    return parser.parse_args(argv)


def build_steps(
    *,
    source_prefix: str,
    output_prefix: str,
    quality_model: str,
    quality_model_version: str,
    names: list[str] | None = None,
    pool_workers: int | None = None,
) -> list[StepSpec]:
    """One quality step per source in ``source_prefix``."""
    sources = sample_sources(source_prefix, names)
    out_root = output_prefix.rstrip("/")
    max_workers = pool_workers if pool_workers is not None else DEFAULT_SCALE.pool.n_workers
    logger.info(
        "score_sample_quality: %d sources -> %s (model_version=%s, workers=%d)",
        len(sources),
        out_root,
        quality_model_version,
        max_workers,
    )
    return [
        StepSpec(
            name=f"{out_root.rsplit('/', 1)[-1]}/{name}",
            deps=[step],
            hash_attrs={"model_version": quality_model_version, "v": 1},
            override_output_path=f"{out_root}/{name}",
            fn=remote(
                lambda output_path, np=step.output_path, src=name: score_normalized(
                    output_path=output_path,
                    normalized=read_artifact(np, NormalizedData),
                    source=src,
                    model_dir=quality_model,
                    max_workers=max_workers,
                    worker_resources=DEFAULT_SCALE.pool.worker,
                ),
                resources=DRIVER_RESOURCES,
            ),
        )
        for name, step in sorted(sources.items())
    ]


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    names = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    steps = build_steps(
        source_prefix=args.source_prefix,
        output_prefix=args.output_prefix,
        quality_model=args.quality_model or quality_model_path(),
        quality_model_version=args.quality_model_version,
        names=names,
        pool_workers=args.pool_workers,
    )
    if args.dry_run:
        for s in steps:
            logger.info("  %s -> %s", s.name, s.output_path)
        logger.info("score_sample_quality: dry run, %d steps not submitted", len(steps))
        return
    StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
