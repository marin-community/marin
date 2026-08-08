# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample an existing Datakit sample tree down to a target document count.

Reads a per-source sample tree (a ``sample_<size>_<hash>`` root whose sources are
:class:`~marin.datakit.normalize.NormalizedData` artifacts) and writes a smaller
tree holding about ``--target-docs`` documents, allocated proportionally to each
source's document count.

Proportional allocation is one uniform fraction. Source *i* keeps
``target_docs * docs_i / total_docs`` documents, so its fraction is
``target_docs / total_docs`` for every source. The per-source work is
:func:`~experiments.datakit.testbed.sampler.sample_normalized_shards`, which takes
whole shards server-side and row-samples one tail shard, so each output source
stays byte-fair and content-fair with its input.

Document counts come from each source's ``sampler/rows_out`` counter, so the input
must be a sampler output. Whole-shard counts there are estimates from a probed
rows-per-shard, so the emitted total lands near the target rather than on it.

Submit on iris::

    uv run iris --cluster=cw-us-east-02a job run --priority batch --cpu 2 --memory 8GB \\
        --enable-extra-resources -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.scripts.sample_by_doc_count \\
            --source-prefix s3://marin-us-east-02a/marin/datakit/sample_10pct_91269634 \\
            --target-docs 50000000 \\
            --output-prefix s3://marin-us-east-02a/marin/user/rav/sample_10pct_91269634_50M
"""

import argparse
import logging

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import sample_sources
from experiments.datakit.testbed.sampler import sample_normalized_shards_step

logger = logging.getLogger(__name__)

# Counter the upstream sampler writes with each source's emitted row count.
DOC_COUNTER = "sampler/rows_out"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True, help="existing sample tree root to sample from")
    parser.add_argument("--output-prefix", required=True, help="root the sampled per-source tree is written under")
    parser.add_argument("--target-docs", type=int, required=True, help="approximate total documents to keep")
    parser.add_argument(
        "--sources",
        help="Comma-separated source names (default: every source in --source-prefix).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-source allocation and exit without writing anything.",
    )
    parser.add_argument("--max-concurrent", type=int, default=16, metavar="N", help="max steps StepRunner runs at once")
    return parser.parse_args(argv)


def source_doc_counts(sources: dict[str, StepSpec]) -> dict[str, int]:
    """Document count per source, read from its ``sampler/rows_out`` counter."""
    counts = {}
    for name, step in sources.items():
        counters = read_artifact(step.output_path, NormalizedData).counters
        if DOC_COUNTER not in counters:
            raise ValueError(
                f"source {name!r} has no {DOC_COUNTER!r} counter at {step.output_path}; "
                "--source-prefix must name a sampler output tree"
            )
        counts[name] = int(counters[DOC_COUNTER])
    return counts


def build_steps(
    *,
    source_prefix: str,
    output_prefix: str,
    target_docs: int,
    names: list[str] | None = None,
) -> list[StepSpec]:
    """One sampler step per source, all sharing the ``target_docs / total_docs`` fraction."""
    sources = sample_sources(source_prefix, names)
    counts = source_doc_counts(sources)
    total_docs = sum(counts.values())
    if total_docs <= 0:
        raise ValueError(f"{source_prefix} reports {total_docs} documents; nothing to sample")

    fraction = min(1.0, target_docs / total_docs)
    out_root = output_prefix.rstrip("/")
    logger.info(
        "sample_by_doc_count: %d sources, %d documents, target %d -> fraction %.6f",
        len(sources),
        total_docs,
        target_docs,
        fraction,
    )
    for name in sorted(counts, key=lambda n: -counts[n]):
        logger.info("  %-60s %12d -> %10d", name, counts[name], round(counts[name] * fraction))

    return [
        sample_normalized_shards_step(
            name=f"{out_root.rsplit('/', 1)[-1]}/{name}",
            normalized=step,
            sample_fraction=fraction,
            override_output_path=f"{out_root}/{name}",
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
        target_docs=args.target_docs,
        names=names,
    )
    if args.dry_run:
        logger.info("sample_by_doc_count: dry run, %d steps not submitted", len(steps))
        return
    StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
