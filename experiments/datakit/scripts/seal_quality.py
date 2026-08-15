# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mark the rescored quality stage succeeded, so ``StepRunner`` will serve it.

``score_corpus`` fans workers out over a manifest and writes score shards
straight into ``hero_data.quality``'s directory. It never goes through
``StepRunner``, so it never writes the ``.executor_status`` file the runner reads
to decide a step is built. The data is complete and the marker is not there.

That is not a harmless gap for the store. Its quality dependencies are hero
steps, which refuse to execute, so an unmarked stage does not get rebuilt -- it
takes the run down on the dependency. This writes the marker the scoring run
would have written, for the sources whose scores are actually complete.

Completeness is checked, not assumed: a source is sealed only when its score
directory carries a shard for every shard of the tokenize leaf the scorer read.
That is the same pairing ``score_corpus manifest`` refuses to build without, so
a short source is one whose run did not finish, and it is left unsealed.

    uv run python -m experiments.datakit.scripts.seal_quality --dry-run
    uv run python -m experiments.datakit.scripts.seal_quality

Run it in the CoreWeave data region with ``MARIN_PREFIX`` set to
``s3://marin-us-east-02a/marin``.
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from marin.execution.artifact import read_artifact
from marin.execution.step_status import STATUS_SUCCESS, get_status_path
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit import hero_data
from experiments.datakit.reference_pipeline import SPLIT

logger = logging.getLogger(__name__)


def _basenames(directory: str) -> set[str]:
    return {os.path.basename(str(match)) for match in StoragePath(f"{directory.rstrip('/')}/*.parquet").glob()}


def shortfall(source: str, quality_model: hero_data.QualityPin) -> tuple[str, set[str]]:
    """Return the source's score directory and the tokenize shards it is missing.

    Takes the tokenize directory from the artifact rather than appending the split
    to the step path, because that is where the store looks. The two agree today;
    reading the artifact is what keeps them agreeing.
    """
    quality_dir = hero_data.quality(source, quality_model).output_path
    tokenize = read_artifact(hero_data.tokenized(source, quality_model.tokenizer).output_path, TokenizedAttrData)
    tokenize_dir = tokenize.output_dirs.get(SPLIT)
    if tokenize_dir is None:
        raise KeyError(f"{source}: tokenize has no split={SPLIT!r}")
    return quality_dir, _basenames(tokenize_dir) - _basenames(quality_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", help="comma-separated source names; omit for every registered source")
    parser.add_argument("--dry-run", action="store_true", help="report what would be sealed and exit")
    args = parser.parse_args(argv)

    configure_logging(logging.INFO)
    configure_coreweave_s3()

    names = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else hero_data.source_names()
    # Listing two directories per source is the whole cost, and it is all latency.
    with ThreadPoolExecutor(max_workers=32) as pool:
        shortfalls = list(pool.map(lambda n: shortfall(n, hero_data.NEMOTRON_88K), names))

    sealed, already, short = 0, 0, 0
    for source, (quality_dir, missing) in zip(names, shortfalls, strict=True):
        status_path = get_status_path(quality_dir)
        if missing:
            short += 1
            logger.error("%s: %d tokenize shards have no score shard, e.g. %s", source, len(missing), sorted(missing)[0])
            continue
        if StoragePath(status_path).exists():
            already += 1
            continue
        if args.dry_run:
            logger.info("would seal %s", quality_dir)
        else:
            StoragePath(status_path).write_text(STATUS_SUCCESS)
        sealed += 1

    logger.info(
        "%d %s, %d already marked, %d incomplete, over %d sources",
        sealed,
        "would be sealed" if args.dry_run else "sealed",
        already,
        short,
        len(names),
    )
    if short:
        raise SystemExit(f"{short} sources have incomplete scores and were left unsealed")


if __name__ == "__main__":
    main()
