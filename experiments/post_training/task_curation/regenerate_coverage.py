# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ``source_coverage.json`` from curriculum releases.

    uv run python -m experiments.post_training.task_curation.regenerate_coverage \\
        --release s3://.../publication-*/VERSION/release.json --coverage s3://.../coverage-*/VERSION/coverage.json \\
        --release ... --coverage ...

Releases apply in order; a later release replaces an earlier one's values for the
sources it matched. Sources no release matched take their pinned declared count and
a recorded gap.
"""

import json
from pathlib import Path

import click
import pyarrow.parquet as pq
from finestore.rl.curation import SourceCoverage
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.curriculum.models import CurriculumCoverage

from experiments.post_training.task_curation.source_coverage import COVERAGE_TABLE, declared_coverage, release_coverage
from experiments.post_training.task_curation.sources import declared_sources


@click.command()
@click.option("--release", "releases", multiple=True, required=True, help="Curriculum release.json, in order.")
@click.option("--coverage", "coverages", multiple=True, required=True, help="Its coverage.json, in the same order.")
@click.option("--output", type=click.Path(path_type=Path), default=COVERAGE_TABLE, show_default=True)
def main(releases: tuple[str, ...], coverages: tuple[str, ...], output: Path) -> None:
    if len(releases) != len(coverages):
        raise click.UsageError("Pass one --coverage per --release")
    configure_coreweave_s3()
    measured: dict[str, SourceCoverage] = {}
    for release_uri, coverage_uri in zip(releases, coverages, strict=True):
        release = json.loads(StoragePath(release_uri).read_text())
        coverage = CurriculumCoverage.model_validate_json(StoragePath(coverage_uri).read_text())
        if coverage.alignment_release_id != release["release_id"]:
            raise click.UsageError(f"{coverage_uri} does not describe {release_uri}")
        statistics = []
        for path in coverage.paths:
            with StoragePath(path).open("rb") as handle:
                statistics.extend(pq.read_table(handle).to_pylist())
        measured.update(release_coverage(release, release_uri, statistics))
    table = {
        source_id: measured.get(source_id) or declared_coverage(source.declaration)
        for source_id, source in sorted(declared_sources().items())
    }
    output.write_text(
        json.dumps({key: value.model_dump(mode="json") for key, value in table.items()}, indent=1) + "\n"
    )
    click.echo(json.dumps({"sources": len(table), "measured": sum(key in measured for key in table)}))


if __name__ == "__main__":
    main()
