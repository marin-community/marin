# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the lexical text population for one published native conversion release.

The lexical stage consumes rendered text, not release parquets: this stage
streams the release's published task bytes, deduplicates exact contracts, and
writes the text/spec partitions the shared bloom measures, recording the
SourceTextPopulation the scan's populations manifest carries. The release pin
stays bound on the record (``dataset_id``/``dataset_revision``); the render
changes no task or runtime bytes.
"""

import argparse
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population
from experiments.post_training.task_curation.lexical_scan import population_key

LOGGER = logging.getLogger(__name__)


def published_rows(tasks_path: str) -> list[tuple[dict, dict]]:
    """Every published row as (task document, runtime allocation).

    The published bytes must be their own canonical serialization: a changed
    serialization would desynchronize the population's spec digests from the
    release the conversion sealed.
    """
    rows: list[tuple[dict, dict]] = []
    with StoragePath(tasks_path).open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, columns=["task_spec", "runtime"]):
            for row in batch.to_pylist():
                encoded = row["task_spec"]
                task = TaskSpec.model_validate_json(encoded)
                if task.model_dump_json() != encoded:
                    raise ValueError(f"Published task bytes are not their canonical serialization: {task.id}")
                runtime = row.get("runtime")
                if isinstance(runtime, str):
                    runtime = json.loads(runtime) if runtime else {}
                rows.append((json.loads(encoded), runtime or {}))
    return rows


def _release_tasks_path(release_repo: str, release_revision: str) -> str:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return hf_hub_download(release_repo, "tasks.parquet", repo_type="dataset", revision=release_revision)


def render_population(
    source_id: str,
    release_repo: str,
    release_revision: str,
    *,
    source_version: str,
    converter_revision: str,
    output_root: str,
) -> SourceTextPopulation:
    """Render the release's public text and record the lexical input population."""
    tasks_path = _release_tasks_path(release_repo, release_revision)
    rows = published_rows(tasks_path)
    if not rows:
        raise ValueError(f"The release population is empty: {release_repo}@{release_revision[:12]}")
    files = write_text_population(rows, output_root)
    LOGGER.info(
        "population rendered: %d unique of %d raw (%d duplicates), text_sha256 %s",
        files.task_count,
        files.raw_task_count,
        files.within_source_duplicates,
        files.text_sha256,
    )
    return SourceTextPopulation(
        path=output_root,
        source_id=source_id,
        dataset_id=release_repo,
        dataset_revision=release_revision,
        source_version=source_version,
        converter_revision=converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--release-repo", required=True)
    parser.add_argument("--release-revision", required=True, help="The published release pin (sha).")
    parser.add_argument("--population-version", required=True, help="The staged population version.")
    parser.add_argument("--converter-revision", required=True, help="The conversion's landed revision.")
    parser.add_argument("--source-version", default="harbor-cohort-taskspec-v1")
    parser.add_argument("--output", type=Path, required=True, help="Write the SourceTextPopulation record JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

    configure_coreweave_s3()

    root = (
        "s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text/"
        f"{population_key(args.source_id)}/{args.population_version}"
    )
    population = render_population(
        args.source_id,
        args.release_repo,
        args.release_revision,
        source_version=args.source_version,
        converter_revision=args.converter_revision,
        output_root=root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(population.model_dump(mode="json"), sort_keys=True) + "\n")
    print(f"population record: {args.output} at {root}")


if __name__ == "__main__":
    main()
