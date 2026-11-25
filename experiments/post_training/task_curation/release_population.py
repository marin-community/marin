# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a published cohort-conversion release as the standard lexical population.

The measurement chain (actor census, lexical decontamination, SAMPLE QC) reads a
cohort's population through the typed ``SourceTextPopulation`` writer. The
conversion releases carry the same content in the cohort carrier schema
(``task_spec`` rows) with their manifest binding the cohort pin, converter
revision and converted count; this module maps that layout into the population
writer the intake flow uses, and refuses a release whose rows disagree with its
own manifest.
"""

import argparse
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population

LOGGER = logging.getLogger(__name__)


def release_documents(release_repo: str, revision: str) -> list[dict]:
    """The release's converted TaskSpec documents, in carrier order."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    path = Path(hf_hub_download(release_repo, "tasks.parquet", repo_type="dataset", revision=revision))
    documents = []
    for row in pq.read_table(path, columns=["task_spec"]).to_pylist():
        documents.append(json.loads(row["task_spec"]))
    return documents


def stage_release_population(
    source_id: str,
    release_repo: str,
    revision: str,
    *,
    manifest: dict,
    output_path: str,
) -> SourceTextPopulation:
    """Write the release's lexical population and refuse a row-count disagreement."""
    manifest_path = Path(hf_hub_download(release_repo, "manifest.json", repo_type="dataset", revision=revision))
    if json.loads(manifest_path.read_bytes())["converted_rows"] != manifest["converted_rows"]:
        raise ValueError("The release manifest disagrees with the staged conversion record")
    documents = release_documents(release_repo, revision)
    if len(documents) != manifest["converted_rows"]:
        raise ValueError(f"The release rows differ from its manifest: {len(documents)} != {manifest['converted_rows']}")
    files = write_text_population((document, {}) for document in documents)
    if files.task_count != manifest["converted_rows"] or files.task_count == 0:
        raise ValueError("The staged population differs from its completed release manifest")
    return SourceTextPopulation(
        path=output_path,
        source_id=source_id,
        dataset_id=release_repo,
        dataset_revision=revision,
        source_version=manifest["protocol"],
        converter_revision=manifest.get("converter_revision", ""),
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
    parser.add_argument("--source-id", required=True, help="The cohort source id.")
    parser.add_argument("--release-repo", required=True, help="The published conversion release repo.")
    parser.add_argument("--release-revision", required=True, help="The release revision to stage.")
    parser.add_argument("--output", required=True, type=Path, help="The staged population output directory.")
    parser.add_argument("--report", type=Path, help="Write the staged population JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    manifest = json.loads(
        Path(
            hf_hub_download(args.release_repo, "manifest.json", repo_type="dataset", revision=args.release_revision)
        ).read_bytes()
    )
    population = stage_release_population(
        args.source_id, args.release_repo, args.release_revision, manifest=manifest, output_path=str(args.output)
    )
    payload = json.dumps(population.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "population.json").write_text(payload)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
