# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the declared forge cohort's declaration-phase Atlas receipts.

One run over the pinned published cohort produces every measurement the source
inventory needs before admission receipts exist:

1. renders the exact-contract text population (``write_text_population``, which
   also measures within-source duplicates and writes their drop manifest);
2. writes the within-source dedup receipt the population build binds
   (``WithinSourceDedupEvidence`` citing the declared pin and converter);
3. copies the published cohort next to the measurements as the durable
   fingerprint input;
4. fingerprints the cohort (``source_duplicates.fingerprint_manifest``) so the
   overlap gate's index can carry the source.

The declaration in ``datasets/campaign_releases.py`` is the single source of
truth: the pin, converter version and source identity are read from it, never
restated here. Run with CoreWeave and Hugging Face credentials exported::

    uv run python -m experiments.post_training.task_curation.forge_census \
        --output-root s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/forge-main/<version>/census-v1 \
        --converter-revision <git-sha-of-this-checkout>
"""

import argparse
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.curation import WithinSourceDedupEvidence
from huggingface_hub import hf_hub_download
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.campaign_releases import sources as campaign_sources
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import RENDERER_VERSION, write_text_population
from experiments.post_training.task_curation.lowered_envelope import LOWERED_TASK_SPEC_COLUMN
from experiments.post_training.task_curation.source_duplicates import fingerprint_manifest

logger = logging.getLogger(__name__)

FORGE_SOURCE_ID = "Marin:forge-main-20261117"
COHORT_FILENAME = "tasks.parquet"
FINGERPRINT_ARTIFACT = "fingerprint/source-identities.json"
DEDUP_SUMMARY = "dedup/summary.json"


def forge_declaration() -> tuple[str, str, str, str, int]:
    """The declared source id, cohort pin, converter version and expected rows."""
    for source in campaign_sources():
        if source.info.id == FORGE_SOURCE_ID:
            config = source.config
            dataset = source.dataset
            assert dataset is not None and config is not None
            return source.info.id, dataset.name, dataset.revision, source.version, config.expected_rows
    raise ValueError(f"Campaign releases do not declare {FORGE_SOURCE_ID}")


def cohort_documents(local: Path) -> tuple[str, str, list[tuple[dict, dict]]]:
    """The published cohort's task specs, as population input documents."""
    table = pq.read_table(local, columns=[LOWERED_TASK_SPEC_COLUMN])
    documents = [(json.loads(row[LOWERED_TASK_SPEC_COLUMN])["task"], {}) for row in table.to_pylist()]
    return LOWERED_TASK_SPEC_COLUMN, str(table.num_rows), documents


def census(output_root: str, converter_revision: str) -> dict:
    """Render the population, write the dedup receipt and the fingerprint survey."""
    source_id, repo, revision, version, expected_rows = forge_declaration()
    root = StoragePath(output_root)
    if (root / DEDUP_SUMMARY).exists():
        raise FileExistsError(f"Census requires a new artifact root: {output_root}")
    local = Path(hf_hub_download(repo, COHORT_FILENAME, repo_type="dataset", revision=revision))
    column, rows, documents = cohort_documents(local)
    if int(rows) != expected_rows:
        raise ValueError(f"Published cohort carries {rows} rows, declaration expects {expected_rows}")

    files = write_text_population(documents, str(root / "population"))
    if files.raw_task_count != expected_rows:
        raise ValueError(f"Population render counted {files.raw_task_count} rows, expected {expected_rows}")
    receipt = WithinSourceDedupEvidence(
        raw_count=files.raw_task_count,
        unique_count=files.task_count,
        within_source_duplicates=files.within_source_duplicates,
        source_dataset_id=repo,
        source_revision=revision,
        converter_revision=converter_revision,
        converter_version=version,
        drop_manifest_uri=str(root / "population" / "within-source-dedup" / "drops.parquet"),
        evidence_url=str(root / DEDUP_SUMMARY),
    )
    durable_input = str(root / COHORT_FILENAME)
    StoragePath(durable_input).write_bytes(local.read_bytes())
    survey = fingerprint_manifest(
        {
            "catalog_root": "atlas:forge-main-declaration",
            "snapshot_id": revision,
            "source_ids": [source_id],
            "inputs": [
                {
                    "source_id": source_id,
                    "paths": [str(local)],
                    "input_uris": [durable_input],
                    "expected_tasks": expected_rows,
                    "column": column,
                }
            ],
        }
    )
    summary = {
        "population": json.loads(
            SourceTextPopulation(
                path=str(root / "population"),
                source_id=source_id,
                dataset_id=repo,
                dataset_revision=revision,
                source_version=version,
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
                    "drop_manifest_uri": receipt.drop_manifest_uri,
                },
            ).model_dump_json()
        ),
        "dedup_receipt": json.loads(receipt.model_dump_json()),
        "fingerprint": {
            "uri": str(root / FINGERPRINT_ARTIFACT),
            "task_count": survey.fingerprints[source_id].task_count
            if survey.fingerprints.get(source_id) is not None
            else None,
            "inputs": survey.fingerprints[source_id].inputs if survey.fingerprints.get(source_id) else None,
        },
    }
    (root / DEDUP_SUMMARY).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / FINGERPRINT_ARTIFACT).write_text(survey.model_dump_json(indent=2) + "\n")
    logger.info(
        "%s: %d raw, %d unique, %d duplicates; fingerprinted %s",
        source_id,
        files.raw_task_count,
        files.task_count,
        files.within_source_duplicates,
        durable_input,
    )
    return {
        "source_id": source_id,
        "repo": repo,
        "revision": revision,
        "converter_version": version,
        "durable_input": durable_input,
        "fingerprint": str(root / FINGERPRINT_ARTIFACT),
        "summary": str(root / DEDUP_SUMMARY),
        "raw_count": files.raw_task_count,
        "unique_count": files.task_count,
        "within_source_duplicates": files.within_source_duplicates,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True, help="new S3 artifact root for the measurements")
    parser.add_argument("--converter-revision", required=True, help="the git revision this census ran from")
    args = parser.parse_args()
    configure_coreweave_s3()
    print(json.dumps(census(args.output_root, args.converter_revision), indent=2))


if __name__ == "__main__":
    main()
