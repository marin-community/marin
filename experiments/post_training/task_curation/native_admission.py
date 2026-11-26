# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the native taskspec conversions' admission documents.

Builds, from the sealed stage artifacts, the documents the standard registry
append consumes (``registry.archive_registry_documents``):

- the lexical measurement receipts bound to the option-A census inputs (one
  CLEAN receipt per zero-contamination measurement, conserving the measured
  population exactly),
- the retired-inventory ledger with the entries' disposition releases re-pinned
  to the taskspec releases: the census binding, the Daytona measurement, the
  grading-environment classification and the release file hashes.

Staging only — the append lands through ``archive_registry_documents``, and the
repo's Git side goes through atlas-push after SAMPLE.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from finestore.rl.curation import (
    DaytonaMeasurement,
    DeconReceiptStatus,
    DecontaminatedRelease,
    LexicalReceipt,
    LexicalUse,
)
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.native_source_conversion import load_inventory_entry
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.retirement import RETIREMENTS_PATH

LOGGER = logging.getLogger(__name__)

CONVERSION_VERSION = "harbor-cohort-taskspec-v1"
GRADING_REASON = "Repository or workspace-state grading reads state the agent can mutate"
RELEASE_FILES = ("README.md", "manifest.json", "tasks.parquet", "conversion-receipts.jsonl", "conversion-exclusions.jsonl")


def census_documents(census_root: str, slug: str) -> tuple[dict, dict]:
    """The staged option-A census result and its actor-census receipt input."""
    result = json.loads(StoragePath(f"{census_root}/{slug}/{slug}/census-result.json").read_bytes())
    receipt_shard = json.loads(StoragePath(f"{census_root}/{slug}-receipts/{slug}-receipts/part-000.json").read_bytes())
    census = result["census"]
    receipt = next(iter(receipt_shard.values()))
    staged_artifact = f"{census_root}/{slug}/{slug}"
    receipt["input"]["evidence_url"] = f"{staged_artifact}/.artifact.json"
    return census, receipt


def _lexical_receipt(source_id: str, receipt_input: dict, measurement: dict) -> dict:
    from finestore.rl.curation import ActorCensusInput, LexicalContaminationEvidence  # noqa: PLC0415
    from experiments.post_training.task_curation.lexical_binding import LexicalMeasurementReceipt as Receipt  # noqa: PLC0415

    # The registered measurement stays raw: a training-use receipt cannot certify
    # a release. The declaration's taskspec lexical evidence builds the CLEAN
    # release receipt from this measurement at admission time.
    receipt = Receipt(
        source_id=source_id,
        input=ActorCensusInput.model_validate(receipt_input),
        use=LexicalUse.TRAIN,
        measurement=LexicalContaminationEvidence.model_validate(dict(measurement)),
    )
    return receipt.model_dump(mode="json")



def _release_file_hashes(release: dict) -> dict[str, str]:
    hashes = {}
    for name in RELEASE_FILES:
        try:
            downloaded = hf_hub_download(release["repo"], name, repo_type="dataset", revision=release["revision"])
        except Exception:  # noqa: BLE001 - a release file the conversion did not write
            continue
        hashes[name] = file_sha256(StoragePath(str(Path(downloaded))))
    return hashes


def update_ledger_entry(entry: dict, census: dict, receipt: dict, release_repo: str, release_revision: str) -> dict:
    """Re-pin one ledger entry's disposition release to its taskspec release."""
    if entry.get("disposition") != "quick_converted":
        raise ValueError(f"The entry is not a quick-converted ledger row: {entry['id']}")
    release = dict(entry["disposition_release"])
    measured = census["task_count"]
    if measured != receipt["input"]["measured_tasks"]:
        raise ValueError(f"The census and its receipt disagree on the population: {entry['id']}")
    release.update(
        {
            "repo": release_repo,
            "revision": release_revision,
            "converted_rows": measured,
            "conversion_version": CONVERSION_VERSION,
            "converter_identity": census["converter_revision"],
            "actor_census": {
                "converter_revision": census["converter_revision"],
                "converter_version": census["converter_version"],
                "input_artifact": receipt["input"]["evidence_url"],
                "path": f"{str(receipt['input']['evidence_url']).rsplit('/', 1)[0]}",
                "task_count": measured,
                "task_specs_sha256": census["task_specs_sha256"],
            },
            "daytona_measurement": DaytonaMeasurement(
                dataset_id=release_repo,
                dataset_revision=release_revision,
                daytona_snapshot_count=census["daytona_snapshot_count"],
                daytona_machine_slots=census["daytona_machine_slots"],
                daytona_execution_failures=census["daytona_execution_failures"],
                daytona_count_basis=census["daytona_count_basis"],
                daytona_unsafe_reasons=tuple(census["daytona_unsafe_reasons"]),
                daytona_unsafe_details=census["daytona_unsafe_details"],
                daytona_unsafe_counts=census["daytona_unsafe_counts"],
                daytona_roles=census["daytona_roles"],
            ).model_dump(mode="json"),
            "grading_environment": {
                "assessed_tasks": measured,
                "required_counts": {"shared": measured},
                "declared_counts": {"shared": measured},
                "status_counts": {"matched": measured},
                "reason_counts": {GRADING_REASON: measured},
                "evidence_url": str(receipt["input"]["evidence_url"]).rsplit("/", 1)[0],
            },
            "file_hashes": _release_file_hashes({"repo": release_repo, "revision": release_revision}),
        }
    )
    return {**entry, "disposition_release": release}


def stage_sources(
    source_ids: tuple[str, ...],
    *,
    census_root: str,
    lexical_report_uri: str,
    staged_root: Path,
) -> dict:
    """Build the lexical receipts and the re-pinned ledger into the staged tree."""
    report = json.loads(StoragePath(lexical_report_uri).read_bytes())
    measurements = report["measurements"]
    ledger = read_registry_json(RETIREMENTS_PATH)
    entries = ledger["entries"]
    by_id = {entry["id"]: entry for entry in entries}
    lexical_receipts: dict[str, dict] = {}
    for source_id in source_ids:
        entry = by_id.get(source_id)
        if entry is None:
            raise ValueError(f"The source is not in the ledger: {source_id}")
        slug = str(source_id).partition(":")[2].replace("/", "-").replace("@", "-").lower()
        census, receipt = census_documents(census_root, slug)
        release = entry["disposition_release"]
        new_repo, new_revision = census["dataset_id"], census["dataset_revision"]
        measurement = measurements.get(source_id)
        if measurement is None:
            raise ValueError(f"The lexical report carries no measurement for the source: {source_id}")
        if (measurement["input_dataset_id"], measurement["input_dataset_revision"]) != (new_repo, new_revision):
            raise ValueError(f"The lexical measurement binds another pin: {source_id}")
        if measurement["measured_tasks"] != census["task_count"]:
            raise ValueError(f"The lexical measurement population differs from the census: {source_id}")
        updated_input = dict(receipt["input"])
        updated_input["evidence_url"] = f"{census_root}/{slug}/{slug}/.artifact.json"
        lexical_receipts[source_id] = _lexical_receipt(source_id, updated_input, measurement)
        updated = update_ledger_entry(
            entry,
            census,
            {**receipt, "input": updated_input},
            new_repo,
            new_revision,
        )
        by_id[source_id] = updated
        LOGGER.info("staged re-pin: %s -> %s@%s", source_id, new_repo, new_revision[:12])
    ledger["entries"] = [by_id[entry["id"]] for entry in entries]

    lexical_dir = staged_root / "experiments/post_training/task_curation/datasets/lexical_measurement_receipts"
    lexical_dir.mkdir(parents=True, exist_ok=True)
    # The staged shard continues the repo's existing family numbering: the
    # archive flow's family rules refuse duplicate identities across shards,
    # and a partial staging would look like dropped sibling rows. The existing
    # family documents re-stage verbatim beside the new shard.
    repo_family = Path(__file__).with_name("datasets") / "lexical_measurement_receipts"
    next_part = max((int(path.stem.split("-")[1]) for path in repo_family.glob("part-*.json")), default=-1) + 1
    for previous in sorted(repo_family.glob("part-*.json")):
        document = read_registry_json(previous)
        # The sources' stale lexical receipts (the superseded 20261112 releases)
        # leave their shards: the re-pinned releases' measurements replace them,
        # and an identity may not survive an append in two shards.
        remaining = {key: row for key, row in document.items() if key not in lexical_receipts}
        if len(remaining) != len(document):
            LOGGER.info(
                "consolidating %d superseded rows out of %s", len(document) - len(remaining), previous.name
            )
        (lexical_dir / previous.name).write_text(json.dumps(remaining, indent=2, sort_keys=True) + "\n")
    (lexical_dir / f"part-{next_part:03d}.json").write_text(json.dumps(lexical_receipts, indent=2, sort_keys=True) + "\n")
    ledger_path = staged_root / "experiments/post_training/task_curation/retired-inventory-only.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return {"sources": list(source_ids), "lexical_receipts": str(lexical_dir / f"part-{next_part:03d}.json"), "ledger": str(ledger_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="Comma-separated source ids to stage.")
    parser.add_argument("--census-root", required=True)
    parser.add_argument("--lexical-report-uri", required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

    configure_coreweave_s3()
    staged = stage_sources(
        tuple(args.sources.split(",")),
        census_root=args.census_root,
        lexical_report_uri=args.lexical_report_uri,
        staged_root=args.staged_root,
    )
    print(json.dumps(staged, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
