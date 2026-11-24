"""Publish and verify the Endless oracle-filtered private release (stage 2 of the admission plan)."""
import argparse
import hashlib
import io
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import (
    DatasetRelease,
    OracleFiltrationEvidence,
    OraclePendingExclusion,
    OraclePendingExclusionReason,
)
from finestore.rl.licenses import LicenseReceipt, require_license
from finestore.rl.status_taxonomy import DataRestriction
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.oracle_filter import (
    OracleExclusionReason,
    OracleFilterReceipt,
)

from experiments.post_training.task_curation.license_publication import (
    licensed_dataset_card,
    lineage_folder_identifier,
    prepare_license_publication,
    prepare_private_dataset,
    publish_private_dataset,
)
from experiments.post_training.task_curation.publication_gate import require_public_data
from experiments.post_training.task_curation.oracle_filter import row_runtime, row_task
from experiments.post_training.task_curation.oracle_release import verify_oracle_release
from experiments.post_training.task_curation.remediation_publish import TASK_SCHEMA

ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/endless/2026.11.19.1"
FILTER = ROOT + "/oracle-filter-v1"
OUT = ROOT + "/oracle-release-v1"
COHORT = {"uri": FILTER + "/inputs/cohort.parquet", "sha256": "691a00bb8c1fcbd4aa825c629ec7387fca34f31da660614a718a1e1d7f5557eb"}
FILTER_RECEIPT = {"uri": FILTER + "/oracle-filter.json", "sha256": "3e6a58ce0a080a627b8e766ed67159d46e5ee163a10e08b7fba134edd90c10d0"}
OUTPUT = {"uri": FILTER + "/tasks.parquet", "sha256": "8fb119202251df5be11d6231a5777c014959d3a3424a90ae927474fbe7549b8a"}
SUMMARY = {"uri": FILTER + "/oracle-filter-summary.json", "sha256": "a33c8985b1ce9a3eb5bed7210e6e8e158af190afb68561ba515c162ac8d8c008"}
PLAN = {"uri": ROOT + "/inputs/native-controls-full-plan-v1.json",
        "sha256": "3ae3db109bf2c9851ea6644d8ebb03c6886f83088ecea4a9cd333be772d32f3b"}
EXPECTED_ORIGINAL = 2069
EXPECTED_ADMITTED = 1976
ORIGINAL_COUNTS = {"oracle_failed": 70, "negative_control_pending": 5,
                   "resource_calibration:infrastructure": 17, "resource_calibration:verifier_timeout": 1}
ORIGINAL_RELEASE = DatasetRelease(repo="FWeindel/validated-tasks", revision="317659ac88c8b706d318020c5fc3813a931fb6ec",
                                  url="https://huggingface.co/datasets/FWeindel/validated-tasks/tree/317659ac88c8b706d318020c5fc3813a931fb6ec")
REPO = "open-athena/fweindel-endless-oracle-filtered-20261124-v1"
VERSION = "2026.11.24.endless-oracle-release.1"
LOG = logging.getLogger("endless-oracle-release")


def checked(pin):
    raw = StoragePath(pin["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("Frozen input bytes changed: " + pin["uri"])
    return raw


def seal(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    if path.exists():
        raise FileExistsError(str(path))
    path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError("Sealed output readback changed: " + str(path))
    return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def grants(plan):
    lineage = json.loads(checked(plan["license_lineage"]))
    records = json.loads(checked(lineage))
    out = {}
    for source_id, record in (records.items() if isinstance(records, dict) else []):
        out[source_id] = LicenseReceipt.model_validate(record if "receipt" not in record else record["receipt"])
    if not out:
        raise ValueError("License lineage carries no grants")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    started = time.monotonic()
    cohort_raw = checked(COHORT)
    filter_raw = checked(FILTER_RECEIPT)
    output_raw = checked(OUTPUT)
    summary = json.loads(checked(SUMMARY))
    plan = json.loads(checked(PLAN))
    from finestore.rl.curation import ContentEvidenceReference  # noqa: PLC0415

    from experiments.post_training.task_curation.datasets.endless.tasks import sources as endless_sources  # noqa: PLC0415
    from experiments.post_training.task_curation.license_binding import bind_license_receipt  # noqa: PLC0415

    source_receipts = endless_sources(ContentEvidenceReference.model_validate(plan["images"]),
                                      ContentEvidenceReference.model_validate(plan["source_census"]))
    lineage = checked(plan["license_lineage"])
    records = json.loads(lineage)
    grants = {source_id: LicenseReceipt.model_validate(record if "receipt" not in record else record["receipt"])
              for source_id, record in records.items()}
    for source in source_receipts:
        bind_license_receipt(source, grants[source.info.id]).require_license()
    grant = grants["FWeindel:validated-tasks/endless_v2"]

    cohort_table = pq.read_table(io.BytesIO(cohort_raw))
    output_table = pq.read_table(io.BytesIO(output_raw))
    filter_receipt = OracleFilterReceipt.model_validate_json(filter_raw)
    pending = [OraclePendingExclusion(task_id=task_id, reason=OraclePendingExclusionReason.INFRASTRUCTURE)
               for task_id, entry in filter_receipt.tasks.items()
               if entry.exclusion_reason is OracleExclusionReason.INFRASTRUCTURE]
    observed_counts = Counter(entry.exclusion_reason.value for entry in filter_receipt.tasks.values()
                              if entry.exclusion_reason is not None)
    if dict(observed_counts) != ORIGINAL_COUNTS:
        raise ValueError("Filter receipt exclusion counts differ from the frozen summary")
    if len(pending) != ORIGINAL_COUNTS["resource_calibration:infrastructure"]:
        raise ValueError("Pending infrastructure rows differ from the frozen summary")
    if cohort_table.num_rows != EXPECTED_ORIGINAL or output_table.num_rows != EXPECTED_ADMITTED:
        raise ValueError("Frozen cohort/output row counts differ from the sealed summary")
    source_release_identity = json.loads(cohort_table.to_pylist()[0]["task_spec"])["source"]
    if (source_release_identity["dataset"], source_release_identity["revision"]) != (ORIGINAL_RELEASE.repo, ORIGINAL_RELEASE.revision):
        raise ValueError("Cohort source identity differs from the original release pin")
    if summary["input_rows"] != EXPECTED_ORIGINAL or summary["output_rows"] != EXPECTED_ADMITTED \
            or summary["exclusions_by_reason"] != ORIGINAL_COUNTS:
        raise ValueError("Filter summary differs from the frozen outcome accounting")
    if args.preflight_output is not None:
        folder_identifiers = {folder: lineage_folder_identifier(record) for folder, record in sorted(records.items())}
        if set(folder_identifiers) != {"FWeindel:validated-tasks/endless_v2", "FWeindel:validated-tasks/endless_v3"}:
            raise ValueError("Folder lineage coverage differs from the frozen campaign folders")
        for row in cohort_table.to_pylist():
            folder = json.loads(row["task_spec"])["source"]["row"].split("/")[0]
            if folder not in {name.rsplit("/", 1)[-1] for name in folder_identifiers}:
                raise ValueError(f"Cohort row folder lacks a lineage decision: {folder}")
        args.preflight_output.write_text(json.dumps({
            "status": "PASS", "cohort_rows": EXPECTED_ORIGINAL, "admitted": EXPECTED_ADMITTED,
            "exclusions": ORIGINAL_COUNTS, "publication_performed": False,
            "lineage_folders": folder_identifiers,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, indent=2) + "\n")
        return
    job = get_job_info()
    assert job is not None and job.attempt_id == 0 and not StoragePath(OUT).exists()
    require_public_data(DataRestriction.OPEN, "FWeindel:validated-tasks/endless_v2", REPO)
    require_license(grant, "FWeindel:validated-tasks/endless_v2", ORIGINAL_RELEASE.repo, ORIGINAL_RELEASE.revision,
                    "train", ban=None)
    with TemporaryDirectory(prefix="endless-oracle-release-") as directory:
        work = Path(directory)
        stage = work / "publication"
        stage.mkdir()
        copy(OUTPUT["uri"], str(stage / "tasks.parquet"), no_clobber=True)
        assert file_sha256(StoragePath(str(stage / "tasks.parquet"))) == OUTPUT["sha256"]
        manifest_document = {
            "input_rows": EXPECTED_ORIGINAL, "admitted_rows": EXPECTED_ADMITTED,
            "recipe_revision": VERSION, "source_revision": plan["source_revision"],
            "filter_job": "/benfeuer/new-rl-ingest-a-endless-oracle-filter-v4",
            "exclusions_by_reason": ORIGINAL_COUNTS,
            "change_basis": "Oracle-filtered retained cohort: filter_cohort over the verified pair ledger with bound "
                            "physical NoP controls; 1,976 oracle-passing tasks retained; typed exclusions sealed with "
                            "their per-row evidence. Zero infrastructure-pending rows in this release.",
            "provider_actions": 0,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest_document, indent=2) + "\n")
        # The repackager ruling: a folder whose lineage result is not approved is excluded,
        # not published. Both Endless folders carry approved exceptions; this fails closed
        # if any cohort row's folder ever lacks one.
        included_folders = {"endless_v2", "endless_v3"}
        for row in output_table.to_pylist():
            folder = json.loads(row["task_spec"])["source"]["row"].split("/")[0]
            if folder not in included_folders:
                raise ValueError(f"Cohort row folder lacks an approved lineage decision: {folder}")
        publication = prepare_license_publication(
            grant, stage, use="train", lineage=records,
            folders=sorted(f"FWeindel:validated-tasks/{name}" for name in sorted(included_folders)),
        )
        folder_table = "\n".join(
            f"| {row['folder'].rsplit('/', 1)[-1]} | {row['upstream']} | {row['identifier']} | `{row['receipt_sha256'][:16]}…` |"
            for row in publication.license_table
        )
        body = (
            "# FWeindel Endless: oracle-filtered retained cohort\n\n"
            "Private TRAIN candidate, not admitted. The oracle-filtered subset of the frozen Endless census "
            "population: 1,976 of 2,069 verified oracle+NoP pairs retained (the original oracle passes and the "
            "physical NoP does not); 93 rows typed-excluded with their per-row outcomes "
            "(70 oracle_failed, 5 negative_control_pending, 18 resource-calibration infrastructure). "
            "Task bytes, prompts, native files and role budgets are unchanged from the frozen census population.\n\n"
            "## License\n\n"
            "Per the repackager ruling (repackager-license-follows-upstream-20261116), the license follows each "
            "folder's original upstream. Every included folder's folder-lineage receipt carries an approved "
            "exception accepting Endless v1-v3 from obiwan96/endless-terminals@26ecf78458e7f756e4d06780d5fbf3dd78e91815 "
            "as MIT:\n\n"
            "| Folder | Upstream project + pin | Identifier | Lineage receipt hash |\n"
            "|---|---|---|---|\n"
            f"{folder_table}\n"
        )
        (stage / "README.md").write_text(licensed_dataset_card(body, publication, pretty_name="FWeindel Endless oracle-filtered tasks"))
        names = ("tasks.parquet", "manifest.json", "README.md", *(p.name for p in publication.notice_files))
        prepared = prepare_private_dataset("FWeindel:validated-tasks/endless_v2", REPO, stage, names, grant, use="train",
                                           lineage=records)
        published = publish_private_dataset(prepared)
        for name in names:
            target = StoragePath(OUT + "/publication/" + name)
            copy(str(stage / name), str(target), no_clobber=True)
            assert file_sha256(target) == prepared.file_hashes[name]
        LOG.info("ENDLESS_RELEASE_PUBLISHED repo=%s revision=%s", REPO, published.revision)
        artifacts = {name: {"uri": str(StoragePath(OUT + "/publication/" + name)), "sha256": prepared.file_hashes[name]}
                     for name in names}
        quick = {"repo": REPO, "revision": published.revision, "manifest": manifest_document,
                 "artifacts": artifacts, "file_hashes": prepared.file_hashes, "license_check": published.license_check}
        quick_pin = seal(StoragePath(OUT + "/publication.json"), quick)
        validation_pin = seal(StoragePath(OUT + "/verification.json"), {
            "cohort": COHORT, "oracle_filter": FILTER_RECEIPT, "output": OUTPUT,
            "cohort_rows": EXPECTED_ORIGINAL, "admitted": EXPECTED_ADMITTED,
            "exclusions": ORIGINAL_COUNTS, "byte_checks": "checked() sha256 over every frozen pin; "
            "verify_oracle_release re-verifies rows, runtimes, canonical admission and the exclusion partition",
            "publication": quick_pin, "at": datetime.now(timezone.utc).isoformat(),
        })
        filtered_release = DatasetRelease(repo=REPO, revision=published.revision,
                                          url=f"https://huggingface.co/datasets/{REPO}/tree/{published.revision}")
        local_cohort, local_filter, local_output = work / "cohort.parquet", work / "oracle-filter.json", work / "tasks.parquet"
        local_cohort.write_bytes(cohort_raw)
        local_filter.write_bytes(filter_raw)
        local_output.write_bytes(output_raw)
        evidence = OracleFiltrationEvidence(
            original_release=ORIGINAL_RELEASE,
            original_cohort=ContentEvidenceReference(**COHORT),
            original_count=EXPECTED_ORIGINAL,
            oracle_filter=ContentEvidenceReference(**FILTER_RECEIPT),
            oracle_filter_version=4,
            filtered_release=filtered_release,
            private=True,
            tasks_parquet=ContentEvidenceReference(**OUTPUT),
            admitted_count=EXPECTED_ADMITTED,
            exclusion_counts=dict(ORIGINAL_COUNTS),
            pending_exclusions=pending,
            validation=ContentEvidenceReference(**validation_pin),
        )
        receipt = verify_oracle_release(evidence, local_cohort, local_filter, local_output)
        evidence_pin = seal(StoragePath(OUT + "/filtration-evidence.json"), evidence.model_dump(mode="json"))
        LOG.info("ENDLESS_ORACLE_RELEASE_VERIFIED tasks=%d", receipt.input_rows)
        result = {
            "status": "ENDLESS_ORACLE_RELEASE_COMPLETE",
            "job": job.job_id.to_wire(), "at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "publication": quick_pin, "filtration_evidence": evidence_pin, "verification": validation_pin,
            "release": {"repo": REPO, "revision": published.revision,
                        "url": f"https://huggingface.co/datasets/{REPO}/tree/{published.revision}"},
            "admitted": EXPECTED_ADMITTED, "excluded": EXPECTED_ORIGINAL - EXPECTED_ADMITTED,
            "exclusions_by_reason": ORIGINAL_COUNTS,
            "scope": "Stage 2 of the endless oracle-filtered admission plan: the oracle-qualified private release "
                     "published and independently verified. No catalog declaration, admission claim or provider actions.",
            "next": "The Endless RlDataSource declaration (atlas-push, lands with admission), the admitted-release "
                    "manifest (oracle_admitted_release), then the exact-pin census, lexical, SAMPLE V4 and admission.",
        }
        seal(StoragePath(OUT + "/result.json"), result)
        print(json.dumps({"status": result["status"], "admitted": EXPECTED_ADMITTED,
                          "repo": REPO, "revision": published.revision}))


if __name__ == "__main__":
    main()
