# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run lossless native Harbor QUICK conversion and publish private inspection releases."""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import TaskFormat
from finestore.rl.licenses import LicenseReceipt, require_license
from finestore.rl.status_taxonomy import DataRestriction
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from taskcompendium.pipeline.source_processing import SourceProcessingMode, run_source_pipeline
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.native_harbor import VERSION, hub_rows, native_task, unpack_native
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.license_publication import publish_licensed_population
from experiments.post_training.task_curation.publication_gate import configured_restriction, require_public_data
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source_policy import CROSSCODE_EVAL_SOURCE_IDS
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, original_source_id

LOGGER = logging.getLogger(__name__)
RETIRED = Path(__file__).with_name("retired-inventory-only.json")
TASK_SCHEMA = pa.schema([("path", pa.string()), ("task_spec", pa.string()), ("task_sha256", pa.string())])
REPO_SLUG_LIMIT = 63
CANONICAL_FILES = {
    "GAIR__OpenSWE__openswe_oss": "data/openswe-oss-a8db93af5335.parquet",
    "GAIR__OpenSWE__openswe_other": "data/openswe-other-a8db93af5335.parquet",
    "R2E-Gym__R2E-Gym-V1": "data/r2egym-903d405799ac.parquet",
    "SWE-Gym__SWE-Gym": "data/swegym-bb94ed9e39bb.parquet",
    "XiaomiMiMo__MiMo-V2.6-RL-oss__code": "data/mimo-code-639865fd3374.parquet",
    "XiaomiMiMo__MiMo-V2.6-RL-oss__cyber": "data/mimo-cyber-639865fd3374.parquet",
    "XiaomiMiMo__MiMo-V2.6-RL-oss__general": "data/mimo-general-639865fd3374.parquet",
    "XiaomiMiMo__MiMo-V2.6-RL-oss__music": "data/mimo-music-639865fd3374.parquet",
    "XiaomiMiMo__MiMo-V2.6-RL-oss__webdev": "data/mimo-webdev-639865fd3374.parquet",
    "smartie-calibforge-admitted": "releases/smartie-continuation-20261016/calibforge/tasks.parquet",
    "smartie-calibforge-shared-v2": "releases/calibforge-shared-20261017-v2/tasks.parquet",
    "smartie-swesmith-admitted": "releases/smartie-continuation-20261016/swesmith/tasks.parquet",
}


def source_slug(source_id: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", original_source_id(source_id).lower()).strip("-")


def publish_quick(
    entry: dict,
    output: Path,
    repo: str,
    license_receipt: LicenseReceipt,
    *,
    intended_use: IntendedUse,
    data_restriction: DataRestriction,
) -> dict:
    """Publish conversion evidence and validate every TaskSpec and uploaded byte stream."""
    require_public_data(data_restriction, entry["id"], repo)
    require_license(license_receipt, entry["id"], intended_use.value, ban=None)
    require_unbanned(entry["id"], entry["dataset_id"], license_receipt)
    manifest = json.loads((output / "manifest.json").read_text())
    destination = output / "tasks.parquet"
    with pq.ParquetWriter(destination, TASK_SCHEMA) as writer:
        for path in sorted((output / "normalize").glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                rows = []
                for row in batch.to_pylist():
                    serialized = row["task_json"]
                    if serialized is None:
                        continue
                    TaskSpec.model_validate_json(serialized)
                    rows.append(
                        {
                            "path": row["original_path"],
                            "task_spec": serialized,
                            "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                        }
                    )
                writer.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA))
    if pq.ParquetFile(destination).metadata.num_rows != manifest["converted_rows"]:
        raise ValueError("Published row count differs from QUICK conversion")
    if manifest["converted_rows"] == 0:
        return {"id": entry["id"], "status": "no_faithful_tasks", "manifest": manifest}
    body = (
        f"# {entry['display_name']}\n\n"
        "Private research artifact. QUICK conversion is unreviewed and is not training admission. "
        "TaskCompendium TaskSpecs retain the native Harbor task.toml, environment and verifier files. "
        "Harbor must execute the original package; file capture is not an equivalent runtime. "
        "Native image tags and build downloads may remain mutable. No environment was substituted.\n\n"
        f"Upstream: {entry['dataset_id']} at {entry['dataset_revision']}. {entry['url']}\n\n"
        f"Converted: {manifest['converted_rows']}; input: {manifest['input_rows']}. "
        f"Mechanical exclusions: {json.dumps(manifest['rejections'], sort_keys=True)}. "
        "No quality result or oracle execution is claimed. Lexical decontamination and fingerprint "
        "coverage are pending the combined backend pass; SAMPLE/FULL must wait for its pinned receipt.\n\n"
        f"Preserved source warnings: {entry['notes'].split(' Admission blocked:')[0]}\n"
    )
    # The QUICK release's licence carries forward from the upstream receipt through the
    # publication chokepoint; the new pin's receipt registers before this returns.
    publication = publish_licensed_population(
        entry["id"],
        repo,
        output,
        ("tasks.parquet", "manifest.json"),
        body=body,
        pretty_name="Native Harbor QUICK inspection release",
        use=intended_use.value,
    )
    return {
        "id": entry["id"],
        "status": "quick_converted",
        "repo": publication.repo,
        "revision": publication.revision,
        "manifest": manifest,
        "file_hashes": publication.file_hashes,
        "license_check": publication.license_check,
    }


def run_entry(
    entry: dict,
    root: Path,
    hub_input: str | None,
    license_receipt: LicenseReceipt,
    version: str,
    *,
    data_restriction: DataRestriction,
) -> dict:
    intended_use = IntendedUse.EVAL if entry["id"] in CROSSCODE_EVAL_SOURCE_IDS else IntendedUse.TRAIN
    require_license(license_receipt, entry["id"], intended_use.value, ban=None)
    require_unbanned(entry["id"], entry["dataset_id"], license_receipt)
    slug = source_slug(entry["id"])
    output = root / slug
    input_root = root / "inputs" / slug
    if entry["origin"] == "Harbor Hub":
        if hub_input is None:
            raise ValueError("Harbor Hub conversion requires its completed pinned acquisition")
        acquired = json.loads((StoragePath(hub_input) / "release.json").read_text())
        if acquired["release"] != f"{entry['dataset_id']}@{entry['dataset_revision']}":
            raise ValueError("Harbor acquisition differs from the declared source pin")
        if acquired["selected"] != acquired["task_refs"] or acquired["population_count"] != len(acquired["selected"]):
            raise ValueError("Native source ingestion requires the complete pinned Harbor population")
        source = SourceFiles(
            entry["dataset_id"], entry["dataset_revision"], ("release.json",), SourceFormat.JSON, read=hub_rows
        )
        staged = hub_input
    else:
        filename = (
            CANONICAL_FILES[entry["name"]]
            if entry["dataset_id"] == "open-athena/task-trove"
            else entry["name"] + "/tasks.parquet"
        )
        downloaded = Path(
            hf_hub_download(entry["dataset_id"], filename, repo_type="dataset", revision=entry["dataset_revision"])
        )
        target = input_root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(downloaded)
        staged = str(input_root)
        source = SourceFiles(
            entry["dataset_id"], entry["dataset_revision"], (filename,), SourceFormat.PARQUET, decode=unpack_native
        )
    recipe = SourceRecipe(
        name=slug,
        version=VERSION,
        source=source,
        convert=native_task,
        intended_use=intended_use,
        rubric=None,
        controls=None,
    )
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=2, chunk_storage_prefix=str(root / "chunks" / slug)) as context,
        ):
            run_source_pipeline(
                recipe,
                context,
                staged,
                str(output),
                mode=SourceProcessingMode.QUICK,
                canonical_source=original_source_id(entry["id"]),
                declared_task_format=TaskFormat.HARBOR,
            )
    finally:
        client.shutdown()
    repo_slug = (
        slug[:REPO_SLUG_LIMIT].rstrip("-")
        + "-"
        + hashlib.sha256(original_source_id(entry["id"]).encode()).hexdigest()[:8]
    )
    return publish_quick(
        entry,
        output,
        f"open-athena/{repo_slug}-taskcompendium-{version}",
        license_receipt,
        intended_use=intended_use,
        data_restriction=data_restriction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--hub-input")
    parser.add_argument("--version", required=True, help="New immutable private repository version")
    parser.add_argument("--license-receipts", type=Path, default=LICENSE_RECEIPT_INDEX)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    entries = {entry["id"]: entry for entry in read_registry_json(RETIRED)["entries"]}
    receipts = read_license_receipts(args.license_receipts)
    args.root.mkdir(parents=True, exist_ok=True)
    for requested in args.source:
        source_id = canonical_source_id(requested)
        entry = entries[source_id]
        receipt = receipts.get(entry["id"])
        intended_use = IntendedUse.EVAL if entry["id"] in CROSSCODE_EVAL_SOURCE_IDS else IntendedUse.TRAIN
        require_license(receipt, entry["id"], intended_use.value, ban=None)
        assert receipt is not None
        result = run_entry(
            entry, args.root, args.hub_input, receipt, args.version, data_restriction=configured_restriction(source_id)
        )
        slug = source_slug(source_id)
        artifact_root = StoragePath(args.evidence_root).parent / "artifacts" / slug
        if result["status"] == "quick_converted":
            for filename in ("tasks.parquet", "manifest.json"):
                destination = artifact_root / filename
                copy(str(args.root / slug / filename), str(destination), no_clobber=True)
                if file_sha256(destination) != result["file_hashes"][filename]:
                    raise ValueError(f"Regional artifact copy differs: {destination}")
            result["task_specs_uri"] = str(artifact_root / "tasks.parquet")
            result["manifest_uri"] = str(artifact_root / "manifest.json")
        (StoragePath(args.evidence_root) / slug / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", auto_mkdir=True
        )
        LOGGER.info("Published %s at %s", source_id, result.get("revision"))


if __name__ == "__main__":
    main()
