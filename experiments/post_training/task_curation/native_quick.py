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
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from huggingface_hub import HfApi, hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from taskcompendium.pipeline.source_processing import SourceProcessingMode, run_source_pipeline
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.native_harbor import VERSION, hub_rows, native_task, unpack_native
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source_identity import canonical_source_id, original_source_id
from experiments.post_training.task_curation.source_policy import CROSSCODE_EVAL_SOURCE_IDS

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


def publish_quick(entry: dict, output: Path, repo: str) -> dict:
    """Publish conversion evidence and validate every TaskSpec and uploaded byte stream."""
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
    (output / "README.md").write_text(
        "---\npretty_name: Native Harbor QUICK inspection release\n---\n\n"
        f"# {entry['display_name']}\n\n"
        "Private research artifact. QUICK conversion is unreviewed and is not training admission. "
        "TaskCompendium TaskSpecs retain the native Harbor task.toml, environment and verifier files. "
        "Harbor must execute the original package; file capture is not an equivalent runtime. "
        "Native image tags and build downloads may remain mutable. No environment was substituted.\n\n"
        f"Upstream: {entry['dataset_id']} at {entry['dataset_revision']}. {entry['url']}\n\n"
        f"Converted: {manifest['converted_rows']}; input: {manifest['input_rows']}. "
        f"Mechanical exclusions: {json.dumps(manifest['rejections'], sort_keys=True)}. "
        "No quality result or oracle execution is claimed. Lexical decontamination and fingerprint "
        "coverage are pending the combined backend pass; SAMPLE/FULL must wait for its pinned receipt. "
        "Upstream licenses and third-party asset terms apply; this conversion grants no new rights.\n\n"
        f"Preserved source warnings: {entry['notes'].split(' Admission blocked:')[0]}\n"
    )
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Conversion destination must be private")
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=output,
        allow_patterns=["tasks.parquet", "manifest.json", "README.md"],
        commit_message="Publish native Harbor QUICK conversion",
    )
    hashes = {}
    for filename in ("tasks.parquet", "manifest.json", "README.md"):
        uploaded = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=commit.oid))
        digest = file_sha256(StoragePath(str(output / filename)))
        if file_sha256(StoragePath(str(uploaded))) != digest:
            raise ValueError(f"Private upload readback differs: {filename}")
        hashes[filename] = digest
    return {
        "id": entry["id"],
        "status": "quick_converted",
        "repo": repo,
        "revision": commit.oid,
        "manifest": manifest,
        "file_hashes": hashes,
    }


def run_entry(entry: dict, root: Path, hub_input: str | None) -> dict:
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
        intended_use=IntendedUse.EVAL if entry["id"] in CROSSCODE_EVAL_SOURCE_IDS else IntendedUse.TRAIN,
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
            )
    finally:
        client.shutdown()
    repo_slug = (
        slug[:REPO_SLUG_LIMIT].rstrip("-")
        + "-"
        + hashlib.sha256(original_source_id(entry["id"]).encode()).hexdigest()[:8]
    )
    return publish_quick(entry, output, f"open-athena/{repo_slug}-taskcompendium-20261112")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--hub-input")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    entries = {entry["id"]: entry for entry in read_registry_json(RETIRED)["entries"]}
    args.root.mkdir(parents=True, exist_ok=True)
    for requested in args.source:
        source_id = canonical_source_id(requested)
        entry = entries[source_id]
        result = run_entry(entry, args.root, args.hub_input)
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
