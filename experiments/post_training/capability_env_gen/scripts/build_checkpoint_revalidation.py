"""Build a revalidation input from a complete generate snapshot and frozen source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from capability_pipeline.checkpoint_revalidation import (
    PROVENANCE_SCHEMA,
    REQUEST_SCHEMA,
    SCHEMA,
    _inventory,
    _sha,
    _verify_selective_snapshot,
    validate_checkpoint_bundle,
)
from capability_pipeline.inference import atomic_json, digest
from capability_pipeline.proposal_adoption import _member


def build_bundle(checkpoint: Path, source_controller: Path, item_name: str,
                 output: Path, *, construction_root: str = "construction") -> dict:
    """Copy verified bytes only; never import or execute the historical controller."""
    if output.exists() or output.is_symlink():
        raise ValueError("revalidation bundle output must be new")
    files, snapshot = _verify_selective_snapshot(checkpoint, item_name, construction_root)
    if snapshot["remote_final"] is not True:
        raise ValueError("revalidation requires a terminal source snapshot")
    construction = checkpoint if construction_root == "." else _member(checkpoint, construction_root)
    item = _member(construction / "items", item_name)
    accepted = json.loads((item / "contract/accepted.json").read_bytes())
    identity = None
    if construction_root == ".":
        run = json.loads((checkpoint / "run.json").read_bytes())
        submission = json.loads((checkpoint / "submission.json").read_bytes())
        staged_accepted = source_controller / "inputs/source/accepted.json"
        if (run.get("stage") != "synthesize" or run.get("accepted_count") != 1
                or submission.get("phase") != "synthesize"
                or source_controller.name not in Path(run.get("accepted", "")).parts
                or json.loads(staged_accepted.read_bytes()) != [accepted]):
            raise ValueError("standalone synthesis checkpoint differs from its frozen submission")
    else:
        proposals = json.loads((checkpoint / "proposal/accepted.json").read_bytes())
        if accepted not in proposals:
            raise ValueError("item contract is absent from frozen accepted proposals")
        identity = json.loads((checkpoint / "generate-run.json").read_bytes())
        identity_fields = {key: value for key, value in identity.items() if key != "identity_sha256"}
        if digest(identity_fields) != identity.get("identity_sha256"):
            raise ValueError("source generate identity is invalid")
        if _sha(checkpoint / "input-pilot.json") != identity.get("pilot_sha256"):
            raise ValueError("source pilot differs from generate identity")
    package_files = {
        name: value for name, value in _inventory(source_controller / "capability_pipeline").items()
        if "__pycache__" not in Path(name).parts and Path(name).suffix in {".py", ".json"}
    }
    if identity is not None and digest(dict(sorted(package_files.items()))) != identity["settings"]["controller_package_sha256"]:
        raise ValueError("historical controller differs from frozen generate identity")
    source_files = {
        "capability_pipeline/" + name: value for name, value in package_files.items()
    }
    for name in ("source.lock.json", "composite_extension.lock.json"):
        relative = "vendor/task_spec/" + name
        source_files[relative] = _sha(source_controller / relative)
        if source_files[relative] != _sha(item / "contract" / name):
            raise ValueError("historical controller lock differs from task contract")
    if identity is not None and source_files["vendor/task_spec/source.lock.json"] != identity["settings"].get("taskcompendium_lock_sha256"):
        raise ValueError("historical TaskCompendium lock differs from generate identity")
    python_files = {name: value for name, value in package_files.items() if name.endswith(".py")}
    source_provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "source_tree_sha256": hashlib.sha256("".join(
            f"{name}\0{python_files[name]}\n" for name in sorted(python_files)
        ).encode()).hexdigest(),
        "synthesis_sha256": package_files["synthesis.py"],
        "taskcompendium_source_lock_sha256": source_files["vendor/task_spec/source.lock.json"],
        "taskcompendium_overlay_lock_sha256": source_files["vendor/task_spec/composite_extension.lock.json"],
    }
    output.mkdir(parents=True)
    for name in files:
        target = _member(output / "restore-seed", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_member(checkpoint, name), target)
    metadata = output / "source-metadata"
    metadata.mkdir()
    for name in ("pull-manifest.json", "snapshot-capture.json"):
        shutil.copy2(checkpoint / name, metadata / name)
    for name, expected in source_files.items():
        source = _member(source_controller, name)
        if _sha(source) != expected:
            raise ValueError("historical controller changed while bundling")
        target = output / "source-controller" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_json(output / "accepted.json", [accepted])
    request = {
        "schema_version": REQUEST_SCHEMA, "state": "approved",
        "item_name": item_name, "construction_root": construction_root,
        "source_controller": source_provenance,
        "source": {
            "pull_manifest_sha256": _sha(metadata / "pull-manifest.json"),
            "snapshot_capture_sha256": _sha(metadata / "snapshot-capture.json"),
            "source_tree_sha256": hashlib.sha256("".join(
                f"{name}\0{files[name]}\n" for name in sorted(files)
            ).encode()).hexdigest(),
        },
    }
    atomic_json(output / "request.json", request)
    atomic_json(output / "manifest.json", {
        "schema_version": SCHEMA, "state": "prepared_pending_maintained_validation",
        "files": _inventory(output),
    })
    manifest_sha, request_sha = _sha(output / "manifest.json"), _sha(output / "request.json")
    validate_checkpoint_bundle(output, expected_manifest_sha256=manifest_sha,
                               expected_request_sha256=request_sha)
    return {"state": "prepared", "bundle": str(output), "item_name": item_name,
            "manifest_sha256": manifest_sha, "request_sha256": request_sha,
            "source_snapshot_id": snapshot["snapshot_id"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-controller", type=Path, required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--construction-root", default="construction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.checkpoint, args.source_controller, args.item,
                                  args.output, construction_root=args.construction_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
