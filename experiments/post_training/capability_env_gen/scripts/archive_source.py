#!/usr/bin/env python3
"""Create and upload a credential-free reproducibility snapshot for one run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

INCLUDE = (
    "capability_pipeline",
    "scripts",
    "data/pilot.json",
    "data/revocations.json",
    "catalog.json",
    "new_catalog.json",
    "Task Generation.md",
    "pyproject.toml",
    "uv.lock",
    "vendor/task_spec",
    "docs/task_contract.md",
    "docs/revocations.md",
    "docs/build_acceptance_001.md",
    "docs/build_acceptance",
    "docs/audits",
    "docs/partial_controls.md",
    "docs/quality_review.md",
    "docs/construction_repair.md",
    "docs/construction_continuation.md",
)
INPUT_NAMES = (
    "accepted.json",
    "input_pilot.json",
    "plans.json",
    "proposals.json",
    "report.json",
    "manifest.json",
    "seed-manifest.json",
    "seed-provenance.json",
    "daytona-health.json",
)
EXCLUDE_PARTS = {"__pycache__", ".git", ".venv"}
EXCLUDE_NAMES = {"models.template.yml", ".github_token", ".parallel_key"}


def selected(root: Path):
    for name in INCLUDE:
        target = root / name
        if target.is_file():
            yield target
        elif target.is_dir():
            for path in sorted(target.rglob("*")):
                if (
                    path.is_file()
                    and not (set(path.parts) & EXCLUDE_PARTS)
                    and path.name not in EXCLUDE_NAMES
                    and not path.name.startswith("._")
                ):
                    yield path


def _complete_input(source: Path) -> dict[str, Path]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("complete source input lacks manifest.json")
    document = json.loads(manifest_path.read_text())
    if document.get("schema_version") not in {
        "capability-portable-runtime-bundle-v1",
        "capability-portable-runtime-bundle-v2",
        "capability-runtime-evaluation-bundle-v1",
        "capability-fixed-submission-regrade-bundle-v1",
    }:
        raise ValueError("complete source input is not a portable runtime bundle")
    declared = document.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("portable runtime bundle manifest has no file inventory")
    inputs: dict[str, Path] = {}
    for name, expected in declared.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise TypeError("portable runtime bundle inventory is malformed")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("portable runtime bundle contains an unsafe path")
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"portable runtime bundle member is unavailable: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"portable runtime bundle member hash mismatch: {name}")
        inputs[name] = path
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(declared):
        raise ValueError("portable runtime bundle file set differs from its manifest")
    inputs["manifest.json"] = manifest_path
    return inputs


def build_archive(
    root: Path,
    *,
    source: Path | None = None,
    pilot: Path | None = None,
    complete_input: bool = False,
):
    """Snapshot named runtime sources and explicit public launch inputs only."""
    root = root.resolve()
    contents: dict[str, bytes] = {}
    for path in selected(root):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(
                f"source snapshot contains a symlink or external file: {path}"
            )
        contents[path.relative_to(root).as_posix()] = path.read_bytes()
    if source is not None:
        if source.is_dir():
            inputs = (
                _complete_input(source)
                if complete_input
                else {
                    name: source / name
                    for name in INPUT_NAMES
                    if (source / name).is_file()
                }
            )
            if not ({"accepted.json", "input_pilot.json", "plan.json"} & inputs.keys()):
                raise ValueError(
                    "source snapshot input lacks accepted proposals, a proposal seed, or a frozen plan"
                )
        else:
            inputs = {"accepted.json": source}
        for name, path in inputs.items():
            if path.is_symlink():
                raise ValueError("source snapshot input must not be a symlink")
            contents["inputs/source/" + name] = path.read_bytes()
    if pilot is not None:
        if pilot.is_symlink():
            raise ValueError("pilot snapshot input must not be a symlink")
        contents["inputs/pilot.json"] = pilot.read_bytes()
    launch_manifest = contents.get("inputs/source/manifest.json")
    if launch_manifest is not None:
        document = json.loads(launch_manifest)
        required = document.get("required_contract_inputs", {})
        if not isinstance(required, dict):
            raise TypeError("launch contract inputs must be a path-to-digest mapping")
        for name, expected in required.items():
            if (
                name not in contents
                or hashlib.sha256(contents[name]).hexdigest() != expected
            ):
                raise ValueError(
                    f"launch contract input differs from source snapshot: {name}"
                )
        accepted_hash = document.get("accepted_sha256")
        if accepted_hash is not None and (
            "inputs/source/accepted.json" not in contents
            or hashlib.sha256(contents["inputs/source/accepted.json"]).hexdigest()
            != accepted_hash
        ):
            raise ValueError(
                "launch accepted-input digest differs from source snapshot"
            )
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(contents.items())
        },
        "launch_input_scope": "explicit pilot and named source handoff files; credentials and arbitrary adjacent files excluded",
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, data in sorted(contents.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(manifest_bytes))
    return gzip.compress(archive.getvalue(), mtime=0), manifest, manifest_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--input", type=Path, help="actual source seed directory or accepted JSON file"
    )
    parser.add_argument(
        "--pilot", type=Path, help="actual launch pilot, including nondefault samples"
    )
    parser.add_argument(
        "--complete-input",
        action="store_true",
        help="archive every hash-declared member of a portable runtime bundle",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    data, manifest, manifest_bytes = build_archive(
        root,
        source=args.input,
        pilot=args.pilot,
        complete_input=args.complete_input,
    )
    digest = hashlib.sha256(data).hexdigest()
    import fsspec
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    fs, prefix = fsspec.core.url_to_fs(args.destination.rstrip("/"))
    fs.pipe(f"{prefix}/_source_snapshot/{digest}/source.tar.gz", data)
    fs.pipe(f"{prefix}/_source_snapshot/{digest}/manifest.json", manifest_bytes)
    fs.pipe(
        f"{prefix}/_source_snapshot/latest.json",
        json.dumps({"sha256": digest, "files": len(manifest["files"])}).encode(),
    )
    print(json.dumps({"sha256": digest, "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
