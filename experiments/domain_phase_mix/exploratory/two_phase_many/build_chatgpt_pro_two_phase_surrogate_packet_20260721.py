# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///
"""Build the portable ChatGPT Pro two-phase surrogate modeling packet."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PACKET = SCRIPT_DIR / "reference_outputs/two_phase_surrogate_collaborator_packet_20260721"
SOURCE_ZIP = SOURCE_PACKET.with_suffix(".zip")
TEMPLATES = SCRIPT_DIR / "chatgpt_pro_two_phase_surrogate_packet_20260721_src"
PACKET = SCRIPT_DIR / "chatgpt_pro_two_phase_surrogate_packet_20260721"
ZIP_PATH = PACKET.with_suffix(".zip")

REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
TEXT_SUFFIXES = {".csv", ".html", ".json", ".md", ".py", ".txt"}
NOISE_NAMES = {".DS_Store", "__pycache__"}
PRIVATE_GCS_PREFIX = re.compile(r"gs://marin-[^/]+/[^/]+/")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_ASSIGNMENT = re.compile(r"(?i)(token|secret|password|api[_-]?key)\s*=\s*[^\s,;]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    return any(part in NOISE_NAMES for part in path.parts) or path.suffix == ".pyc"


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path, *, suffixes: set[str] | None = None) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if ignored(path) or (suffixes is not None and path.suffix not in suffixes):
            continue
        copy_file(path, destination / path.relative_to(source))


def copy_latest_evidence() -> None:
    copy_tree(
        REFERENCE_OUTPUTS / "mechanistic_surrogate_discovery_20260719/final_synthesis",
        PACKET / "evidence/mechanistic_surrogate_discovery/final_synthesis",
    )
    copy_tree(
        REFERENCE_OUTPUTS / "compact_retained_state_sample_efficiency_3e18_20260721",
        PACKET / "evidence/compact_sample_efficiency",
    )
    copy_tree(
        REFERENCE_OUTPUTS / "delphi_compact_optimum_path_validation_results_20260721",
        PACKET / "evidence/compact_raw_optimum_validation",
    )
    copy_tree(
        REFERENCE_OUTPUTS / "delphi_grp_compact_raw_optimum_paths_20260721",
        PACKET / "evidence/grp_compact_raw_optimum_paths",
        suffixes={".csv", ".json", ".md"},
    )
    extension_names = (
        "compact_bucket_mechanisms_3e18_20260721",
        "compact_learned_bucket_exponents_3e18_20260721",
        "compact_learned_bucket_exponents_tight_solver_3e18_20260721",
        "compact_learned_bucket_rates_3e18_20260721",
        "compact_learned_state_vectors_3e18_20260721",
        "compact_nonlinear_solver_audit_3e18_20260721",
        "compact_policy_optimizer_audit_3e18_20260721",
    )
    for name in extension_names:
        copy_tree(
            REFERENCE_OUTPUTS / name,
            PACKET / "evidence/compact_extension_audits" / name,
            suffixes={".csv", ".json", ".md"},
        )


def sanitize_text(text: str) -> str:
    redacted = text.replace(str(REPO_ROOT), "<MARIN_REPO>").replace(str(Path.home()), "<USER_HOME>")
    redacted = PRIVATE_GCS_PREFIX.sub("gs://<redacted-private-prefix>/", redacted)
    redacted = EMAIL.sub("<REDACTED_EMAIL>", redacted)
    return SECRET_ASSIGNMENT.sub(lambda match: match.group(1) + "=<REDACTED>", redacted)


def sanitize_packet() -> None:
    for path in PACKET.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(errors="ignore")
            redacted = sanitize_text(text)
            if redacted != text:
                path.write_text(redacted)


def audit_packet() -> None:
    failures: list[str] = []
    for path in PACKET.rglob("*"):
        if ignored(path):
            failures.append(f"noise file: {path.relative_to(PACKET)}")
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="ignore")
        if "/Users/" in text:
            failures.append(f"absolute home path: {path.relative_to(PACKET)}")
        if EMAIL.search(text):
            failures.append(f"email address: {path.relative_to(PACKET)}")
        if PRIVATE_GCS_PREFIX.search(text):
            failures.append(f"private GCS prefix: {path.relative_to(PACKET)}")
    if failures:
        raise RuntimeError("Packet audit failed:\n" + "\n".join(failures[:50]))


def write_metadata() -> None:
    catalog = json.loads((PACKET / "data/catalog.json").read_text())
    heldout = catalog["datasets"]["delphi_3e18_heldouts"]
    metadata = {
        "packet": PACKET.name,
        "schema_version": 1,
        "source_packet": SOURCE_PACKET.name,
        "source_packet_zip_sha256": sha256(SOURCE_ZIP),
        "official_prompt_inspiration": "https://cdn.openai.com/pdf/04d1d1e4-bc75-476a-97cf-49055cd98d31/cdc_prompt.pdf",
        "delphi_3e18_heldout_rows": heldout["row_count"],
        "delphi_3e18_heldout_target_coverage": heldout["target_coverage"],
        "sealed_evidence": [
            {
                "name": "Compact sub-280 raw-optimum learning-curve panel",
                "included": False,
                "reason": "Outcomes remain sealed for later comparison.",
            }
        ],
    }
    (PACKET / "PACKET_METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_manifest() -> None:
    excluded = {"MANIFEST.json", "CHECKSUMS.sha256"}
    files = []
    for path in sorted(item for item in PACKET.rglob("*") if item.is_file()):
        relative = str(path.relative_to(PACKET))
        if relative in excluded:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "packet": PACKET.name,
        "schema_version": 1,
        "build_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "file_count": len(files),
        "files": files,
        "privacy": "Portable scientific packet with local paths, credentials, and private storage prefixes removed.",
    }
    (PACKET / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksums = [f"{record['sha256']}  {record['path']}" for record in files]
    (PACKET / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n")


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in PACKET.rglob("*") if item.is_file()):
            archive.write(path, Path(PACKET.name) / path.relative_to(PACKET))


def main() -> None:
    if not SOURCE_PACKET.is_dir() or not SOURCE_ZIP.is_file():
        raise FileNotFoundError("Build the current collaborator packet before building the ChatGPT Pro packet")
    if PACKET.exists():
        shutil.rmtree(PACKET)
    copy_tree(SOURCE_PACKET, PACKET)
    copy_latest_evidence()
    copy_tree(TEMPLATES, PACKET)
    for stale in (PACKET / "MANIFEST.json", PACKET / "CHECKSUMS.sha256"):
        stale.unlink(missing_ok=True)
    sanitize_packet()
    write_metadata()
    audit_packet()
    write_manifest()
    build_zip()
    print(f"Built {PACKET}")
    print(f"Archive {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
