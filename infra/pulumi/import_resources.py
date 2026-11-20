# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate and apply reviewed Program-first Pulumi import transactions."""

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

# Pulumi Automation starts gRPC threads before invoking subprocesses. Its fork
# hook is unnecessary because each child immediately execs pulumi or gcloud.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from iac.imports import (
    IMPORT_MANIFEST_MODE,
    ImportCatalog,
    ImportManifest,
    ImportManifestSummary,
    import_manifest_summary,
    resolve_import_manifest,
    validate_reviewed_manifest,
    write_import_manifest,
)
from iac.program import build_stack
from pulumi import automation as auto

PULUMI_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PULUMI_DIR.parents[1]
CONFIRMATION_DIGEST_LENGTH = 12
IMPORT_TEMP_DIRECTORY_PREFIX = "marin-iac-import-"
GENERATED_MANIFEST_FILENAME = "generated.json"


def _load_manifest(path: Path) -> ImportManifest:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("import manifest must be a JSON object")
    return manifest


def _require_local_manifest(path: Path, *, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    if resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("import manifests contain provider IDs and must be kept outside the repository")
    if must_exist and stat.S_IMODE(resolved.stat().st_mode) != IMPORT_MANIFEST_MODE:
        raise ValueError("reviewed import manifest must have mode 0600")
    return resolved


def _generate_manifest(stack_name: str, output_path: Path) -> ImportManifest:
    catalog = ImportCatalog()
    stack = auto.select_stack(
        stack_name=stack_name,
        work_dir=str(PULUMI_DIR),
    )
    stack.preview(
        program=lambda: build_stack(catalog),
        import_file=str(output_path),
        suppress_outputs=True,
    )
    generated = _load_manifest(output_path)
    return resolve_import_manifest(generated, catalog.specs)


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{resource_type}={count}" for resource_type, count in counts.items()) or "none"


def _print_summary(summary: ImportManifestSummary) -> None:
    print(f"imports: {_format_counts(summary.imports_by_type)}")
    print(f"unresolved creates: {_format_counts(summary.unresolved_by_type)}")
    print(f"components: {summary.component_count}")
    print(f"sha256: {summary.digest}")


def generate(stack_name: str, output_path: Path) -> None:
    """Generate the current create set and fill cataloged provider IDs."""

    output_path = _require_local_manifest(output_path, must_exist=False)
    with tempfile.TemporaryDirectory(prefix=IMPORT_TEMP_DIRECTORY_PREFIX) as temp_directory:
        generated_path = Path(temp_directory) / GENERATED_MANIFEST_FILENAME
        manifest = _generate_manifest(stack_name, generated_path)
    write_import_manifest(output_path, manifest)
    _print_summary(import_manifest_summary(manifest))
    print(f"wrote owner-only manifest: {output_path}")


def _run_pulumi(arguments: Sequence[str]) -> None:
    subprocess.run(["pulumi", *arguments], cwd=PULUMI_DIR, check=True)


def apply(stack_name: str, reviewed_path: Path) -> None:
    """Validate and execute an unchanged subset of the current import candidates."""

    reviewed_path = _require_local_manifest(reviewed_path, must_exist=True)
    reviewed = _load_manifest(reviewed_path)
    with tempfile.TemporaryDirectory(prefix=IMPORT_TEMP_DIRECTORY_PREFIX) as temp_directory:
        generated_path = Path(temp_directory) / GENERATED_MANIFEST_FILENAME
        current = _generate_manifest(stack_name, generated_path)

    summary = validate_reviewed_manifest(reviewed, current)
    _print_summary(summary)
    _run_pulumi(
        [
            "import",
            "--stack",
            stack_name,
            "--file",
            str(reviewed_path),
            "--preview-only",
            "--generate-code=false",
        ]
    )

    confirmation_token = f"import {summary.digest[:CONFIRMATION_DIGEST_LENGTH]}"
    confirmation = input(f"Type '{confirmation_token}' to apply this protected import: ")
    if confirmation != confirmation_token:
        raise ValueError("import confirmation did not match the reviewed manifest digest")

    _run_pulumi(
        [
            "import",
            "--stack",
            stack_name,
            "--file",
            str(reviewed_path),
            "--generate-code=false",
            "--yes",
        ]
    )
    _run_pulumi(["preview", "--stack", stack_name])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate a local import manifest")
    generate_parser.add_argument("--stack", required=True, help="Pulumi stack name")
    generate_parser.add_argument("--output", required=True, type=Path, help="path outside the repository")

    apply_parser = subparsers.add_parser("apply", help="validate and execute a reviewed manifest")
    apply_parser.add_argument("--stack", required=True, help="Pulumi stack name")
    apply_parser.add_argument("--file", required=True, type=Path, help="reviewed mode-0600 manifest")
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(arguments)
    if args.command == "generate":
        generate(args.stack, args.output)
    else:
        apply(args.stack, args.file)


if __name__ == "__main__":
    main()
