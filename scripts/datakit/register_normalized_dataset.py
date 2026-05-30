# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register a normalized Datakit dataset in the artifact registry.

Each :class:`~marin.datakit.sources.DatakitSource` exposes a `normalized` `StepSpec`
whose `output_path` is the canonical (content-addressed) location of the normalized
bytes. This script maps a source name to an artifact id, picks a CalVer version, and
records `(id, version) -> output_path` so downstream code can load it by name:

    Artifact.from_id("nemotron_cc_v2_1/high_quality", "2026.05.29", NormalizedData)

Examples
--------
List the available source names:

    uv run python scripts/datakit/register_normalized_dataset.py --list

Dry-run (print what would be registered, write nothing):

    uv run python scripts/datakit/register_normalized_dataset.py \
        nemotron_cc_v2_1/high_quality --dry-run

Register into the default registry (MARIN_ARTIFACT_REGISTRY, else the canonical
gs://marin-us-central1/artifact_registry), verifying the artifact loads first:

    uv run python scripts/datakit/register_normalized_dataset.py nemotron_cc_v2_1/high_quality

Register into a throwaway local registry instead:

    uv run python scripts/datakit/register_normalized_dataset.py cp/biodiversity \
        --registry-root /tmp/marin/artifact_registry
"""

import argparse
import datetime
import sys

from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.artifact import Artifact
from marin.execution.artifact_registry import (
    ArtifactAlreadyExistsError,
    ArtifactRegistry,
    FilesystemArtifactRegistry,
    get_default_registry,
)


def source_name_to_artifact_id(source_name: str) -> str:
    """Map a Datakit source name to a `<namespace>/<name>` artifact id.

    Source names already carry their own slashes (`cp/biodiversity`,
    `safety_pt/moral_education/score_4_morals`); an artifact id is exactly one
    slash, so the first segment becomes the namespace and the rest is joined with
    `.`. Slash-free names (`climblab-ja`) are placed under the `datasets` namespace.
    """
    namespace, _, remainder = source_name.partition("/")
    if not remainder:
        return f"datasets/{namespace}"
    return f"{namespace}/{remainder.replace('/', '.')}"


def default_version() -> str:
    """Today's date as a CalVer `YYYY.MM.DD` version string."""
    return datetime.date.today().strftime("%Y.%m.%d")


def register_source(
    source: DatakitSource,
    version: str,
    registry: ArtifactRegistry,
    *,
    verify: bool,
    dry_run: bool,
) -> int:
    artifact_id = source_name_to_artifact_id(source.name)
    uri = source.normalized.output_path

    print(f"source : {source.name} (~{source.rough_token_count_b:.2f}B tokens)")
    print(f"id     : {artifact_id}")
    print(f"version: {version}")
    print(f"uri    : {uri}")

    if verify:
        print("verify : loading the normalized artifact via Artifact.from_path ...")
        # Raises FileNotFoundError if neither the sidecar nor a SUCCESS marker is present.
        Artifact.from_path(uri)
        print("verify : ok")

    if dry_run:
        print("dry-run: nothing written")
        return 0

    try:
        entry = registry.register(artifact_id, version, uri)
    except ArtifactAlreadyExistsError as e:
        existing = e.existing
        print(f"exists : {artifact_id}@{version} already registered -> {existing.uri}")
        # Idempotent if it points at the same bytes; otherwise the caller picked a
        # colliding version and must choose a new one (the registry is append-only).
        return 0 if existing.uri == uri else 1

    print(f"ok     : registered {entry.id}@{entry.version} -> {entry.uri}")
    if entry.relative_path is not None:
        print(f"         (relative_path={entry.relative_path})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="Datakit source name, e.g. 'nemotron_cc_v2_1/high_quality'")
    parser.add_argument("--version", default=None, help="CalVer version (default: today, YYYY.MM.DD)")
    parser.add_argument(
        "--registry-root",
        default=None,
        help="Registry root URI. Default: the process default (MARIN_ARTIFACT_REGISTRY / canonical root).",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Do not load the artifact to confirm it exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing to the registry")
    parser.add_argument("--list", action="store_true", help="List available source names and exit")
    args = parser.parse_args()

    sources = all_sources()

    if args.list:
        for name in sorted(sources):
            print(name)
        return 0

    if not args.name:
        parser.error("a source name is required (use --list to see the options)")
    if args.name not in sources:
        parser.error(f"unknown source {args.name!r}; use --list to see the options")

    registry = FilesystemArtifactRegistry(args.registry_root) if args.registry_root else get_default_registry()

    return register_source(
        sources[args.name],
        args.version or default_version(),
        registry,
        verify=not args.skip_verify,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
