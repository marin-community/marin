# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the route-tier changes a requirement backfill causes on one population.

Mirrors the landed tier table's ``route_tiers`` (milestone 1, ``ad57e31a62``; read, not
re-landed): a spec that omits its capabilities keeps the single tier its declared source
requires, while a spec that declares capabilities routes across every satisfying tier,
cheapest first. Emits the per-family counts, the digest-pin distribution that decides the
local tier's image acceptance, and the old-to-new route pairs.
"""

import argparse
import json
import logging
import re
from collections import Counter

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

LOGGER = logging.getLogger(__name__)

DIGEST_PIN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")

LOCAL_CAPABILITIES = frozenset({"shell", "filesystem", "python", "process", "network", "git_repository"})
DAYTONA_CAPABILITIES = LOCAL_CAPABILITIES


def digest_pinned(image: str) -> bool:
    return DIGEST_PIN.match(image) is not None


def old_route(docker_image: str | None, packages_lock: str | None, package_spec: bool) -> tuple[str, ...]:
    """The declared-source route a capabilities-omitting spec keeps today."""
    if docker_image is not None:
        return ("daytona",)
    if packages_lock is not None:
        return ("local",)
    if package_spec:
        return ("local",)
    return ()


def new_route(docker_image: str | None, capabilities: tuple[str, ...], network_declared: bool) -> tuple[str, ...]:
    """The tier table over declared capabilities: ShellSim, local bubblewrap, Daytona."""
    tiers = []
    if not capabilities:
        return old_route(docker_image, None, False)
    if set(capabilities) <= LOCAL_CAPABILITIES and not network_declared and docker_image is not None:
        # Local bubblewrap accepts image sources only when digest-pinned.
        if digest_pinned(docker_image):
            tiers.append("local")
    if docker_image is not None:
        tiers.append("daytona")
    return tuple(tiers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--parent-population", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    def requirements(prefix: str) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        pairs = []
        shards = sorted(str(path) for path in (StoragePath(prefix) / "specs/*.parquet").glob())
        for shard in shards:
            with StoragePath(shard).open("rb") as stream:
                table = pq.read_table(stream, columns=["task_spec"])
            for encoded in table.column("task_spec").to_pylist():
                spec = json.loads(encoded)
                environment = spec.get("environment_requirements") or {}
                docker_image = environment.get("docker_image")
                capabilities = tuple(sorted(environment.get("capabilities") or ()))
                packages_lock = environment.get("packages_lock")
                package_spec = environment.get("package_spec") is not None
                pairs.append(
                    (
                        old_route(docker_image, packages_lock, package_spec),
                        new_route(docker_image, capabilities, "network" in capabilities),
                    )
                )
        return pairs

    old_pairs = requirements(args.parent_population)
    new_pairs = requirements(args.population)
    if len(old_pairs) != len(new_pairs):
        raise ValueError("The parent and backfilled populations disagree on row count")
    changes = Counter()
    images = Counter()
    for (old, _), (_, new) in zip(old_pairs, new_pairs, strict=True):
        changes[f"{'+'.join(old) or 'none'} -> {'+'.join(new) or 'none'}"] += 1
    shards = sorted(str(path) for path in (StoragePath(args.population) / "specs/*.parquet").glob())
    for shard in shards:
        with StoragePath(shard).open("rb") as stream:
            table = pq.read_table(stream, columns=["task_spec"])
        for encoded in table.column("task_spec").to_pylist():
            image = (json.loads(encoded).get("environment_requirements") or {}).get("docker_image")
            if image is not None:
                images["digest_pinned" if digest_pinned(image) else "unpinned"] += 1
    report = {
        "source_id": args.source_id,
        "tasks": len(old_pairs),
        "route_changes": dict(changes),
        "image_digest_pins": dict(images),
    }
    output = StoragePath(args.output)
    (output / "route-changes.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
