# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check every GHCR package in the built-image registry against its allowance.

For every GHCR package the registry map references, this reads the package's
actual visibility (``gh api /orgs/marin-community/packages/container/<name>``)
and compares it with the most open visibility the image's sources allow. The
allowance comes from the metadata the map already records, resolved the way the
Atlas resolves it: a ``private_always`` data tier allows only private use — the
image must not sit in any GHCR package, whose org scope is more open than the
private store; a recorded ``public_display`` publication scope allows a public
package; every other source allows org-internal use. GitHub.com GHCR reports
``public`` and ``private`` only, so a GHCR-private marin-community package is
org-internal on this scale (``private < internal < public``).

A package is over-exposed when its actual visibility is more open than the
most restrictive source's allowance. References outside the audited org are
listed as out of scope, never checked. The check is read-only: it never
changes a package's visibility. It exits non-zero when any package is
over-exposed, when a referenced package is missing, or when a query fails — an
unverifiable package never passes.
"""

import argparse
import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.status_taxonomy import DataRestriction
from huggingface_hub import hf_hub_download

from experiments.post_training.task_curation.image_registry import MAP_COLUMNS, MAP_FILENAME, REGISTRY_REPO
from infra.marina.apps.rl_data_catalog.server.publication_scope import PublicationScope

GHCR_HOST = "ghcr.io"
CHECKED_ORG = "marin-community"


class ImageVisibility(StrEnum):
    """How open a built image may be, ordered private < internal < public."""

    PRIVATE = "private"
    INTERNAL = "internal"
    PUBLIC = "public"


OPENNESS = {ImageVisibility.PRIVATE: 0, ImageVisibility.INTERNAL: 1, ImageVisibility.PUBLIC: 2}

GHCR_ACTUAL_VISIBILITY = {
    "private": ImageVisibility.INTERNAL,
    "internal": ImageVisibility.INTERNAL,
    "public": ImageVisibility.PUBLIC,
}
"""GitHub.com GHCR reports public/private only; a GHCR-private package is
org-internal on this scale. An ``internal`` string (GitHub Enterprise) maps to
the same rank."""


class PackageVisibilityError(RuntimeError):
    """Reading one package's visibility failed."""


class MissingPackage(PackageVisibilityError):
    """The map references a GHCR package that no longer exists."""


class CheckOutcome(StrEnum):
    MATCH = "match"
    OVER_EXPOSED = "over-exposed"
    MISSING = "missing-package"
    QUERY_ERROR = "query-error"
    OUT_OF_SCOPE = "out-of-scope"


@dataclass(frozen=True)
class PackageCheck:
    """One package's actual visibility against its metadata allowance.

    ``actual`` is None when the package's visibility could not be read
    (missing package, failed query); such a check always fails.
    """

    package: str
    actual: ImageVisibility | None
    allowance: ImageVisibility
    outcome: CheckOutcome
    sources: tuple[str, ...]
    image_references: tuple[str, ...]


@dataclass(frozen=True)
class RegistryCheck:
    """The verdicts of one check run, plus the references it could not check."""

    verdicts: tuple[PackageCheck, ...]
    out_of_scope: tuple[str, ...]

    @property
    def failures(self) -> tuple[PackageCheck, ...]:
        return tuple(verdict for verdict in self.verdicts if verdict.outcome is not CheckOutcome.MATCH)


def row_allowance(row: dict[str, str]) -> ImageVisibility:
    """The most open visibility one map row's source metadata allows."""
    if row["data_restriction"] == DataRestriction.PRIVATE_ALWAYS.value:
        return ImageVisibility.PRIVATE
    if row["publication_scope"] == PublicationScope.PUBLIC_DISPLAY.value:
        return ImageVisibility.PUBLIC
    return ImageVisibility.INTERNAL


def ghcr_package(reference: str) -> str | None:
    """The ghcr.io package of one image reference without tag or digest, or None."""
    name = reference.split("@", 1)[0]
    host, separator, path = name.partition("/")
    if host != GHCR_HOST or not separator:
        return None
    package, colon, _tag = path.rpartition(":")
    return (package if colon else path).lower()


def checked_package(reference: str) -> str | None:
    """The audited org's package this reference lives in, or None when it does not."""
    package = ghcr_package(reference)
    if package is None or package.partition("/")[0] != CHECKED_ORG:
        return None
    return package


def ghcr_visibility(package: str) -> str:
    """Read one package's actual visibility through the read-only gh CLI.

    The org-scoped endpoint takes the package name without its org prefix;
    a nested name keeps its slash, URL-encoded.
    """
    name = package.removeprefix(f"{CHECKED_ORG}/")
    encoded = name.replace("/", "%2F")
    result = subprocess.run(
        ("gh", "api", f"/orgs/{CHECKED_ORG}/packages/container/{encoded}"),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "HTTP 404" in result.stderr:
            raise MissingPackage(f"Package does not exist: {CHECKED_ORG}/{package}")
        raise PackageVisibilityError(f"gh api failed for {package}: {result.stderr.strip()}")
    visibility = json.loads(result.stdout).get("visibility")
    if visibility not in GHCR_ACTUAL_VISIBILITY:
        raise PackageVisibilityError(f"Unknown package visibility {visibility!r}: {package}")
    return visibility


def check_registry(rows: Iterable[dict[str, str]], *, visibility: Callable[[str], str]) -> RegistryCheck:
    """Compare every package the map references with the allowance its sources allow."""
    grouped: dict[str, list[dict[str, str]]] = {}
    out_of_scope: set[str] = set()
    for row in rows:
        package = checked_package(row["image_reference"])
        if package is None:
            out_of_scope.add(row["image_reference"])
        else:
            grouped.setdefault(package, []).append(row)

    verdicts: list[PackageCheck] = []
    for package, package_rows in sorted(grouped.items()):
        allowance = min((row_allowance(row) for row in package_rows), key=lambda open_to: OPENNESS[open_to])
        sources = tuple(sorted({row["source_id"] for row in package_rows}))
        references = tuple(sorted({row["image_reference"] for row in package_rows}))
        try:
            actual = GHCR_ACTUAL_VISIBILITY[visibility(package)]
        except MissingPackage:
            verdicts.append(PackageCheck(package, None, allowance, CheckOutcome.MISSING, sources, references))
            continue
        except PackageVisibilityError:
            verdicts.append(PackageCheck(package, None, allowance, CheckOutcome.QUERY_ERROR, sources, references))
            continue
        outcome = CheckOutcome.MATCH if OPENNESS[actual] <= OPENNESS[allowance] else CheckOutcome.OVER_EXPOSED
        verdicts.append(PackageCheck(package, actual, allowance, outcome, sources, references))
    return RegistryCheck(tuple(verdicts), tuple(sorted(out_of_scope)))


def render(check: RegistryCheck) -> str:
    """The typed result table: one row per checked package, then the out-of-scope list."""
    lines = [f"{'package':<56} {'actual':<9} {'allowance':<10} {'outcome':<15} sources"]
    for verdict in check.verdicts:
        actual = verdict.actual.value if verdict.actual is not None else "-"
        lines.append(
            f"{verdict.package:<56} {actual:<9} {verdict.allowance.value:<10} "
            f"{verdict.outcome.value:<15} {', '.join(verdict.sources)}"
        )
    for verdict in check.verdicts:
        if verdict.outcome is not CheckOutcome.MATCH:
            lines.extend(f"  {verdict.package}: {reference}" for reference in verdict.image_references)
    if check.out_of_scope:
        lines.append(f"Out of scope ({len(check.out_of_scope)} references outside {CHECKED_ORG} on ghcr.io):")
        lines.extend(f"  {reference}" for reference in check.out_of_scope)
    return "\n".join(lines)


def read_map(map_path: Path | None, repo: str, revision: str | None) -> list[dict[str, str]]:
    """The map rows from a local parquet or the registry dataset at a revision."""
    if map_path is not None:
        local = map_path
    else:
        local = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=MAP_FILENAME, revision=revision))
    table = pq.read_table(local)
    missing = [column for column in MAP_COLUMNS if column not in table.column_names]
    if missing:
        raise ValueError(f"Map lacks its columns {missing}: {local}")
    return table.select(list(MAP_COLUMNS)).to_pylist()


def main() -> int:
    """Run the visibility check and exit non-zero on any mismatch or failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REGISTRY_REPO, help="private dataset repo the map lives on")
    parser.add_argument("--revision", default=None, help="pin the map revision; default is the repo's latest commit")
    parser.add_argument("--map", type=Path, default=None, help="read this local map parquet instead of the dataset")
    args = parser.parse_args()

    rows = read_map(args.map, args.repo, args.revision)
    check = check_registry(rows, visibility=ghcr_visibility)
    print(render(check))
    failures = check.failures
    if failures:
        print(f"{len(failures)} of {len(check.verdicts)} packages violate their metadata allowance")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
