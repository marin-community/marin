# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a retired native Harbor source's recorded population into executable TaskSpecs.

The zero-task SWE sources (ruling 2026-11-26) failed before any oracle trial: the
contracts pass recorded the ``mutable_docker_base`` blocker on every task because
their package Dockerfiles pull third-party base images by mutable tag, and the
mechanical wave failed closed with no oracle trial and no image build. The repair
runs at conversion time: each task's Dockerfile ``FROM`` bases resolve to their
registry digests and the FROM lines are rewritten to the digest-pinned references —
the same image bytes under an immutable name, with every pin recorded in the
conversion receipt. The population that converts is the source's recorded QUICK
release (the census-measured native TaskSpec packages), verified byte-exact against
the registry's recorded file hash.

Everything else reuses the cohort conversion unchanged: the executable grader is
the package's own ``tests/test.sh`` harness re-expressed for the rollout sandbox,
with the cohort converter's typed rejections. The run is strict: any row that
still carries a static blocker after pinning (or that the converter rejects) fails
the run, because a shrunken population would change the census contract without
typed admission evidence.
"""

import argparse
import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.licenses import LicenseReceipt
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import DOCKER_IMAGE_PATTERN, AnswerType, DockerBuildExecution, TaskSpec
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.calibration_inventory import task_blockers
from experiments.post_training.task_curation.harbor_cohort_conversion import (
    CONVERSION_PROTOCOL,
    CohortConversionRejected,
    convert_cohort_task,
)
from experiments.post_training.task_curation.harbor_cohort_release import (
    converter_revision,
    publish_conversion,
    release_repo,
    write_conversion_artifacts,
)
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.license_publication import check_publication
from experiments.post_training.task_curation.publication_gate import require_public_data
from experiments.post_training.task_curation.registry import RegistryReference, read_registry_document
from experiments.post_training.task_curation.source import DataRestriction, SourceInfo, SourceOrigin, SourceReference
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned

LOGGER = logging.getLogger(__name__)

RETIRED_INVENTORY = RegistryReference(
    repo="open-athena/marin-rl-data-atlas-registry",
    revision="1e406e232a49daaa030b416d892acdd54f967853",
    path="experiments/post_training/task_curation/retired-inventory-only.json",
    sha256="d2329f648bff0edd2bc1799c0f5792a4d3dd04a5e4893bbb1311aafeb96a9f6f",
)
DOCKERFILE_RESOURCE = "environment/Dockerfile"
DOCKER_HUB_REGISTRY = "registry-1.docker.io"
DOCKER_HUB_AUTH = "auth.docker.io"
DOCKER_HUB_SERVICE = "registry.docker.io"
MANIFEST_ACCEPT = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.oci.image.manifest.v1+json"
)
DIGEST_PATTERN = r"sha256:[0-9a-f]{64}"


class BasePinRejected(ValueError):
    """A Dockerfile base the registry cannot resolve; the row stays unconverted."""


@dataclass(frozen=True)
class BasePin:
    """One resolved FROM base: the original reference and its immutable digest form."""

    base: str
    pinned: str


@dataclass(frozen=True)
class NativeConversionSource:
    """The retired inventory entry as a conversion source: declared pin and receipt."""

    info: SourceInfo
    dataset: SourceReference
    intended_use: IntendedUse
    release_repo: str
    release_revision: str
    task_specs_sha256: str


@dataclass(frozen=True)
class NativeRowConversion:
    """One converted row: the executable TaskSpec, its ledger entry and the base pins."""

    task: TaskSpec
    input_task_sha256: str
    output_task_sha256: str
    base_pins: tuple[BasePin, ...]


def load_inventory_entry(source_id: str) -> dict:
    """The retired inventory entry for one source, from the pinned registry document."""
    document = read_registry_document(RETIRED_INVENTORY)
    entry = next((item for item in document["entries"] if item.get("id") == source_id), None)
    if entry is None:
        raise ValueError(f"Source {source_id} is not in the pinned retired inventory")
    release = entry.get("disposition_release") or {}
    if not release.get("repo") or not release.get("revision"):
        raise ValueError(f"Source {source_id} carries no recorded conversion release")
    return entry


def conversion_source(entry: dict, receipt: LicenseReceipt | None) -> NativeConversionSource:
    """Build the conversion source view: declared pin, recorded release, receipt."""
    release = entry["disposition_release"]
    info = SourceInfo(
        data_restriction=DataRestriction.OPEN,
        id=entry["id"],
        upstream_dataset=entry.get("name") or entry["id"].partition(":")[2],
        upstream_name=entry.get("display_name") or entry.get("name") or entry["id"],
        build_path="Task Trove",
        origin=SourceOrigin.TASKTROVE,
        family=entry.get("family") or "swe-repo",
        tags=tuple(entry.get("tags") or ()),
        count=entry.get("task_count"),
        notes=entry.get("notes") or "",
        dataset=SourceReference(
            name=entry["dataset_id"],
            revision=entry["dataset_revision"],
            url=entry.get("url"),
        ),
        license_receipt=receipt,
    )
    return NativeConversionSource(
        info=info,
        dataset=info.dataset,
        intended_use=IntendedUse.TRAIN,
        release_repo=release["repo"],
        release_revision=release["revision"],
        task_specs_sha256=release["file_hashes"]["tasks.parquet"],
    )


def _docker_hub_repo(name: str) -> str:
    """Docker Hub stores official images under the library/ namespace."""
    return name if "/" in name else f"library/{name}"


_DIGEST_CACHE: dict[str, str] = {}


def resolve_base_digest(reference: str) -> str:
    """Resolve one base image reference to its digest-pinned form via the registry.

    ``registry-1.docker.io`` serves Docker Hub (official images under ``library/``);
    any other host is its own registry with the same v2 API. An already
    digest-pinned reference passes through unchanged. Resolutions cache per
    process: a cohort's tasks share bases, and the shared registries rate-limit.
    """
    if re.fullmatch(DOCKER_IMAGE_PATTERN, reference):
        return reference
    if reference in _DIGEST_CACHE:
        return _DIGEST_CACHE[reference]
    pinned = _resolve_base_digest(reference)
    _DIGEST_CACHE[reference] = pinned
    return pinned


def _resolve_base_digest(reference: str) -> str:
    registry_host = DOCKER_HUB_REGISTRY
    name = reference
    if "://" in reference or reference.startswith(("docker.io/", "registry-1.docker.io/")):
        raise BasePinRejected(f"Unsupported base reference form: {reference}")
    first, _, rest = reference.partition("/")
    if "." in first and rest:
        registry_host = first if first != "docker.io" else DOCKER_HUB_REGISTRY
        name = rest
    name, _, digest = name.partition("@")
    if digest:
        return f"{name}@{digest}"
    repo, _, tag = name.rpartition(":")
    if not repo or "/" in tag:
        repo, tag = name, "latest"
    scope_repo = _docker_hub_repo(repo) if registry_host == DOCKER_HUB_REGISTRY else repo
    token_host = DOCKER_HUB_AUTH if registry_host == DOCKER_HUB_REGISTRY else registry_host
    token_service = DOCKER_HUB_SERVICE if registry_host == DOCKER_HUB_REGISTRY else registry_host
    token_url = (
        f"https://{token_host}/token?service={urllib.parse.quote(token_service)}"
        f"&scope=repository:{urllib.parse.quote(scope_repo)}:pull"
    )
    with urllib.request.urlopen(token_url, timeout=60) as response:
        token = json.loads(response.read())["token"]
    request = urllib.request.Request(
        f"https://{registry_host}/v2/{scope_repo}/manifests/{urllib.parse.quote(tag)}",
        headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        manifest_digest = response.headers.get("Docker-Content-Digest")
    if not manifest_digest or not re.fullmatch(DIGEST_PATTERN, manifest_digest):
        raise BasePinRejected(f"The registry returned no usable manifest digest for {reference}")
    return f"{name}@{manifest_digest}"


def pinned_dockerfile(text: str, resolve: Callable[[str], str] = resolve_base_digest) -> tuple[str, tuple[BasePin, ...]]:
    """Rewrite every external FROM base to its digest-pinned reference.

    Stage names, ``--platform`` flags, ``scratch`` and already-pinned references
    pass through unchanged; a base the registry cannot resolve is a typed
    rejection.
    """
    stages: set[str] = set()
    pins: dict[str, BasePin] = {}
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        tokens = stripped.split() if stripped and not stripped.startswith("#") else []
        if not tokens or tokens[0].upper() != "FROM":
            lines.append(raw)
            continue
        fields = [token for token in tokens[1:] if not token.startswith("--")]
        base = fields[0] if fields else ""
        if len(fields) >= 3 and fields[1].upper() == "AS":
            stages.add(fields[2])
        if base in stages or base == "scratch" or re.fullmatch(DOCKER_IMAGE_PATTERN, base):
            lines.append(raw)
            continue
        if base not in pins:
            try:
                pinned = resolve(base)
            except (ValueError, OSError) as error:
                raise BasePinRejected(f"FROM base {base} does not resolve: {error}") from error
            pins[base] = BasePin(base=base, pinned=pinned)
        lines.append(stripped.replace(base, pins[base].pinned, 1))
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), tuple(pins.values())


def pin_package_dockerfile(
    task: TaskSpec, resolve: Callable[[str], str] = resolve_base_digest
) -> tuple[TaskSpec, tuple[BasePin, ...]]:
    """Pin the package's build Dockerfile in place; refuse a task without one."""
    dockerfile = next((resource for resource in task.resources.verifier if resource.path == DOCKERFILE_RESOURCE), None)
    if dockerfile is None:
        raise CohortConversionRejected(f"{task.id}: the package carries no {DOCKERFILE_RESOURCE}")
    pinned_text, pins = pinned_dockerfile(resource_bytes(dockerfile).decode(), resolve)
    if not pins:
        return task, ()
    pinned_resource = inline_resource(DOCKERFILE_RESOURCE, pinned_text.encode()).model_copy(
        update={"mode": dockerfile.mode, "mtime_ns": dockerfile.mtime_ns}
    )
    updated = tuple(
        pinned_resource if resource.path == DOCKERFILE_RESOURCE else resource for resource in task.resources.verifier
    )
    return task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": updated})}), pins


def convert_native_source_row(
    task: TaskSpec, *, input_task_sha256: str, resolve: Callable[[str], str] = resolve_base_digest
) -> NativeRowConversion:
    """Pin the package's base images, convert the grader, and refuse any static blocker.

    The pin check runs on the pinned native package (the only shape whose static
    verification reads the package Dockerfile); the grader-contract check runs on
    the converted row through the declarative factory route, the census's own
    classification for unbuilt-image tasks.
    """
    pinned, pins = pin_package_dockerfile(task, resolve)
    native_blockers = task_blockers(pinned)
    if native_blockers:
        raise CohortConversionRejected(f"{task.id}: static blockers remain after pinning: {sorted(native_blockers)}")
    outcome = convert_cohort_task(pinned, input_task_sha256=input_task_sha256)
    if outcome.task.answer_type not in (AnswerType.WORKSPACE_STATE, AnswerType.TEXT):
        raise CohortConversionRejected(f"{task.id}: unexpected converted answer type {outcome.task.answer_type}")
    converted_blockers = task_blockers(outcome.task, docker_build=DockerBuildExecution.DECLARATIVE)
    if converted_blockers:
        raise CohortConversionRejected(
            f"{task.id}: static blockers remain on the converted task: {sorted(converted_blockers)}"
        )
    return NativeRowConversion(
        task=outcome.task,
        input_task_sha256=outcome.input_task_sha256,
        output_task_sha256=outcome.output_task_sha256,
        base_pins=pins,
    )


def load_release_population(source: NativeConversionSource, *, limit: int | None) -> list[dict]:
    """Download the recorded release population and verify it byte-exact against the registry."""
    path = Path(
        hf_hub_download(source.release_repo, "tasks.parquet", repo_type="dataset", revision=source.release_revision)
    )
    digest = file_sha256(StoragePath(str(path)))
    if digest != source.task_specs_sha256:
        raise ValueError(
            f"The recorded release population changed: {source.release_repo}@{source.release_revision} "
            f"sha256 {digest} != {source.task_specs_sha256}"
        )
    rows = pq.read_table(path, columns=["task_spec"]).to_pylist()
    if limit is not None:
        rows = rows[:limit]
    return rows


def convert_population(source: NativeConversionSource, *, limit: int | None) -> tuple[list[dict], list[dict], dict]:
    """Convert the recorded population; strict — any rejection or blocker fails the run."""
    rows = load_release_population(source, limit=limit)
    converted: list[dict] = []
    receipts: list[dict] = []
    for row in rows:
        task = TaskSpec.model_validate_json(row["task_spec"])
        outcome = convert_native_source_row(task, input_task_sha256=task_spec_sha256(task))
        serialized = outcome.task.model_dump_json()
        converted.append(
            {
                "path": task.source.row,
                "task_spec": serialized,
                "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            }
        )
        receipts.append(
            {
                "task_id": task.id,
                "protocol": CONVERSION_PROTOCOL,
                "input_task_sha256": outcome.input_task_sha256,
                "output_task_sha256": outcome.output_task_sha256,
                "changed_fields": ["environment_requirements", "execution_requirements", "grader", "tags", "labels"],
                "base_pins": [{"base": pin.base, "pinned": pin.pinned} for pin in outcome.base_pins],
            }
        )
    return converted, receipts, {"input_rows": len(rows), "converted_rows": len(converted)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="The retired native source id to convert.")
    parser.add_argument("--output-root", required=True, help="The conversion output directory (local scratch).")
    parser.add_argument("--artifact-version", required=True, help="The release repo version suffix.")
    parser.add_argument("--limit", type=int, help="Convert only the first N rows (local validation slices).")
    parser.add_argument(
        "--converter-revision",
        help="The landed revision the conversion runs from; the bootstrap passes the archive revision.",
    )
    parser.add_argument(
        "--run", action="store_true", help="Publish the private release; omit to validate and plan only."
    )
    parser.add_argument("--report", type=Path, help="Write the result JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    entry = load_inventory_entry(args.source_id)
    receipts = read_license_receipts(LICENSE_RECEIPT_INDEX)
    receipt = receipts.get(args.source_id)
    require_unbanned(args.source_id, entry["dataset_id"], receipt)
    source = conversion_source(entry, receipt)
    require_public_data(source.info.data_restriction, source.info.id, source.dataset.name)
    converted, receipts_out, counts = convert_population(source, limit=args.limit)
    manifest = write_conversion_artifacts(
        converted, receipts_out, counts, output_root=args.output_root, source_id=source.info.id
    )
    manifest["converter_revision"] = converter_revision(args.converter_revision)
    manifest["mode"] = "publish" if args.run else "plan"
    manifest["declared_pin"] = {"dataset_id": entry["dataset_id"], "revision": entry["dataset_revision"]}
    manifest["input_release"] = {
        "repo": source.release_repo,
        "revision": source.release_revision,
        "tasks_parquet_sha256": source.task_specs_sha256,
    }
    # The published release is self-describing: rewrite the staged manifest before the
    # upload so it carries the declared pin and the verified input-release identity.
    (Path(args.output_root) / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not args.run:
        print(
            json.dumps({"plan": manifest, "release_repo": release_repo(source.info.id, args.artifact_version)}, indent=2)
        )
        return
    outcome = publish_conversion(
        source,  # type: ignore[arg-type]
        output_root=args.output_root,
        version=args.artifact_version,
        manifest=manifest,
    )
    result = {
        **manifest,
        "release_repo": outcome.release_repo,
        "release_revision": outcome.release_revision,
        "task_specs_sha256": outcome.task_specs_sha256,
        "license_check": check_publication(source.info.id, outcome.release_repo, outcome.release_revision, "train"),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
