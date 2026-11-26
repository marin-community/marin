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
import base64
import hashlib
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.licenses import LicenseDecision, LicenseEvidence, LicenseReceipt, LicenseSourcePin
from huggingface_hub import HfApi, hf_hub_download
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
)
from experiments.post_training.task_curation.license_binding import (
    LICENSE_RECEIPT_INDEX,
    AppendWouldRewriteSharedShard,
    append_license_receipt,
    read_license_receipts,
)
from experiments.post_training.task_curation.license_publication import (
    check_publication,
    licensed_dataset_card,
    prepare_license_publication,
)
from experiments.post_training.task_curation.license_resolution import LicenseTarget, resolve_license
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
RATE_LIMIT_MAX_WAIT_SECONDS = 600.0
_FROM = re.compile(r"\s+")


class BasePinRejected(ValueError):
    """A Dockerfile base the registry cannot resolve; the row stays unconverted."""


class BasePinMissing(BasePinRejected):
    """A base image tag that no longer exists upstream: a typed per-task exclusion."""


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
    inventory_entry: dict


@dataclass(frozen=True)
class NativeRowConversion:
    """One converted row: the executable TaskSpec, its ledger entry and the base pins."""

    task: TaskSpec
    input_task_sha256: str
    output_task_sha256: str
    base_pins: tuple[BasePin, ...]


def original_upstream(entry: dict) -> tuple[str, str]:
    """The source's true original upstream, descending through the TaskTrove aggregate.

    The retired natives' declared dataset is the ``open-athena/task-trove`` aggregate
    whose release card carries the whole catalog's multi-license list — the exact-pin
    check escalates on it (E2) without describing any cohort. The intake's per-config
    provenance records the cohort's actual upstream repository and revision (the
    repackager rule: the card follows the original upstream); any other declared
    dataset IS the original upstream.
    """
    dataset_id, revision = entry["dataset_id"], entry["dataset_revision"]
    if dataset_id != "open-athena/task-trove":
        return dataset_id, revision
    name = entry.get("name") or entry["id"].partition(":")[2]
    provenance = json.loads(
        Path(
            hf_hub_download(dataset_id, f"provenance/{name}/source.json", repo_type="dataset", revision=revision)
        ).read_bytes()
    )
    repository, upstream_revision = provenance["repository"], provenance["revision"]
    if not repository or not upstream_revision:
        raise CohortConversionRejected(f"{entry['id']}: the intake provenance records no upstream pin for {name}")
    return repository, upstream_revision


def release_receipt(
    source: NativeConversionSource, release_repo: str, release_revision: str, card_bytes: bytes
) -> LicenseReceipt:
    """The release's own license receipt: the card follows the original upstream.

    The repackager rule (rulings/repackager-license-follows-upstream-20261116.md): a
    repackager's aggregate card is not the tasks' license. The receipt resolves the
    source's declared original upstream through the checker — never by hand — and
    carries the source's recorded decisions, so an approved user exception (ruling
    2026-11-16 for the SWE-Gym and R2E-Gym cohorts) survives the repackaging with its
    original pin, date and scope. The checker still arbitrates: a source without an
    exception publishes the same lineage and fails closed on any escalation.
    """
    receipt = source.info.license_receipt
    if receipt is None:
        raise CohortConversionRejected(f"{source.info.id}: the release publication requires a license receipt")
    upstream_dataset, upstream_revision = original_upstream(source.inventory_entry)
    upstream = resolve_license(
        LicenseTarget(
            upstream_dataset,
            upstream_dataset,
            upstream_revision,
            f"https://huggingface.co/datasets/{upstream_dataset}/tree/{upstream_revision}",
        )
    )
    if upstream.resolved_source is not None:
        raise CohortConversionRejected(
            f"{source.info.id}: the original upstream {upstream_dataset}@{upstream_revision[:12]} declares "
            f"another wrapper source ({upstream.resolved_source.dataset_id}); resolve that pin instead"
        )
    primary_card = next(item for item in upstream.evidence if item.kind == "dataset_card")
    carried = []
    for decision in receipt.decisions:
        # The schema binds a decision to its own receipt pin; the exception's lineage
        # (original pin, date, scope) is recorded in the rationale at the release pin,
        # per the coordinator's ruling 2026-11-26 that the published card carries it.
        carried.append(
            LicenseDecision(
                kind=decision.kind,
                approved=decision.approved,
                who=decision.who,
                when=datetime.now(UTC).isoformat(),
                source_id=source.info.id,
                dataset_id=release_repo,
                revision=release_revision,
                scope=list(decision.scope),
                rationale=(
                    f"{decision.rationale} Lineage: recorded at {decision.dataset_id}@{decision.revision} "
                    f"({decision.when}); carried to this release pin by the repackager rule."
                ),
                evidence_urls=list(decision.evidence_urls),
                issue_url=decision.issue_url,
            )
        )
    return LicenseReceipt(
        source_id=source.info.id,
        dataset_id=release_repo,
        revision=release_revision,
        resolved_at=upstream.resolved_at,
        resolver_version=upstream.resolver_version,
        evidence=[*upstream.evidence, _card_evidence(release_repo, release_revision, card_bytes)],
        bundled_third_party=upstream.bundled_third_party,
        model_generated_outputs=upstream.model_generated_outputs,
        decisions=carried,
        resolved_source=LicenseSourcePin(
            dataset_id=upstream_dataset,
            revision=upstream_revision,
            url=f"https://huggingface.co/datasets/{upstream_dataset}/tree/{upstream_revision}",
            declaration=primary_card,
        ),
        wrapper_chain=[],
    )


def _card_evidence(release_repo: str, revision: str, card_bytes: bytes) -> LicenseEvidence:
    """The release card's own wrapper declaration, quoting its license_source lines."""
    quoted = [
        line
        for line in card_bytes.decode("utf-8").splitlines()
        if line.startswith(("license_source:", "  dataset_id:", "  revision:", "  url:"))
    ][:6]
    if not quoted:
        raise CohortConversionRejected("The release card declares no license_source to bind")
    return LicenseEvidence(
        kind="wrapper_source",
        url=f"https://huggingface.co/datasets/{release_repo}/resolve/{revision}/README.md",
        sha256=hashlib.sha256(card_bytes).hexdigest(),
        license_ids=[],
        quoted_lines=quoted,
        restrictive_terms=False,
        unidentified_third_party_terms=False,
    )


def exception_lineage(receipt: LicenseReceipt) -> str:
    """The card's human-readable exception record: original pin, date, scope, grant."""
    lines = []
    for decision in receipt.decisions:
        if getattr(decision, "kind", "") != "exception" or not getattr(decision, "approved", False):
            continue
        grant = ", ".join(
            item.license_ids[0] if item.license_ids else "declared upstream"
            for item in receipt.evidence
            if item.kind == "dataset_card"
        )
        lines.append(
            f"License exception: {decision.rationale} (decided {decision.when}, scope: "
            f"{'/'.join(decision.scope)}, upstream grant: {grant})."
        )
    return "\n\n".join(lines)


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
        inventory_entry=entry,
    )


def _docker_hub_repo(name: str) -> str:
    """Docker Hub stores official images under the library/ namespace."""
    return name if "/" in name else f"library/{name}"


_DIGEST_CACHE: dict[str, str] = {}
_DOCKER_HUB_TOKEN: str | None = None


def load_digest_cache(uri: str) -> int:
    """Seed the in-process digest cache from its durable in-region copy."""
    from rigging.filesystem.storage_path import StoragePath  # noqa: PLC0415

    path = StoragePath(uri)
    if not path.exists():
        return 0
    stored = json.loads(path.read_text())
    if not isinstance(stored, dict):
        raise ValueError(f"The digest cache at {uri} is not an object")
    _DIGEST_CACHE.update({str(key): str(value) for key, value in stored.items()})
    return len(_DIGEST_CACHE)


def save_digest_cache(uri: str) -> int:
    """Persist the resolved-digest cache to its in-region copy so a rerun skips resolved tags."""
    from rigging.filesystem.storage_path import StoragePath  # noqa: PLC0415

    StoragePath(uri).write_text(json.dumps(_DIGEST_CACHE, indent=2, sort_keys=True) + "\n")
    return len(_DIGEST_CACHE)


def _docker_hub_bearer() -> str | None:
    """The authenticated Docker Hub token source, when the runtime carries one.

    Submit-time credentials arrive as DOCKERHUB_USERNAME/DOCKERHUB_TOKEN env vars
    (the repair-candidate lane's established pattern — the Mac's Docker Desktop
    credential, exchanged here for a pull token via auth.docker.io basic auth, which
    raises the anonymous rate limit). Mounted secrets under the same names are the
    fallback. The credentials are never logged.
    """
    global _DOCKER_HUB_TOKEN
    if _DOCKER_HUB_TOKEN is not None:
        return _DOCKER_HUB_TOKEN
    from rigging.secrets import get_secret  # noqa: PLC0415

    for name in ("DOCKER_HUB_TOKEN", "DOCKERHUB_TOKEN"):
        value = get_secret(name)
        if value:
            _DOCKER_HUB_TOKEN = value.strip()
            return _DOCKER_HUB_TOKEN
    return None


def _docker_hub_credentials() -> tuple[str, str] | None:
    """The username/secret pair for the pull-token exchange, when present."""
    username = os.environ.get("DOCKERHUB_USERNAME")
    secret = os.environ.get("DOCKERHUB_TOKEN") or _docker_hub_bearer()
    if username and secret:
        return username, secret
    return None


def _open_with_rate_limit_retry(request: urllib.request.Request, what: str):
    """Open a registry URL, honoring the server's rate-limit windows.

    A 429 sleeps out the throttle window — Retry-After verbatim when the server
    names one, otherwise a doubling wait capped at the window cap — and retries
    without a failure ceiling: the conversion job stays alive until the registry
    admits the request (coordinator ruling 2026-11-26, the anonymous-limit rerun).
    Any other error propagates immediately.
    """
    attempt = 0
    while True:
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            delay = error.headers.get("Retry-After")
            seconds = (
                float(delay)
                if delay and delay.replace(".", "", 1).isdigit()
                else min(2.0 ** min(attempt, 10), RATE_LIMIT_MAX_WAIT_SECONDS)
            )
            seconds = min(seconds, RATE_LIMIT_MAX_WAIT_SECONDS)
            attempt += 1
            LOGGER.warning("registry rate limited on %s; wait %.0fs (attempt %d) then retry", what, seconds, attempt)
            time.sleep(seconds)


def resolve_base_digest(reference: str) -> str:
    """Resolve one base image reference to its digest-pinned form via the registry.

    ``registry-1.docker.io`` serves Docker Hub (official images under ``library/``);
    any other host is its own registry with the same v2 API. An already
    digest-pinned reference passes through unchanged. Resolutions cache per
    process (and in the durable cache when one is configured): a cohort's tasks
    share bases, and the shared registries rate-limit. A mounted Docker Hub token
    authenticates the requests when the runtime carries one.
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
    credentials = _docker_hub_credentials()
    if credentials:
        username, secret = credentials
        basic = base64.b64encode(f"{username}:{secret}".encode()).decode()
        token_request = urllib.request.Request(token_url, headers={"Authorization": f"Basic {basic}"})
        with _open_with_rate_limit_retry(token_request, f"{reference} token exchange") as response:
            token = json.loads(response.read())["token"]
    else:
        bearer = _docker_hub_bearer()
        if bearer:
            token = bearer
        else:
            with _open_with_rate_limit_retry(token_url, f"{reference} token") as response:
                token = json.loads(response.read())["token"]
    request = urllib.request.Request(
        f"https://{registry_host}/v2/{scope_repo}/manifests/{urllib.parse.quote(tag)}",
        headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT},
    )
    try:
        with _open_with_rate_limit_retry(request, f"{reference} manifest") as response:
            manifest_digest = response.headers.get("Docker-Content-Digest")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise BasePinMissing(f"The base image {reference} no longer exists upstream") from None
        raise
    if not manifest_digest or not re.fullmatch(DIGEST_PATTERN, manifest_digest):
        raise BasePinRejected(f"The registry returned no usable manifest digest for {reference}")
    return f"{repo}@{manifest_digest}"


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
    """Convert the recorded population; strict — any rejection or blocker fails the run.

    A base tag that no longer exists upstream is a typed per-task exclusion, never a
    substitute image (ruling 2026-11-26): the row leaves the converted population with
    its evidence and the manifest records the exclusion.
    """
    rows = load_release_population(source, limit=limit)
    converted: list[dict] = []
    receipts: list[dict] = []
    exclusions: list[dict] = []
    digest_cache = os.environ.get("NATIVE_CONVERSION_DIGEST_CACHE")
    resolutions_at_last_save = 0
    for row in rows:
        task = TaskSpec.model_validate_json(row["task_spec"])
        try:
            outcome = convert_native_source_row(task, input_task_sha256=task_spec_sha256(task))
        except BasePinMissing as error:
            exclusions.append(
                {
                    "task_id": task.id,
                    "reason": "base_image_missing",
                    "detail": str(error),
                    "input_task_sha256": task_spec_sha256(task),
                }
            )
            continue
        if digest_cache and len(_DIGEST_CACHE) - resolutions_at_last_save >= 250:
            save_digest_cache(digest_cache)
            resolutions_at_last_save = len(_DIGEST_CACHE)
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
    counts = {
        "input_rows": len(rows),
        "converted_rows": len(converted),
        "excluded_rows": len(exclusions),
    }
    if counts["converted_rows"] + counts["excluded_rows"] != counts["input_rows"]:
        raise ValueError("The conversion lost rows: converted plus excluded must cover every input row")
    return converted, receipts, {**counts, "exclusions": exclusions}


def write_native_conversion_artifacts(
    converted: list[dict], receipts: list[dict], counts: dict, *, output_root: str, source_id: str
) -> dict:
    """Write the converted population, receipts, typed exclusions and the manifest.

    Converted plus excluded rows must cover every input row: a silently shrunken
    population would change the census contract without typed admission evidence.
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    from experiments.post_training.task_curation.harbor_cohort_release import TASK_SCHEMA  # noqa: PLC0415

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "tasks.parquet"
    with pq.ParquetWriter(destination, TASK_SCHEMA) as writer:
        for start in range(0, len(converted), 32):
            writer.write_table(pa.Table.from_pylist(converted[start : start + 32], schema=TASK_SCHEMA))
    receipts_path = output / "conversion-receipts.jsonl"
    receipts_path.write_text("".join(json.dumps(receipt, sort_keys=True) + "\n" for receipt in receipts))
    exclusions_path = output / "conversion-exclusions.jsonl"
    exclusions_path.write_text(
        "".join(json.dumps(exclusion, sort_keys=True) + "\n" for exclusion in counts.get("exclusions", []))
    )
    manifest = {
        "protocol": CONVERSION_PROTOCOL,
        "source_id": source_id,
        "converter": "experiments.post_training.task_curation.native_source_conversion",
        "task_specs_sha256": file_sha256(StoragePath(str(destination))),
        "input_rows": counts["input_rows"],
        "converted_rows": counts["converted_rows"],
        "excluded_rows": counts["excluded_rows"],
        "rejections": {"base_image_missing": counts["excluded_rows"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if counts["converted_rows"] + counts["excluded_rows"] != counts["input_rows"]:
        raise ValueError("The conversion lost rows: converted plus excluded must cover every input row")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="The retired native source id to convert.")
    parser.add_argument("--output-root", required=True, help="The conversion output directory (local scratch).")
    parser.add_argument("--artifact-version", required=True, help="The release repo version suffix.")
    parser.add_argument("--limit", type=int, help="Convert only the first N rows (local validation slices).")
    parser.add_argument(
        "--digest-cache",
        help="The durable in-region digest cache (an s3:// object): resolved FROM-base digests "
        "persist here so a rerun skips tags it has already resolved.",
    )
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

    if args.digest_cache:
        seeded = load_digest_cache(args.digest_cache)
        LOGGER.info("digest cache seeded with %d resolutions from %s", seeded, args.digest_cache)
        os.environ["NATIVE_CONVERSION_DIGEST_CACHE"] = args.digest_cache

    entry = load_inventory_entry(args.source_id)
    receipts = read_license_receipts(LICENSE_RECEIPT_INDEX)
    receipt = receipts.get(args.source_id)
    require_unbanned(args.source_id, entry["dataset_id"], receipt)
    source = conversion_source(entry, receipt)
    require_public_data(source.info.data_restriction, source.info.id, source.dataset.name)
    converted, receipts_out, counts = convert_population(source, limit=args.limit)
    if args.digest_cache:
        saved = save_digest_cache(args.digest_cache)
        LOGGER.info("digest cache persisted %d resolutions to %s", saved, args.digest_cache)
    manifest = write_native_conversion_artifacts(
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
    # The repackager rule: the published card follows the original upstream, and an
    # approved user exception survives the repackaging with its lineage (original
    # pin, date, scope, upstream grant). The checker still arbitrates the result.
    # The rebinding prepares into fresh scratch: the first publish's notice files
    # belong to the wrapper card, and the upstream's own notices have different bytes.
    api = HfApi()
    card_path = Path(
        api.hf_hub_download(outcome.release_repo, "README.md", repo_type="dataset", revision=outcome.release_revision)
    )
    bound = release_receipt(source, outcome.release_repo, outcome.release_revision, card_path.read_bytes())
    try:
        index_commit: str | None = append_license_receipt(
            LICENSE_RECEIPT_INDEX,
            bound,
            message=f"Bind {source.info.id}'s native TaskSpec conversion release to its original upstream license",
        )
    except AppendWouldRewriteSharedShard as error:
        # The bind is provenance: the card declares the original upstream and the
        # checker resolves it directly. A source that already holds a receipt in a
        # shared shard is refused by the append-only writer; the receipts owners'
        # variant-family path re-pins it later.
        LOGGER.warning("receipt append refused for %s: %s", source.info.id, error)
        index_commit = None
    rebind_root = Path(args.output_root).parent / f"{Path(args.output_root).name}-rebind"
    if rebind_root.exists():
        import shutil  # noqa: PLC0415

        shutil.rmtree(rebind_root)
    publication = prepare_license_publication(bound, rebind_root, use=source.intended_use.value)
    body = card_path.read_text(encoding="utf-8").split("---", 2)[-1].strip()
    lineage = exception_lineage(bound)
    if lineage:
        body = f"{body}\n\n{lineage}\n"
    (rebind_root / "README.md").write_text(
        licensed_dataset_card(body, publication, pretty_name="Native Harbor TaskSpec conversion release")
    )
    card_commit = api.upload_folder(
        repo_id=outcome.release_repo,
        repo_type="dataset",
        folder_path=str(rebind_root),
        allow_patterns=["README.md"],
        commit_message="Bind the release card to its original upstream license",
    )
    result = {
        **manifest,
        "release_repo": outcome.release_repo,
        "release_revision": card_commit.oid,
        "task_specs_sha256": outcome.task_specs_sha256,
        "license_index_commit": index_commit,
        "license_check": check_publication(
            source.info.id, outcome.release_repo, card_commit.oid, source.intended_use.value
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
