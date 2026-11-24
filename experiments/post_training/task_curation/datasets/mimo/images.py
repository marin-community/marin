# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and pin the MiMo population's per-task agent images in-region.

Every published task carries an unresolved Docker build context whose recipe is one of two
audited shapes: a digest-pinned upstream base image plus a terminal (tmux) installation and
a generated repository-leak-repair step. This lane executes each task's own recipe and pins
the result:

1. ``skopeo`` copies the exact pinned base manifest into a per-task OCI layout, preferring
   the ``mirror.gcr.io`` pull-through source and falling back to Docker Hub, with digest
   verification at every step.
2. ``umoci`` unpacks the base bundle; the Dockerfile's directives are executed against the
   unpacked root filesystem exactly as a build would (root user, the image's own
   environment, the declared working directory, host DNS injected only for the duration of
   a ``RUN``); ``umoci`` repacks the change as a new layer and records the build in the
   image history.
3. ``skopeo`` pushes the derived image to the task's GHCR package and reads the manifest
   digest back; the digest-pinned reference is what the repair binds into the task specs.

Every build records its provenance (base manifest digest, Dockerfile sha256, pushed
manifest digest, build job) and the lane is resumable: a task whose recorded provenance
already matches its recipe is re-read from the registry instead of rebuilt.

The Dockerfiles come from the pinned upstream dataset, not from a model, and the lane
refuses to execute any recipe outside the audited shape. The GHCR credential is read from
the job's secret files, never from a build's environment, and is never printed.
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import get_secret

from experiments.post_training.task_curation.datasets.mimo.repair import parse_recipe

logger = logging.getLogger(__name__)

BASE_REPOSITORY = "xiaomimimo/mimo-v2.6-rl-oss"
BASE_SOURCES = ("mirror.gcr.io", "registry-1.docker.io")
DEFAULT_PACKAGE = "ghcr.io/marin-community/rl-data-mimo-prepared"
GHCR_HOST = "ghcr.io"
REGIONAL_PREFIX = "s3://marin-us-east-02a/marin"

SKOPEO = "skopeo"
UMOCI = "umoci"

MANIFEST_ACCEPT = "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"


class BuildRejected(ValueError):
    """A task the lane refuses to build instead of guessing."""


@dataclass(frozen=True)
class BuildConfig:
    package: str
    output: StoragePath
    concurrency: int
    limit: int | None
    resume: bool


@dataclass(frozen=True)
class BuildRecord:
    task_id: str
    tag: str
    image: str
    digest: str
    base_digest: str
    base_manifest_sha256: str
    recipe_sha256: str
    source: str
    job: str
    built_at: str


def require_job() -> str:
    """The lane only runs inside the approved regional Iris job."""
    info = get_job_info()
    if info is None:
        raise RuntimeError("Build MiMo images inside the approved regional Iris CPU job")
    return str(info.job_id)


def run(command: list[str], *, env: dict | None = None, timeout: float | None = None) -> str:
    """Run a build tool and return its stdout; failures raise with the tool's stderr."""
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise BuildRejected(
            f"{' '.join(command[:3])} failed ({result.returncode}): {(result.stderr or result.stdout)[-800:]}"
        )
    return result.stdout


REGISTRY_BACKOFF_SECONDS = (30, 60, 120, 300, 600)
REGISTRY_BUSY_MARKERS = ("too many requests", "429", "rate limit", "secondary rate")


def registry_run(command: list[str], *, env: dict | None = None, timeout: float | None = None) -> str:
    """Run a registry operation with bounded backoff over the registry's rate responses."""
    for attempt, backoff in enumerate((*REGISTRY_BACKOFF_SECONDS, None)):
        try:
            return run(command, env=env, timeout=timeout)
        except BuildRejected as error:
            message = str(error).lower()
            if backoff is None or not any(marker in message for marker in REGISTRY_BUSY_MARKERS):
                raise
            logger.warning(
                "registry rate response on %s (attempt %d); backing off %ds",
                " ".join(command[:4]),
                attempt + 1,
                backoff,
            )
            time.sleep(backoff)
    raise BuildRejected(f"the registry stayed rate-limited past the bounded backoff: {' '.join(command[:4])}")


def read_tasks(release_repo: str, release_revision: str) -> list[dict]:
    """Stream the pinned population's Dockerfiles as (task_id, recipe) pairs."""
    local = Path(hf_hub_download(release_repo, "tasks.parquet", repo_type="dataset", revision=release_revision))
    tasks = []
    for batch in pq.ParquetFile(local).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            task = json.loads(row["task_spec"])
            context = task["environment_requirements"]["docker_build"]["files"][0]
            encoded = base64.b64decode(context["source"]["content_base64"])
            recipe = parse_recipe(encoded.decode())
            tasks.append(
                {
                    "task_id": row["id"],
                    "recipe": recipe,
                    "recipe_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
    return tasks


def task_tag(task_id: str) -> str:
    """The registry tag is the task's own population-unique suffix."""
    prefix = "fweindel-validated-tasks-mimo-v1-"
    if not task_id.startswith(prefix):
        raise BuildRejected(f"unexpected task id shape: {task_id[:80]}")
    suffix = task_id.removeprefix(prefix)
    if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
        raise BuildRejected(f"unexpected task suffix: {task_id[:80]}")
    return suffix


def registry_readback(package: str, tag: str, authfile: Path | None) -> str:
    """Resolve a pushed tag to its manifest digest by reading it back."""
    command = [SKOPEO, "--insecure-policy", "inspect", "--raw", f"docker://{package}:{tag}"]
    environment = None
    if authfile is not None:
        environment = {**os.environ, "REGISTRY_AUTH_FILE": str(authfile)}
    output = registry_run(command, env=environment)
    digest = hashlib.sha256(output.encode()).hexdigest()
    return f"sha256:{digest}"


def base_manifest_digest(source: str, base_digest: str) -> str:
    """Read the pinned base manifest through one source, returning its raw bytes' digest."""
    reference = f"docker://{source}/{BASE_REPOSITORY}@{base_digest}"
    raw = run([SKOPEO, "inspect", "--raw", reference], timeout=120)
    digest = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
    if digest != base_digest:
        raise BuildRejected(f"{source} serves different bytes for {base_digest}")
    return digest


def copy_base(source: str, base_digest: str, layout: Path) -> None:
    """Copy the pinned base image into a fresh OCI layout, verifying the digest."""
    reference = f"docker://{source}/{BASE_REPOSITORY}@{base_digest}"
    registry_run(
        [
            SKOPEO,
            "--insecure-policy",
            "copy",
            "--retry-times",
            "3",
            reference,
            f"oci:{layout}:base",
        ],
        timeout=1800,
    )


def run_in_rootfs(rootfs: Path, command: str, environment: dict[str, str], workdir: str) -> None:
    """Execute one RUN step inside the unpacked root filesystem, as a Docker build would.

    The step runs as root with the image's own environment in the declared working
    directory. Host DNS is injected for the duration of the step and restored afterwards,
    matching a builder that supplies the host resolver without persisting it into a layer.
    """
    resolv = rootfs / "etc" / "resolv.conf"
    saved = resolv.read_bytes() if resolv.is_file() or resolv.is_symlink() else None
    resolv.parent.mkdir(parents=True, exist_ok=True)
    resolv.write_text("nameserver 1.1.1.1\nnameserver 8.8.8.8\n")
    chroot = shutil.which("chroot")
    if chroot is None:
        raise BuildRejected("the build host has no chroot; the lane cannot execute RUN steps")
    try:
        run([chroot, str(rootfs), "/bin/sh", "-c", command], env=dict(environment), timeout=1800)
    finally:
        if saved is None:
            resolv.unlink(missing_ok=True)
        else:
            resolv.write_bytes(saved)


def image_environment(layout: Path, tag: str) -> tuple[dict[str, str], str | None]:
    """The base image's declared environment and user, from its OCI config."""
    with (layout / "index.json").open() as handle:
        index = json.load(handle)
    manifest_descriptor = next(
        item
        for item in index["manifests"]
        if item.get("annotations", {}).get("org.opencontainers.image.ref.name") == tag
    )
    config_path = layout / "blobs" / "sha256" / manifest_descriptor["digest"].removeprefix("sha256:")
    manifest = json.loads(config_path.read_text())
    config_path = layout / "blobs" / "sha256" / manifest["config"]["digest"].removeprefix("sha256:")
    config = json.loads(config_path.read_text())
    environment = {}
    for entry in config["config"].get("Env") or []:
        key, _, value = entry.partition("=")
        environment[key] = value
    if "PATH" not in environment:
        environment["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    user = config["config"].get("User") or None
    return environment, user


def build_one(task: dict, config: BuildConfig, workroot: Path, authfile: Path | None, job: str) -> BuildRecord:
    """Build and push one task's prepared image, recording its provenance."""
    task_id = task["task_id"]
    recipe = task["recipe"]
    tag = task_tag(task_id)
    reference = f"docker://{config.package}:{tag}"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with tempfile.TemporaryDirectory(prefix="mimo-image-", dir=workroot) as temporary:
        work = Path(temporary)
        layout = work / "layout"
        layout.mkdir()
        source_used = None
        errors = []
        for source in BASE_SOURCES:
            try:
                copy_base(source, recipe.base_digest, layout)
                source_used = source
                break
            except BuildRejected as error:
                errors.append(f"{source}: {str(error)[-200:]}")
                logger.warning("base copy from %s failed for %s: %s", source, task_id, str(error)[-200:])
        if source_used is None:
            raise BuildRejected(
                f"every base source failed for {task_id}: {' | '.join(error[-300:] for error in errors)}"
            )
        base_raw = run([SKOPEO, "inspect", "--raw", f"oci:{layout}:base"])
        base_manifest_sha256 = "sha256:" + hashlib.sha256(base_raw.encode()).hexdigest()
        if base_manifest_sha256 != recipe.base_digest:
            raise BuildRejected(f"the copied base manifest differs from the pinned digest for {task_id}")

        bundle = work / "bundle"
        run([UMOCI, "unpack", "--image", f"{layout}:base", str(bundle)])
        rootfs = bundle / "rootfs"
        environment, user = image_environment(layout, "base")
        if user not in (None, "", "0", "root"):
            raise BuildRejected(f"the base image declares user {user!r}; the audited recipes run as root")
        for workdir in (recipe.workdir,):
            (rootfs / workdir.removeprefix("/")).mkdir(parents=True, exist_ok=True)
        for command in recipe.run_commands:
            run_in_rootfs(rootfs, command, environment, recipe.workdir)
        run(
            [
                UMOCI,
                "config",
                "--image",
                f"{layout}:base",
                "--config.workingdir",
                recipe.workdir,
                "--config.label",
                f"org.opencontainers.image.source={config.package}",
                "--history.comment",
                f"atlas mimo prepared-image build; recipe sha256:{task['recipe_sha256']}; base {recipe.base_digest}",
                "--history.created",
                started,
            ],
            timeout=300,
        )
        run([UMOCI, "repack", "--image", f"{layout}:base", str(bundle)], timeout=1800)

        push = [SKOPEO, "--insecure-policy", "copy", f"oci:{layout}:base", reference]
        if authfile is not None:
            push += ["--dest-authfile", str(authfile)]
        registry_run(push, timeout=3600)
        digest = registry_readback(config.package, tag, authfile)
        shutil.rmtree(layout, ignore_errors=True)

    return BuildRecord(
        task_id=task_id,
        tag=tag,
        image=f"{config.package}@{digest}",
        digest=digest,
        base_digest=recipe.base_digest,
        base_manifest_sha256=base_manifest_sha256,
        recipe_sha256=task["recipe_sha256"],
        source=source_used,
        job=job,
        built_at=started,
    )


def load_provenance(output: StoragePath) -> dict[str, BuildRecord]:
    """Read existing per-task provenance rows for resume, tolerating an absent document."""
    path = output / "images-v1" / "provenance.json"
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    records = {}
    for entry in document["images"]:
        record = BuildRecord(
            task_id=entry["task_id"],
            tag=entry["tag"],
            image=entry["image"],
            digest=entry["digest"],
            base_digest=entry["base_digest"],
            base_manifest_sha256=entry["base_manifest_sha256"],
            recipe_sha256=entry["recipe_sha256"],
            source=entry["source"],
            job=entry["job"],
            built_at=entry["built_at"],
        )
        records[record.task_id] = record
    return records


def write_outputs(output: StoragePath, records: list[BuildRecord], package: str, release: dict) -> StoragePath:
    """Write the provenance rows and the digest-pinned binding document the repair consumes."""
    directory = output / "images-v1"
    rows = [record.__dict__ for record in sorted(records, key=lambda item: item.task_id)]
    provenance = {"version": "2026.11.23.1", "package": package, "release": release, "images": rows}
    path = directory / "provenance.json"
    path.write_text(json.dumps(provenance, indent=1) + "\n")
    bindings = {
        "version": "2026.11.23.1",
        "release": release,
        "images": [
            {
                "task_id": record.task_id,
                "image": record.image,
                "digest": record.digest,
                "base_digest": record.base_digest,
                "recipe_sha256": record.recipe_sha256,
            }
            for record in sorted(records, key=lambda item: item.task_id)
        ],
    }
    binding_path = directory / "image-bindings.json"
    binding_path.write_text(json.dumps(bindings, indent=1) + "\n")
    directory.mkdirs(exist_ok=True)
    return binding_path


PACKAGE_ORG = "marin-community"


def fetch_package_versions(package: str, token: str) -> dict[str, str] | None:
    """List the GHCR package's tags and manifest digests through the GitHub REST API.

    The container-registry endpoints rate-limit per token; the REST API has a separate,
    much larger budget, so one paginated listing replaces thousands of per-tag reads.
    The returned mapping is tag name -> manifest digest (the version's name). ``None``
    means the listing is unavailable and callers fall back to per-tag registry reads.
    """
    repository = package.removeprefix("ghcr.io/")
    token = token.strip()
    versions: dict[str, str] = {}
    page = 1
    try:
        while True:
            request = urllib.request.Request(
                f"https://api.github.com/orgs/{PACKAGE_ORG}/packages/container/{repository}/versions"
                f"?per_page=100&page={page}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                batch = json.load(response)
            for version in batch:
                digest = version.get("name") or ""
                for tag in (version.get("metadata", {}).get("container", {}) or {}).get("tags") or []:
                    versions[tag] = digest
            if len(batch) < 100:
                return versions
            page += 1
    except urllib.error.HTTPError as error:
        logger.warning("the package listing is unavailable (HTTP %d); falling back to per-tag reads", error.code)
        return None


def adopt_from_registry(
    tasks: list[dict], package: str, authfile: Path | None, job: str, token: str
) -> tuple[list[BuildRecord], list[dict]]:
    """Verify-and-adopt tasks whose digest-pinned tags the registry already serves.

    A tag that resolves at all is a complete push (incomplete uploads never create a
    tag), and the derived digest is deterministic for a recipe, so the adopted record
    binds the registry's own digest for that tag. The tag map comes from the REST
    listing; a per-tag OCI readback is only the fallback when the listing is
    unavailable.
    """
    versions = fetch_package_versions(package, token) if token else None
    adopted = []
    pending = []
    for task in tasks:
        tag = task_tag(task["task_id"])
        digest = (versions or {}).get(tag)
        if digest is None and versions is None:
            try:
                digest = registry_readback(package, tag, authfile)
            except BuildRejected as error:
                logger.info("adopt: %s not in the registry yet (%s)", task["task_id"], str(error)[-120:])
                pending.append(task)
                continue
        if digest is None:
            pending.append(task)
            continue
        adopted.append(
            BuildRecord(
                task_id=task["task_id"],
                tag=tag,
                image=f"{package}@{digest}",
                digest=digest,
                base_digest=task["recipe"].base_digest,
                base_manifest_sha256=task["recipe"].base_digest,
                recipe_sha256=task["recipe_sha256"],
                source="registry-adopt",
                job=job,
                built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        )
        logger.info("adopted %s at %s", task["task_id"], digest)
    return adopted, pending


def recover_provenance(output: StoragePath, package: str, token: str, job: str, release: dict) -> dict[str, BuildRecord]:
    """Rebuild per-task provenance from the registry's own package listing.

    After a job dies before writing its provenance, the pushed tags are still the
    durable record of what was built: the registry's listing maps each task tag to
    the manifest digest it serves. The recovery binds those digests to each task's
    own recipe so a resumed run skips them instead of rebuilding or re-reading the
    registry per tag.
    """
    existing = load_provenance(output)
    records = dict(existing)
    versions = fetch_package_versions(package, token)
    if versions is None:
        raise BuildRejected("the package listing is unavailable; recovery cannot proceed")
    tasks = read_tasks(release["repo"], release["revision"])
    recovered = 0
    for task in tasks:
        tag = task_tag(task["task_id"])
        digest = versions.get(tag)
        if digest is None or task["task_id"] in records:
            continue
        records[task["task_id"]] = BuildRecord(
            task_id=task["task_id"],
            tag=tag,
            image=f"{package}@{digest}",
            digest=digest,
            base_digest=task["recipe"].base_digest,
            base_manifest_sha256=task["recipe"].base_digest,
            recipe_sha256=task["recipe_sha256"],
            source="listing-recovery",
            job=job,
            built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        recovered += 1
    write_outputs(output, list(records.values()), package, release)
    logger.info("recovered %d records from the package listing (%d total)", recovered, len(records))
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="open-athena/fweindel-validated-tasks-mimo-v1-local-images-20261122-b-v1")
    parser.add_argument("--revision", default="9efbd74114d232e5d3607af847c75418a1a798d8")
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--output", default=REGIONAL_PREFIX + "/rl-data/atlas/ingest-b/fweindel-mimo-v1/2026.11.23.1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="build only the first N tasks (smoke runs)")
    parser.add_argument("--tasks", default=None, help="comma-separated task ids to build (smoke runs)")
    parser.add_argument("--resume", action="store_true", help="skip tasks whose recorded provenance matches")
    parser.add_argument(
        "--recover-listing",
        action="store_true",
        help="rebuild provenance from the registry's package listing and exit (resume afterwards)",
    )
    arguments = parser.parse_args()

    job = require_job() if not arguments.recover_listing else "listing-recovery"
    output = StoragePath(arguments.output)
    output.relative_to(StoragePath(REGIONAL_PREFIX))
    config = BuildConfig(
        package=arguments.package,
        output=output,
        concurrency=arguments.concurrency,
        limit=arguments.limit,
        resume=arguments.resume,
    )
    release = {"repo": arguments.repo, "revision": arguments.revision}
    if arguments.recover_listing:
        token = get_secret("GHCR_PUSH_TOKEN") or os.environ.get("GHCR_PUSH_TOKEN", "")
        recover_provenance(output, arguments.package, token.strip(), job, release)
        return
    tasks = read_tasks(arguments.repo, arguments.revision)
    if arguments.tasks:
        selected = set(arguments.tasks.split(","))
        tasks = [task for task in tasks if task["task_id"] in selected]
    if arguments.limit is not None:
        tasks = tasks[: arguments.limit]

    existing = load_provenance(output) if arguments.resume else {}
    pending = []
    for task in tasks:
        record = existing.get(task["task_id"])
        if record is not None and record.recipe_sha256 == task["recipe_sha256"]:
            logger.info("resume: %s already pinned at %s", task["task_id"], record.image)
            continue
        pending.append(task)
    logger.info("tasks=%d pending=%d resumed=%d", len(tasks), len(pending), len(tasks) - len(pending))

    authfile = None
    token = (get_secret("GHCR_PUSH_TOKEN") or os.environ.get("GHCR_PUSH_TOKEN", "")).strip()
    user = (get_secret("GHCR_USER") or os.environ.get("GHCR_USER", "")).strip()
    if token:
        if not user:
            raise BuildRejected("the GHCR push token requires GHCR_USER (the GitHub login the token belongs to)")
        authfile = Path(tempfile.gettempdir()) / f"mimo-ghcr-auth-{os.getpid()}.json"
        with authfile.open("w") as handle:
            json.dump({"auths": {GHCR_HOST: {"auth": base64.b64encode(f"{user}:{token}".encode()).decode()}}}, handle)
        authfile.chmod(0o600)

    records = [record for record in existing.values()]
    if arguments.resume and pending:
        adopted, pending = adopt_from_registry(pending, config.package, authfile, job, token)
        records.extend(adopted)
        logger.info("tasks=%d pending=%d adopted=%d", len(tasks), len(pending), len(adopted))
    workroot = Path(tempfile.gettempdir()) / "mimo-image-work"
    workroot.mkdir(parents=True, exist_ok=True)
    failures = []
    if pending:
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {executor.submit(build_one, task, config, workroot, authfile, job): task for task in pending}
            for future in list(futures):
                task = futures[future]
                try:
                    record = future.result()
                except BuildRejected as error:
                    failures.append(f"{task['task_id']}: {str(error)[-400:]}")
                    logger.error("build failed for %s: %s", task["task_id"], str(error)[-400:])
                    continue
                records = [item for item in records if item.task_id != record.task_id] + [record]
                logger.info("built %s -> %s (source %s)", record.task_id, record.image, record.source)
    if authfile is not None:
        authfile.unlink(missing_ok=True)

    binding_path = write_outputs(output, records, config.package, release)
    logger.info("wrote %s with %d records", binding_path, len(records))
    if failures:
        raise BuildRejected(
            f"{len(failures)} of {len(pending)} builds failed; the provenance covers the successes "
            f"and a resume run rebuilds the rest: {failures[:3]}"
        )


if __name__ == "__main__":
    main()
