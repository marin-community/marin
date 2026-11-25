# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract each accepted cohort's Docker build context for the regional image build.

The conversion releases carry per-task Docker build contexts; within a cohort
the sampled tasks share one context (measured: jest-v5's 20 sampled tasks
resolve to a single build). This module materializes each accepted cohort's
context from one of its release rows, archives it, stages the archive to the
regional bucket, and records the cohort-to-context manifest the Kaniko build
job consumes. The build job digest-pins one registry image per cohort.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.docker_build import materialize_docker_build

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CohortContext:
    source_id: str
    release_repo: str
    release_revision: str
    context_uri: str
    context_sha256: str
    dockerfile: str
    files: tuple[str, ...]


def cohort_context(source_id: str, release_repo: str, revision: str, workdir: Path) -> CohortContext:
    """Materialize the cohort's build context from one release row and archive it."""
    path = Path(hf_hub_download(release_repo, "tasks.parquet", repo_type="dataset", revision=revision))
    build = None
    for row in pq_rows(path):
        task = TaskSpec.model_validate_json(row["task_spec"])
        candidate = task.environment_requirements.docker_build
        if candidate is not None:
            build = candidate
            break
    if build is None:
        raise ValueError(f"No build context in {release_repo}@{revision[:12]} (a pinned image may already cover it)")
    import shutil  # noqa: PLC0415

    context_root = workdir / "context"
    if context_root.exists():
        shutil.rmtree(context_root)
    source = materialize_docker_build(build, context_root)
    context = source.context
    archive = workdir / f"{slug(source_id)}.tar.gz"
    import tarfile  # noqa: PLC0415

    with tarfile.open(archive, "w:gz") as tar:
        for member in sorted(context.rglob("*")):
            tar.add(member, arcname=member.relative_to(context).as_posix())
    digest = file_sha256(StoragePath(str(archive)))
    files = tuple(member.relative_to(context).as_posix() for member in sorted(context.rglob("*")) if member.is_file())
    dockerfile = next((name for name in files if name.lower() == "dockerfile"), None)
    if dockerfile is None:
        raise ValueError(f"The context carries no Dockerfile: {source_id}")
    return CohortContext(
        source_id=source_id,
        release_repo=release_repo,
        release_revision=revision,
        context_uri="",  # assigned by the caller after staging
        context_sha256=digest,
        dockerfile=dockerfile,
        files=files,
    )


def slug(source_id: str) -> str:
    return source_id.partition(":")[2].replace("/", "-").replace("@", "-").lower()[:48].rstrip("-")


def pq_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    return pq.read_table(path, columns=["task_spec"]).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="JSON list of {source_id, release_repo, release_revision}."
    )
    parser.add_argument("--workdir", type=Path, required=True, help="Local scratch for the contexts.")
    parser.add_argument("--context-root", required=True, help="The regional bucket root for the staged archives.")
    parser.add_argument("--output", type=Path, required=True, help="The cohort-to-context manifest to write.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    plan = json.loads(args.manifest.read_text())
    contexts = []
    for entry in plan:
        source_id = entry["source_id"]
        workdir = args.workdir / slug(source_id)
        workdir.mkdir(parents=True, exist_ok=True)
        context = cohort_context(source_id, entry["release_repo"], entry["release_revision"], workdir)
        uri = f"{args.context_root.rstrip('/')}/{slug(source_id)}/context.tar.gz"
        archive_path = workdir / f"{slug(source_id)}.tar.gz"
        import subprocess  # noqa: PLC0415

        subprocess.run(
            ["uv", "run", "--no-sync", "fsutil", "cp", str(archive_path), uri],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        contexts.append(
            {
                "source_id": source_id,
                "release_repo": context.release_repo,
                "release_revision": context.release_revision,
                "context_uri": uri,
                "context_sha256": context.context_sha256,
                "dockerfile": context.dockerfile,
                "files": list(context.files),
            }
        )
        LOGGER.info("staged %s (%d files, sha256 %s)", uri, len(context.files), context.context_sha256[:12])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contexts, indent=2) + "\n")
    print(json.dumps({"staged": len(contexts)}, indent=1))


if __name__ == "__main__":
    main()
