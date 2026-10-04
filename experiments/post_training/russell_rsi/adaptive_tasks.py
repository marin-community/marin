# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind development feedback to one bounded generated task cohort."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from rigging.config_discovery import find_project_root
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle, install_runtime_bundle

from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import SourceSnapshot, source_group_id

GENERATION_MAX_TOKENS = 16384


@dataclass(frozen=True)
class AdaptiveTasksConfig:
    snapshots_uri: str
    snapshots_sha256: str
    inventory_uri: str
    traces_uri: str
    output_path: str
    relay_job: str
    image: str
    runtime_bundle: RuntimeBundle
    max_candidates: int
    minimum_train_rows: int
    admission_concurrency: int
    dependency_wheels_uri: str
    source_identities: tuple[str, ...]  # Freeze the source cohort in the artifact fingerprint.


def require_train_rows(output: Path, minimum_train_rows: int) -> None:
    """Reject a published cohort that does not meet the training minimum."""
    admitted = json.loads((output / "admission-summary.json").read_text())["train_rows"]
    if admitted < minimum_train_rows:
        raise ValueError(f"Admitted {admitted} train tasks; the RL batch requires {minimum_train_rows}")


def prepare_adaptive_tasks(config: AdaptiveTasksConfig) -> None:
    """Generate and admit tasks after the frozen development evaluation finishes."""
    from experiments.post_training.russell_rsi.feedback import abstract_failure_skills  # noqa: PLC0415
    from experiments.post_training.russell_rsi.tasks import (  # noqa: PLC0415
        accept_candidates,
        generate_candidates,
        generation_request,
    )

    manifest = install_runtime_bundle(config.runtime_bundle)
    with tempfile.TemporaryDirectory(prefix="russell-adaptive-") as directory:
        root = Path(directory)
        analyst = root / "feedback-analysis.json"
        analyst_storage = StoragePath(prefix_join(config.output_path, "feedback-analysis.json"))
        if analyst_storage.exists():
            analyst_storage.download_to(str(analyst))
        try:
            feedback = asyncio.run(
                abstract_failure_skills(
                    config.traces_uri,
                    config.relay_job,
                    analyst,
                )
            )
        finally:
            if analyst.exists():
                analyst_storage.upload_from(str(analyst))
        (root / "inventory.jsonl").write_bytes(StoragePath(config.inventory_uri).read_bytes())
        snapshots = StoragePath(config.snapshots_uri).read_bytes()
        if hashlib.sha256(snapshots).hexdigest() != config.snapshots_sha256:
            raise ValueError("Training source snapshot digest does not match its pinned identity")
        records = [json.loads(line) for line in snapshots.decode().splitlines() if line.strip()]
        (root / "snapshots.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records if row["split"] == "train")
        )
        wheels = root / "wheels"
        wheels.mkdir()
        wheel_manifest = StoragePath(prefix_join(config.dependency_wheels_uri, "manifest.json")).read_bytes()
        (wheels / "manifest.json").write_bytes(wheel_manifest)
        for relative in set(json.loads(wheel_manifest)["repository_wheels"].values()) - {None}:
            target = wheels / relative
            target.mkdir(parents=True)
            sources = StoragePath(prefix_join(config.dependency_wheels_uri, relative + "/*.whl")).glob()
            if not sources:
                raise ValueError(f"Dependency bundle {relative} contains no wheels")
            for source in sources:
                source.download_to(str(target / Path(str(source)).name))
        candidates = root / "candidates"
        output = root / "accepted"
        generation_uri = prefix_join(config.output_path, "generation")
        generation_path = StoragePath(generation_uri)
        candidates.mkdir()
        for source in (generation_path / "**/*").glob():
            if source.isdir():
                continue
            relative = source.relative_to(generation_path)
            target = candidates / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source.download_to(str(target))
        identity = {
            "snapshots_sha256": hashlib.sha256((root / "snapshots.jsonl").read_bytes()).hexdigest(),
            "inventory_sha256": hashlib.sha256((root / "inventory.jsonl").read_bytes()).hexdigest(),
            "feedback_sha256": hashlib.sha256(feedback.encode()).hexdigest(),
            "source_identities": list(config.source_identities),
            "max_candidates": config.max_candidates,
            "generation_max_tokens": GENERATION_MAX_TOKENS,
            "generation_requests": {
                source_group_id(snapshot): (
                    hashlib.sha256(
                        json.dumps(
                            generation_request(snapshot, failure_summary=feedback, max_tokens=GENERATION_MAX_TOKENS),
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                )
                for snapshot in [SourceSnapshot.model_validate(row) for row in records if row["split"] == "train"][
                    : config.max_candidates
                ]
            },
            "dependency_manifest_sha256": hashlib.sha256(wheel_manifest).hexdigest(),
            "runtime_archive_sha256": config.runtime_bundle.archive_sha256,
            "image": config.image,
        }
        identity_storage = StoragePath(prefix_join(config.output_path, "cohort-identity.json"))
        if identity_storage.exists() and json.loads(identity_storage.read_text()) != identity:
            raise ValueError("Adaptive cohort identity changed; use a new artifact version")
        identity_path = root / "cohort-identity.json"
        identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n")
        identity_storage.upload_from(str(identity_path))

        async def persist(path: Path) -> None:
            if path.is_relative_to(candidates):
                destination = prefix_join(generation_uri, path.relative_to(candidates).as_posix())
            elif path.is_relative_to(output):
                destination = prefix_join(config.output_path, path.relative_to(output).as_posix())
            else:
                raise ValueError("Admission evidence is outside its artifact directories")
            await asyncio.to_thread(StoragePath(destination).upload_from, str(path))

        try:
            asyncio.run(
                generate_candidates(
                    argparse.Namespace(
                        snapshots=root / "snapshots.jsonl",
                        output=candidates,
                        relay_job=config.relay_job,
                        token_env=GLM_TOKEN_ENV,
                        failure_summary=feedback,
                        max_candidates=config.max_candidates,
                        max_tokens=GENERATION_MAX_TOKENS,
                    )
                )
            )
        finally:
            StoragePath(generation_uri).upload_from(str(candidates) + "/", recursive=True)
        asyncio.run(
            accept_candidates(
                argparse.Namespace(
                    inventory=root / "inventory.jsonl",
                    candidates=candidates,
                    output=output,
                    backend="qemu",
                    image=config.image,
                    prepared_bundle=Path(config.runtime_bundle.installation_parent) / manifest["directory_name"],
                    max_candidates=config.max_candidates,
                    timeout=120,
                    dependency_bundles=wheels,
                ),
                concurrency=config.admission_concurrency,
                persist=persist,
            )
        )
        require_train_rows(output, config.minimum_train_rows)


def run_adaptive_tasks_in_project(config: AdaptiveTasksConfig) -> None:
    """Execute task admission with the bundled rollout package dependencies."""
    workspace = find_project_root()
    if workspace is None:
        raise RuntimeError("Task admission requires the bundled Marin workspace")
    with tempfile.TemporaryDirectory(prefix="russell-task-config-") as directory:
        config_path = Path(directory) / "config.json"
        config_path.write_text(json.dumps(asdict(config)))
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(workspace / "lib/rolloutengine"),
                "--with",
                "openai==2.24.0",
                "--with-editable",
                str(workspace / "lib/iris"),
                "python",
                "-m",
                "experiments.post_training.russell_rsi.adaptive_tasks",
                "--config",
                str(config_path),
            ],
            cwd=workspace,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    values = json.loads(arguments.config.read_text())
    values["runtime_bundle"] = RuntimeBundle(**values["runtime_bundle"])
    prepare_adaptive_tasks(AdaptiveTasksConfig(**values))
