# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind development feedback to one bounded generated task cohort."""

import argparse
import asyncio
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from rigging.config_discovery import find_project_root
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle, install_runtime_bundle

from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV


@dataclass(frozen=True)
class AdaptiveTasksConfig:
    snapshots_uri: str
    inventory_uri: str
    traces_uri: str
    output_path: str
    relay_job: str
    image: str
    runtime_bundle: RuntimeBundle
    max_candidates: int
    minimum_train_rows: int
    dependency_wheels_uri: str
    source_identities: tuple[str, ...]  # Freeze the source cohort in the artifact fingerprint.


def prepare_adaptive_tasks(config: AdaptiveTasksConfig) -> None:
    """Generate and admit tasks after the frozen development evaluation finishes."""
    from taskcompendium.parquet import read_tasks  # noqa: PLC0415

    from experiments.post_training.russell_rsi.feedback import abstract_failure_skills  # noqa: PLC0415
    from experiments.post_training.russell_rsi.tasks import accept_candidates, generate_candidates  # noqa: PLC0415

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
        records = [
            json.loads(line) for line in StoragePath(config.snapshots_uri).read_text().splitlines() if line.strip()
        ]
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
        candidates.mkdir()
        for source in StoragePath(generation_uri + "/**/*").glob():
            if source.isdir():
                continue
            relative = str(source).removeprefix(generation_uri + "/")
            target = candidates / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source.download_to(str(target))
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
                        max_tokens=8192,
                    )
                )
            )
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
                    )
                )
            )
        finally:
            StoragePath(generation_uri).upload_from(str(candidates) + "/", recursive=True)
        admitted = sum(1 for _ in read_tasks(str(output / "train.parquet")))
        if admitted < config.minimum_train_rows:
            raise ValueError(f"Admitted {admitted} train tasks; the RL batch requires {config.minimum_train_rows}")
        StoragePath(config.output_path).upload_from(str(output) + "/", recursive=True)


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
