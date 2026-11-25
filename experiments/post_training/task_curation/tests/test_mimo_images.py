# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The image lane's pure-Python entry point, end to end, with the shell tools doubled.

Every failure so far (digest prefix, readback credentials, storage-path API) lived in the
code around the tools, so this test runs the exact ``main`` entry point locally with
``skopeo``/``umoci``/``chroot`` behind a fake that honors the same commands: it materializes
a real OCI layout, unpacks a bundle, records pushes, and serves the readback. The storage
writes, provenance, resume, failure accounting and readback bookkeeping all execute for real.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.datasets.mimo import images as lane
from experiments.post_training.task_curation.datasets.mimo.images import task_tag
from experiments.post_training.task_curation.datasets.mimo.repair import load_bindings
from experiments.post_training.task_curation.tests.test_mimo_repair import resource, retained_row

TASK_ID = "fweindel-validated-tasks-mimo-v1-" + "c" * 64
PACKAGE = "ghcr.io/marin-community/rl-data-mimo-test"


def smoke_row() -> tuple[dict, str, bytes]:
    """The retained row, its recipe's base digest, and the manifest that digest pins.

    The manifest bytes come first and the fixture's Dockerfile pins their real sha256, so
    the lane's digest verification holds against the doubled registry exactly as it does
    against the real one.
    """
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": "", "size": 0},
            "layers": [
                {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": "sha256:" + "0" * 64, "size": 10}
            ],
        }
    ).encode()
    digest_hex = hashlib.sha256(manifest).hexdigest()
    recipe_digest = f"sha256:{digest_hex}"
    dockerfile = (
        f"FROM docker.io/xiaomimimo/mimo-v2.6-rl-oss@{recipe_digest}\n"
        "WORKDIR /workspace/repo\n"
        "RUN install-tmux\n"
        "# ot-validation: repository-leak-repair-v1 (generated; replaced, never stacked)\n"
        "RUN repair-repository\n"
    ).encode()
    row = retained_row(TASK_ID)
    task = json.loads(row["task_spec"])
    task["environment_requirements"]["docker_build"]["files"][0] = resource("Dockerfile", dockerfile)
    row["task_spec"] = json.dumps(task)
    return row, recipe_digest, manifest


class FakeTools:
    """Double for the lane's run(): records commands and simulates the registry and OCI tools."""

    def __init__(self, base_manifest: bytes):
        self.commands = []
        self.base_manifest = base_manifest
        self.pushed = {}
        self.fail_on = set()

    def __call__(self, command, *, env=None, timeout=None):
        self.commands.append(command)
        joined = " ".join(command)
        for pattern in self.fail_on:
            if pattern in joined:
                raise lane.BuildRejected(f"simulated failure: {pattern}")
        if command[0] == "umoci" and command[1] == "unpack":
            bundle = Path(command[-1])
            (bundle / "rootfs" / "testbed").mkdir(parents=True)
            (bundle / "rootfs" / "testbed" / "mimo_test_command.sh").write_text("exit 0\n")
            # Some bases occupy the declared workdir with a plain file, exactly as the
            # /workspace/repo bases that broke the retry did.
            (bundle / "rootfs" / "workspace").mkdir(parents=True)
            (bundle / "rootfs" / "workspace" / "repo").write_text("occupant\n")
            return ""
        if command[0] == "umoci" and command[1] in ("config", "repack"):
            return ""
        if command[0].endswith("/chroot") or command[0] == "chroot":
            return ""
        if command[0] == "skopeo" and "copy" in command:
            source, destination = self.copy_arguments(command)
            if source.startswith("docker://"):
                return self.materialize_layout(destination)
            self.pushed[destination.removeprefix("docker://")] = self.base_manifest
            return ""
        if command[0] == "skopeo" and "inspect" in command and "--raw" in command:
            reference = command[-1]
            if reference.startswith("oci:"):
                return self.base_manifest.decode()
            key = reference.removeprefix("docker://")
            if key not in self.pushed:
                raise lane.BuildRejected(f"unauthorized or missing tag: {reference}")
            return self.pushed[key].decode()
        raise AssertionError(f"the lane issued an un-doubleable command: {joined}")

    @staticmethod
    def copy_arguments(command) -> tuple[str, str]:
        """The two positional references of a skopeo copy, flags and their values skipped."""
        positionals = []
        skip_value = False
        for token in command[command.index("copy") + 1 :]:
            if skip_value:
                skip_value = False
                continue
            if token.startswith("-"):
                skip_value = token in ("--retry-times", "--dest-authfile", "--src-authfile", "--authfile")
                continue
            positionals.append(token)
        if len(positionals) != 2:
            raise AssertionError(f"the doubled copy has unexpected positionals: {command}")
        return positionals[0], positionals[1]

    def materialize_layout(self, destination) -> str:
        layout = Path(destination.removeprefix("oci:").split(":")[0])
        digest_hex = hashlib.sha256(self.base_manifest).hexdigest()
        blobs = layout / "blobs" / "sha256"
        blobs.mkdir(parents=True, exist_ok=True)
        config = json.dumps({"config": {"Env": ["PATH=/usr/bin"], "WorkingDir": None, "User": None}}).encode()
        config_digest = hashlib.sha256(config).hexdigest()
        (blobs / config_digest).write_bytes(config)
        manifest = json.loads(self.base_manifest)
        manifest["config"]["digest"] = f"sha256:{config_digest}"
        manifest_bytes = json.dumps(manifest).encode()
        (blobs / digest_hex).write_bytes(manifest_bytes)
        (layout / "index.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "manifests": [
                        {
                            "mediaType": "application/vnd.oci.image.manifest.v1+json",
                            "digest": f"sha256:{digest_hex}",
                            "size": len(manifest_bytes),
                            "annotations": {"org.opencontainers.image.ref.name": destination.split(":")[-1]},
                        }
                    ],
                }
            )
        )
        return ""


def fixture_parquet(tmp_path: Path, row: dict) -> str:
    path = tmp_path / "tasks.parquet"
    table = pa.Table.from_pylist([row])
    with path.open("wb") as stream:
        with pq.ParquetWriter(stream, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def run_main(monkeypatch, tmp_path: Path, row: dict, tools: FakeTools, *extra: str, output: Path | None = None) -> None:
    parquet = fixture_parquet(tmp_path, row)
    info = SimpleNamespace(job_id="/benfeuer/test-lane")
    monkeypatch.setattr(lane, "run", tools)
    monkeypatch.setattr(lane, "get_job_info", lambda: info)
    monkeypatch.setattr(lane, "hf_hub_download", lambda repo, filename, repo_type=None, revision=None: parquet)
    monkeypatch.setattr(lane, "REGIONAL_PREFIX", str(tmp_path))
    monkeypatch.setattr(lane, "DEFAULT_PACKAGE", PACKAGE)
    monkeypatch.setenv("GHCR_PUSH_TOKEN", "token-value")
    monkeypatch.setenv("GHCR_USER", "tester")
    monkeypatch.setattr(
        "sys.argv",
        [
            "images",
            "--tasks",
            row["id"],
            "--output",
            str(output or tmp_path / "regional"),
            "--concurrency",
            "2",
            *extra,
        ],
    )
    lane.main()


def test_entry_point_builds_one_task_end_to_end_and_resumes(tmp_path, monkeypatch):
    row, recipe_digest, manifest = smoke_row()
    tools = FakeTools(manifest)
    output = tmp_path / "regional"
    run_main(monkeypatch, tmp_path, row, tools)

    provenance = json.loads((output / "images-v1" / "provenance.json").read_text())
    assert len(provenance["images"]) == 1
    record = provenance["images"][0]
    assert record["task_id"] == row["id"]
    assert record["base_manifest_sha256"] == recipe_digest
    assert record["image"].startswith(PACKAGE + "@sha256:")
    assert record["digest"].startswith("sha256:")
    bindings = load_bindings(output / "images-v1" / "image-bindings.json")
    assert bindings[row["id"]].image == record["image"]
    assert bindings[row["id"]].base_digest == recipe_digest
    assert any("chroot" in " ".join(command) for command in tools.commands)
    assert any(command[0] == "umoci" for command in tools.commands)
    pushes = []
    for command in tools.commands:
        if command[:3] != ["skopeo", "--insecure-policy", "copy"]:
            continue
        destination = tools.copy_arguments(command)[1]
        if destination.startswith("docker://"):
            pushes.append(destination)
    assert len(pushes) == 1 and pushes[0].startswith("docker://" + PACKAGE)
    # The retry reproduction: the workdir the base occupied with a file is now a directory.
    rootfs = next(Path(command[1]) for command in tools.commands if command[0].endswith("/chroot"))
    assert rootfs.is_dir() and (rootfs / "workspace" / "repo").is_dir()

    # A resume run re-reads the registry instead of rebuilding.
    resume = FakeTools(manifest)
    run_main(monkeypatch, tmp_path, row, resume, "--resume")
    assert not any(command[0] == "umoci" for command in resume.commands)
    assert not any(command[:3] == ["skopeo", "--insecure-policy", "copy"] for command in resume.commands)


def test_entry_point_adopts_pushed_tags_on_resume(tmp_path, monkeypatch):
    row, _recipe_digest, manifest = smoke_row()
    tools = FakeTools(manifest)
    run_main(monkeypatch, tmp_path, row, tools)
    first = json.loads((tmp_path / "regional" / "images-v1" / "provenance.json").read_text())
    assert len(first["images"]) == 1
    tag = task_tag(row["id"])

    # A resume run against a fresh output with no provenance document adopts the
    # already-pushed tag from the registry's REST listing instead of rebuilding.
    monkeypatch.setattr(
        lane,
        "fetch_package_versions",
        lambda package, token: {tag: first["images"][0]["digest"]} if token else {},
    )
    umoci_before = sum(1 for command in tools.commands if command[0] == "umoci")
    run_main(monkeypatch, tmp_path, row, tools, "--resume", output=tmp_path / "regional2")
    assert sum(1 for command in tools.commands if command[0] == "umoci") == umoci_before
    adopted = json.loads((tmp_path / "regional2" / "images-v1" / "provenance.json").read_text())["images"]
    assert len(adopted) == 1 and adopted[0]["source"] == "registry-adopt"
    assert adopted[0]["digest"].startswith("sha256:")
    assert adopted[0]["image"] == first["images"][0]["image"]


def test_entry_point_records_failures_and_still_writes_provenance(tmp_path, monkeypatch):
    row, _recipe_digest, manifest = smoke_row()
    tools = FakeTools(manifest)
    tools.fail_on.update(("docker://mirror.gcr.io", "docker://registry-1.docker.io"))
    with pytest.raises(lane.BuildRejected, match="1 of 1 builds failed"):
        run_main(monkeypatch, tmp_path, row, tools)
    provenance = json.loads((tmp_path / "regional" / "images-v1" / "provenance.json").read_text())
    assert provenance["images"] == []
