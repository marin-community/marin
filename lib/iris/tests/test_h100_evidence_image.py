# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[3]
DOCKERFILE = REPO_ROOT / "lib" / "iris" / "Dockerfile"
PACKAGE_MANIFEST = REPO_ROOT / "lib" / "iris" / "images" / "h100-evidence-debian12-amd64.sha256"
IMAGE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ops-docker-images.yaml"

EXPECTED_PACKAGES = {
    "cuda-cccl-13-2_13.2.86-1_amd64.deb",
    "cuda-crt-13-2_13.2.86-1_amd64.deb",
    "cuda-cudart-13-2_13.2.86-1_amd64.deb",
    "cuda-cudart-dev-13-2_13.2.86-1_amd64.deb",
    "cuda-culibos-dev-13-2_13.2.86-1_amd64.deb",
    "cuda-cuobjdump-13-2_13.2.86-1_amd64.deb",
    "cuda-driver-dev-13-2_13.2.86-1_amd64.deb",
    "cuda-nvcc-13-2_13.2.86-1_amd64.deb",
    "cuda-toolkit-13-2-config-common_13.2.86-1_all.deb",
    "cuda-toolkit-13-config-common_13.2.86-1_all.deb",
    "cuda-toolkit-config-common_13.2.86-1_all.deb",
    "libnvptxcompiler-13-2_13.2.86-1_amd64.deb",
    "libnvvm-13-2_13.2.86-1_amd64.deb",
    "nsight-compute-2026.1.1_2026.1.1.2-1_amd64.deb",
}


def test_h100_evidence_package_manifest_is_closed_and_hash_pinned():
    records = [line.split() for line in PACKAGE_MANIFEST.read_text().splitlines()]

    assert {filename for _, filename in records} == EXPECTED_PACKAGES
    assert len(records) == len(EXPECTED_PACKAGES)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest, _ in records)


def test_h100_evidence_target_inherits_task_and_checks_every_required_tool():
    dockerfile = DOCKERFILE.read_text()
    target = dockerfile.split("FROM task AS task-h100-evidence", maxsplit=1)[1]

    assert "h100-evidence-debian12-amd64.sha256" in target
    assert 'test "$(dpkg --print-architecture)" = amd64' in target
    for tool in ("nvcc", "ptxas", "cuobjdump", "ncu", "nsys"):
        assert re.search(rf"/[^ ;]*{tool} --version", target)


def test_h100_evidence_workflow_publishes_only_full_sha_tag_for_amd64():
    workflow = yaml.safe_load(IMAGE_WORKFLOW.read_text())
    job = workflow["jobs"]["iris-h100-evidence-image"]
    build = next(step for step in job["steps"] if step.get("id") == "build")

    assert job["if"] == "github.event_name == 'workflow_dispatch'"
    assert build["env"]["IMAGE"].endswith(":${{ needs.iris-tags.outputs.full_hash }}")
    assert "--target task-h100-evidence" in build["run"]
    assert "--platform linux/amd64" in build["run"]
    assert "containerimage.digest" in build["run"]
    assert ":latest" not in build["run"]
    assert "outputs.date" not in build["run"]
