# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[3]
DOCKERFILE = REPO_ROOT / "lib" / "iris" / "Dockerfile"
PACKAGE_MANIFEST = REPO_ROOT / "lib" / "iris" / "images" / "h100-evidence-debian12-amd64.sha256"
H100_IMAGE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ops-h100-evidence-image.yaml"
BROAD_IMAGE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ops-docker-images.yaml"

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

LEGACY_IMAGE_JOBS = {
    "iris-tags",
    "iris-images",
    "iris-manifests",
    "finelog-image",
    "marin-tpu-ci-images",
}
WEEKLY_IMAGE_JOBS = {"iris-tags", "iris-images", "iris-manifests", "finelog-image"}
MANUAL_ONLY_PATTERN = re.compile(
    r"github\.event_name == 'workflow_dispatch' && inputs\.image_set == '(?P<image_set>all|h100-evidence)'"
)
MANUAL_OR_SCHEDULE_PATTERN = re.compile(
    r"\(github\.event_name == 'workflow_dispatch' && inputs\.image_set == '(?P<image_set>all|h100-evidence)'\)"
    r"\s*\|\|\s*"
    r"\(github\.event_name == 'schedule' && github\.event\.schedule == '(?P<schedule>[^']+)'\)"
)


def _job_runs(job: dict, *, event_name: str, image_set: str = "", schedule: str = "") -> bool:
    condition = job["if"].strip()
    manual_only = MANUAL_ONLY_PATTERN.fullmatch(condition)
    if manual_only:
        return event_name == "workflow_dispatch" and image_set == manual_only.group("image_set")

    manual_or_schedule = MANUAL_OR_SCHEDULE_PATTERN.fullmatch(condition)
    if manual_or_schedule:
        manual = event_name == "workflow_dispatch" and image_set == manual_or_schedule.group("image_set")
        scheduled = event_name == "schedule" and schedule == manual_or_schedule.group("schedule")
        return manual or scheduled

    raise AssertionError(f"job has an unsupported condition: {condition}")


def _jobs_for(workflow: dict, *, event_name: str, image_set: str = "", schedule: str = "") -> set[str]:
    return {
        name
        for name, job in workflow["jobs"].items()
        if _job_runs(job, event_name=event_name, image_set=image_set, schedule=schedule)
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


def test_h100_evidence_workflow_dispatch_builds_one_exact_source_image():
    workflow = yaml.safe_load(H100_IMAGE_WORKFLOW.read_text())
    triggers = workflow[True]

    assert set(triggers) == {"workflow_dispatch", "workflow_call"}
    assert triggers["workflow_dispatch"]["inputs"] == {
        "ref": {"description": "Git ref to build", "required": True, "type": "string"}
    }
    assert triggers["workflow_call"] == {
        "inputs": {"ref": {"description": "Git ref to build", "required": True, "type": "string"}},
        "outputs": {
            "image_ref": {
                "description": "Full-SHA image reference with OCI digest",
                "value": "${{ jobs.build-h100-evidence-image.outputs.image_ref }}",
            }
        },
    }
    assert workflow["permissions"] == {"contents": "read", "packages": "write"}
    assert set(workflow["jobs"]) == {"build-h100-evidence-image"}

    job = workflow["jobs"]["build-h100-evidence-image"]
    assert "strategy" not in job
    checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v5")
    source = next(step for step in job["steps"] if step.get("id") == "source")
    build = next(step for step in job["steps"] if step.get("id") == "build")

    assert checkout["with"] == {"ref": "${{ inputs.ref }}", "persist-credentials": False}
    assert source["run"].count("git rev-parse HEAD") == 1
    assert "^[0-9a-f]{40}$" in source["run"]
    assert build["env"]["IMAGE"] == (
        "ghcr.io/marin-community/iris-task-h100-evidence:${{ steps.source.outputs.full_sha }}"
    )
    assert "--target task-h100-evidence" in build["run"]
    assert "--platform linux/amd64" in build["run"]
    assert build["run"].count("docker buildx build ") == 1
    assert re.findall(r"--tag ([^ ]+)", build["run"]) == ['"$IMAGE"']
    assert "containerimage.digest" in build["run"]
    assert 'docker buildx imagetools inspect "$image_ref"' in build["run"]
    assert ":latest" not in build["run"]
    assert "DATE_TAG" not in build["run"]
    assert "HASH_TAG" not in build["run"]


def test_broad_ops_manual_image_set_selects_legacy_or_h100_exclusively():
    workflow = yaml.safe_load(BROAD_IMAGE_WORKFLOW.read_text())
    dispatch = workflow[True]["workflow_dispatch"]
    bridge = workflow["jobs"]["h100-evidence-image"]

    assert dispatch["inputs"] == {
        "image_set": {
            "description": "Image set to build",
            "required": True,
            "default": "all",
            "type": "choice",
            "options": ["all", "h100-evidence"],
        }
    }
    assert set(workflow["jobs"]) == LEGACY_IMAGE_JOBS | {"h100-evidence-image"}
    assert _jobs_for(workflow, event_name="workflow_dispatch", image_set="all") == LEGACY_IMAGE_JOBS
    assert _jobs_for(workflow, event_name="workflow_dispatch", image_set="h100-evidence") == {"h100-evidence-image"}
    assert _jobs_for(workflow, event_name="schedule", schedule="0 2 * * 0") == WEEKLY_IMAGE_JOBS
    assert _jobs_for(workflow, event_name="schedule", schedule="0 3 * * *") == {"marin-tpu-ci-images"}
    assert bridge == {
        "if": "github.event_name == 'workflow_dispatch' && inputs.image_set == 'h100-evidence'",
        "permissions": {"contents": "read", "packages": "write"},
        "uses": "./.github/workflows/ops-h100-evidence-image.yaml",
        "with": {"ref": "${{ github.sha }}"},
    }
