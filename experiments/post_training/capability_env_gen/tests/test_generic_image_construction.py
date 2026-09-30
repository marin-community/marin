import json

import pytest
from test_generic_image_capture import _fixture

from capability_pipeline.generic_image_construction import (
    freeze_request,
    request_needed,
)


def test_construction_freezes_exact_custom_pointer_request(tmp_path):
    workspace, tools, plan, _ = _fixture(tmp_path)
    task = workspace / "task"
    task.mkdir()
    ref = plan["images"][0]["source_snapshot"]["ref"]
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": ref}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": ref}}))
    plan.pop("capture_implementation")
    (task / "image-capture-request.json").write_text(json.dumps(plan))
    assert request_needed(workspace)["needed"] is True
    result = freeze_request(workspace, tmp_path / "frozen", tools)
    assert result["state"] == "review_required"
    assert len(result["pointers"]) == 2
    frozen = json.loads((tmp_path / "frozen/input/plan.json").read_text())
    assert frozen["capture_implementation"]["dtx_sha256"]
    assert (tmp_path / "frozen/input/workspace/public.txt").read_text() == "public"
    assert freeze_request(workspace, tmp_path / "frozen", tools)["plan_sha256"] == result["plan_sha256"]


@pytest.mark.parametrize("name", [
    "registry.example/tasks/base", "python:3.12-slim",
    "docker.io/library/python:3.12-slim", "registry.example:5000/tasks/base:v1",
])
def test_construction_skips_canonical_registry_image(tmp_path, name):
    workspace = tmp_path / "workspace"
    task = workspace / "task"
    task.mkdir(parents=True)
    ref = name + "@sha256:" + "a" * 64
    (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": ref}}, "steps": []}))
    (task / "binding.json").write_text(json.dumps({"environment": {"image": ref}}))
    assert request_needed(workspace)["needed"] is False


def test_daytona_recipe_hash_is_not_a_published_oci_digest(tmp_path):
    workspace = tmp_path / "workspace"
    task = workspace / "task"
    task.mkdir(parents=True)
    pointer = "envgen.daytona/cap-slot5-arbiter-verifier/dockerfile@sha256:" + "a" * 64
    (task / "specification.json").write_text(json.dumps({
        "requirements": {"state": {"image": None}},
        "steps": [{"verifier": {"runtime": {"image": pointer}}}],
    }))
    (task / "binding.json").write_text(json.dumps({"environment": {}}))
    inventory = request_needed(workspace)
    assert inventory["needed"] is True
    assert inventory["custom_pointers"] == [{
        "role": "private_verifier",
        "pointer": "specification.steps.0.verifier.runtime.image",
        "image": pointer,
    }]


def test_authored_capture_request_overrides_digest_shaped_pointer(tmp_path):
    workspace = tmp_path / "workspace"
    task = workspace / "task"
    task.mkdir(parents=True)
    pointer = "internal.example/verifier@sha256:" + "b" * 64
    (task / "specification.json").write_text(json.dumps({
        "requirements": {"state": {"image": None}},
        "steps": [{"verifier": {"runtime": {"image": pointer}}}],
    }))
    (task / "binding.json").write_text(json.dumps({"environment": {}}))
    (task / "image-capture-request.json").write_text(json.dumps({
        "images": [{"role": "private_verifier", "authored_image_pointer": pointer}],
    }))
    assert request_needed(workspace)["custom_pointers"][0]["image"] == pointer


def _migrated_workspace(tmp_path, requested_pointer):
    workspace = tmp_path / "workspace"
    task = workspace / "task"
    task.mkdir(parents=True)
    published = "envreg.example/capability-env-gen/verifier@sha256:" + "c" * 64
    (task / "specification.json").write_text(json.dumps({
        "requirements": {"state": {"image": None}},
        "steps": [{"verifier": {"runtime": {"image": published}}}],
    }))
    (task / "binding.json").write_text(json.dumps({"environment": {}}))
    (task / "image-capture-request.json").write_text(json.dumps({
        "images": [{"role": "private_verifier", "authored_image_pointer": requested_pointer}],
    }))
    return workspace


def test_request_replaced_by_applied_migration_is_not_needed(tmp_path):
    # Migration rewrites the task documents but not the builder's request, so
    # afterwards the request no longer binds a task pointer.
    pointer = "envgen.daytona/verifier/dockerfile@sha256:" + "d" * 64
    workspace = _migrated_workspace(tmp_path, pointer)
    with pytest.raises(ValueError, match="does not bind"):
        request_needed(workspace)
    inventory = request_needed(workspace, migrated_pointers={pointer})
    assert inventory["needed"] is False
    assert inventory["custom_pointers"] == []


def test_migrated_pointers_do_not_excuse_an_unrelated_request(tmp_path):
    workspace = _migrated_workspace(tmp_path, "envgen.daytona/other/dockerfile@sha256:" + "e" * 64)
    with pytest.raises(ValueError, match="does not bind"):
        request_needed(workspace, migrated_pointers={"envgen.daytona/verifier/dockerfile@sha256:" + "d" * 64})
