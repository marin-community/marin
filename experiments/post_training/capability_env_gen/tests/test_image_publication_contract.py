import hashlib

import pytest

from capability_pipeline.image_publication_contract import (
    PublicationContractError,
    validate_publication_contract,
)


def contract(tmp_path):
    source = tmp_path / "Dockerfile"
    source.write_text("FROM scratch\n")
    config = {
        "Env": ["PATH=/bin"],
        "WorkingDir": "",
        "User": "",
        "Entrypoint": ["/entrypoint"],
        "Cmd": ["service"],
    }
    lifecycle = {
        "ready_probe": "test -f /tmp/ready",
        "shutdown": "service stop",
        "shutdown_probe": "test ! -f /run/service.pid",
        "reset_validation": "fresh blocked sandbox passes task gates",
    }
    return {
        "schema_version": "capability-image-publication-plan-v1",
        "state": "review_required",
        "review_blockers": ["inherited Cmd conflicts with entrypoint"],
        "source_files": [{"path": "Dockerfile", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
        "images": [
            {"role": "candidate", "repository": "capability-env-gen/task-candidate", "source_recipe": {"path": "Dockerfile", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}, "source_snapshot": {"id": "0" * 8 + "-" + "0" * 4 + "-" + "0" * 4 + "-" + "0" * 4 + "-" + "0" * 12, "name": "candidate", "ref": "provider/ref"}, "image_config": config, "capture_lifecycle": lifecycle, "privacy_boundary": {"reject_paths": ["/private"]}, "required_ready_hashes": {"/workspace/task.txt": "a" * 64}},
            {"role": "private_verifier", "repository": "capability-env-gen/task-verifier", "source_recipe": {"path": "Dockerfile", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}, "source_snapshot": {"id": "1" * 8 + "-" + "1" * 4 + "-" + "1" * 4 + "-" + "1" * 4 + "-" + "1" * 12, "name": "verifier", "ref": "provider/private"}, "image_config": config, "capture_lifecycle": lifecycle, "privacy_boundary": {"reject_paths": []}},
        ],
    }


def test_review_contract_binds_sources_and_separate_roles(tmp_path):
    document = contract(tmp_path)
    assert validate_publication_contract(document, workspace=tmp_path) is document


def test_approval_fails_with_unresolved_blocker(tmp_path):
    document = contract(tmp_path)
    document["state"] = "approved_for_capture"
    with pytest.raises(PublicationContractError, match="unresolved blockers"):
        validate_publication_contract(document, workspace=tmp_path)


def test_helper_rejects_source_drift_and_repository_reuse(tmp_path):
    document = contract(tmp_path)
    (tmp_path / "Dockerfile").write_text("changed")
    with pytest.raises(PublicationContractError, match="source hash mismatch"):
        validate_publication_contract(document, workspace=tmp_path)
    document = contract(tmp_path)
    document["images"][1]["repository"] = document["images"][0]["repository"]
    with pytest.raises(PublicationContractError, match="reused"):
        validate_publication_contract(document, workspace=tmp_path)
