import json

import pytest

from capability_pipeline.generic_image_publication import validate_review
from capability_pipeline.image_plan_review import run_review
from capability_pipeline.image_review_contract import CHECKS


class Reviewer:
    model = "glm-orion/glm-5.3"

    def __init__(self, action="approve"):
        self.action = action

    def invoke(self, workspace, session_dir, prompt, attempt):
        assert attempt == 0
        assert session_dir.name.startswith("image-review-")
        manifest = json.loads((workspace / "input-manifest.json").read_text())
        path = "controller/image-plan.json"
        decision = {
            "plan_sha256": manifest["files"][path],
            "snapshot_hash": manifest["snapshot_hash"],
            "decision": "approve", "issues": [],
            "checks": [{"id": check, "state": "passed", "citations": [{
                "path": path, "sha256": manifest["files"][path],
                "supports": "Controlled metadata fixture, no live evidence claimed.",
            }]} for check in CHECKS],
        }
        if self.action == "repair":
            decision.update(decision="repair", issues=["Missing source provenance."])
        if self.action == "unbound":
            decision["checks"][0]["citations"][0]["sha256"] = "0" * 64
        if self.action == "tamper":
            source = workspace / "input" / path
            source.chmod(0o644)
            source.write_text('{"changed":true}')
        (workspace / "decision.json").write_text(json.dumps(decision))
        # Reviewer-supplied approvals must never survive a rejected review.
        (workspace / "approval.json").write_text('{"state":"approved"}')
        return {"returncode": 0, "timed_out": self.action == "timeout",
                "stdout": "", "stderr": ""}


def inputs(tmp_path):
    item = tmp_path / "item"
    (item / "contract").mkdir(parents=True)
    (item / "workspace/task").mkdir(parents=True)
    (item / "contract/accepted.json").write_text('{}')
    (item / "workspace/task/specification.json").write_text('{}')
    plan = tmp_path / "plan.json"
    plan.write_text('{"fixture":"metadata only"}')
    return item, plan, tmp_path / "review"


def test_fresh_review_emits_controller_approval_bound_to_raw_decision(tmp_path):
    item, plan, review = inputs(tmp_path)
    result = run_review(item_root=item, plan_path=plan, review_root=review,
                        agent=Reviewer(), builder_session_ids={"builder-s1"})
    assert result["state"] == "approve"
    approval = validate_review(review / "approval.json", plan,
                               builder_session_ids={"builder-s1"})
    assert approval["model"] == "glm-5.3"
    assert approval["raw_artifact"] == "decision.json"
    assert result["approval_path"] == str(review / "approval.json")


@pytest.mark.parametrize("action", ["repair", "unbound", "tamper", "timeout"])
def test_review_cannot_approve_missing_failed_or_mutated_evidence(tmp_path, action):
    item, plan, review = inputs(tmp_path)
    result = run_review(item_root=item, plan_path=plan, review_root=review,
                        agent=Reviewer(action), builder_session_ids=set())
    assert result["state"] != "approve"
    assert result["issues"]
    assert not (review / "approval.json").exists()


def test_other_model_cannot_supply_pipeline_review(tmp_path):
    item, plan, review = inputs(tmp_path)
    agent = Reviewer()
    agent.model = "other"
    with pytest.raises(ValueError, match="requires GLM5.3"):
        run_review(item_root=item, plan_path=plan, review_root=review,
                   agent=agent, builder_session_ids=set())
    assert not review.exists()
