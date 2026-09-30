"""Image-plan reviews that never decided used to hold items at pending_image_review forever.

Live shapes (catalog-full-construct-003, 2026-09-29): 5 items with no result.json
(``image_plan_review_incomplete``), 15 whose decision failed validation
(``image_plan_review_pending``, "image plan citation is unbound") and 11 whose
approval no longer bound the task (``review_or_task_binding_changed``).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline import image_pipeline as pipeline
from tests.test_image_pipeline import _args, _fixture


def _captured_runner(plan):
    def runner(command):
        Path(command[command.index("--output") + 1]).write_text(json.dumps({
            "schema_version": "capability-rootfs-capture-v1",
            "state": "captured_pending_privacy_and_publication", "role": "candidate",
            "plan_sha256": pipeline._sha(plan),
            "source_snapshot": {"name": "authored-snapshot", "id": "snapshot-id", "ref": "snapshot-ref"},
            "cleanup": {"absence_verified": True},
        }))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return runner


def _review_root(item, attempt):
    return item.parent / "image-reviews" / item.name / attempt.name


def _approving_review(calls):
    def review(**kw):
        calls.append(kw["review_root"])
        kw["review_root"].mkdir(parents=True)
        (kw["review_root"] / "approval.json").write_text("{}")
        return {"state": "approve"}
    return review


def test_interrupted_review_without_result_is_retired_and_rerun(tmp_path, monkeypatch):
    item, attempt, plan, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    root = _review_root(item, attempt)
    (root / "input").mkdir(parents=True)  # the packet was prepared, then the job died
    calls = []
    monkeypatch.setattr(pipeline, "run_review", _approving_review(calls))
    result = pipeline.process_image_construction(**_args(item, tools, scripts, _captured_runner(plan)))
    assert result["state"] == "pending_publication", result
    assert calls == [root]
    assert (root.parent / f"{root.name}.incomplete-1" / "input").is_dir()


def test_undecided_review_is_retried_with_explicit_attempts_and_backoff(tmp_path, monkeypatch):
    item, attempt, _, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    root = _review_root(item, attempt)
    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"state": "pending", "issues": ["image plan citation is unbound"]}))

    def review(**kw):
        kw["review_root"].mkdir(parents=True)
        return {"state": "pending", "issues": ["image plan citation is unbound"]}

    monkeypatch.setattr(pipeline, "run_review", review)
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no capture without approval")))
    assert result["state"] == "pending_review" and result["retryable"] is True
    assert result["attempts"] == 2 and result["max_attempts"] == pipeline.MAX_REVIEW_RETIREMENTS + 1
    assert result["backoff_seconds"] == 300
    assert result["reason"] == "image_plan_review_pending"


def test_review_retirements_are_bounded_then_terminal(tmp_path, monkeypatch):
    item, attempt, _, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    root = _review_root(item, attempt)
    for index in range(1, pipeline.MAX_REVIEW_RETIREMENTS + 1):
        (root.parent / f"{root.name}.incomplete-{index}").mkdir(parents=True)
    root.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "run_review", lambda **kw: pytest.fail("the bound is spent"))
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no capture")))
    assert result["state"] == "failed_terminal" and result["failure_stage"] == "image_review"
    assert result["reason"] == "image_plan_review_incomplete_after_4_attempts"
    assert result["retryable"] is False
    assert root.is_dir()  # the last evidence stays in place


def test_stale_approval_is_retired_and_reviewed_afresh(tmp_path, monkeypatch):
    item, attempt, plan, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    root = _review_root(item, attempt)
    root.mkdir(parents=True)
    (root / "approval.json").write_text("{}")
    checks = []

    def validate(*args, **kwargs):
        checks.append(args[0])
        if len(checks) == 1:
            raise ValueError("builder artifacts changed after review")
        return {}

    monkeypatch.setattr(pipeline, "validate_review", validate)
    calls = []
    monkeypatch.setattr(pipeline, "run_review", _approving_review(calls))
    result = pipeline.process_image_construction(**_args(item, tools, scripts, _captured_runner(plan)))
    assert result["state"] == "pending_publication", result
    assert calls == [root] and (root.parent / f"{root.name}.stale-1" / "approval.json").is_file()


def test_review_transport_failure_is_retryable(tmp_path, monkeypatch):
    item, _, _, _, tools, scripts = _fixture(tmp_path, monkeypatch)

    def review(**kw):
        kw["review_root"].mkdir(parents=True)
        raise RuntimeError("GLM relay unavailable")

    monkeypatch.setattr(pipeline, "run_review", review)
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no capture")))
    assert result["state"] == "pending_review" and result["retryable"] is True
    assert result["reason"] == "image_review_transport_failed" and result["failure_class"] == "transport"
    # The conveyor's image_review_transport budget owns attempts and backoff, not the review bound.
    assert result["transport_attempts"] == 1
    assert "attempts" not in result and "max_attempts" not in result and "backoff_seconds" not in result


def test_recorded_repair_decision_stays_repairable(tmp_path, monkeypatch):
    item, attempt, _, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    root = _review_root(item, attempt)
    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"state": "repair", "issues": ["Source closure incomplete"]}))
    monkeypatch.setattr(pipeline, "run_review", lambda **kw: pytest.fail("a decision is final"))
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no capture")))
    assert result["state"] == "repairable" and result["issues"] == ["Source closure incomplete"]


# -- a GLM outage is not a review outcome (integration review 2026-09-29) ---------------------


def _undecided_root(root, *, returncode, timed_out=False):
    """What run_review leaves when the reviewer did not produce a valid decision."""
    root.mkdir(parents=True)
    (root / "execution.json").write_text(json.dumps({"returncode": returncode, "timed_out": timed_out}))
    (root / "result.json").write_text(json.dumps({"state": "pending", "issues": ["image plan reviewer did not complete"]}))


@pytest.mark.parametrize("returncode,timed_out", [(7, False), (0, True)])
def test_transport_failed_review_is_retired_outside_the_review_bound(tmp_path, monkeypatch, returncode, timed_out):
    item, attempt, plan, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    root = _review_root(item, attempt)
    for index in range(1, pipeline.MAX_REVIEW_RETIREMENTS + 1):  # the review bound is already spent
        (root.parent / f"{root.name}.incomplete-{index}").mkdir(parents=True)
    _undecided_root(root, returncode=returncode, timed_out=timed_out)
    calls = []
    monkeypatch.setattr(pipeline, "run_review", _approving_review(calls))
    result = pipeline.process_image_construction(**_args(item, tools, scripts, _captured_runner(plan)))
    assert result["state"] == "pending_publication", result
    assert calls == [root]
    assert (root.parent / f"{root.name}.transport-1" / "execution.json").is_file()
    assert len(pipeline._retired_reviews(root)) == pipeline.MAX_REVIEW_RETIREMENTS


def test_decided_but_malformed_review_keeps_the_bound(tmp_path, monkeypatch):
    item, attempt, _, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    root = _review_root(item, attempt)
    for index in range(1, pipeline.MAX_REVIEW_RETIREMENTS + 1):
        (root.parent / f"{root.name}.incomplete-{index}").mkdir(parents=True)
    _undecided_root(root, returncode=0)  # the reviewer finished; its decision failed validation
    monkeypatch.setattr(pipeline, "run_review", lambda **kw: pytest.fail("the bound is spent"))
    result = pipeline.process_image_construction(**_args(
        item, tools, scripts, lambda _: pytest.fail("no capture")))
    assert result["state"] == "failed_terminal" and result["failure_stage"] == "image_review"


def test_transport_exception_marks_the_directory_for_a_transport_retirement(tmp_path, monkeypatch):
    item, attempt, plan, _, tools, scripts = _fixture(tmp_path, monkeypatch)
    root = _review_root(item, attempt)

    def down(**kw):
        kw["review_root"].mkdir(parents=True)  # the packet was prepared, then the relay failed
        raise OSError("connection refused")

    monkeypatch.setattr(pipeline, "run_review", down)
    first = pipeline.process_image_construction(**_args(item, tools, scripts, lambda _: pytest.fail("no capture")))
    assert first["failure_class"] == "transport" and (root / pipeline.REVIEW_TRANSPORT_MARKER).is_file()
    calls = []
    monkeypatch.setattr(pipeline, "run_review", _approving_review(calls))
    second = pipeline.process_image_construction(**_args(item, tools, scripts, _captured_runner(plan)))
    assert second["state"] == "pending_publication"
    assert (root.parent / f"{root.name}.transport-1").is_dir()
    assert pipeline._retired_reviews(root) == []


def test_review_retries_carry_the_reviewer_session_bound(tmp_path, monkeypatch):
    item, _, _, _, tools, scripts = _fixture(tmp_path, monkeypatch)

    def undecided(**kw):
        kw["review_root"].mkdir(parents=True)
        return {"state": "pending", "issues": ["image plan citation is unbound"]}

    monkeypatch.setattr(pipeline, "run_review", undecided)
    args = _args(item, tools, scripts, lambda _: pytest.fail("no capture"))
    args["agent"] = SimpleNamespace(model="glm-5.3", session_time=28_800)
    result = pipeline.process_image_construction(**args)
    assert result["reason"] == "image_plan_review_pending" and result["step_timeout_seconds"] == 28_800


def test_glm_outage_during_review_never_becomes_failed_terminal(tmp_path, monkeypatch):
    """Reviewer repro: omp exits non-zero on every image-plan review.  Before the fix the fifth
    call was failed_terminal (4 retirements); now every call is a transport retry, classified
    into the conveyor's image_review_transport budget, and the item approves once GLM is back."""
    import shutil

    from test_publication_exchange import _world

    from capability_pipeline import conveyor

    world = _world(tmp_path, monkeypatch)
    review_root = world.review_base / world.attempt.name
    approved = tmp_path / "approved-review"
    shutil.copytree(review_root, approved)
    shutil.rmtree(review_root)
    monkeypatch.setattr("capability_pipeline.image_plan_review.prepare_packet",
                        lambda item_root, review_root, extra: (review_root.mkdir(parents=True, exist_ok=True) or
                                                               {"files": {"controller/image-plan.json": "0" * 64},
                                                                "snapshot_hash": "s", "item_files": []}))

    class Down:
        model = "glm-5.3"
        session_time = 600
        invocations = 0

        def invoke(self, *args):
            Down.invocations += 1
            return {"returncode": 7, "timed_out": False, "stdout": "", "stderr": "relay unreachable"}

    def construct(agent):
        return pipeline.process_image_construction(
            item_root=world.item, capture_tools=world.tools, scripts_root=world.scripts, agent=agent,
            builder_session_ids={"builder-1"}, command_runner=world.runner, review_base=world.review_base,
            publication_queue=world.queue)

    results = [construct(Down()) for _ in range(pipeline.MAX_REVIEW_RETIREMENTS + 4)]
    assert Down.invocations == len(results)
    assert {r["state"] for r in results} == {"pending_review"}
    assert all(r["failure_class"] == "transport" and r["retryable"] is True for r in results)
    assert [r["transport_attempts"] for r in results] == list(range(1, len(results) + 1))
    assert pipeline._retired_reviews(review_root) == []
    assert len(pipeline._transport_retirements(review_root)) == len(results) - 1
    status = {"state": "pending_image_review", "custom_images": results[-1]}
    classification = conveyor.classify_result(status, repair_actionable=lambda value: False)
    assert (classification.klass, classification.stage) == (conveyor.WAITING, "image_review_transport")

    def approve(**kw):  # GLM is back: the next review approves
        shutil.copytree(approved, kw["review_root"])
        return {"state": "approve"}

    monkeypatch.setattr(pipeline, "run_review", approve)
    recovered = construct(Down())
    assert recovered["state"] == "pending_publication", recovered
