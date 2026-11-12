"""One patch graded by SWE-bench's single-instance Docker harness."""

from contextlib import closing
from enum import StrEnum
import json

from eval.contracts.eval_spec import Grade
from eval.runtime.checkpoints import input_digest

_TEST_TIMEOUT = 1800


class PatchOutcome(StrEnum):
    EMPTY = "empty_patch"
    REJECTED = "patch_rejected"
    TIMEOUT = "test_timeout"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


def _patch_grade(spec, status: PatchOutcome, report=None):
    resolved = status is PatchOutcome.RESOLVED
    completed = status in (PatchOutcome.RESOLVED, PatchOutcome.UNRESOLVED)
    audit = {
        "report_status": status.value,
        "report_outcome": {
            "instance_id": spec.doc["instance_id"],
            "completed": completed, "incomplete": False, "submitted": True,
            "empty_patch": status is PatchOutcome.EMPTY, "resolved": resolved,
            "unresolved": status is PatchOutcome.UNRESOLVED,
            "error": status in (PatchOutcome.REJECTED, PatchOutcome.TIMEOUT),
        },
    }
    if report is not None:
        audit["report"] = report
    if status is PatchOutcome.TIMEOUT:
        audit["test_timed_out"] = True
    return Grade({"accuracy": float(resolved)}, audit)


def swe_patch(spec, responses):
    patch = str(responses[-1])
    if not patch:
        return _patch_grade(spec, PatchOutcome.EMPTY)

    import docker
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL, KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, LOG_INSTANCE, LOG_REPORT, LOG_TEST_OUTPUT, RUN_EVALUATION_LOG_DIR,
    )
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance = make_test_spec(dict(spec.doc), namespace="swebench", instance_image_tag="v1")
    prediction = {KEY_INSTANCE_ID: spec.doc["instance_id"], KEY_MODEL: "evalchemy", KEY_PREDICTION: patch}
    run_id = "evalchemy-" + input_digest(prediction)[:16]
    with closing(docker.from_env()) as client:
        outcome = run_instance(instance, prediction, rm_image=False, force_rebuild=False, client=client,
                               run_id=run_id, timeout=_TEST_TIMEOUT)
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / prediction[KEY_MODEL] / prediction[KEY_INSTANCE_ID]
    if not outcome["completed"]:
        # The harness also returns incomplete for candidate failures. Require its
        # generated timeout artifacts so unrelated infrastructure timeouts still fail.
        log_path = log_dir / LOG_INSTANCE
        log = log_path.read_text()
        if APPLY_PATCH_FAIL in log:
            return _patch_grade(spec, PatchOutcome.REJECTED)
        test_output = log_dir / LOG_TEST_OUTPUT
        if (f"Test timed out after {_TEST_TIMEOUT} seconds." in log and test_output.is_file()
                and test_output.read_text().endswith(f"\n\nTimeout error: {_TEST_TIMEOUT} seconds exceeded.")):
            return _patch_grade(spec, PatchOutcome.TIMEOUT)
        raise RuntimeError(f"SWE-bench did not complete instance {spec.source_id}; see {log_path}")
    report = json.loads((log_dir / LOG_REPORT).read_text())
    status = PatchOutcome.RESOLVED if report[spec.doc["instance_id"]]["resolved"] else PatchOutcome.UNRESOLVED
    return _patch_grade(spec, status, report)
