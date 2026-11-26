# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The launch-time judge-route gate for declared source judges."""

from taskcompendium.models import TaskSpec, judge_spec
from verifyit.modes.grade_judge import JudgeConnection

NO_JUDGE = "no_judge"
DEFAULT_FALLBACK = "default_fallback"
SERVED = "served"
SUBSTITUTED = "substituted"


def judge_route_verdict(task: TaskSpec, connection: JudgeConnection | None) -> str:
    """Classify a task's declared judge against the deployed judge route, refusing broken ones.

    The deployment serves exactly one judge model (the connection's default) and substitutes the
    declared models it cannot serve (the ``substitute_models`` map; an entry mapping a model to
    itself records a served model). Anything else would be sent verbatim and fail every rollout
    closed at grading time - the marin #12374 class - so the CPU gate refuses it before any GPU
    allocation.
    """
    judge = judge_spec(task.grader)
    if judge is None:
        return NO_JUDGE
    declared = judge.model.strip()
    if not declared:
        return DEFAULT_FALLBACK
    if connection is not None:
        served = (connection.default_model or "").strip()
        if declared in connection.substitute_models:
            return SUBSTITUTED
        if served and declared == served:
            return SERVED
    raise ValueError(
        f"Source {task.source.dataset} declares judge model {declared!r} that the deployed judge endpoint "
        "neither serves nor substitutes; every rollout would fail closed at grading time (marin #12374). "
        "Add the model to task_sessions.remote_judge.substitute_models, or serve it, before GPU training."
    )
