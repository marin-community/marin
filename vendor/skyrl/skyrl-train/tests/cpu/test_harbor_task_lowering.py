"""Harbor task lowering keeps task.toml-declared deadlines in the prepared rows."""

from pathlib import Path

from skyrl_train.dataset.harbor import lower_harbor_task
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION
from taskcompendium.models import Source, TaskSpec

from tests.cpu.task_specs import text_task

# Session limits mirror the canonical full-trial eval profile; its verifier_timeout of 300 s is
# the value that must never override a task's own declared verifier deadline.
TRIAL_SESSION = {
    "task_session": HARBOR_TRIAL_SESSION,
    "max_turns": 1024,
    "max_total_tokens": None,
    "model_turn_timeout": 1800.0,
    "tool_turn_timeout": 1800.0,
    "total_turn_timeout": 1800.0,
    "attempt_timeout": 2460.0,
    "verifier_timeout": 300.0,
    "cleanup_timeout": 60.0,
}


def harbor_task(directory: Path, task: TaskSpec, task_toml: str) -> Path:
    (directory / "task.toml").write_text(task_toml)
    (directory / "instruction.md").write_text("Do the thing.\n")
    return lower_harbor_task(
        directory,
        task.model_copy(update={"source": Source(dataset="fixture", revision="1", row="0", importer_revision="1")}),
        session=TRIAL_SESSION,
    )


def test_trial_lowering_keeps_declared_verifier_and_agent_timeouts(tmp_path):
    lowered = harbor_task(
        tmp_path,
        text_task("tb2/declared"),
        """
[verifier]
timeout_sec = 1800.0

[agent]
timeout_sec = 1800.0
""",
    )
    assert lowered.limits.verifier_timeout == 1800.0
    assert lowered.limits.total_turn_timeout == 1800.0
    # Other limits still come from the caller's session.
    assert lowered.limits.attempt_timeout == 2460.0
    assert lowered.limits.max_turns == 1024
    assert lowered.harness == HARBOR_TRIAL_SESSION


def test_trial_lowering_defaults_to_harbor_defaults_when_task_toml_is_silent(tmp_path):
    lowered = harbor_task(tmp_path, text_task("tb2/silent"), "schema_version = '1.1'\n")
    # A missing [verifier] section means Harbor's own 600 s default, not the session's 300 s.
    assert lowered.limits.verifier_timeout == 600.0
    # A missing [agent] section declares no agent deadline; the deployment settings govern it.
    assert lowered.limits.total_turn_timeout is None
