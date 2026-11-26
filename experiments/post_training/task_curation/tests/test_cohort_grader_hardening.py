# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cohort grader hardening (coordinator ruling 2026-11-26): the OpenSWE
reward path recomputes the verdict from the copied eval output and scores zero
on any disagreement with the copied reward file, and the collect sequence kills
leftover agent processes before the verifier runs and before the artifacts
copy."""

import json
import subprocess
import sys

import pytest
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.openswe.grader import VERIFIER_SCRIPT
from experiments.post_training.task_curation.harbor_cohort_conversion import (
    COPIED_EVAL_OUTPUT,
    KILL_AGENT_LEFTOVERS_SCRIPT,
    OPENSWE_RECOMPUTE_SCRIPT,
    REWARD_DIVERGENCE_PATH,
    REWARD_JSON_PATH,
    REWARD_TEXT_PATH,
    _reward_files,
    convert_cohort_task,
)

# The guard and recompute scripts run on the grading machine at fixed absolute
# paths; the tests stage them under a temporary root and substitute the paths.
SUBSTITUTIONS = (
    ("/source-reward.json", "source-reward.json"),
    ("/source-reward.txt", "source-reward.txt"),
    ("/source-eval-output.txt", "source-eval-output.txt"),
    ("/source-test-output.txt", "source-test-output.txt"),
    (REWARD_JSON_PATH, "converted-reward.json"),
    (REWARD_TEXT_PATH, "converted-reward.txt"),
    (REWARD_DIVERGENCE_PATH, "converted-reward-divergence.txt"),
)


def localized(script: str, root) -> str:
    for absolute, relative in SUBSTITUTIONS:
        script = script.replace(absolute, f"{root}/{relative}")
    return script


def run_guard(root, *, reward_json: bytes | None, reward_text: bytes | None, eval_output: bytes):
    """Stage the copied artifacts and run the production fallback guard's text path.

    The guard runs at the grading machine's fixed absolute paths; the test
    substitutes every production path with its staged location under ``root`` —
    the composition itself is the production argv from ``_reward_files``.
    """
    if reward_json is not None:
        (root / "source-reward.json").write_bytes(reward_json)
    if reward_text is not None:
        (root / "source-reward.txt").write_bytes(reward_text)
    (root / "source-eval-output.txt").write_bytes(eval_output)
    _, _, guard = _reward_files(False, OPENSWE_RECOMPUTE_SCRIPT)
    script = guard.argv[2]
    for absolute, relative in SUBSTITUTIONS:
        script = script.replace(absolute, f"{root}/{relative}")
    completed = subprocess.run(("bash", "-c", script), capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


def test_open_swe_recompute_scores_zero_when_a_planted_reward_disagrees(tmp_path):
    """A planted passing reward over a failing bug state cannot score: the
    recomputed marker rule disagrees, the reward becomes 0, and the divergence
    lands beside it with both values."""
    run_guard(tmp_path, reward_json=None, reward_text=b"1\n", eval_output=b"OPENSWE_EXIT_CODE=1\n")
    assert (tmp_path / "converted-reward.txt").read_bytes() == b"0\n"
    divergence = json.loads(json.dumps((tmp_path / "converted-reward-divergence.txt").read_text()))
    assert divergence is not None
    assert "recorded=1" in (tmp_path / "converted-reward-divergence.txt").read_text()
    assert "recomputed=0" in (tmp_path / "converted-reward-divergence.txt").read_text()


def test_open_swe_recompute_keeps_an_agreeing_reward(tmp_path):
    run_guard(tmp_path, reward_json=None, reward_text=b"1\n", eval_output=b"OPENSWE_EXIT_CODE=0\n")
    assert (tmp_path / "converted-reward.txt").read_bytes() == b"1\n"
    assert not (tmp_path / "converted-reward-divergence.txt").exists()


def test_open_swe_recompute_fails_ambiguous_output_closed(tmp_path):
    """A doubled echo makes the verdict untrustworthy: the recomputed rule scores
    0, so a planted passing reward scores 0 too."""
    run_guard(
        tmp_path,
        reward_json=None,
        reward_text=b"1\n",
        eval_output=b"OPENSWE_EXIT_CODE=0\nOPENSWE_EXIT_CODE=0\n",
    )
    assert (tmp_path / "converted-reward.txt").read_bytes() == b"0\n"
    assert "recomputed=0" in (tmp_path / "converted-reward-divergence.txt").read_text()


def test_open_swe_recompute_leaves_a_missing_reward_failing(tmp_path):
    """No copied reward means a crashed harness: the grading must fail, not
    score, so the recompute touches nothing."""
    run_guard(tmp_path, reward_json=None, reward_text=None, eval_output=b"OPENSWE_EXIT_CODE=0\n")
    assert not (tmp_path / "converted-reward.txt").exists()
    assert not (tmp_path / "converted-reward-divergence.txt").exists()


def test_open_swe_recompute_does_not_touch_the_json_path(tmp_path):
    """A verifier that recorded reward.json keeps the plain copy: the JSON
    graders' recompute rules are the recorded follow-up, not a silent rewrite."""
    run_guard(
        tmp_path,
        reward_json=b'{"reward": 1}\n',
        reward_text=b"1\n",
        eval_output=b"OPENSWE_EXIT_CODE=1\n",
    )
    assert (tmp_path / "converted-reward.json").read_bytes() == b'{"reward": 1}\n'
    assert not (tmp_path / "converted-reward-divergence.txt").exists()


def test_open_swe_profile_attaches_the_recompute_and_the_other_families_do_not():
    """The recompute is family-scoped: only a package carrying the profile's
    eval_script gets the marker rule; the fallback families keep the plain copy
    until their verified derivation rules land."""
    openswe_verifier = VERIFIER_SCRIPT
    assert "eval-output.txt" in openswe_verifier  # the recompute reads what the verifier captured
    artifacts, _, guard = _reward_files(False, OPENSWE_RECOMPUTE_SCRIPT)
    assert COPIED_EVAL_OUTPUT in (artifact.target for artifact in artifacts)
    assert OPENSWE_RECOMPUTE_SCRIPT in guard.argv[2]
    plain_artifacts, _, plain_guard = _reward_files(False)
    assert COPIED_EVAL_OUTPUT in (artifact.target for artifact in plain_artifacts)
    assert OPENSWE_RECOMPUTE_SCRIPT not in plain_guard.argv[2]


def test_collect_kills_leftovers_before_the_verifier_and_the_copies():
    """The kill command is the collect sequence's first command for every cohort
    grader: the verifier has not run yet and the artifacts have not been copied.
    The OpenSWE profile's package additionally carries the recompute in its
    guard; the fallback families keep the plain copy."""

    def package(*, eval_script: bytes | None) -> TaskSpec:
        files = [
            inline_resource("tests/test.sh", VERIFIER_SCRIPT.encode()).model_copy(update={"mode": "755"}),
            inline_resource("instruction.md", b"Fix the repository."),
            inline_resource(
                "task.toml",
                b'[environment]\ndocker_image = "registry/control-task-image@sha256:' + b"a" * 64 + b'"\n\n'
                b"[verifier]\ntimeout_sec = 900.0\n",
            ),
            inline_resource("environment/Dockerfile", b"FROM ubuntu@sha256:" + b"b" * 60 + b"\n"),
        ]
        if eval_script is not None:
            files.append(inline_resource("tests/eval_script.sh", eval_script).model_copy(update={"mode": "755"}))
        return TaskSpec(
            id="Marin:cohort-fixture/task-1",
            source=Source(dataset="fixture", revision="a" * 40, row="fixture/task-1", importer_revision="v1"),
            context=ConversationInput(events=(TextMessage(role="user", content="Fix the repository."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.WORKSPACE_STATE,
            answer_format=PlainText(),
            grader=NoGrader(reason="Harbor grades the completed trial", contract={}),
            resources=ResourceGroups(verifier=tuple(files)),
            tags=("harbor",),
        )

    openswe = convert_cohort_task(package(eval_script=b"#!/bin/bash\ntrue\n"), input_task_sha256="0" * 64)
    first = openswe.task.grader.collect[0]
    assert KILL_AGENT_LEFTOVERS_SCRIPT in " ".join(first.argv)
    assert OPENSWE_RECOMPUTE_SCRIPT in " ".join(openswe.task.grader.argv)
    assert COPIED_EVAL_OUTPUT in (artifact.target for artifact in openswe.task.grader.artifacts)

    plain = convert_cohort_task(package(eval_script=None), input_task_sha256="0" * 64)
    assert KILL_AGENT_LEFTOVERS_SCRIPT in " ".join(plain.task.grader.collect[0].argv)
    assert OPENSWE_RECOMPUTE_SCRIPT not in " ".join(plain.task.grader.argv)


@pytest.mark.skipif(sys.platform != "linux", reason="the kill sweep walks /proc; Linux guests only")
def test_kill_sweep_terminates_a_background_writer_and_spares_services(tmp_path):
    """On a Linux guest the sweep kills a planted background process that is not
    in the sweep's ancestry and not a sandbox service, while its own ancestry
    and service-named processes survive."""
    planted = subprocess.Popen(
        ["bash", "-c", "exec -a planted-agent-writer sleep 300"],
        start_new_session=True,
    )
    service = subprocess.Popen(
        ["bash", "-c", "exec -a daytona-toolbox-fake sleep 300"],
        start_new_session=True,
    )
    try:
        sweep = subprocess.run(
            ["bash", "-c", KILL_AGENT_LEFTOVERS_SCRIPT],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert sweep.returncode == 0, sweep.stderr
        assert planted.poll() is not None, "the planted background writer survived the sweep"
        assert service.poll() is None, "the service-named process was killed"
    finally:
        for process in (planted, service):
            process.kill()
            process.wait()
