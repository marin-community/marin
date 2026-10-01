# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym imports preserve source semantics across exact and script routes."""

import hashlib
import json
import subprocess
import sys
import tarfile
from dataclasses import replace
from io import BytesIO

import pytest
import reasoning_gym

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.convert import read_archive
from taskcompendium.importers.tasktrove.reasoning_gym import IMPORTER_REVISION, import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import ConversationTrace, TextMessage, VerifierKind
from taskcompendium.submission import GradingAttempt, PlainText, chat_request
from taskcompendium.verifier_registry import grade_answer, resolve_verifier
from taskcompendium.verifiers.script import materialize_private_resources

from .harbor_replay import run_replay_trial

TASKTROVE_SOURCE = "laion__nemotron-gym-reasoning-gym-v2"
TASKTROVE_PATH = "reasoning-gym-synthetic.tar.gz"
RELEASE_URI = "synthetic-tasktrove-clean"
RELEASE_REVISION = "synthetic-revision"
RUNTIME_IMAGE = f"synthetic/verifier@sha256:{'a' * 64}"
PREAMBLE = (
    "You are solving a procedurally-generated reasoning task from Reasoning Gym. Read the problem below and write "
    "your final answer to `/app/answer.txt`. The verifier will try the upstream Reasoning Gym scorer first, then fall "
    "back to normalized exact-match.\n\n---\n\n"
)


def _archive_bytes(dataset: str = "course_schedule", entry: dict | None = None) -> tuple[bytes, dict]:
    if entry is None:
        entry = reasoning_gym.create_dataset(dataset, seed=231, size=1)[0]
    entry["metadata"]["private_marker"] = "synthetic-hidden-marker"
    files = {
        "task.toml": (
            (
                f'[metadata]\nfamily = "other"\nconverter = "nemotron_reasoning"\nmode = "reasoning-gym"\n'
                f'tasktrove_source = "{TASKTROVE_SOURCE}"\ntasktrove_path = "{TASKTROVE_PATH}"\n'
                f'tags = ["reasoning", "reasoning-gym", "{dataset.replace("_", "-")}", "nemotron"]\n'
            ).encode()
        ),
        "instruction.md": (PREAMBLE + entry["question"]).encode(),
        "tests/verifier.toml": f'mode = "reasoning-gym"\ndataset = "{dataset}"\n'.encode(),
        "tests/entry.json": json.dumps(entry, separators=(",", ":")).encode(),
    }
    payload = BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, contents in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, BytesIO(contents))
    return payload.getvalue(), entry


def _entry(dataset: str, answer: str, **metadata) -> dict:
    return {
        "question": "Synthetic problem instructions.",
        "answer": answer,
        "metadata": {"source_dataset": dataset, **metadata},
    }


def _imported(dataset: str = "course_schedule", entry: dict | None = None):
    data, entry = _archive_bytes(dataset, entry)
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    return import_task(archive, runtime_image=RUNTIME_IMAGE, timeout_seconds=30.0), entry


def _run_script(specification, candidate: str, tmp_path, *, isolated_python: bool = False):
    configuration = resolve_verifier(specification.verifier)
    tests = tmp_path / "tests"
    materialize_private_resources(configuration.resources, tests)
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "submission.json").write_text(
        json.dumps({"protocol_version": 1, "answer_type": "text", "convention_id": "plain", "answer": candidate})
    )
    flags = ["-I", "-S"] if isolated_python else []
    subprocess.run(
        [sys.executable, *flags, str(tests / configuration.entrypoint), str(tests), str(verifier)], check=True
    )
    return json.loads((verifier / "result.json").read_text())


def test_import_preserves_problem_tags_and_archive_digest():
    data, entry = _archive_bytes()
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    specification = import_task(archive, runtime_image=RUNTIME_IMAGE, timeout_seconds=30.0)

    assert specification.context.events[0].content == entry["question"].strip()
    assert specification.source.importer_revision == IMPORTER_REVISION
    assert specification.tags == ("reasoning", "reasoning-gym", "course-schedule", "nemotron")
    assert archive.archive_sha256 == hashlib.sha256(data).hexdigest()
    assert entry["metadata"]["private_marker"] not in specification.context.model_dump_json()
    assert specification.environment_requirements.capabilities == ()


def test_script_lowering_keeps_generated_entry_out_of_model_request(tmp_path):
    specification, entry = _imported()
    convention = PlainText(id="plain")
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    marker = entry["metadata"]["private_marker"]
    assert marker not in json.dumps(chat_request(specification, convention))
    assert marker not in (task / "instruction.md").read_text()
    assert marker in (task / "private_resources/entry.json").read_text()


def test_archive_with_different_problem_and_entry_is_rejected():
    data, _ = _archive_bytes()
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    changed = replace(archive, files={**archive.files, "instruction.md": (PREAMBLE + "Different problem.").encode()})
    with pytest.raises(ValueError, match="instruction differs from its generated entry"):
        import_task(changed, runtime_image=RUNTIME_IMAGE, timeout_seconds=30.0)


@pytest.mark.parametrize(
    "dataset,expected,candidate,reward",
    [
        ("ab", "A", "A", 1.0),
        ("ab", "A", "a", 0.0),
        ("self_reference", "2", "2", 1.0),
        ("self_reference", "2", "2 extra", 0.0),
        ("path_star", "1 2 3", "1\n2\t3", 1.0),
        ("path_star", "1 2 3", "1 3 2", 0.0),
    ],
)
async def test_exact_routes_match_source_without_script_settings(dataset, expected, candidate, reward):
    data, entry = _archive_bytes(dataset, _entry(dataset, expected))
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    specification = import_task(archive)
    assert specification.verifier.kind is VerifierKind.EXACT_ANSWER
    attempt = GradingAttempt(
        ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=candidate))),
        object(),
    )
    result = await grade_answer(specification, PlainText(id="plain"), attempt)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)
    assert reasoning_gym.get_score_answer_fn(dataset)(candidate.strip(), entry) == reward


async def test_exact_route_replays_through_harbor(tmp_path):
    data, _ = _archive_bytes("path_star", _entry("path_star", "1 2 3"))
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    specification = import_task(archive)
    task = lower_to_harbor(specification, PlainText(id="plain"), HarborEnvironmentConfig(), tmp_path / "task")
    trial = await run_replay_trial(task, {"role": "assistant", "content": "1\n2\t3"}, tmp_path / "trials", "correct")
    outcome = json.loads((tmp_path / "trials/correct/verifier/taskcompendium-result.json").read_text())
    assert trial.exception_info is None, trial.exception_info
    assert outcome == {"status": "graded", "reward": 1.0, "error": None}


@pytest.mark.parametrize(
    "dataset,expected,candidate,reward",
    [
        ("course_schedule", "True", "True", 1.0),
        ("course_schedule", "True", "answer True", 4 / 11),
        ("course_schedule", "True", "False", 0.0),
        ("letter_jumble", "alpha beta", "ALPHA BETA", 1.0),
        ("letter_jumble", "alpha beta", "alpha wrong", 0.5),
        ("letter_jumble", "alpha beta", "alpha beta extra", 1.0),
        ("family_relationships", "sister", "SISTER", 1.0),
        ("family_relationships", "sister", "\u017fi\u017fter", 0.0),
        ("decimal_chain_sum", "1.0000000000000000001", "1.0000000000000000002", 0.0),
    ],
)
def test_private_script_preserves_source_partial_credit_and_normalization(
    tmp_path, dataset, expected, candidate, reward
):
    specification, entry = _imported(dataset, _entry(dataset, expected))
    result = _run_script(specification, candidate, tmp_path)
    assert result == {"status": "scored", "reward": reward}
    assert reasoning_gym.get_score_answer_fn(dataset)(candidate.strip(), entry) == reward


def test_private_script_failure_is_unscored(tmp_path):
    specification, _ = _imported("word_sorting", _entry("word_sorting", "alpha, beta"))
    result = _run_script(specification, "alpha, beta", tmp_path)
    assert result["status"] == "infra_error"
    assert "reward" not in result


@pytest.mark.parametrize("valid,expected", [(True, 1.0), (False, 0.01)])
def test_generated_graph_coloring_with_null_gold_preserves_script_reward(tmp_path, valid, expected):
    entry = reasoning_gym.create_dataset(
        "graph_color", seed=231, size=1, min_num_vertices=4, max_num_vertices=4, num_colors=4, edge_probability=0.8
    )[0]
    assert entry["answer"] is None
    coloring = (
        entry["metadata"]["possible_answer"] if valid else dict.fromkeys(entry["metadata"]["puzzle"]["vertices"], 1)
    )
    candidate = json.dumps(coloring)
    specification, entry = _imported("graph_color", entry)
    assert _run_script(specification, candidate, tmp_path) == {"status": "scored", "reward": expected}
    assert reasoning_gym.get_score_answer_fn("graph_color")(candidate, entry) == expected


@pytest.mark.parametrize("dataset", ["ab", "self_reference", "path_star"])
def test_exact_routes_reject_null_gold(dataset):
    entry = _entry(dataset, "42")
    entry["answer"] = None
    data, _ = _archive_bytes(dataset, entry)
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    with pytest.raises(ValueError, match="requires a nonempty string answer"):
        import_task(archive)


@pytest.mark.parametrize("answer_field", [{}, {"answer": [1, 2]}])
def test_null_gold_support_preserves_malformed_archive_errors(answer_field):
    entry = _entry("graph_color", "42")
    del entry["answer"]
    entry.update(answer_field)
    data, _ = _archive_bytes("graph_color", entry)
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    with pytest.raises(ValueError, match="string or null answer field"):
        import_task(archive, runtime_image=RUNTIME_IMAGE, timeout_seconds=30.0)


def test_private_script_missing_runtime_dependency_is_unscored(tmp_path):
    specification, _ = _imported()
    result = _run_script(specification, "True", tmp_path, isolated_python=True)
    assert result["status"] == "infra_error"
    assert "reward" not in result


@pytest.mark.parametrize("dataset", ["arc_agi", "rearc", "composite", "not-a-reviewed-dataset"])
def test_unsupported_evaluators_never_fall_back_to_exact(dataset):
    data, _ = _archive_bytes(dataset, _entry(dataset, "42"))
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    with pytest.raises(ValueError, match=r"(Unsupported|Unaudited) Reasoning Gym evaluator"):
        import_task(archive, runtime_image=RUNTIME_IMAGE, timeout_seconds=30.0)


def test_partial_credit_evaluator_requires_explicit_script_runtime():
    data, _ = _archive_bytes("letter_jumble", _entry("letter_jumble", "alpha beta"))
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    with pytest.raises(ValueError, match="requires an explicit runtime_image and timeout_seconds"):
        import_task(archive)
