# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve source judge inputs and separate producer failures from scored responses."""

import base64
import json
import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest
from taskcompendium.models import AnswerType, EnvironmentRequirements, Source, TaskSpec
from taskcompendium.pipeline.models import DatasetRecipe, ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.rewardkit.runtime import run_source
from experiments.post_training.task_curation.pipeline import SourceRuntime, SourceRuntimeConfig
from experiments.post_training.task_curation.sources import rl_data_pipelines
from experiments.post_training.tasktrove.taskbinary import TaskFiles, read_task_binary

SOURCES = {source.source_key: source for source in rl_data_pipelines().values()}


@dataclass(frozen=True)
class SourceFixture:
    original: TaskFiles
    recipe: DatasetRecipe
    row: RawRow


@pytest.fixture
def source_fixture():
    fixture = Path(__file__).parents[2] / "tasktrove/fixtures/nemotron_multichallenge.tar.gz"
    original = read_task_binary(fixture.read_bytes())
    recipe = SOURCES["Task Trove:laion__nemotron-gym-multichallenge-advanced-v4"].recipe(
        SourceRuntimeConfig(
            images={
                "Task Trove:laion__nemotron-gym-multichallenge-advanced-v4": SourceRuntime(
                    backend="iris-gvisor", image="example.org/rewardkit@sha256:" + "a" * 64
                )
            },
            controller_url="http://fixture",
            verifier_secret_env={"TOGETHER_API_KEY": ("env:PRIVATE_TEST_KEY",)},
        )
    )
    source = Source(dataset=recipe.source.dataset, revision=recipe.source.revision, row="0", importer_revision="1")
    row = RawRow(
        "multichallenge-fixture",
        source,
        {
            "instruction": original.text("instruction.md"),
            "verifier_data": json.loads(original.text("tests/verifier_data.json")),
            "files": {path: base64.b64encode(content).decode() for path, content in original.files.items()},
        },
    )
    return SourceFixture(original, recipe, row)


def test_rewardkit_binding_preserves_source_contract_and_private_text_boundary(source_fixture):
    original = source_fixture.original
    result = source_fixture.recipe.policy.normalize(source_fixture.row)
    assert isinstance(result, NormalizedTask)
    task = TaskSpec.model_validate_json(result.task.model_dump_json())
    assert task.answer_type == AnswerType.TEXT
    assert task.environment_requirements == EnvironmentRequirements()
    assert not task.resources.all and not task.resources.worker and not task.resources.oracle
    private = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    for path, content in original.files.items():
        target = path.removeprefix("tests/") if path.startswith("tests/") else "__source/" + path
        assert private[target] == content
    # A visible subdirectory would make RewardKit ignore the original flat judge.
    assert all("/" not in path or path.startswith("__") for path in private)
    assert "PRIVATE_TEST_KEY" not in task.model_dump_json()
    assert json.loads(private["config.json"])["contract"]["aggregation"]["scoring"]["aggregation"] == "all_pass"


@pytest.mark.parametrize(
    "path,original,replacement",
    [
        (
            "tests/judge.toml",
            'files = ["/tests/conversation.txt", "/app/response.txt"]',
            'files = ["/proc/self/environ"]',
        ),
        ("tests/judge.toml", "[judge]", '[judge]\napi_base = "https://unexpected.invalid"'),
        ("tests/judge.toml", 'type = "numeric"', 'files = ["/proc/self/environ"]\ntype = "numeric"'),
        ("tests/test.sh", "set -euo pipefail", 'set -euo pipefail\necho "$TOGETHER_API_KEY"'),
    ],
)
def test_credentialed_source_rejects_endpoint_file_scope_and_executable_changes(
    source_fixture, path, original, replacement
):
    recipe, row = source_fixture.recipe, source_fixture.row
    encoded = row.data["files"]
    content = base64.b64decode(encoded[path]).decode()
    assert original in content
    encoded[path] = base64.b64encode(content.replace(original, replacement).encode()).decode()
    result = recipe.policy.normalize(row)
    assert isinstance(result, ImportRejection)
    assert result.reason in {"unsupported_rewardkit_judge", "unsupported_rewardkit_runtime"}


@pytest.mark.parametrize("value", [0.0, 0.125, 1.0])
def test_source_reward_is_not_thresholded_or_replaced(tmp_path, value):
    (tmp_path / "test.sh").write_text(
        f"printf '%s' '{json.dumps({'reward': value})}' > {shlex.quote(str(tmp_path / 'reward.json'))}\n"
    )
    assert run_source(tmp_path, tmp_path, 5) == {
        "status": "scored",
        "reward": value,
        "detail": {"source": "harbor-rewardkit"},
    }


def test_failed_source_with_stale_positive_reward_is_infrastructure_error_without_secret_logs(tmp_path):
    (tmp_path / "reward.json").write_text('{"reward": 1.0}')
    (tmp_path / "test.sh").write_text("echo 'provider-header-private-value' >&2\nexit 7\n")
    verdict = run_source(tmp_path, tmp_path, 5)
    assert verdict["status"] == "infra_error"
    assert verdict["reward"] == 0
    assert verdict["detail"]["exit_code"] == 7
    assert "provider-header-private-value" not in json.dumps(verdict)


@pytest.mark.parametrize(
    "name,content",
    [
        ("deterministic_gate", b"def broken(private_source_text\n"),
        ("verifier.py", b"def broken(private_source_text\n"),
        ("sitecustomize.py", b"def broken(private_source_text\n"),
        ("verifier_data.json", b'{"private_reference_text":'),
        ("criterion_partition.json", b'{"private_reference_text":'),
        ("judge.toml", b'[judge]\nreference = "private_reference_text'),
    ],
)
def test_malformed_private_suite_fails_before_original_runner_or_provider(tmp_path, name, content):
    marker = tmp_path / "runner-started"
    (tmp_path / "test.sh").write_text(f"touch {shlex.quote(str(marker))}\nexit 7\n")
    (tmp_path / name).write_bytes(content)
    verdict = run_source(tmp_path, tmp_path, 5)
    assert verdict["status"] == "invalid_task"
    assert verdict["detail"]["file"] == name
    assert not marker.exists()
    assert "private_source_text" not in json.dumps(verdict)
    assert "private_reference_text" not in json.dumps(verdict)
    assert (tmp_path / name).read_bytes() == content


def test_preflight_does_not_execute_private_code_or_change_partial_reward(tmp_path):
    (tmp_path / "verifier.py").write_text("raise RuntimeError('Only original runner may execute this')\n")
    (tmp_path / "verifier_data.json").write_text('{"reference": "original"}')
    (tmp_path / "judge.toml").write_text('[judge]\nmode = "individual"\n')
    (tmp_path / "test.sh").write_text(
        f"printf '%s' '{{\"reward\": 0.125}}' > {shlex.quote(str(tmp_path / 'reward.json'))}\n"
    )
    assert run_source(tmp_path, tmp_path, 5)["reward"] == 0.125


@pytest.mark.parametrize("payload", [{"other": 1}, {"reward": 1, "extra": 0}, {"reward": True}, {"reward": 2}])
def test_malformed_source_reward_cannot_pass_a_runtime_control(tmp_path, payload):
    (tmp_path / "test.sh").write_text("exit 0\n")
    (tmp_path / "reward.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        run_source(tmp_path, tmp_path, 5)
