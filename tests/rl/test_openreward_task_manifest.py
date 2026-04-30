# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from marin.rl.integrations.openreward import (
    OpenRewardPromptBlockType,
    build_openreward_task_manifest,
    load_openreward_task_manifest,
    prepare_openreward_task_manifest,
    resolve_task_indices,
    save_openreward_task_manifest,
)


@dataclass(frozen=True)
class FakeTask:
    environment_name: str
    task_spec: dict


@dataclass(frozen=True)
class FakeTool:
    name: str
    description: str
    input_schema: dict | None


@dataclass(frozen=True)
class FakeTextBlock:
    text: str
    detail: dict | None = None
    type: str = "text"


@dataclass(frozen=True)
class FakeImageBlock:
    data: str
    mimeType: str
    detail: dict | None = None
    type: str = "image"


class FakeSession:
    def __init__(self, prompt_blocks, tools):
        self._prompt_blocks = prompt_blocks
        self._tools = tools

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def get_prompt(self):
        return self._prompt_blocks

    def list_tools(self, provider_format=None):
        assert provider_format is None
        return self._tools


class FakeEnvironment:
    name = "math-agent"
    deployment_name = "marin/openreward-math-agent"

    def __init__(self):
        self.tasks = [
            FakeTask(environment_name="math-agent-v1", task_spec={"problem_id": "train-0"}),
            FakeTask(environment_name="math-agent-v1", task_spec={"problem_id": "train-1"}),
            FakeTask(environment_name="math-agent-v1", task_spec={"problem_id": "train-2"}),
        ]
        self.prompts = {
            0: [FakeTextBlock(text="Solve 1+1."), FakeImageBlock(data="abc", mimeType="image/png")],
            1: [FakeTextBlock(text="Solve 2+2.", detail={"difficulty": "easy"})],
            2: [FakeTextBlock(text="Solve 3+3.")],
        }
        self.tools = {
            0: [FakeTool(name="submit_answer", description="Submit an answer", input_schema={"type": "object"})],
            1: [FakeTool(name="request_hint", description="Ask for a hint", input_schema=None)],
            2: [FakeTool(name="submit_answer", description="Submit an answer", input_schema={"type": "object"})],
        }
        self.session_task_specs = []

    def num_tasks(self, split: str) -> int:
        assert split == "train"
        return len(self.tasks)

    def get_task(self, split: str, index: int) -> FakeTask:
        assert split == "train"
        return self.tasks[index]

    def session(self, task=None, secrets=None, *, split=None, index=None):
        assert split is None
        assert index is None
        assert task is not None
        assert secrets is None or secrets == {"OPENAI_API_KEY": "secret"}
        task_index = self.tasks.index(task)
        self.session_task_specs.append(task.task_spec["problem_id"])
        return FakeSession(self.prompts[task_index], self.tools[task_index])


def test_resolve_task_indices_uses_slice_semantics():
    assert resolve_task_indices(5, start=1, stop=4) == [1, 2, 3]
    assert resolve_task_indices(5, start=-2) == [3, 4]


def test_resolve_task_indices_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate task index"):
        resolve_task_indices(5, indices=[1, -4])


def test_build_openreward_task_manifest_snapshots_prompt_blocks_and_tools():
    environment = FakeEnvironment()

    manifest = build_openreward_task_manifest(
        environment,
        "train",
        indices=[0, 1],
        secrets={"OPENAI_API_KEY": "secret"},
    )

    assert manifest.deployment_name == "marin/openreward-math-agent"
    assert manifest.environment_name == "math-agent-v1"
    assert manifest.split == "train"
    assert manifest.task_count == 2
    assert environment.session_task_specs == ["train-0", "train-1"]

    first_task = manifest.tasks[0]
    assert first_task.task_index == 0
    assert first_task.task_spec == {"problem_id": "train-0"}
    assert first_task.prompt_blocks[0].type == OpenRewardPromptBlockType.TEXT
    assert first_task.prompt_blocks[0].text == "Solve 1+1."
    assert first_task.prompt_blocks[1].type == OpenRewardPromptBlockType.IMAGE
    assert first_task.prompt_blocks[1].mime_type == "image/png"
    assert first_task.tools[0].name == "submit_answer"
    assert first_task.tools[0].input_schema == {"type": "object"}

    second_task = manifest.tasks[1]
    assert second_task.task_index == 1
    assert second_task.prompt_blocks[0].detail == {"difficulty": "easy"}
    assert second_task.tools[0].name == "request_hint"


def test_save_and_load_openreward_task_manifest_round_trip(tmp_path):
    manifest = build_openreward_task_manifest(FakeEnvironment(), "train", start=0, stop=2)
    output_path = tmp_path / "openreward-manifest.json"

    save_openreward_task_manifest(manifest, str(output_path))
    loaded_manifest = load_openreward_task_manifest(str(output_path))

    assert loaded_manifest == manifest


def test_prepare_openreward_task_manifest_uses_client_factory(monkeypatch):
    environment = FakeEnvironment()
    calls = {}

    class FakeEnvironmentsAPI:
        def get(self, name: str, variant: str | None = None, base_url: str | None = None):
            del base_url
            calls["environment_name"] = name
            calls["variant"] = variant
            return environment

    class FakeOpenReward:
        def __init__(self, api_key: str | None = None, base_url: str | None = None):
            calls["api_key"] = api_key
            calls["base_url"] = base_url
            self.environments = FakeEnvironmentsAPI()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

    monkeypatch.setattr(
        "marin.rl.integrations.openreward.manifest.load_openreward_client",
        lambda: FakeOpenReward,
    )

    manifest = prepare_openreward_task_manifest(
        "marin/openreward-math-agent",
        "train",
        api_key="api-key",
        base_url="https://openreward.example",
        variant="math",
        start=1,
        stop=3,
    )

    assert calls["api_key"] == "api-key"
    assert calls["base_url"] == "https://openreward.example"
    assert calls["environment_name"] == "marin/openreward-math-agent"
    assert calls["variant"] == "math"
    assert [task.task_index for task in manifest.tasks] == [1, 2]
