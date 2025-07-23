import os

import pytest
import yaml

from marin.agents.dataset_agent import DatasetAgent, DatasetAgentStep
from marin.agents.hparam_agent import HparamAgentStep, HyperparameterAgent
from marin.resources import CpuOnlyConfig


@pytest.fixture
def mock_dataset():
    # Simulate a minimal dataset with a 'text' field
    class MockDS:
        features = {"text": str}

        def select(self, idxs):
            return [{"text": f"sample {i}"} for i in idxs]

        def __contains__(self, key):
            return key == "validation"

        def __len__(self):
            return 5

    return MockDS()


def test_dataset_agent_config(monkeypatch, mock_dataset):
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: mock_dataset)
    agent = DatasetAgent(model="gpt-4o", provider="openai", mode="auto")
    monkeypatch.setattr(
        agent,
        "prompt",
        lambda *a, **kw: '{"valid": true, "config": "data:\\n  train_paths: [mock-ds]\\n  tokenizer: specified_tokenizer_here"}',
    )
    result = agent.validate("mock-ds", default_tokenizer="gpt2")
    assert "config_snippet" in result
    assert "tokenizer: gpt2" in result["config_snippet"]
    assert "validation_paths" in result["config_snippet"]
    assert "rationale" in result
    assert "agent_steps" in result


def test_dataset_agent_recipe(monkeypatch, mock_dataset, tmp_path):
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: mock_dataset)
    agent = DatasetAgent(model="gpt-4o", provider="openai", mode="auto")
    monkeypatch.setattr(
        agent,
        "prompt",
        lambda *a, **kw: '{"valid": true, "config": "data:\\n  train_paths: [mock-ds]\\n  tokenizer: default_tokenizer"}',
    )
    os.makedirs(tmp_path / "recipes", exist_ok=True)
    os.chdir(tmp_path)
    result = agent.validate("mock-ds", recipe_mode=True, default_tokenizer="gpt2")
    assert "recipe" in result
    recipe_path = tmp_path / "recipes" / "dataset_add_mock-ds.yaml"
    assert recipe_path.exists() or os.path.exists(recipe_path)
    loaded = yaml.safe_load(result["recipe"])
    assert loaded["dataset_id"] == "mock-ds"


def test_hparam_agent_suggest(monkeypatch):
    agent = HyperparameterAgent(model="gpt-4o", provider="openai", mode="auto")
    monkeypatch.setattr(
        agent, "prompt", lambda *a, **kw: '{"suggestions": ["train_batch_size: 8\\nlearning_rate: 0.001"]}'
    )
    default_hparams = {"resources": str(CpuOnlyConfig(num_cpus=1)), "train_batch_size": 4, "learning_rate": 0.001}
    dataset_metadata = {"num_examples": 1000, "source": "mock-ds"}
    result = agent.suggest(default_hparams, dataset_metadata)
    assert "marin_configs" in result
    assert "SimpleTrainConfig" in result["marin_configs"][0]
    assert "subtasks" in result
    assert "hardware_info" in result


def test_hparam_agent_executable_subtasks(monkeypatch):
    agent = HyperparameterAgent(model="gpt-4o", provider="openai", mode="auto")
    monkeypatch.setattr(
        agent, "prompt", lambda *a, **kw: '{"suggestions": ["train_batch_size: 8\\nlearning_rate: 0.001"]}'
    )
    default_hparams = {"resources": str(CpuOnlyConfig(num_cpus=1)), "train_batch_size": 4, "learning_rate": 0.001}
    dataset_metadata = {"num_examples": 1000, "source": "mock-ds"}
    result = agent.suggest(default_hparams, dataset_metadata, decompose_executable=True)
    assert "executable_subtasks" in result
    for subtask in result["executable_subtasks"]:
        subtask()  # Should print something, but not raise


def test_dataset_agent_step(monkeypatch, mock_dataset):
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: mock_dataset)
    step = DatasetAgentStep(
        name="test_step",
        dataset_id_or_path="mock-ds",
        agent_kwargs={"model": "gpt-4o", "provider": "openai", "mode": "auto"},
    )
    monkeypatch.setattr(
        step.agent,
        "prompt",
        lambda *a, **kw: '{"valid": true, "config": "data:\\n  train_paths: [mock-ds]\\n  tokenizer: gpt2"}',
    )
    result = step.run()
    assert "config_snippet" in result


def test_hparam_agent_step(monkeypatch):
    step = HparamAgentStep(
        name="test_hparam_step",
        current_config={"resources": str(CpuOnlyConfig(num_cpus=1)), "train_batch_size": 4, "learning_rate": 0.001},
        dataset_metadata={"num_examples": 1000, "source": "mock-ds"},
        agent_kwargs={"model": "gpt-4o", "provider": "openai", "mode": "auto"},
    )
    monkeypatch.setattr(
        step.agent, "prompt", lambda *a, **kw: '{"suggestions": ["train_batch_size: 8\\nlearning_rate: 0.001"]}'
    )
    result = step.run()
    assert "marin_configs" in result
