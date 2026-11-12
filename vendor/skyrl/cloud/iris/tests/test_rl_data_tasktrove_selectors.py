from pathlib import Path
from types import SimpleNamespace
import json

from omegaconf import OmegaConf

from cloud.iris import hf_datasets, rl_data
from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner
from marinskyrl.resource_locator import HFDatasetSelector, is_hugging_face_repo_id, parse_hf_dataset_selector


def test_dataset_selector_is_distinct_from_plain_repo_id():
    value = "fixture-org/fixture-trove@rev-1::a/nested"

    assert not is_hugging_face_repo_id(value)
    assert parse_hf_dataset_selector(value) == HFDatasetSelector(
        repo_id="fixture-org/fixture-trove",
        revision="rev-1",
        subdir="a/nested",
    )


def test_cache_names_do_not_confuse_repo_suffixes_with_subdirectories():
    repo_suffix = HFDatasetSelector("fixture-org/a__b", revision="commit").cache_name()
    subdirectory = HFDatasetSelector("fixture-org/a", revision="commit", subdir="b").cache_name()

    assert repo_suffix != subdirectory


def test_download_selects_only_the_requested_subdirectory(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    selected = snapshot / "a"
    selected.mkdir(parents=True)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(hf_datasets, "snapshot_download", download)
    monkeypatch.delenv("HF_CACHE_DIR", raising=False)

    assert hf_datasets.download_hf_dataset("fixture-org/fixture-trove@rev-1::a") == str(selected)
    assert calls == [
        {
            "repo_id": "fixture-org/fixture-trove",
            "cache_dir": str(Path.home() / ".cache/huggingface/hub"),
            "revision": "rev-1",
            "repo_type": "dataset",
            "allow_patterns": ["a/**"],
        }
    ]


def test_task_selectors_use_distinct_revision_pinned_caches(monkeypatch, tmp_path):
    commits = {"a": "commit-a", "b": "commit-b"}
    commands = []

    def resolve(value):
        selector = parse_hf_dataset_selector(value)
        assert selector is not None and selector.subdir is not None
        return HFDatasetSelector(selector.repo_id, commits[selector.subdir], selector.subdir)

    def run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("--output_dir") + 1])
        (output / "task-1").mkdir(parents=True)
        (output / "task-1" / "instruction.md").write_text("task")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(rl_data, "resolve_hf_dataset_selector", resolve)
    monkeypatch.setattr(rl_data.subprocess, "run", run)
    monkeypatch.setattr(rl_data, "_fix_task_permissions", lambda *_args, **_kwargs: None)

    resolved = rl_data.resolve_rl_train_data_with_sources(
        ["fixture-org/fixture-trove::a", "fixture-org/fixture-trove::b"],
        scratch_dir=str(tmp_path),
        verbose=False,
    )

    assert resolved.sources == (
        "fixture-org/fixture-trove@commit-a::a",
        "fixture-org/fixture-trove@commit-b::b",
    )
    assert resolved.paths[0] != resolved.paths[1]
    assert all((Path(path) / "task-1" / "instruction.md").read_text() == "task" for path in resolved.paths)
    assert [command[command.index("--parquet") + 1] for command in commands] == list(resolved.sources)


def test_revision_resolution_uses_the_requested_revision(monkeypatch):
    calls = []

    class FakeApi:
        def dataset_info(self, repo_id, revision):
            calls.append((repo_id, revision))
            return SimpleNamespace(sha="immutable-sha")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    selector = hf_datasets.resolve_hf_dataset_selector("fixture-org/fixture-trove@rev-1::a")

    assert selector.canonical() == "fixture-org/fixture-trove@immutable-sha::a"
    assert calls == [("fixture-org/fixture-trove", "rev-1")]


def test_runner_resolves_and_records_training_and_validation_selectors(monkeypatch, tmp_path):
    resolved = {
        "fixture-org/fixture-trove::train": rl_data.ResolvedRLData(
            paths=("/tasks/train",),
            sources=("fixture-org/fixture-trove@immutable-sha::train",),
        ),
        "fixture-org/fixture-trove::validation": rl_data.ResolvedRLData(
            paths=("/tasks/validation",),
            sources=("fixture-org/fixture-trove@immutable-sha::validation",),
        ),
    }

    def resolve(values, *, kind):
        assert kind == "parquet"
        return resolved[values[0]]

    monkeypatch.setattr("cloud.iris.training_driver.resolve_rl_train_data_with_sources", resolve)
    launch_config = OmegaConf.create(
        {
            "run": {"mode": "train"},
            "inputs": {"data_kind": "parquet"},
            "skyrl": {
                "data": {"train_data": [], "val_data": [], "terminal_bench_data": []},
            },
        }
    )
    runner = LocalRLRunner(
        LocalRLConfig(
            job_name="job",
            model_path="org/model",
            train_data=["fixture-org/fixture-trove::train"],
            val_data=["fixture-org/fixture-trove::validation"],
            experiments_dir=str(tmp_path / "experiments"),
            resolved_config_uri=(tmp_path / "resolved.json").as_uri(),
            launch_config=launch_config,
        )
    )
    monkeypatch.setattr(runner, "_setup_environment", lambda: None)
    monkeypatch.setattr(runner, "_run_skyrl", lambda _config: 0)

    assert runner.run() == 0

    recorded = json.loads((tmp_path / "resolved.json").read_text())
    assert recorded["train_data_sources"] == ["fixture-org/fixture-trove@immutable-sha::train"]
    assert recorded["val_data_sources"] == ["fixture-org/fixture-trove@immutable-sha::validation"]
    assert recorded["config"]["skyrl"]["data"]["train_data"] == ["/tasks/train"]
    assert recorded["config"]["skyrl"]["data"]["val_data"] == ["/tasks/validation"]
