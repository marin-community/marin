# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.runner import HarborRunConfig, HarborTrial, _write_samples, run_harbor
from marin.inference.types import OpenAIEndpoint, RunningModel


def test_materialize_harbor_dataset_downloads_hf_revision_as_local_tasks(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / ".gitattributes").write_text("*.gz filter=lfs")
    (snapshot / "task-one").mkdir()
    calls: list[dict] = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("marin.evaluation.harbor.dataset.snapshot_download", download)

    path = materialize_harbor_dataset(
        "hf://DCAgent2/terminal_bench_2",
        "main",
        tmp_path / "workdir",
        hf_token=None,
    )

    assert path == Path(snapshot)
    assert calls == [
        {
            "repo_id": "DCAgent2/terminal_bench_2",
            "repo_type": "dataset",
            "revision": "main",
            "local_dir": str(tmp_path / "workdir" / "hf_dataset"),
            "cache_dir": str(tmp_path / "workdir" / "hf_cache"),
            "token": False,
        }
    ]
    assert not (snapshot / ".gitattributes").exists()


def test_write_samples_uses_a_path_safe_name_for_hf_dataset(tmp_path):
    trial = HarborTrial(task_id="task-one", reward=0.0, status="completed", trajectory=None, error=None)

    path = _write_samples([trial], "hf://DCAgent2/terminal_bench_2", str(tmp_path))

    assert path == str(tmp_path / "samples_harbor.parquet")
    assert Path(path).exists()


def test_run_harbor_derives_url_and_served_name_from_running_model(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(config_file: Path) -> None:
        captured.update(json.loads(config_file.read_text()))
        trial_dir = Path(captured["jobs_dir"]) / captured["job_name"] / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner._run_driver", run_driver)
    monkeypatch.setattr("marin.evaluation.harbor.runner.materialize_harbor_dataset", lambda *args, **kwargs: None)
    model = RunningModel(
        endpoint=OpenAIEndpoint(
            base_url="https://iris.example/proxy/t/token/serve.model/v1",
            model="qwen3-0.6b",
        )
    )

    result = run_harbor(
        model,
        HarborRunConfig(dataset=f"toy-{tmp_path.name}", version="1.0", agent="terminus-2"),
        str(tmp_path),
        hf_token=None,
    )

    assert captured["model_name"] == "hosted_vllm/qwen3-0.6b"
    assert captured["agent_kwargs"]["api_base"] == model.endpoint.base_url
    assert result.total_trials == 1
    assert result.accuracy == 1.0
