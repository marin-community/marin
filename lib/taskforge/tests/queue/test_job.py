# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from taskforge.llm.client import Pool
from taskforge.loop.events import Terminal
from taskforge.loop.program import LEDGER_DIR
from taskforge.queue.config import LaptopGlm, ParallelKeyFile, load_run_config
from taskforge.queue.job import IRIS_JOB_ENV, glm_endpoint, host_secrets, restore, run_root
from taskforge.queue.run import FailedItems, item_terminal
from taskforge.sandbox.factories import MachineHost
from taskforge.triage.verdict import TriageDecision

EXAMPLE = Path(__file__).parents[2] / "docs" / "policy.example.json"


@pytest.fixture
def iris_config():
    return load_run_config(EXAMPLE)


def test_an_iris_run_takes_its_secrets_out_of_every_child_environment(monkeypatch, iris_config):
    monkeypatch.setenv("GLM_API_TOKEN", "glm-secret")
    monkeypatch.setenv("PARALLEL_KEY", "parallel-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    job_env = {"GLM_API_TOKEN": "glm-secret", "PARALLEL_KEY": "parallel-secret", "HF_TOKEN": "hf", "KEEP": "1"}
    monkeypatch.setenv(IRIS_JOB_ENV, json.dumps(job_env))

    secrets = host_secrets(iris_config)

    assert (secrets.glm_token, secrets.parallel_key) == ("glm-secret", "parallel-secret")
    assert not {"GLM_API_TOKEN", "PARALLEL_KEY", "HF_TOKEN"} & set(os.environ)
    assert json.loads(os.environ[IRIS_JOB_ENV]) == {"KEEP": "1"}
    assert "secret" not in repr(secrets)


def test_a_relay_token_variable_that_is_unset_fails_before_anything_runs(monkeypatch, iris_config):
    monkeypatch.delenv("GLM_API_TOKEN", raising=False)
    monkeypatch.setenv("PARALLEL_KEY", "parallel-secret")

    with pytest.raises(ValueError, match="GLM_API_TOKEN"):
        host_secrets(iris_config)


def test_a_laptop_run_reads_its_secrets_from_files_and_leaves_the_environment_alone(monkeypatch, tmp_path, iris_config):
    (tmp_path / "glm.txt").write_text("# interactive pool\nGLM_API_TOKEN=glm-file-token\n")
    (tmp_path / "parallel").write_text("PARALLEL_KEY=parallel-file-key\n")
    monkeypatch.setenv("HF_TOKEN", "hf")
    config = replace(
        iris_config,
        host=MachineHost.LAPTOP,
        image_cache=tmp_path / "images",
        root=tmp_path / "run",
        glm=LaptopGlm("http://127.0.0.1:18000/v1/", tmp_path / "glm.txt", Pool.HIGH),
        web=ParallelKeyFile(tmp_path / "parallel"),
    )

    secrets = host_secrets(config)
    endpoint = glm_endpoint(config, secrets.glm_token)

    assert (secrets.glm_token, secrets.parallel_key) == ("glm-file-token", "parallel-file-key")
    assert os.environ["HF_TOKEN"] == "hf"
    assert (endpoint.base_url, endpoint.pool) == ("http://127.0.0.1:18000/v1", Pool.HIGH)
    assert run_root(config) == tmp_path / "run"


def test_an_iris_run_root_lives_in_the_attempt_output_dir(monkeypatch, tmp_path, iris_config):
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path / "out"))

    assert run_root(iris_config) == tmp_path / "out" / "taskforge_run"


async def test_a_restored_archive_resumes_and_skips_its_terminal_items(tmp_path, queue_run, fakes):
    first = queue_run(rubric=fakes.rubric(TriageDecision.REJECT))
    await first({"a": "a"}, fakes.policy(), width=2)
    archive = first.root

    restored = tmp_path / "attempt-1" / "taskforge_run"
    restore(str(archive), restored)

    assert item_terminal(restored / LEDGER_DIR, "a--0") is Terminal.REJECTED
    again = replace(first, root=restored)
    summary = await again({"a": "a"}, fakes.policy(), width=2, failed=FailedItems.SKIP)
    assert summary.items == {"a--0": Terminal.REJECTED}
    assert first.rubric.assessed == ["a/0"]
    assert first.source.calls == ["a"]


def test_restore_refuses_a_run_root_that_already_has_files(tmp_path):
    (tmp_path / "archive" / LEDGER_DIR).mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    (root / "summary.json").write_text("{}")

    with pytest.raises(ValueError, match="empty run root"):
        restore(str(tmp_path / "archive"), root)


def test_restore_refuses_an_archive_without_a_ledger(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "policy.json").write_text("{}")

    with pytest.raises(ValueError, match="no ledger/"):
        restore(str(tmp_path / "archive"), tmp_path / "root")
