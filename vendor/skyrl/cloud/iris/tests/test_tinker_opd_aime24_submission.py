import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SUBMITTER_PATH = Path(__file__).parents[3] / "skyrl-train" / "ci" / "opd" / "tinker_repro" / "submit_iris.py"
sys.path.insert(0, str(SUBMITTER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("tinker_opd_aime24_submitter", SUBMITTER_PATH)
assert SPEC is not None and SPEC.loader is not None
submitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = submitter
SPEC.loader.exec_module(submitter)


def test_submit_sends_secret_only_in_environment_and_requests_cpu_worker(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Client:
        def submit(self, **kwargs: Any) -> SimpleNamespace:
            captured["submit"] = kwargs
            return SimpleNamespace(job_id="/ben/tinker-opd-aime24")

    @contextmanager
    def open_iris_client(*, cluster_name: str, workspace: Path) -> Iterator[Client]:
        yield Client()

    monkeypatch.setattr(submitter, "open_iris_client", open_iris_client)
    api_key = "sentinel-tinker-api-key"
    job_id = submitter.submit(
        submitter.SubmissionConfig(
            checkpoint="tinker://released/sampler_weights/final",
            save_dir="s3://evaluation/results",
            max_examples=1,
        ),
        tinker_api_key=api_key,
    )

    request = captured["submit"]
    assert job_id == "/ben/tinker-opd-aime24"
    assert request["environment"].env_vars == {"TINKER_API_KEY": api_key}
    assert api_key not in request["entrypoint"].command
    assert not request["resources"].to_proto().HasField("device")
    assert request["max_retries_failure"] == 0
    assert request["max_task_failures"] == 0


def test_cli_defaults_to_non_secret_dry_run(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setattr(submitter, "submit", lambda *args, **kwargs: pytest.fail("dry run submitted a job"))
    monkeypatch.setattr(
        submitter,
        "load_secrets_env_into_os_environ",
        lambda path: pytest.fail("dry run read the secrets file"),
    )
    secrets_env = tmp_path / "explicit-secrets.env"

    assert (
        submitter.main(
            [
                "--checkpoint",
                "tinker://released/sampler_weights/final",
                "--save-dir",
                "s3://evaluation/unique-run",
                "--max-examples",
                "1",
                "--secrets-env",
                str(secrets_env),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    plan = json.loads(output.split("\nDry run only.", 1)[0])
    assert plan["cluster"] == "cw-rno2a"
    assert plan["checkpoint"] == "tinker://released/sampler_weights/final"
    assert plan["save_dir"] == "s3://evaluation/unique-run"
    assert plan["max_examples"] == 1
    assert plan["secrets_env"] == str(secrets_env)
    assert "TINKER_API_KEY" not in output


def test_cli_submit_loads_export_assignment_before_key_lookup(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    secrets_env = tmp_path / "submit-secrets.env"
    secrets_env.write_text("export TINKER_API_KEY='sentinel-tinker'\n")
    captured: dict[str, Any] = {}

    def submit(config: Any, *, tinker_api_key: str) -> str:
        captured["config"] = config
        captured["tinker_api_key"] = tinker_api_key
        return "/ben/tinker-opd-aime24"

    monkeypatch.setattr(submitter, "submit", submit)

    assert (
        submitter.main(
            [
                "--checkpoint",
                "tinker://released/sampler_weights/final",
                "--save-dir",
                "s3://evaluation/unique-submit",
                "--secrets-env",
                str(secrets_env),
                "--submit",
            ]
        )
        == 0
    )

    assert captured == {
        "config": submitter.SubmissionConfig(
            checkpoint="tinker://released/sampler_weights/final",
            save_dir="s3://evaluation/unique-submit",
            max_examples=None,
            secrets_env=str(secrets_env),
        ),
        "tinker_api_key": "sentinel-tinker",
    }
    assert "/ben/tinker-opd-aime24" in capsys.readouterr().out


def test_cli_submission_requires_key(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="TINKER_API_KEY must be set"):
        submitter.main(
            [
                "--checkpoint",
                "tinker://released/sampler_weights/final",
                "--save-dir",
                "s3://evaluation/unique-run",
                "--secrets-env",
                str(tmp_path / "missing.env"),
                "--submit",
            ]
        )
    capsys.readouterr()
