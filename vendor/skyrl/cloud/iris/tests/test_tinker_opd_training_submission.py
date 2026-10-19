import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

MODULE_DIR = Path(__file__).parents[3] / "skyrl-train" / "ci" / "opd" / "tinker_repro"
sys.path.insert(0, str(MODULE_DIR))
SUBMITTER_PATH = MODULE_DIR / "submit_training_iris.py"
SPEC = importlib.util.spec_from_file_location("tinker_opd_training_submitter", SUBMITTER_PATH)
assert SPEC is not None and SPEC.loader is not None
submitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = submitter
SPEC.loader.exec_module(submitter)


def full_sft_config(acknowledgement: Decimal | None) -> Any:
    return submitter.SubmissionConfig(
        plan=submitter.build_training_plan(
            submitter.Stage.SFT_FULL,
            run_id="repro-20260914",
            output_uri="s3://marin-us-east-02a/experiments/repro-20260914/sft",
        ),
        cost_acknowledgement=acknowledgement,
    )


def test_submit_full_stage_keeps_credentials_out_of_argv_and_disables_retries(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Client:
        def submit(self, **kwargs: Any) -> SimpleNamespace:
            captured["submit"] = kwargs
            return SimpleNamespace(job_id="/ben/tinker-sft-full")

    @contextmanager
    def open_iris_client(*, cluster_name: str, workspace: Path) -> Iterator[Client]:
        yield Client()

    monkeypatch.setattr(submitter, "open_iris_client", open_iris_client)
    credentials = submitter.Credentials(
        tinker_api_key="sentinel-tinker-key",
        wandb_api_key="sentinel-wandb-key",
        hf_token="sentinel-hf-key",
    )
    job_id = submitter.submit(full_sft_config(Decimal("10000")), credentials=credentials)

    assert job_id == "/ben/tinker-sft-full"
    request = captured["submit"]
    assert request["environment"].env_vars == {
        "TINKER_API_KEY": "sentinel-tinker-key",
        "WANDB_API_KEY": "sentinel-wandb-key",
        "HF_TOKEN": "sentinel-hf-key",
    }
    command = request["entrypoint"].command
    assert not any("sentinel" in argument for argument in command)
    # The in-task script re-checks the acknowledgement, so it must be forwarded.
    assert command[-2:] == ["--acknowledge-cost-usd", "10000"]
    assert not request["resources"].to_proto().HasField("device")
    assert request["max_retries_failure"] == 0
    assert request["max_task_failures"] == 0


def test_cli_defaults_to_dry_run_without_reading_credentials(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
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
                "--stage",
                "sft_full",
                "--run-id",
                "repro-20260914",
                "--output-uri",
                "s3://marin-us-east-02a/experiments/repro-20260914/sft",
                "--secrets-env",
                str(secrets_env),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    plan = json.loads(output.split("\nDry run only.", 1)[0])
    assert plan["training"]["stage"] == "sft_full"
    assert plan["required_cost_acknowledgement_usd"] == "10000"
    assert plan["priority"] == "interactive"
    assert plan["secrets_env"] == str(secrets_env)
    assert "TINKER_API_KEY" not in output


def test_cli_submit_loads_export_assignments_before_credential_lookup(monkeypatch, capsys, tmp_path: Path) -> None:
    for key in ("TINKER_API_KEY", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    secrets_env = tmp_path / "submit-secrets.env"
    secrets_env.write_text(
        "export TINKER_API_KEY=sentinel-tinker\n"
        "export WANDB_API_KEY='sentinel-wandb'\n"
        'export HF_TOKEN="sentinel-hf"\n'
    )
    captured: dict[str, Any] = {}

    def submit(config: Any, *, credentials: Any) -> str:
        captured["config"] = config
        captured["credentials"] = credentials
        return "/ben/tinker-sft-plumbing"

    monkeypatch.setattr(submitter, "submit", submit)

    assert (
        submitter.main(
            [
                "--stage",
                "sft_plumbing",
                "--run-id",
                "repro-20260914",
                "--output-uri",
                "s3://marin-us-east-02a/experiments/repro-20260914/sft-plumbing",
                "--secrets-env",
                str(secrets_env),
                "--submit",
            ]
        )
        == 0
    )

    assert captured["config"].secrets_env == str(secrets_env)
    assert captured["credentials"] == submitter.Credentials(
        "sentinel-tinker",
        "sentinel-wandb",
        "sentinel-hf",
    )
    assert "/ben/tinker-sft-plumbing" in capsys.readouterr().out


def test_cli_defaults_secrets_env_to_configured_approved_file(monkeypatch, capsys, tmp_path: Path) -> None:
    secrets_env = tmp_path / "operator-secrets.env"
    secrets_env.touch()
    monkeypatch.setenv("OT_AGENT_SECRETS_ENV", str(secrets_env))
    monkeypatch.setattr(
        submitter,
        "load_secrets_env_into_os_environ",
        lambda path: pytest.fail("dry run read the secrets file"),
    )

    assert (
        submitter.main(
            [
                "--stage",
                "sft_plumbing",
                "--run-id",
                "repro-20260914",
                "--output-uri",
                "s3://marin-us-east-02a/experiments/repro-20260914/sft-default",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    plan = json.loads(output.split("\nDry run only.", 1)[0])
    assert plan["secrets_env"] == str(secrets_env)


def test_default_secrets_env_falls_back_to_documents_file(monkeypatch, tmp_path: Path) -> None:
    secrets_env = tmp_path / "Documents" / "secrets.env"
    secrets_env.parent.mkdir()
    secrets_env.touch()
    monkeypatch.delenv("OT_AGENT_SECRETS_ENV", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert submitter.default_secrets_env() == str(secrets_env)


def test_cli_submit_full_stage_without_acknowledgement_cannot_reach_submission(monkeypatch, capsys) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "unused")
    monkeypatch.setenv("WANDB_API_KEY", "unused")
    monkeypatch.setattr(submitter, "submit", lambda *args, **kwargs: pytest.fail("unauthorized full run submitted"))

    with pytest.raises(SystemExit, match="requires --acknowledge-cost-usd 10000"):
        submitter.main(
            [
                "--stage",
                "sft_full",
                "--run-id",
                "repro-20260914",
                "--output-uri",
                "s3://marin-us-east-02a/experiments/repro-20260914/sft",
                "--submit",
            ]
        )
    capsys.readouterr()
