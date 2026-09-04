# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from click.testing import CliRunner
from marina.cli import cli


def write_job(apps_dir: Path, app: str, name: str, code: str, *, secrets: tuple[str, ...] = ()) -> None:
    root = apps_dir / app
    root.mkdir(parents=True)
    secret_line = f"secrets = {list(secrets)!r}\n" if secrets else ""
    (root / "app.toml").write_text(
        f"""title = "{app}"
description = "test app"
[[jobs]]
name = "{name}"
runner = "hourly"
schedule = "0 * * * *"
command = {[sys.executable, "-c", code]!r}
timeout = 10
cpu = 1
memory_gib = 1
{secret_line}"""
    )


def test_run_attempts_later_jobs_and_filters_runner_secrets(tmp_path: Path, database_url: str) -> None:
    apps_dir = tmp_path / "apps"
    write_job(apps_dir, "alpha", "fails", "raise SystemExit(2)", secrets=("ALPHA_TOKEN",))
    output = tmp_path / "observed"
    write_job(
        apps_dir,
        "beta",
        "records",
        f"from pathlib import Path; import os; Path({str(output)!r}).write_text(str('ALPHA_TOKEN' in os.environ))",
        secrets=("BETA_TOKEN",),
    )

    result = CliRunner().invoke(
        cli,
        ["run", "hourly", "--apps-dir", str(apps_dir)],
        env={"MARINA_DATABASE_URL": database_url, "ALPHA_TOKEN": "alpha", "BETA_TOKEN": "beta"},
    )

    assert result.exit_code == 1
    assert "job failures: alpha.fails" in result.output
    assert output.read_text() == "False"


def test_run_migrate_only_does_not_execute_jobs(tmp_path: Path, database_url: str) -> None:
    apps_dir = tmp_path / "apps"
    output = tmp_path / "unexpected"
    write_job(apps_dir, "alpha", "job", f"from pathlib import Path; Path({str(output)!r}).touch()")

    result = CliRunner().invoke(
        cli,
        ["run", "hourly", "--apps-dir", str(apps_dir), "--migrate-only"],
        env={"MARINA_DATABASE_URL": database_url},
    )

    assert result.exit_code == 0
    assert not output.exists()
