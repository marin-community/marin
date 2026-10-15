# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from click.testing import CliRunner
from marina.applets import AppletStore, package_applet, read_applet_package
from marina.cli import cli
from marina.database_setup import ensure_applet_provisioning
from marina.db import UrlDatabase
from marina.journeys import running_kernel
from marina.server import MarinaConfig


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


def test_applet_host_commands_change_persisted_binding(tmp_path: Path, database_url: str) -> None:
    store = AppletStore(UrlDatabase(database_url))
    ensure_applet_provisioning(store.engine)
    store.migrate()
    package = read_applet_package(package_applet(Path(__file__).parents[1] / "examples" / "static-html-applet"))
    published = store.publish(package, "anonymous")
    apps_dir = tmp_path / "apps"
    apps_dir.mkdir()
    config = MarinaConfig(
        apps_dir=apps_dir,
        data_root=str(tmp_path),
        iap_audience=None,
        database=store.database,
        applet_operators=frozenset({"anonymous"}),
        applet_domains=frozenset({"cli.example"}),
    )
    with running_kernel(config) as kernel:
        runner = CliRunner()
        bound = runner.invoke(
            cli, ["applets", "hosts", "bind", "cli.example", str(published.applet_id), "--url", kernel.origin]
        )
        assert bound.exit_code == 0, bound.output
        assert store.host_applet("cli.example") == published.applet_id
        listed = runner.invoke(cli, ["applets", "hosts", "list", "--url", kernel.origin])
        assert listed.exit_code == 0, listed.output
        assert json.loads(listed.output)["hosts"]["cli.example"] == str(published.applet_id)
        unbound = runner.invoke(cli, ["applets", "hosts", "unbind", "cli.example", "--url", kernel.origin])
        assert unbound.exit_code == 0, unbound.output
        assert store.host_bindings()["cli.example"] is None
