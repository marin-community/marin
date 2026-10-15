# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
import textwrap
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from click.testing import CliRunner
from marina.applets import AppletStore, package_applet, read_applet_package
from marina.cli import cli
from marina.database_setup import ensure_applet_provisioning
from marina.db import UrlDatabase
from marina.testing import isolated_database


@pytest.fixture
def job_registry(database_url: str) -> Iterator[tuple[AppletStore, str]]:
    with isolated_database(database_url) as url:
        store = AppletStore(UrlDatabase(url))
        ensure_applet_provisioning(store.engine)
        store.migrate()
        try:
            yield store, url
        finally:
            store.engine.dispose()


def release_job(root: Path, store: AppletStore, name: str, code: str, *, secrets: tuple[str, ...] = ()) -> UUID:
    source = root / name
    (source / "dist").mkdir(parents=True)
    (source / "dist/index.html").write_text(f"<h1>{name}</h1>")
    (source / "applet.toml").write_text(
        f'name = "{name}"\ntitle = "{name}"\ndescription = "test"\n'
        '[[jobs]]\nname = "work"\nrunner = "hourly"\nmodule = "work"\ntimeout = 10\n'
        f"secrets = {list(secrets)!r}\n"
    )
    (source / "work.py").write_text(
        "def main():\n" + textwrap.indent(code, "    ") + '\n\nif __name__ == "__main__":\n    main()\n'
    )
    applet_id = UUID(int=1 if name == "alpha" else 2)
    store.publish(read_applet_package(package_applet(source)), "test", applet_id, 0, frozenset({"test"}))
    return applet_id


def runner_config(root: Path, permissions: dict[UUID, tuple[str, ...]]) -> Path:
    config = root / "runtime.toml"
    config.write_text(
        '[runners.hourly]\nschedule = "0 * * * *"\ntimeout = 60\ncpu = 1\nmemory_gib = 1\n'
        'secrets = ["ALPHA_TOKEN", "BETA_TOKEN"]\n'
        + "".join(
            f'[job_permissions."{applet_id}"]\nrunner = "hourly"\nsecrets = {list(secrets)!r}\n'
            for applet_id, secrets in permissions.items()
        )
    )
    return config


def test_run_attempts_later_released_jobs_and_filters_secrets(tmp_path: Path, job_registry) -> None:
    store, url = job_registry
    first = release_job(tmp_path, store, "alpha", "raise SystemExit(2)", secrets=("ALPHA_TOKEN",))
    output = tmp_path / "observed"
    second = release_job(
        tmp_path,
        store,
        "beta",
        f'from pathlib import Path\nimport os\nPath({str(output)!r}).write_text(str("ALPHA_TOKEN" in os.environ))',
        secrets=("BETA_TOKEN",),
    )
    config = runner_config(tmp_path, {first: ("ALPHA_TOKEN",), second: ("BETA_TOKEN",)})
    result = CliRunner().invoke(
        cli,
        ["run", "hourly"],
        env={
            "MARINA_DATABASE_URL": url,
            "MARINA_RUNTIME_CONFIG": str(config),
            "ALPHA_TOKEN": "alpha",
            "BETA_TOKEN": "beta",
        },
    )
    assert result.exit_code == 1, result.output
    assert output.read_text() == "False"


def test_migration_gate_requires_explicit_release_and_preserves_revision(tmp_path: Path, job_registry) -> None:
    store, url = job_registry
    applet_id = UUID(int=1)
    config = tmp_path / "runtime.toml"
    config.write_text(f'[required_apps]\nalpha = "{applet_id}"\n')
    env = {"MARINA_DATABASE_URL": url, "MARINA_RUNTIME_CONFIG": str(config)}
    runner = CliRunner()
    assert runner.invoke(cli, ["migrate", "--require-apps"], env=env).exit_code == 1
    output = tmp_path / "unexpected"
    release_job(tmp_path, store, "alpha", f"from pathlib import Path\nPath({str(output)!r}).touch()")
    result = runner.invoke(cli, ["migrate", "--require-apps"], env=env)
    assert result.exit_code == 0, result.output
    assert store.current_version(applet_id) == 1
    assert not output.exists()


def test_runner_uses_snapshot_when_an_earlier_job_releases_new_code(tmp_path: Path, job_registry) -> None:
    store, url = job_registry
    output = tmp_path / "observed"
    second = release_job(tmp_path, store, "beta", f'from pathlib import Path\nPath({str(output)!r}).write_text("old")')
    next_source = tmp_path / "beta-next"
    shutil.copytree(tmp_path / "beta", next_source)
    (next_source / "work.py").write_text((next_source / "work.py").read_text().replace('"old"', '"new"'))
    first = release_job(
        tmp_path,
        store,
        "alpha",
        textwrap.dedent(
            f"""\
        import os
        from pathlib import Path
        from uuid import UUID
        from marina.applets import AppletStore, package_applet, read_applet_package
        from marina.db import UrlDatabase
        store = AppletStore(UrlDatabase(os.environ["MARINA_DATABASE_URL"]))
        try:
            if store.current_version(UUID("{second}")) == 1:
                package = read_applet_package(package_applet(Path({str(next_source)!r})))
                store.publish(package, "test", UUID("{second}"), 1)
        finally:
            store.engine.dispose()
        """
        ),
    )
    config = runner_config(tmp_path, {first: (), second: ()})
    args = ["run", "hourly"]
    env = {"MARINA_DATABASE_URL": url, "MARINA_RUNTIME_CONFIG": str(config)}
    runner = CliRunner()
    first_run = runner.invoke(cli, args, env=env)
    assert first_run.exit_code == 0, first_run.output
    assert store.current_version(second) == 2
    assert output.read_text() == "old"
    second_run = runner.invoke(cli, args, env=env)
    assert second_run.exit_code == 0, second_run.output
    assert output.read_text() == "new"
