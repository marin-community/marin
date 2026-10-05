# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from scripts.ci.dependency_update import (
    BranchPushMode,
    CheckRow,
    MergeDecision,
    PullRequestSnapshot,
    evaluate_merge,
    evaluate_required_checks,
    prepare_update_branch,
    publish_update,
    required_check_rows,
    validate_changed_files,
    validated_pull_request,
)
from scripts.ci.dependency_update_policy import (
    EXTERNAL_RUNTIME_POLICIES,
    NATIVE_PACKAGE_POLICY,
    ExternalRuntime,
    PullRequestPolicy,
)
from scripts.ci.package_release import PACKAGES, requirement_paths_for_packages

EXPECTED_SHA = "a" * 40
SKYRL_POLICY = EXTERNAL_RUNTIME_POLICIES[ExternalRuntime.MARIN_SKYRL]


def _pull_request(policy: PullRequestPolicy = SKYRL_POLICY, **overrides) -> PullRequestSnapshot:
    values = {
        "author": "app/marin-external-runtime-updater",
        "base_branch": "main",
        "files": tuple(sorted(policy.allowed_files)),
        "head_branch": policy.head_branch,
        "head_sha": EXPECTED_SHA,
        "state": "OPEN",
        "title": policy.title,
        "url": "https://github.com/marin-community/marin/pull/123",
    }
    values.update(overrides)
    return PullRequestSnapshot(**values)


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_repository(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(repository)], check=True, capture_output=True)
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "uv.lock").write_text("initial\n")
    _git(repository, "add", "uv.lock")
    _git(repository, "commit", "-m", "initial")
    _git(repository, "remote", "add", "origin", str(remote))
    _git(repository, "push", "-u", "origin", "main")
    return repository, remote, _git(repository, "rev-parse", "HEAD")


@pytest.mark.parametrize("policy", [*EXTERNAL_RUNTIME_POLICIES.values(), NATIVE_PACKAGE_POLICY])
def test_returns_the_dedicated_apps_exact_generated_pull_request(policy: PullRequestPolicy) -> None:
    pull_request = _pull_request(policy)

    validated = validated_pull_request(
        pull_request,
        policy=policy,
        expected_app_slug="marin-external-runtime-updater",
        expected_head_sha=EXPECTED_SHA,
    )

    assert validated == pull_request


@pytest.mark.parametrize(
    "override",
    [
        {"author": "octocat"},
        {"base_branch": "release"},
        {"head_branch": "feature/unrelated"},
        {"head_sha": "b" * 40},
        {"title": "Update dependencies"},
        {"files": (*tuple(sorted(SKYRL_POLICY.allowed_files)), "src/backdoor.py")},
        {"files": (*tuple(sorted(SKYRL_POLICY.allowed_files)), "config/external/harbor/uv.lock")},
        {"files": ()},
    ],
    ids=["author", "base", "head", "sha", "title", "files", "other-project", "empty"],
)
def test_rejects_a_pull_request_outside_the_generated_boundary(override: dict) -> None:
    with pytest.raises(ValueError):
        validated_pull_request(
            _pull_request(**override),
            policy=SKYRL_POLICY,
            expected_app_slug="marin-external-runtime-updater",
            expected_head_sha=EXPECTED_SHA,
        )


def test_required_check_gate_distinguishes_missing_pending_and_failed_checks() -> None:
    required = ("marin-integration", "marin-lint", "rust-checks", "unit-tests")

    missing = evaluate_required_checks(
        [CheckRow(name="marin-lint", bucket="pass")],
        required=required,
    )
    pending = evaluate_required_checks(
        [
            CheckRow(name="marin-integration", bucket="pass"),
            CheckRow(name="marin-lint", bucket="pass"),
            CheckRow(name="rust-checks", bucket="pending"),
            CheckRow(name="unit-tests", bucket="pass"),
        ],
        required=required,
    )
    failed = evaluate_required_checks(
        [
            CheckRow(name="marin-integration", bucket="pass"),
            CheckRow(name="marin-lint", bucket="fail"),
            CheckRow(name="rust-checks", bucket="pass"),
            CheckRow(name="unit-tests", bucket="pass"),
        ],
        required=required,
    )

    assert missing.missing == ("marin-integration", "rust-checks", "unit-tests")
    assert pending.pending == ("rust-checks",)
    assert failed.failing == ("marin-lint",)
    assert evaluate_merge("OPEN", missing) is MergeDecision.WAIT
    assert evaluate_merge("OPEN", pending) is MergeDecision.WAIT
    assert evaluate_merge("OPEN", failed) is MergeDecision.FAIL


def test_required_check_gate_ignores_duplicate_unrelated_checks() -> None:
    gate = evaluate_required_checks(
        [
            CheckRow(name="changes", bucket="pass"),
            CheckRow(name="changes", bucket="pass"),
            CheckRow(name="marin-lint", bucket="pass"),
        ],
        required=("marin-lint",),
    )

    assert gate.passed


def test_merge_gate_only_releases_an_open_pull_request_after_all_required_checks_pass() -> None:
    checks = evaluate_required_checks(
        [
            CheckRow(name=name, bucket="pass")
            for name in ("marin-integration", "marin-lint", "rust-checks", "unit-tests")
        ],
        required=("marin-integration", "marin-lint", "rust-checks", "unit-tests"),
    )

    assert evaluate_merge("OPEN", checks) is MergeDecision.MERGE
    assert evaluate_merge("MERGED", checks) is MergeDecision.DONE
    assert evaluate_merge("CLOSED", checks) is MergeDecision.FAIL


def test_no_registered_github_checks_is_a_missing_gate_not_a_cli_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.ci.dependency_update.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="no checks"),
    )

    assert required_check_rows("123", "marin-community/marin") == ()


def test_changed_files_are_sorted_and_restricted_to_the_policy() -> None:
    changed = validate_changed_files(
        [
            "lib/marin/src/marin/external_dependencies.py",
            "config/external/MarinSkyRL/uv.lock",
            "config/external/MarinSkyRL/uv.lock",
        ],
        policy=SKYRL_POLICY,
    )

    assert changed == ("config/external/MarinSkyRL/uv.lock", "lib/marin/src/marin/external_dependencies.py")
    with pytest.raises(ValueError):
        validate_changed_files(["config/external/MarinSkyRL/uv.lock", "src/backdoor.py"], policy=SKYRL_POLICY)


def test_prepare_update_branch_resets_main_with_a_lease_for_an_existing_branch(monkeypatch, tmp_path: Path) -> None:
    repository, _remote, main_sha = _git_repository(tmp_path)
    _git(repository, "switch", "-c", NATIVE_PACKAGE_POLICY.head_branch)
    (repository / "uv.lock").write_text("stale update\n")
    _git(repository, "commit", "-am", "stale update")
    _git(repository, "push", "origin", NATIVE_PACKAGE_POLICY.head_branch)
    remote_sha = _git(repository, "rev-parse", "HEAD")
    _git(repository, "switch", "main")
    monkeypatch.chdir(repository)
    monkeypatch.setattr(
        "scripts.ci.dependency_update._gh_json",
        lambda *_args: [{"url": "https://github.com/marin-community/marin/pull/123"}],
    )

    branch = prepare_update_branch(policy=NATIVE_PACKAGE_POLICY, repository="marin-community/marin")

    assert branch.expected_remote_sha == remote_sha
    assert branch.pull_request_url == "https://github.com/marin-community/marin/pull/123"
    assert branch.push_mode is BranchPushMode.FORCE_WITH_LEASE
    assert _git(repository, "branch", "--show-current") == NATIVE_PACKAGE_POLICY.head_branch
    assert _git(repository, "rev-parse", "HEAD") == main_sha


def test_publish_update_stages_the_allowlist_and_creates_an_app_pull_request(monkeypatch, tmp_path: Path) -> None:
    repository, remote, _main_sha = _git_repository(tmp_path)
    _git(repository, "switch", "-c", NATIVE_PACKAGE_POLICY.head_branch)
    (repository / "uv.lock").write_text("published update\n")
    body_file = repository / "body.md"
    body_file.write_text("Update native packages.\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "if sys.argv[1:3] == ['pr', 'view']:\n"
        "    print(json.dumps({'url': 'https://github.com/marin-community/marin/pull/123'}))\n"
    )
    fake_gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.chdir(repository)

    published = publish_update(
        policy=NATIVE_PACKAGE_POLICY,
        repository="marin-community/marin",
        body_file=body_file,
        expected_remote_sha="",
        pull_request_url="",
        push_mode=BranchPushMode.CREATE,
    )

    assert published.url == "https://github.com/marin-community/marin/pull/123"
    remote_sha = subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", NATIVE_PACKAGE_POLICY.head_branch],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert published.head_sha == remote_sha
    assert _git(repository, "show", f"{remote_sha}:uv.lock") == "published update"


def test_external_update_cli_resolves_one_project_from_main_and_rejects_other_project_files(tmp_path: Path) -> None:
    repository, _remote, _main_sha = _git_repository(tmp_path)
    upstream, _upstream_remote, _upstream_sha = _git_repository(tmp_path / "upstream")
    schema = upstream / "marinskyrl/recipe_schema"
    schema.mkdir(parents=True)
    (schema / "__init__.py").write_text("from .old import VALUE\n")
    (schema / "old.py").write_text("VALUE = 1\n")
    _git(upstream, "add", "marinskyrl")
    _git(upstream, "commit", "-m", "author schema")
    original_schema_commit = _git(upstream, "rev-parse", "HEAD")
    (schema / "old.py").unlink()
    (schema / "new.py").write_text("VALUE = 2\n")
    (schema / "__init__.py").write_text("from .new import VALUE\n")
    _git(upstream, "add", "marinskyrl")
    _git(upstream, "commit", "-m", "updated author schema")
    new_commit = _git(upstream, "rev-parse", "HEAD")
    source = Path(__file__).resolve().parents[2]
    shutil.copytree(source / "config/external", repository / "config/external")
    shutil.copy2(source / "config/update-external.py", repository / "config/update-external.py")
    scripts = repository / "scripts/ci"
    scripts.mkdir(parents=True)
    for package in ("scripts", "scripts/ci"):
        shutil.copy2(source / package / "__init__.py", repository / package / "__init__.py")
    for name in ("dependency_update.py", "dependency_update_policy.py", "package_release.py"):
        shutil.copy2(source / "scripts/ci" / name, scripts / name)
    pins = repository / "lib/marin/src/marin/external_dependencies.py"
    pins.parent.mkdir(parents=True)
    shutil.copy2(source / "lib/marin/src/marin/external_dependencies.py", pins)
    skyrl_lock = repository / "config/external/MarinSkyRL/uv.lock"
    content = skyrl_lock.read_text()
    package = next(entry for entry in tomllib.loads(content)["package"] if entry["name"] == "marinskyrl")
    skyrl_lock.write_text(content.replace(package["source"]["git"].rsplit("#", 1)[1], original_schema_commit))
    git_environment = {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{upstream.as_uri()}.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://github.com/marin-community/MarinSkyRL.git",
    }
    subprocess.run(
        [sys.executable, "config/update-external.py", "vllm"],
        cwd=repository,
        env=git_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repository, "add", "config", "scripts", "lib")
    _git(repository, "commit", "-m", "external project inputs")
    _git(repository, "push", "origin", "main")
    main_sha = _git(repository, "rev-parse", "HEAD")
    locks = {project: repository / f"config/external/{project.value}/uv.lock" for project in ExternalRuntime}
    originals = {project: path.read_bytes() for project, path in locks.items()}
    distributions = {
        dependency.config_name: dependency.distribution
        for dependency in runpy.run_path(str(pins))["EXTERNAL_DEPENDENCIES"]
    }
    initial_commits = {
        project.value: next(
            entry["source"]["git"].rsplit("#", 1)[1]
            for entry in tomllib.loads(path.read_text())["package"]
            if entry["name"] == distributions[project.value]
        )
        for project, path in locks.items()
    }
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    resolver = fake_bin / "uv"
    resolver.write_text(
        f"#!{sys.executable}\n"
        "import sys, tomllib\n"
        "from pathlib import Path\n"
        "directory = Path(sys.argv[sys.argv.index('--project') + 1])\n"
        "distribution = sys.argv[sys.argv.index('--upgrade-package') + 1]\n"
        "path = directory / 'uv.lock'\n"
        "original = path.read_text()\n"
        "package = next(p for p in tomllib.loads(original)['package'] if p['name'] == distribution)\n"
        "git_source = package['source']['git']\n"
        f"updated = git_source.rsplit('#', 1)[0] + '#{new_commit}'\n"
        "path.write_text(original.replace(git_source, updated))\n"
    )
    github = fake_bin / "gh"
    github.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "if sys.argv[1:3] == ['pr', 'list']:\n"
        "    print('[]')\n"
        "elif sys.argv[1] == 'api':\n"
        f"    print(json.dumps([{{'status': 'ahead', 'total_commits': 1, 'commits': [{{'sha': '{new_commit}', "
        "'commit': {'message': 'Upstream runtime update'}}]}]))\n"
        "else:\n"
        "    raise RuntimeError('unexpected GitHub operation')\n"
    )
    resolver.chmod(0o755)
    github.chmod(0o755)
    environment = {**git_environment, "PATH": f"{fake_bin}:{os.environ['PATH']}", "PYTHONPATH": ""}
    workflow = yaml.safe_load((source / ".github/workflows/ops-external-dependencies.yaml").read_text())
    select_projects = next(step for step in workflow["jobs"]["projects"]["steps"] if step.get("id") == "projects")
    matrix_output = tmp_path / "projects-output"
    subprocess.run(
        ["bash", "-c", select_projects["run"]],
        cwd=repository,
        env={**environment, "GITHUB_OUTPUT": str(matrix_output)},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    projects = json.loads(matrix_output.read_text().removeprefix("projects="))
    branches = []
    for name in projects:
        project = ExternalRuntime(name)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.ci.dependency_update",
                "prepare",
                "--kind",
                "external-runtime",
                "--project",
                project.value,
                "--repository",
                "marin-community/marin",
                "--github-output",
                str(tmp_path / "outputs"),
            ],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert _git(repository, "rev-parse", "HEAD") == main_sha
        branches.append(_git(repository, "branch", "--show-current"))
        summary = tmp_path / "summary.md"
        subprocess.run(
            [
                sys.executable,
                "config/update-external.py",
                project.value,
                "--summary-file",
                str(summary),
            ],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        changed_command = [
            sys.executable,
            "-m",
            "scripts.ci.dependency_update",
            "changed-files",
            "--kind",
            "external-runtime",
            "--project",
            project.value,
        ]
        changed = subprocess.run(
            changed_command,
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        expected = {f"config/external/{project.value}/uv.lock", str(pins.relative_to(repository))}
        if project is ExternalRuntime.MARIN_SKYRL:
            expected |= {
                "lib/marin/src/marin/skyrl_recipe.provenance.json",
                "lib/marin/src/marin/skyrl_recipe/__init__.py",
                "lib/marin/src/marin/skyrl_recipe/old.py",
                "lib/marin/src/marin/skyrl_recipe/new.py",
            }
            assert not (repository / "lib/marin/src/marin/skyrl_recipe/old.py").exists()
            assert (repository / "lib/marin/src/marin/skyrl_recipe/new.py").read_text() == "VALUE = 2\n"
        assert set(changed.stdout.splitlines()) == expected
        if project is ExternalRuntime.MARIN_SKYRL:
            copied = repository / "lib/marin/src/marin/skyrl_recipe"
            provenance = copied.with_suffix(".provenance.json")
            original_copy = {path: path.read_bytes() for path in copied.iterdir()}
            original_provenance = provenance.read_bytes()
            manifest = json.loads(original_provenance)
            digest = hashlib.sha256()
            for path, content in sorted(original_copy.items()):
                digest.update(path.name.encode() + b"\0" + content + b"\0")
            assert manifest["commit"] == new_commit
            assert manifest["sha256"] == digest.hexdigest()
            check_command = [sys.executable, "config/update-external.py", "--check"]
            offline_bin = tmp_path / "offline-bin"
            offline_bin.mkdir()
            for tool in ("git", "uv"):
                executable = offline_bin / tool
                executable.write_text("#!/bin/sh\nexit 99\n")
                executable.chmod(0o755)
            offline_environment = {**environment, "PATH": f"{offline_bin}:{environment['PATH']}"}
            clean = subprocess.run(
                check_command, cwd=repository, env=offline_environment, capture_output=True, text=True, timeout=30
            )
            assert clean.returncode == 0, clean.stderr
            for drift in ("content", "extra", "missing"):
                if drift.startswith("content"):
                    (copied / "new.py").write_text("VALUE = 3\n")
                elif drift == "extra":
                    (copied / "extra.py").write_text("VALUE = 4\n")
                else:
                    (copied / "new.py").unlink()
                before = {path: path.read_bytes() for path in (*copied.iterdir(), provenance, pins)}
                rejected_copy = subprocess.run(
                    check_command, cwd=repository, env=offline_environment, capture_output=True, text=True, timeout=30
                )
                assert rejected_copy.returncode != 0, drift
                assert {path: path.read_bytes() for path in (*copied.iterdir(), provenance, pins)} == before
                (copied / "extra.py").unlink(missing_ok=True)
                for path, content in original_copy.items():
                    path.write_bytes(content)
                provenance.write_bytes(original_provenance)
        _git(repository, "add", *sorted(expected))
        _git(repository, "commit", "-m", "generated runtime update")
        remote_check = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json, sys
from scripts.ci.dependency_update import PullRequestSnapshot, validated_pull_request
from scripts.ci.dependency_update_policy import EXTERNAL_RUNTIME_POLICIES, ExternalRuntime
policy = EXTERNAL_RUNTIME_POLICIES[ExternalRuntime(sys.argv[1])]
snapshot = PullRequestSnapshot(
    author='app/marin-external-runtime-updater', base_branch=policy.base_branch,
    files=tuple(json.loads(sys.argv[2])), head_branch=policy.head_branch,
    head_sha=sys.argv[3], state='OPEN', title=policy.title, url='https://example.test/pr',
)
print(validated_pull_request(snapshot, policy=policy,
    expected_app_slug='marin-external-runtime-updater', expected_head_sha=sys.argv[3]).head_sha)
""",
                project.value,
                json.dumps(sorted(expected)),
                _git(repository, "rev-parse", "HEAD"),
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert remote_check.returncode == 0, remote_check.stderr
        assert remote_check.stdout.strip() == _git(repository, "rev-parse", "HEAD")
        for other, path in locks.items():
            if other != project:
                assert path.read_bytes() == originals[other]
        generated = runpy.run_path(str(pins))
        assert {pin.config_name: pin.commit for pin in generated["EXTERNAL_DEPENDENCIES"]} == {
            **initial_commits,
            project.value: new_commit,
        }
        rows = [line.split("|")[1].strip(" `") for line in summary.read_text().splitlines() if line.startswith("| `")]
        assert rows == [project.value]
        other_project = next(other for other in ExternalRuntime if other != project)
        foreign_lock = locks[other_project]
        foreign_lock.write_bytes(originals[other_project] + b"\n# foreign update\n")
        rejected = subprocess.run(
            changed_command,
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert rejected.returncode != 0
        _git(repository, "restore", str(foreign_lock.relative_to(repository)))
    assert len(set(branches)) == len(ExternalRuntime)
    assert (repository / "uv.lock").read_text() == "initial\n"


def test_native_package_policy_matches_every_compatibility_floor() -> None:
    compatibility_floors = {path.as_posix() for path in requirement_paths_for_packages(PACKAGES)}

    assert NATIVE_PACKAGE_POLICY.allowed_files == {"uv.lock", *compatibility_floors}
