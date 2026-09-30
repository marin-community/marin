"""Agent confinement: builders act only in their workspace and never see job secrets.

Incidents (catalog-full-construct-003, 2026-09-29): a builder's local selfcheck
unpacked a fixture tarball with absolute members over the job's /etc/resolv.conf
(shard-033-g2), and another builder's host "dry run" rmtree'd /app (shard-055-c6).
"""

from __future__ import annotations

import io
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from capability_pipeline import builder_confinement as bc
from capability_pipeline import synthesis
from scripts.sync_supervisor import EtcSentinel

JOB_SECRETS = {
    "CW_KEY_ID": "cw-id",
    "CW_KEY_SECRET": "cw-secret",
    "AWS_ACCESS_KEY_ID": "aws-id",
    "AWS_SECRET_ACCESS_KEY": "aws-secret",
    "FSSPEC_S3": '{"key": "cw-id", "secret": "cw-secret"}',
    "WANDB_API_KEY": "wandb",
    "IRIS_JOB_ENV": '{"CW_KEY_SECRET": "cw-secret"}',
    "HF_TOKEN": "hf",
    "REGISTRY_PASSWORD": "registry",
    "DOCKER_AUTH_CONFIG": "{}",
}
AGENT_NEEDS = {
    "PATH": "/usr/bin:/bin",
    "GLM_BASE_URL": "http://relay/v1",
    "GLM_API_TOKEN": "glm",
    "SILO_API_TOKEN": "silo",
    "SILO_BROKER_RESOLVE_URL": "http://broker",
    "CAPABILITY_SANDBOX_PROVIDER": "silo",
    "PARALLEL_API_KEY": "parallel",
    "OMP_NUM_THREADS": "2",
    "LC_ALL": "C.UTF-8",
    "DT_KEY": "label",
}
DROPPED_NON_SECRETS = {
    # Pointing an agent's uv at /app/.venv lets one `uv sync` gut the controller venv.
    "UV_PROJECT_ENVIRONMENT": "/app/.venv",
    "VIRTUAL_ENV": "/app/.venv",
    "CARGO_HOME": "/cargo",
    "HF_HOME": "/hf/cache",
    "UV_CACHE_DIR": "/uv/cache",
    "KUBERNETES_SERVICE_HOST": "10.0.0.1",
    "HOME": "/root",
}


def test_agent_environment_is_an_allowlist():
    env = bc.agent_environment({**JOB_SECRETS, **AGENT_NEEDS, **DROPPED_NON_SECRETS}, {"HOME": "/agent"})
    assert not set(JOB_SECRETS) & set(env)
    assert not set(DROPPED_NON_SECRETS) - {"HOME"} & set(env)
    assert {name: env[name] for name in AGENT_NEEDS} == AGENT_NEEDS
    assert env["HOME"] == "/agent"


def test_agent_environment_refuses_forbidden_overrides():
    with pytest.raises(bc.ConfinementError, match="CW_KEY_SECRET"):
        bc.agent_environment({}, {"CW_KEY_SECRET": "x"})


def test_mode_parsing(monkeypatch):
    monkeypatch.setenv(bc.MODE_ENV, "off")
    assert bc.mode() == "off"
    monkeypatch.setenv(bc.MODE_ENV, "sometimes")
    with pytest.raises(bc.ConfinementError):
        bc.mode()
    monkeypatch.delenv(bc.MODE_ENV)
    if os.geteuid() != 0:
        assert bc.mode() == "unprivileged"


def _fake_omp(path: Path) -> Path:
    """An 'omp' that records what an agent tool process would see."""
    path.write_text(
        "#!/bin/sh\n"
        'env > "$PWD/agent-env.txt"\n'
        'printf "%s\\n" "$@" > "$PWD/agent-argv.txt"\n'
        "exit 0\n"
    )
    path.chmod(0o755)
    return path


def test_credentials_do_not_reach_agent_subprocesses(tmp_path, monkeypatch):
    """A real subprocess: the object-store keys never enter the agent's environment."""
    for name, value in {**JOB_SECRETS, **AGENT_NEEDS}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
    monkeypatch.setattr(bc, "mode", lambda: "unprivileged")
    workspace = tmp_path / "items/x/workspace"
    workspace.mkdir(parents=True)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("build")
    agent = synthesis.OMPAgent(str(_fake_omp(tmp_path / "omp")), None, 30, 0)

    outcome = agent.invoke(workspace, tmp_path / "items/x/sessions/s1/transcript", prompt, 0)

    assert outcome["returncode"] == 0
    assert outcome["confinement"]["mode"] == "unprivileged"
    seen = dict(
        line.split("=", 1) for line in (workspace / "agent-env.txt").read_text().splitlines() if "=" in line
    )
    assert not set(JOB_SECRETS) & set(seen), sorted(set(JOB_SECRETS) & set(seen))
    for value in ("cw-secret", "aws-secret"):
        assert value not in (workspace / "agent-env.txt").read_text()
    assert seen["SILO_API_TOKEN"] == "silo"
    assert seen["GLM_BASE_URL"] == "http://relay/v1"


def _uid_mode(monkeypatch, tmp_path, chowns):
    monkeypatch.setattr(bc, "mode", lambda: "uid")
    monkeypatch.setattr(bc, "_setup_process", lambda: {"setpriv": "/usr/bin/setpriv"})
    monkeypatch.setattr(bc, "reap", lambda uid: 0)
    monkeypatch.setenv(bc.STATE_ROOT_ENV, str(tmp_path / "agents"))

    def chown(path, uid, gid, *, follow_symlinks=True):
        chowns.append((str(path), uid, gid, follow_symlinks))

    monkeypatch.setattr(bc.os, "chown", chown)
    monkeypatch.setattr(bc.os, "fchown", lambda fd, uid, gid: chowns.append((f"fd:{fd}", uid, gid, False)))


def test_uid_mode_wraps_agent_in_privilege_drop(tmp_path, monkeypatch):
    chowns = []
    _uid_mode(monkeypatch, tmp_path, chowns)
    for name, value in {**JOB_SECRETS, **AGENT_NEEDS}.items():
        monkeypatch.setenv(name, value)
    home = tmp_path / "root-home"
    (home / ".omp/agent").mkdir(parents=True)
    (home / ".omp/glm_token").write_text("tok\n")
    (home / ".omp/agent/models.yml").write_text(f'apiKey: "!cat {home}/.omp/glm_token"\n')
    monkeypatch.setenv("HOME", str(home))
    workspace = tmp_path / "items/x/workspace"
    (workspace / "task").mkdir(parents=True)
    (workspace / "task/spec.json").write_text("{}")
    session = tmp_path / "items/x/sessions/s1/transcript"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("build")
    calls = []

    def fake_run(argv, *, cwd, timeout, env, umask):
        calls.append({"argv": argv, "cwd": cwd, "env": env, "umask": umask})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(synthesis, "_run", fake_run)
    outcome = synthesis.OMPAgent("/tmp/pipeline-tools/omp", None, 30, 0).invoke(workspace, session, prompt, 0)

    uid = outcome["confinement"]["uid"]
    assert bc.UID_BASE <= uid < bc.UID_BASE + bc.UID_SPAN
    assert uid == bc.preferred_uid(workspace)
    argv = calls[0]["argv"]
    assert argv[: argv.index("--") + 1] == bc.privilege_drop_argv("/usr/bin/setpriv", uid, uid)
    assert {"--no-new-privs", "--clear-groups", "--bounding-set=-all"} <= set(argv)
    assert argv[argv.index("--") + 1] == "/tmp/pipeline-tools/omp"
    assert calls[0]["cwd"] == workspace
    assert calls[0]["umask"] == bc.AGENT_UMASK
    env = calls[0]["env"]
    assert not set(JOB_SECRETS) & set(env)
    state = bc.agent_state_dir(workspace)
    assert env["HOME"] == str(state / "home") and env["TMPDIR"] == str(state / "tmp")
    assert env["UV_CACHE_DIR"].startswith(str(state))
    # The workspace, the transcript dir and the private state are handed to the uid,
    # never by following a link.
    owned = {path for path, owner, _, _ in chowns if owner == uid}
    assert str(workspace / "task/spec.json") in owned and str(session) in owned and str(state) in owned
    assert all(follow is False for _, _, _, follow in chowns)
    # The model token is re-homed where the uid can read it.
    models = (state / "home/.omp/agent/models.yml").read_text()
    assert str(home) not in models and "!cat " + str(state / "home/.omp") in models
    assert stat.S_IMODE((state / "home/.omp/agent/models.yml").stat().st_mode) == 0o600


def test_uids_are_deterministic_and_distinct_when_concurrent(monkeypatch):
    monkeypatch.setattr(bc, "UID_SPAN", 2)
    first = bc._allocate_uid("0" * 64)
    second = bc._allocate_uid("0" * 63 + "2")  # same preferred slot
    try:
        assert first != second
        assert bc._allocate_uid("0" * 64) == first  # same workspace shares its uid
        bc._release_uid(first)
    finally:
        bc._release_uid(first)
        bc._release_uid(second)


def test_chown_tree_never_follows_links_out(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "shadow").write_text("root only")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("x")
    (workspace / "file-link").symlink_to(outside / "shadow")
    (workspace / "dir-link").symlink_to(outside)
    os.link(outside / "shadow", workspace / "hardlink")
    calls = []

    counts = bc.chown_tree(
        workspace, 2_000_123, 2_000_123, chown=lambda p, u, g, *, follow_symlinks: calls.append((str(p), follow_symlinks))
    )

    touched = {path for path, _ in calls}
    assert {str(workspace), str(workspace / "a.txt"), str(workspace / "file-link"), str(workspace / "dir-link")} == touched
    assert not any(path.startswith(str(outside)) for path in touched)
    assert all(follow is False for _, follow in calls)
    assert counts["skipped_hardlinks"] == 1


def test_harden_shared_directories(tmp_path):
    shared, private = tmp_path / "tmp", tmp_path / "app"
    shared.mkdir()
    private.mkdir()
    shared.chmod(0o777)
    private.chmod(0o777)
    changes = bc.harden_shared_directories([str(shared)], [str(private)])
    assert stat.S_IMODE(shared.stat().st_mode) == 0o1777
    assert stat.S_IMODE(private.stat().st_mode) == 0o755
    assert len(changes) == 2
    assert bc.harden_shared_directories([str(shared)], [str(private)]) == []


def test_escaping_links_are_removed_after_a_session(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inside").write_text("x")
    (workspace / "log").symlink_to("/etc/resolv.conf")
    (workspace / "ok").symlink_to("inside")
    removed = bc.remove_escaping_symlinks(workspace)
    assert removed == [str(workspace / "log")]
    assert (workspace / "ok").is_symlink()


def test_agent_links_in_create_only_roots_are_removed(tmp_path):
    root = tmp_path / "repair"
    root.mkdir()
    (root / "before").mkdir()
    (root / "continuation-2.md").symlink_to(root / "before")
    (root / "receipt.json").write_text("{}")
    assert bc.remove_agent_symlinks(root, os.getuid()) == [str(root / "continuation-2.md")]
    assert (root / "receipt.json").is_file()


def test_stage_file_replaces_planted_links(tmp_path):
    outside = tmp_path / "app-venv-bin"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "tools").symlink_to(outside)
    source = tmp_path / "dt.py"
    source.write_text("print('dt')")
    staged = bc.stage_file(workspace, "tools/daytona/dt.py", source)
    assert staged == workspace / "tools/daytona/dt.py"
    assert not (workspace / "tools").is_symlink()
    assert list(outside.iterdir()) == []
    (workspace / "tools/daytona/dt.py").unlink()
    (workspace / "tools/daytona/dt.py").symlink_to(outside / "victim")
    bc.stage_file(workspace, "tools/daytona/dt.py", source)
    assert not (outside / "victim").exists()
    with pytest.raises(ValueError):
        bc.stage_file(workspace, "../escape", source)


def test_repair_grant_is_create_only_and_restored(tmp_path, monkeypatch):
    chowns = []
    _uid_mode(monkeypatch, tmp_path, chowns)
    workspace = tmp_path / "items/x/workspace"
    workspace.mkdir(parents=True)
    repair_root = tmp_path / "repairs/attempt-1"
    (repair_root / "before").mkdir(parents=True)
    repair_root.chmod(0o755)
    with bc.grant_create(workspace, repair_root):
        session = bc.open_session(workspace, repair_root / "transcript")
        assert stat.S_IMODE(repair_root.stat().st_mode) == 0o1777
        session.close()
    assert stat.S_IMODE(repair_root.stat().st_mode) == 0o755
    handed = {path for path, owner, _, _ in chowns if owner == session.uid}
    assert str(repair_root / "transcript") in handed
    assert not any(path.startswith(str(repair_root / "before")) for path in handed)
    assert (str(repair_root), 0, 0, False) in chowns


def test_private_toolchain_copy_is_per_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "mode", lambda: "uid")
    monkeypatch.setenv(bc.STATE_ROOT_ENV, str(tmp_path / "agents"))
    shared = tmp_path / "taskcompendium-builder-x"
    (shared / "src").mkdir(parents=True)
    (shared / "src/mod.py").write_text("x = 1")
    (shared / ".venv").mkdir()
    first = bc.private_toolchain(shared, tmp_path / "a/workspace")
    second = bc.private_toolchain(shared, tmp_path / "b/workspace")
    assert first != second
    assert (first / "src/mod.py").read_text() == "x = 1" and not (first / ".venv").exists()
    monkeypatch.setattr(bc, "mode", lambda: "unprivileged")
    assert bc.private_toolchain(shared, tmp_path / "a/workspace") == shared


def test_etc_sentinel_restores_and_reports(tmp_path):
    resolv = tmp_path / "resolv.conf"
    resolv.write_text("nameserver 10.96.0.10\n")
    sentinel = EtcSentinel((str(resolv),))
    assert sentinel.check() == []
    resolv.write_text("search internal.northlane.example\nnameserver 10.20.0.53\n")
    events = sentinel.check()
    assert events and events[0]["restored"] is True and events[0]["path"] == str(resolv)
    assert resolv.read_text() == "nameserver 10.96.0.10\n"
    assert sentinel.check() == []


def test_etc_sentinel_reports_unrestorable_change_once(tmp_path):
    resolv = tmp_path / "resolv.conf"
    resolv.write_text("nameserver 10.96.0.10\n")
    sentinel = EtcSentinel((str(resolv),), restore=False)
    resolv.write_text("nameserver 10.20.0.53\n")
    assert len(sentinel.check()) == 1
    assert sentinel.check() == []


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="3.14 defaults to the 'data' tar filter")
def test_incident_mechanism_absolute_tar_members_escape_the_target(tmp_path):
    """Why a per-tool or per-format fix cannot suffice: stdlib extraction escapes."""
    victim = tmp_path / "etc/resolv.conf"
    victim.parent.mkdir()
    victim.write_text("nameserver 10.96.0.10\n")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        data = b"nameserver 10.20.0.53\n"
        member = tarfile.TarInfo(str(victim))  # build-fixture.py wrote "/etc/resolv.conf"
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    tarball = tmp_path / "fixture.tar.gz"
    tarball.write_bytes(payload.getvalue())
    (tmp_path / "temproot").mkdir()
    with pytest.warns(DeprecationWarning):
        shutil.unpack_archive(str(tarball), str(tmp_path / "temproot"))
    assert victim.read_text() == "nameserver 10.20.0.53\n"
    assert list((tmp_path / "temproot").iterdir()) == []


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() != 0 or not shutil.which("setpriv") or sys.platform != "linux",
    reason="needs root and util-linux setpriv (the construction job's environment)",
)
def test_real_privilege_drop_denies_writes_outside_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(bc.STATE_ROOT_ENV, str(tmp_path / "agents"))
    monkeypatch.setenv("CW_KEY_SECRET", "cw-secret")
    monkeypatch.setattr(bc, "harden_shared_directories", lambda *a, **k: [])
    tmp_path.chmod(0o755)
    victim = tmp_path / "resolv.conf"
    victim.write_text("nameserver 10.96.0.10\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = bc.open_session(workspace, tmp_path / "transcript")
    try:
        script = f'printf x >> {victim} 2>/dev/null; echo ok > {workspace}/made; env'
        done = subprocess.run(
            session.wrap(["/bin/sh", "-c", script]), env=session.env, capture_output=True, text=True, check=False
        )
    finally:
        session.close()
    assert victim.read_text() == "nameserver 10.96.0.10\n"
    assert (workspace / "made").read_text() == "ok\n"
    assert "cw-secret" not in done.stdout


def test_home_render_never_writes_through_planted_links(tmp_path):
    """The agent owns its home; root's next render must not follow its links."""
    source = tmp_path / "root-home"
    (source / ".omp/agent").mkdir(parents=True)
    (source / ".omp/glm_token").write_text("tok\n")
    (source / ".omp/agent/models.yml").write_text(f'apiKey: "!cat {source}/.omp/glm_token"\n')
    victim = tmp_path / "etc-resolv.conf"
    victim.write_text("nameserver 10.96.0.10\n")
    home = tmp_path / "agent-home"
    (home / ".omp/agent").mkdir(parents=True)
    (home / ".omp/agent/models.yml").symlink_to(victim)
    (home / ".omp/secret-0-glm_token").symlink_to(victim)
    bc.render_agent_home(source, home)
    assert victim.read_text() == "nameserver 10.96.0.10\n"
    assert not (home / ".omp/agent/models.yml").is_symlink()
    assert "!cat " + str(home / ".omp/secret-0-glm_token") in (home / ".omp/agent/models.yml").read_text()
    # A whole planted directory link is replaced by a real directory.
    shutil.rmtree(home / ".omp")
    (home / ".omp").symlink_to(tmp_path)
    bc.render_agent_home(source, home)
    assert not (home / ".omp").is_symlink() and not (tmp_path / "agent").exists()


def test_escaping_link_chains_through_nested_links_are_removed(tmp_path):
    outside = tmp_path / "etc"
    outside.mkdir()
    (outside / "resolv.conf").write_text("x")
    root = tmp_path / "review"
    (root / "sub").mkdir(parents=True)
    (root / "sub/s").symlink_to(outside)
    (root / "reviewer.log").symlink_to("sub/s/resolv.conf")  # lexically inside, really outside
    assert str(root / "reviewer.log") in bc.remove_escaping_symlinks(root)


def test_private_toolchain_replaces_a_planted_link(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "mode", lambda: "uid")
    monkeypatch.setenv(bc.STATE_ROOT_ENV, str(tmp_path / "agents"))
    shared = tmp_path / "shared"
    (shared / "src").mkdir(parents=True)
    victim = tmp_path / "app"
    victim.mkdir()
    workspace = tmp_path / "w/workspace"
    state = bc.agent_state_dir(workspace)
    state.mkdir(parents=True)
    (state / "toolchain").symlink_to(victim)
    copy = bc.private_toolchain(shared, workspace)
    assert not copy.is_symlink() and (copy / "src").is_dir()
    assert list(victim.iterdir()) == []


def test_identity_passes_through_when_no_privilege_is_dropped(tmp_path, monkeypatch):
    """The operator escape hatch must not break omp's own config lookup (HOME)."""
    monkeypatch.setattr(bc, "mode", lambda: "off")
    base = {**JOB_SECRETS, **AGENT_NEEDS, "HOME": "/root", "CARGO_HOME": "/cargo", "USER": "root"}
    session = bc.open_session(tmp_path / "w", tmp_path / "t", base_env=base)
    assert session.prefix == [] and session.uid is None
    assert session.env["HOME"] == "/root" and session.env["CARGO_HOME"] == "/cargo"
    assert not set(JOB_SECRETS) & set(session.env)


def test_root_toolchains_become_traversable_but_omp_state_closes(tmp_path):
    home = tmp_path / "root"
    (home / ".rustup/toolchains").mkdir(parents=True)
    (home / ".omp/logs").mkdir(parents=True)
    home.chmod(0o700)
    (home / ".omp").chmod(0o755)
    assert bc.open_root_toolchains(home) == {"RUSTUP_HOME": str(home / ".rustup")}
    assert stat.S_IMODE(home.stat().st_mode) == 0o701
    assert stat.S_IMODE((home / ".omp").stat().st_mode) == 0o700
    assert bc.open_root_toolchains(tmp_path / "no-rust") == {}


def test_controller_environment_keeps_its_credentials(tmp_path, monkeypatch):
    """Filtering is per child: sync_supervisor / publication_exchange still see CW_KEY_*."""
    for name, value in {**JOB_SECRETS, **AGENT_NEEDS}.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    chowns = []
    _uid_mode(monkeypatch, tmp_path, chowns)
    session = bc.open_session(tmp_path / "w", tmp_path / "t")
    session.close()
    assert "CW_KEY_SECRET" not in session.env
    assert {k: v for k, v in os.environ.items() if k != bc.STATE_ROOT_ENV} == {
        k: v for k, v in before.items() if k != bc.STATE_ROOT_ENV
    }
