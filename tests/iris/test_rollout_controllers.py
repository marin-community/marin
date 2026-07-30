# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import subprocess
from pathlib import Path

import pytest
from iris.cluster.config import (
    AuthConfig,
    ControllerVmConfig,
    CoreweaveControllerConfig,
    CoreweavePlatformConfig,
    DefaultsConfig,
    GcpControllerConfig,
    IrisClusterConfig,
    PlatformConfig,
    StorageConfig,
)
from iris.rpc import controller_pb2

from scripts.iris.rollout_controllers import (
    ExecutionHealth,
    HealthUnit,
    Need,
    Requirement,
    Snapshot,
    TreeIssue,
    TreeState,
    backend_health,
    check_requirement,
    check_requirements,
    compare,
    deploy_candidates,
    parse_ahead_behind,
    parse_dirty_files,
    read_tree_state,
    requirements,
    rollout_order,
    run_probe,
    tree_warnings,
)


def coreweave_config(*, state_dir: str = "s3://bucket/state", signing_key: list[str] | None = None):
    return IrisClusterConfig(
        platform=PlatformConfig(
            coreweave=CoreweavePlatformConfig(
                kubeconfig_path="~/.kube/coreweave-iris",
                kube_context="marin-gpu_US-EAST-02A",
            )
        ),
        controller=ControllerVmConfig(coreweave=CoreweaveControllerConfig(port=10000)),
        storage=StorageConfig(remote_state_dir=state_dir),
        auth=AuthConfig(signing_key=signing_key) if signing_key else None,
    )


def gcp_config(*, inject_env: list[str] | None = None):
    return IrisClusterConfig(
        controller=ControllerVmConfig(gcp=GcpControllerConfig(zone="us-central1-a")),
        storage=StorageConfig(remote_state_dir="gs://bucket/state"),
        defaults=DefaultsConfig(inject_env=inject_env or []),
    )


def snapshot(**overrides) -> Snapshot:
    base = {
        "cluster": "marin",
        "captured_at": "2026-07-29T00:00:00+00:00",
        "reachable": True,
        "tree_hash": "aaa",
        "version": "aaa (main)",
        "execution_health": (ExecutionHealth(backend_id="worker-daemon", unit=HealthUnit.WORKER, healthy=10, total=10),),
        "jobs": {"running": 4, "pending": 1, "building": 0},
    }
    return Snapshot(**{**base, **overrides})


def health(
    *,
    healthy: int = 10,
    total: int = 10,
    backend_id: str = "worker-daemon",
    unit: HealthUnit = HealthUnit.WORKER,
) -> tuple[ExecutionHealth, ...]:
    return (ExecutionHealth(backend_id=backend_id, unit=unit, healthy=healthy, total=total),)


def test_backend_health_uses_ready_schedulable_kubernetes_nodes():
    backend = controller_pb2.Controller.BackendSummary(
        backend_id="kubernetes",
        running_task_count=40,
        detail=controller_pb2.Controller.BackendStatus(
            kubernetes=controller_pb2.Controller.GetKubernetesClusterStatusResponse(
                total_nodes=3,
                nodes=[
                    controller_pb2.Controller.NodeStatus(name="ready", ready=True, schedulable=True),
                    controller_pb2.Controller.NodeStatus(name="not-ready", ready=False, schedulable=True),
                    controller_pb2.Controller.NodeStatus(name="cordoned", ready=True, schedulable=False),
                ],
            )
        ),
    )

    result = backend_health((backend,))

    assert result == (ExecutionHealth(backend_id="kubernetes", unit=HealthUnit.NODE, healthy=1, total=3),)
    summary = snapshot(execution_health=result, jobs={"running": 40}).summary()
    assert "kubernetes=1/3 healthy nodes" in summary
    assert "workers=" not in summary


def test_backend_health_uses_worker_daemon_liveness():
    backend = controller_pb2.Controller.BackendSummary(
        backend_id="worker-daemon",
        detail=controller_pb2.Controller.BackendStatus(
            worker=controller_pb2.Controller.WorkerFleetDetail(
                healthy_worker_count=8,
                total_worker_count=10,
            )
        ),
    )

    assert backend_health((backend,)) == (
        ExecutionHealth(backend_id="worker-daemon", unit=HealthUnit.WORKER, healthy=8, total=10),
    )


def test_rollout_order_leads_with_dev_then_production_then_smallest_first():
    # A bad deploy must land on the least hardware first.
    order = rollout_order({"cw-big": 216, "marin": 20_000, "cw-small": 2, "cw-mid": 32, "marin-dev": 18_000})
    assert order == ("marin-dev", "marin", "cw-small", "cw-mid", "cw-big")
    assert rollout_order({"cw-b": 2, "cw-a": 2, "marin": 1}) == ("marin", "cw-a", "cw-b")


def test_deploy_candidates_drop_ci_owned_configs():
    configs = {"marin": Path("marin.yaml"), "ci-gcp-smoke": Path("ci-gcp-smoke.yaml")}
    assert set(deploy_candidates(configs)) == {"marin"}


def test_kubernetes_requirements_cover_s3_keys_kubeconfig_and_signing_key():
    config = coreweave_config(signing_key=["env:IRIS_SIGNING_KEY", "gcp-secret://p/secrets/k/versions/1"])
    needs = requirements(config, s3_env=["CW_KEY_ID", "CW_KEY_SECRET"])

    assert {n.target for n in needs if n.kind is Need.ENV} == {"CW_KEY_ID", "CW_KEY_SECRET"}
    context_needs = [n for n in needs if n.kind is Need.KUBE_CONTEXT]
    assert [(n.target, n.source) for n in context_needs] == [("marin-gpu_US-EAST-02A", "~/.kube/coreweave-iris")]
    # The env: reference addresses the pod, so only the persistent source is checked.
    assert [n.target for n in needs if n.kind is Need.SECRET] == ["gcp-secret://p/secrets/k/versions/1"]
    assert "kubectl" in {n.target for n in needs if n.kind is Need.COMMAND}


def test_env_only_signing_key_becomes_an_operator_env_requirement():
    # With no persistent source behind it, the deploy reads the key from this shell.
    needs = requirements(coreweave_config(signing_key=["env:IRIS_SIGNING_KEY"]), s3_env=[])
    assert [n.target for n in needs if n.kind is Need.ENV] == ["IRIS_SIGNING_KEY"]
    assert not [n for n in needs if n.kind is Need.SECRET]


def test_gcs_backed_kubernetes_cluster_needs_no_s3_keys():
    needs = requirements(coreweave_config(state_dir="gs://bucket/state"), s3_env=["CW_KEY_ID", "CW_KEY_SECRET"])
    assert not [n for n in needs if n.kind is Need.ENV]


def test_every_cluster_probes_a_working_docker_build_toolchain():
    # `docker` on PATH passes while the daemon is down, failing after the gate.
    for config in (coreweave_config(), gcp_config()):
        probes = {n.target for n in requirements(config, s3_env=[]) if n.kind is Need.COMMAND_RUNS}
        assert probes == {"docker info", "docker buildx version"}


def test_probe_fails_on_a_nonzero_exit_and_reports_the_exit_code():
    result = run_probe(Requirement(Need.COMMAND_RUNS, "python --no-such-flag", "why"))
    assert not result.ok
    assert "exit 2" in result.detail


def test_probe_fails_when_the_command_is_absent():
    result = run_probe(Requirement(Need.COMMAND_RUNS, "definitely-not-a-real-binary --version", "why"))
    assert not result.ok
    assert "not on PATH" in result.detail


def test_gcp_cluster_needs_gcloud_and_its_injected_env():
    needs = requirements(gcp_config(inject_env=["WANDB_API_KEY"]))
    assert [n.target for n in needs if n.kind is Need.ENV] == ["WANDB_API_KEY"]
    assert "gcloud" in {n.target for n in needs if n.kind is Need.COMMAND}
    assert "kubectl" not in {n.target for n in needs if n.kind is Need.COMMAND}


def test_env_requirement_fails_when_unset_or_empty():
    need = Requirement(Need.ENV, "CW_KEY_ID", "why")
    assert check_requirement(need, environ={"CW_KEY_ID": "abc"}).ok
    assert not check_requirement(need, environ={}).ok
    assert not check_requirement(need, environ={"CW_KEY_ID": ""}).ok


KUBECONFIG_DOC = """
apiVersion: v1
contexts:
  - name: marin-gpu_US-EAST-02A
    context: {cluster: marin-gpu, user: u}
  - name: marin_US-WEST-04A
    context: {cluster: marin, user: u}
"""


def kube_context_need(context: str, kubeconfig: str) -> Requirement:
    return Requirement(Need.KUBE_CONTEXT, context, "why", source=kubeconfig)


def test_kube_context_check_passes_when_the_kubeconfig_defines_it(tmp_path):
    path = tmp_path / "coreweave-iris"
    path.write_text(KUBECONFIG_DOC)
    result = check_requirement(kube_context_need("marin-gpu_US-EAST-02A", str(path)), environ={})
    assert result.ok


def test_kube_context_check_fails_when_the_cluster_context_is_missing(tmp_path):
    path = tmp_path / "coreweave-iris"
    path.write_text(KUBECONFIG_DOC)
    result = check_requirement(kube_context_need("marin-us-east-08a_US-EAST-08A", str(path)), environ={})
    assert not result.ok
    assert "not among the 2 context(s)" in result.detail


def test_kube_context_check_reports_an_absent_kubeconfig(tmp_path):
    result = check_requirement(kube_context_need("any", str(tmp_path / "missing")), environ={})
    assert not result.ok
    assert "kubeconfig is absent" in result.detail


def test_kube_context_check_expands_user_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "kc").write_text(KUBECONFIG_DOC)
    assert check_requirement(kube_context_need("marin_US-WEST-04A", "~/kc"), environ={}).ok


def test_exported_kubeconfig_overrides_the_configured_path(tmp_path):
    # The k8s controller manager drops the configured path when KUBECONFIG is set.
    configured = tmp_path / "configured"
    configured.write_text(KUBECONFIG_DOC)
    exported = tmp_path / "exported"
    exported.write_text("apiVersion: v1\ncontexts: []\n")
    need = kube_context_need("marin_US-WEST-04A", str(configured))
    assert not check_requirement(need, environ={"KUBECONFIG": str(exported)}).ok


def test_a_merged_kubeconfig_list_satisfies_the_context(tmp_path):
    first = tmp_path / "first"
    first.write_text("apiVersion: v1\ncontexts: []\n")
    second = tmp_path / "second"
    second.write_text(KUBECONFIG_DOC)
    need = kube_context_need("marin_US-WEST-04A", str(tmp_path / "configured"))
    environ = {"KUBECONFIG": f"{first}{os.pathsep}{second}"}
    assert check_requirement(need, environ=environ).ok


def test_a_context_without_a_pinned_path_resolves_against_the_default_kubeconfig(tmp_path, monkeypatch):
    # The deploy binds a bare context within kubectl's default resolution.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".kube").mkdir()
    (tmp_path / ".kube" / "config").write_text(KUBECONFIG_DOC)
    assert check_requirement(kube_context_need("marin_US-WEST-04A", ""), environ={}).ok


def test_unresolvable_secret_becomes_a_failed_check_not_an_exception():
    results = check_requirements([Requirement(Need.SECRET, "gcp-secret://nope", "why")], environ={})
    assert [r.ok for r in results] == [False]


def test_verify_flags_a_controller_running_the_wrong_tree():
    verdict = compare(snapshot(tree_hash="aaa"), snapshot(tree_hash="bbb"), expect_tree_hash="ccc")
    assert not verdict.healthy
    # The concern carries both hashes, or the operator cannot tell what shipped.
    assert any("bbb" in concern and "ccc" in concern for concern in verdict.concerns)


def test_verify_flags_lost_workers():
    verdict = compare(
        snapshot(execution_health=health(healthy=10)),
        snapshot(execution_health=health(healthy=6)),
        expect_tree_hash="aaa",
    )
    assert not verdict.healthy
    assert any("10" in concern and "6" in concern for concern in verdict.concerns)


def test_verify_allows_one_lost_worker_as_churn():
    verdict = compare(
        snapshot(execution_health=health(healthy=10)),
        snapshot(execution_health=health(healthy=9)),
        expect_tree_hash="aaa",
    )
    assert verdict.healthy
    assert not verdict.concerns
    assert any("churn" in note and "1" in note for note in verdict.notes)


def test_verify_flags_worker_loss_above_five_percent():
    baseline = snapshot(execution_health=health(healthy=100, total=100))
    permitted = compare(
        baseline,
        snapshot(execution_health=health(healthy=95, total=100)),
        expect_tree_hash="aaa",
    )
    too_many = compare(
        baseline,
        snapshot(execution_health=health(healthy=94, total=100)),
        expect_tree_hash="aaa",
    )
    assert permitted.healthy
    assert not too_many.healthy


def test_verify_blocks_when_a_backend_health_target_disappears():
    verdict = compare(snapshot(), snapshot(execution_health=()), expect_tree_hash="aaa")

    assert not verdict.healthy
    assert any("targets changed" in concern for concern in verdict.concerns)


def test_verify_flags_an_unreachable_controller_before_anything_else():
    verdict = compare(snapshot(), snapshot(reachable=False, error="RpcError: unavailable"), expect_tree_hash="aaa")
    assert not verdict.healthy
    # One concern only: there is no post-restart reading for the other checks.
    assert len(verdict.concerns) == 1
    assert "RpcError: unavailable" in verdict.concerns[0]
    assert not verdict.notes


def test_verify_passes_when_the_expected_tree_came_back_with_its_workers():
    verdict = compare(snapshot(tree_hash="aaa"), snapshot(tree_hash="bbb"), expect_tree_hash="bbb")
    assert verdict.healthy
    assert not verdict.concerns


def test_growing_queue_and_unchanged_tree_are_notes_not_blockers():
    verdict = compare(
        snapshot(jobs={"running": 4, "pending": 1, "building": 0}),
        snapshot(jobs={"running": 4, "pending": 9, "building": 0}),
        expect_tree_hash="aaa",
    )
    assert verdict.healthy
    assert any("pending" in note and "9" in note for note in verdict.notes)
    assert any("unchanged" in note for note in verdict.notes)


def test_snapshot_round_trip_keeps_the_baseline_readable_by_verify():
    before = snapshot()
    assert Snapshot.from_json(before.to_json()) == before


def tree(**overrides) -> TreeState:
    base = {
        "tree_hash": "abc1234",
        "branch": "main",
        "base_commit": "def5678",
        "dirty_files": (),
        "upstream": "origin/main",
        "ahead": 0,
        "behind": 0,
    }
    return TreeState(**{**base, **overrides})


def test_clean_tree_on_upstream_needs_no_confirmation():
    assert tree_warnings(tree()) == ()


def test_each_kind_of_divergence_raises_its_own_warning():
    assert [w.issue for w in tree_warnings(tree(dirty_files=("a.py",)))] == [TreeIssue.DIRTY]
    assert [w.issue for w in tree_warnings(tree(behind=7))] == [TreeIssue.BEHIND]
    assert [w.issue for w in tree_warnings(tree(ahead=2))] == [TreeIssue.AHEAD]
    assert [w.issue for w in tree_warnings(tree(dirty_files=("a.py",), ahead=2, behind=7))] == [
        TreeIssue.DIRTY,
        TreeIssue.BEHIND,
        TreeIssue.AHEAD,
    ]


def test_dirty_warning_names_the_files_and_counts_the_rest():
    # The operator decides from this message, so "tree is dirty" is not enough.
    warning = tree_warnings(tree(dirty_files=tuple(f"f{index}.py" for index in range(9))))[0]
    assert "9" in warning.message
    assert "f0.py" in warning.message
    assert "(+4 more)" in warning.message


def test_untracked_files_stay_dirty_when_the_operator_git_config_hides_them(tmp_path, monkeypatch):
    # Regression: status.showUntrackedFiles=no suppressed the `??` lines, so preflight
    # called the tree clean while the image build still copied the file in.
    repo = tmp_path / "repo"
    repo.mkdir()
    run = functools.partial(subprocess.run, cwd=repo, check=True, capture_output=True)
    run(["git", "init", "--initial-branch=main", "--quiet"])
    run(["git", "config", "user.email", "test@example.com"])
    run(["git", "config", "user.name", "Test"])
    run(["git", "config", "status.showUntrackedFiles", "no"])
    (repo / "tracked.py").write_text("x = 1\n")
    run(["git", "add", "tracked.py"])
    run(["git", "commit", "--quiet", "-m", "init"])
    run(["git", "branch", "base"])
    (repo / "untracked.py").write_text("y = 2\n")

    monkeypatch.chdir(repo)
    state = read_tree_state(upstream="base", fetch=False)
    assert state.dirty_files == ("untracked.py",)


def test_parse_dirty_files_strips_status_codes():
    porcelain = ' M lib/iris/OPS.md\n?? new.py\nR  old.py -> new.py\n?? "docs/a b.md"\n\n'
    assert parse_dirty_files(porcelain) == ("lib/iris/OPS.md", "new.py", "old.py -> new.py", '"docs/a b.md"')


def test_parse_dirty_files_survives_a_stripped_leading_status_column():
    # Regression: the git helper strips the output, so " M uv.lock" loses its column.
    assert parse_dirty_files("M uv.lock\n?? new.py") == ("uv.lock", "new.py")


def test_parse_ahead_behind_maps_left_to_behind_and_right_to_ahead():
    # `git rev-list --left-right --count origin/main...HEAD` prints "<behind>\t<ahead>".
    assert parse_ahead_behind("7\t2") == (2, 7)
    assert parse_ahead_behind("0\t0") == (0, 0)


def test_parse_ahead_behind_rejects_unexpected_output():
    with pytest.raises(ValueError):
        parse_ahead_behind("3")
