import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import capability_pipeline.runtime as runtime_module
from capability_pipeline.daytona_policy import (
    verifier_bootstrap_sha256,
    verifier_snapshot_recipe,
)
from capability_pipeline.daytona_resources import VERIFIER_DEFAULT, snapshot_name
from capability_pipeline.runtime import (
    authored_oracle_plan,
    authored_replay_agent_kwargs,
    authored_replay_inventory,
    control_replay_commands,
    control_replay_inventory,
    expected_extraction_transport,
    extract_json,
    normalize_openai_base,
    persist_attack_report,
    provider_isolation_record,
    runtime_gate_issues,
    sha256,
    tree_sha256,
    verifier_isolation_record,
)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"cpu": True, "memory_gb": 2, "disk_gb": 10},
        {"cpu": 2, "memory_gb": 0, "disk_gb": 10},
    ],
)
def test_candidate_resource_file_rejects_implicit_or_invalid_requests(tmp_path, value):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runtime_module.load_candidate_resources(path)


def test_candidate_resource_file_preserves_units_and_rejects_links(tmp_path):
    path = tmp_path / "resources.json"
    value = {"cpu": 2, "memory_gb": 2, "disk_gb": 10}
    path.write_text(json.dumps(value))
    assert runtime_module.load_candidate_resources(path) == value
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular JSON"):
        runtime_module.load_candidate_resources(link)


def test_control_workspace_replay_materializes_exact_files(tmp_path):
    source = tmp_path / "task" / "controls" / "positive_reference"
    (source / "out").mkdir(parents=True)
    artifact = source / "out" / "cleaned.gpkg"
    artifact.write_bytes(b"gpkg\x00fixed")
    script = source / "check.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    case = {
        "workspace": "controls/positive_reference",
        "commands": ["test -f out/cleaned.gpkg"],
    }

    commands = control_replay_commands(tmp_path / "task", case)
    assert any("if [ -L ./out/cleaned.gpkg ]; then exit 1; fi" in command for command in commands)
    assert all("[ ! -L" not in command for command in commands)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    for command in commands:
        subprocess.run(
            command, cwd=sandbox, shell=True, check=True, executable="/bin/bash"
        )

    assert (sandbox / "out" / "cleaned.gpkg").read_bytes() == b"gpkg\x00fixed"
    assert os.access(sandbox / "check.sh", os.X_OK)
    inventory = control_replay_inventory(tmp_path / "task", case)
    assert inventory["workspace"] == "controls/positive_reference"
    assert inventory["action"] == "materialize_declared_fixed_submission"
    assert inventory["transport"] == "trusted_replay_base64_sha256"
    assert inventory["materialization_action_count"] == 7
    assert inventory["directories"] == ["out"]
    assert [item["path"] for item in inventory["files"]] == [
        "check.sh",
        "out/cleaned.gpkg",
    ]
    assert all(len(item["sha256"]) == 64 for item in inventory["files"])


def test_control_workspace_replay_chunks_large_binary_for_shell_exec(tmp_path):
    source = tmp_path / "task" / "controls" / "large"
    source.mkdir(parents=True)
    payload = bytes(range(256)) * 801
    (source / "artifact.bin").write_bytes(payload)

    commands = control_replay_commands(
        tmp_path / "task", {"workspace": "controls/large"}
    )

    assert max(map(len, commands)) < 34 * 1024
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    for command in commands:
        subprocess.run(
            command, cwd=sandbox, shell=True, check=True, executable="/bin/bash"
        )
    assert (sandbox / "artifact.bin").read_bytes() == payload


def test_control_workspace_replay_preserves_empty_directories(tmp_path):
    source = tmp_path / "task" / "controls" / "empty"
    (source / "one" / "two").mkdir(parents=True)
    case = {"workspace": "controls/empty"}

    commands = control_replay_commands(tmp_path / "task", case)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    for command in commands:
        subprocess.run(
            command, cwd=sandbox, shell=True, check=True, executable="/bin/bash"
        )

    assert (sandbox / "one" / "two").is_dir()
    inventory = control_replay_inventory(tmp_path / "task", case)
    assert inventory["directories"] == ["one", "one/two"]
    assert inventory["files"] == []
    assert inventory["materialization_action_count"] == 2


@pytest.mark.parametrize("declared", ["/absolute", "../escape", "controls/./case"])
def test_control_workspace_replay_rejects_unsafe_declared_path(tmp_path, declared):
    with pytest.raises(RuntimeError, match="unsafe"):
        control_replay_commands(tmp_path, {"workspace": declared})


def test_control_workspace_replay_rejects_source_symlink_ancestors(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    (task / "controls").symlink_to(outside, target_is_directory=True)
    (outside / "case").mkdir()
    with pytest.raises(RuntimeError, match="traverse a symlink"):
        control_replay_commands(task, {"workspace": "controls/case"})


def test_control_workspace_replay_rejects_symlinks_and_sensitive_paths(tmp_path):
    source = tmp_path / "controls" / "case"
    source.mkdir(parents=True)
    (source / "private").mkdir()
    (source / "private" / "answer").write_text("hidden")
    with pytest.raises(RuntimeError, match="private or credential"):
        control_replay_commands(tmp_path, {"workspace": "controls/case"})
    (source / "private" / "answer").unlink()
    (source / "private").rmdir()
    (source / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="contain symlinks"):
        control_replay_commands(tmp_path, {"workspace": "controls/case"})


def test_control_workspace_replay_rejects_remote_symlink_parent(tmp_path):
    source = tmp_path / "task" / "controls" / "case" / "out"
    source.mkdir(parents=True)
    (source / "answer").write_text("fixed")
    commands = control_replay_commands(
        tmp_path / "task", {"workspace": "controls/case"}
    )
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "outside").mkdir()
    (sandbox / "out").symlink_to(sandbox / "outside", target_is_directory=True)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            commands[0], cwd=sandbox, shell=True, check=True, executable="/bin/bash"
        )
    assert not (sandbox / "outside" / "answer").exists()


def test_control_workspace_replay_rejects_remote_symlink_destination(tmp_path):
    source = tmp_path / "task" / "controls" / "case"
    source.mkdir(parents=True)
    (source / "answer").write_text("fixed")
    commands = control_replay_commands(
        tmp_path / "task", {"workspace": "controls/case"}
    )
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = sandbox / "outside"
    outside.write_text("preserve")
    (sandbox / "answer").symlink_to(outside)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            commands[0], cwd=sandbox, shell=True, check=True, executable="/bin/bash"
        )
    assert outside.read_text() == "preserve"


def test_authored_replay_preserves_multistep_workspace_semantics(tmp_path):
    for name in ("first", "target"):
        source = tmp_path / "controls" / name
        source.mkdir(parents=True)
        (source / f"{name}.txt").write_text(name)
    first = {"workspace": "controls/first", "response": "first"}
    target = {
        "workspace": "controls/target",
        "response": "target",
        "step_index": 1,
    }
    kwargs = authored_replay_agent_kwargs(
        target,
        {0: first, 1: target},
        2,
        tmp_path,
        terminal_available=True,
        workspace_staging_available=True,
    )
    assert [step["response"] for step in kwargs["steps"]] == ["first", "target"]
    assert "first.txt" in kwargs["steps"][0]["commands"][0]
    assert "target.txt" in kwargs["steps"][1]["commands"][0]
    provenance = authored_replay_inventory(target, {0: first, 1: target}, 2, tmp_path)
    assert [(item["step_index"], item["workspace"]) for item in provenance] == [
        (0, "controls/first"),
        (1, "controls/target"),
    ]


def test_shellsim_control_workspace_uses_direct_staging_before_actions(tmp_path):
    source = tmp_path / "controls" / "reference"
    source.mkdir(parents=True)
    (source / "answer.txt").write_bytes(b"answer\n")
    kwargs = authored_replay_agent_kwargs(
        {"workspace": "controls/reference", "response": "done", "commands": ["pwd"]},
        {}, 1, tmp_path, terminal_available=True,
        workspace_staging_available=True, direct_workspace_staging=True,
    )
    assert kwargs["commands"] == ["pwd"]
    assert kwargs["trusted_workspace"]["files"][0]["path"] == "answer.txt"
    assert kwargs["trusted_workspace"]["files"][0]["size"] == 7


def test_authored_replay_fails_closed_for_unsupported_surfaces(tmp_path):
    with pytest.raises(RuntimeError, match="Docker or ShellSim terminal"):
        authored_replay_agent_kwargs(
            {"workspace": "controls/case"},
            {},
            1,
            tmp_path,
            terminal_available=True,
            workspace_staging_available=False,
        )
    with pytest.raises(RuntimeError, match="transcript"):
        authored_replay_agent_kwargs(
            {"transcript": []},
            {},
            1,
            tmp_path,
            terminal_available=False,
            workspace_staging_available=False,
        )
    with pytest.raises(RuntimeError, match="require a terminal"):
        authored_replay_agent_kwargs(
            {"commands": ["true"]},
            {},
            1,
            tmp_path,
            terminal_available=False,
            workspace_staging_available=False,
        )


def test_authored_oracle_plan_retains_two_positives_for_the_same_step():
    first = {"id": "step-0-reference", "class": "positive", "step_index": 0}
    variant = {"id": "step-0-variant", "class": "positive", "step_index": 0}
    step_one = {"id": "step-1-reference", "class": "positive", "step_index": 1}
    controls = {
        "cases": [
            first,
            {"id": "wrong", "class": "negative", "step_index": 0},
            variant,
            step_one,
        ]
    }

    positives, seeds = authored_oracle_plan(controls, 2)

    assert positives == [first, variant, step_one]
    assert seeds == {0: first, 1: step_one}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://relay.example", "https://relay.example/v1"),
        ("https://relay.example/", "https://relay.example/v1"),
        ("https://relay.example/v1", "https://relay.example/v1"),
        ("https://relay.example/v1/", "https://relay.example/v1"),
    ],
)
def test_openai_base_normalization(value, expected):
    assert normalize_openai_base(value) == expected


def test_extract_solver_json_ignores_non_json_prefix():
    assert extract_json('analysis omitted\n{"response":"42"}\n') == {"response": "42"}


def test_persist_attack_report_retains_composite_private_annotations(tmp_path):
    attack = {
        "artifact": "independent-adversary.json",
        "artifact_sha256": "stale",
        "report": {
            "cases": [
                {
                    "steps": [
                        {
                            "private_verifier": {
                                "config_sha256": "c" * 64,
                                "machine_checks": [{"sandbox_id": "fresh-private"}],
                            }
                        }
                    ]
                }
            ]
        },
    }
    persist_attack_report(attack, tmp_path)
    artifact = tmp_path / attack["artifact"]
    assert json.loads(artifact.read_text()) == attack["report"]
    assert attack["artifact_sha256"] == sha256(artifact)


def test_runtime_gate_rejects_incomplete_independent_attack():
    controls = {
        "cases": [
            {
                "id": "positive",
                "class": "positive",
                "category": "known_correct",
                "source_author": "author",
            }
        ]
    }
    evidence = {
        "attestation": {
            "oracle": {"authored": True},
            "solver": {"state": "passed"},
            "adversarial": {
                "authored_controls_executed": True,
                "independent_attack_executed": False,
            },
        },
        "cases": [
            {
                "id": "positive",
                "category": "known_correct",
                "source_author": "author",
                "control_type": "independent_solver",
                "result": {"status": "graded", "reward": 1.0},
            }
        ],
    }
    assert runtime_gate_issues(controls, evidence) == [
        "independent attack suite needs adjudication or retry"
    ]
    evidence["attestation"]["adversarial"]["independent_attack_executed"] = True
    assert runtime_gate_issues(controls, evidence) == []


def test_expected_extraction_transport_is_strict():
    case = {"expect": {"status": "extraction_error"}}
    result = {"status": "extraction_error", "reward": None}
    trial = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type="ExtractionError"),
        verifier_result=None,
    )
    assert expected_extraction_transport(case, result, trial)
    assert not expected_extraction_transport(
        case,
        {"status": "extraction_error", "reward": 0.0},
        trial,
    )
    assert not expected_extraction_transport(
        case,
        result,
        SimpleNamespace(
            exception_info=SimpleNamespace(exception_type="RuntimeError"),
            verifier_result=None,
        ),
    )


def test_package_tree_hash_binds_paths_and_bytes(tmp_path):
    (tmp_path / "a").write_text("one")
    first = tree_sha256(tmp_path)
    (tmp_path / "a").write_text("two")
    assert tree_sha256(tmp_path) != first
    (tmp_path / "a").rename(tmp_path / "b")
    assert tree_sha256(tmp_path) != first


def test_daytona_provider_record_binds_enforced_network_and_fresh_sandbox(tmp_path):
    first = tmp_path / "first.json"
    first.write_text(
        json.dumps(
            {
                "adapter": "taskcompendium-daytona",
                "daytona_sdk_version": "0.200.2",
                "image": "registry/image@sha256:" + "a" * 64,
                "network_block_all": True,
                "sandbox_id": "sandbox-1",
                "snapshot": "snapshot-1",
            }
        )
    )
    seen = set()
    record = provider_isolation_record(first, tmp_path, seen)
    assert record["sandbox_id"] == "sandbox-1"
    assert len(record["provider_artifact_sha256"]) == 64
    with pytest.raises(RuntimeError, match="fresh sandboxes"):
        provider_isolation_record(first, tmp_path, seen)


def test_daytona_provider_record_rejects_unenforced_network(tmp_path):
    artifact = tmp_path / "provider.json"
    artifact.write_text(
        json.dumps(
            {
                "adapter": "taskcompendium-daytona",
                "daytona_sdk_version": "0.200.2",
                "image": "registry/image@sha256:" + "a" * 64,
                "network_block_all": False,
                "sandbox_id": "sandbox-1",
                "snapshot": "snapshot-1",
            }
        )
    )
    with pytest.raises(RuntimeError, match="enforced isolation"):
        provider_isolation_record(artifact, tmp_path, set())


def test_daytona_private_verifier_record_is_bound_and_fresh():
    adapter = sha256(Path(runtime_module.__file__).with_name("daytona_verifier.py"))
    result = {
        "status": "graded",
        "reward": 1.0,
        "detail": {
            "verifier_isolation": "daytona-network-block-all",
            "verifier_sandbox_id": "verifier-1",
            "verifier_snapshot": "snapshot-1",
            "verifier_adapter_sha256": adapter,
            "verifier_bootstrap_sha256": verifier_bootstrap_sha256(),
        },
    }
    seen = set()
    record = verifier_isolation_record(result, seen, {"candidate-1"})
    assert record["sandbox_id"] == "verifier-1"
    cleanup = {
        "state": "deleted",
        "attempts": [{"state": "requested"}],
        "observations": [{"state": "not_found"}],
    }
    with_cleanup = {
        **result,
        "detail": {**result["detail"], "verifier_cleanup": cleanup},
    }
    assert verifier_isolation_record(with_cleanup, set(), set())["cleanup"] == cleanup
    for incomplete in (
        None,
        {},
        {**cleanup, "state": "unconfirmed"},
        {**cleanup, "observations": []},
    ):
        with_cleanup["detail"]["verifier_cleanup"] = incomplete
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
            verifier_isolation_record(with_cleanup, set(), set())
    with pytest.raises(RuntimeError, match="reused"):
        verifier_isolation_record(result, seen, {"candidate-1"})
    with pytest.raises(RuntimeError, match="reused"):
        verifier_isolation_record(
            {
                **result,
                "detail": {
                    **result["detail"],
                    "verifier_sandbox_id": "candidate-1",
                },
            },
            set(),
            {"candidate-1"},
        )


def test_daytona_private_verifier_record_binds_nondefault_supervisor_recipe():
    image = "registry.example/c32/verifier@sha256:" + "a" * 64
    supervisor = "/opt/py312/bin/python3"
    recipe_sha256 = (
        __import__("hashlib")
        .sha256(verifier_snapshot_recipe(image, supervisor).encode())
        .hexdigest()
    )
    result = {
        "detail": {
            "verifier_isolation": "daytona-network-block-all",
            "verifier_sandbox_id": "verifier-312",
            "verifier_snapshot": snapshot_name(
                "cap-verifier",
                verifier_snapshot_recipe(image, supervisor),
                VERIFIER_DEFAULT,
            ),
            "verifier_adapter_sha256": sha256(
                Path(runtime_module.__file__).with_name("daytona_verifier.py")
            ),
            "verifier_bootstrap_sha256": verifier_bootstrap_sha256(),
            "verifier_bootstrap_command_sha256": verifier_bootstrap_sha256(supervisor),
            "verifier_snapshot_recipe_sha256": recipe_sha256,
            "verifier_requested_resource_profile": VERIFIER_DEFAULT.receipt(),
            "verifier_supervisor_python": supervisor,
        }
    }

    record = verifier_isolation_record(
        result,
        set(),
        set(),
        runtime_image=image,
        supervisor_python=supervisor,
    )

    assert record["supervisor_python"] == supervisor
    assert record["snapshot_recipe_sha256"] == recipe_sha256


def test_daytona_verifier_legacy_recipe_name_requires_profile_be_absent():
    image = "registry.example/c32/verifier@sha256:" + "a" * 64
    supervisor = "/opt/py312/bin/python3"
    recipe = verifier_snapshot_recipe(image, supervisor)
    recipe_sha256 = __import__("hashlib").sha256(recipe.encode()).hexdigest()
    result = {
        "detail": {
            "verifier_isolation": "daytona-network-block-all",
            "verifier_sandbox_id": "legacy-verifier",
            "verifier_snapshot": f"cap-verifier-{recipe_sha256[:20]}",
            "verifier_adapter_sha256": sha256(
                Path(runtime_module.__file__).with_name("daytona_verifier.py")
            ),
            "verifier_bootstrap_sha256": verifier_bootstrap_sha256(),
            "verifier_bootstrap_command_sha256": verifier_bootstrap_sha256(supervisor),
            "verifier_snapshot_recipe_sha256": recipe_sha256,
            "verifier_supervisor_python": supervisor,
        }
    }
    record = verifier_isolation_record(
        result, set(), set(), runtime_image=image, supervisor_python=supervisor
    )
    assert record["snapshot"] == f"cap-verifier-{recipe_sha256[:20]}"


@pytest.mark.parametrize("bind_runtime_image", [True, False])
def test_daytona_verifier_present_malformed_resource_profile_never_uses_legacy_name(
    bind_runtime_image,
):
    image = "registry.example/c32/verifier@sha256:" + "a" * 64
    supervisor = "/opt/py312/bin/python3"
    recipe_sha256 = (
        __import__("hashlib")
        .sha256(verifier_snapshot_recipe(image, supervisor).encode())
        .hexdigest()
    )
    result = {
        "detail": {
            "verifier_isolation": "daytona-network-block-all",
            "verifier_sandbox_id": "verifier-312",
            "verifier_snapshot": f"cap-verifier-{recipe_sha256[:20]}",
            "verifier_adapter_sha256": sha256(
                Path(runtime_module.__file__).with_name("daytona_verifier.py")
            ),
            "verifier_bootstrap_sha256": verifier_bootstrap_sha256(),
            "verifier_bootstrap_command_sha256": verifier_bootstrap_sha256(supervisor),
            "verifier_snapshot_recipe_sha256": recipe_sha256,
            "verifier_requested_resource_profile": {"cpu": 2},
            "verifier_supervisor_python": supervisor,
        }
    }
    with pytest.raises(RuntimeError, match="malformed verifier resource profile"):
        verifier_isolation_record(
            result,
            set(),
            set(),
            runtime_image=image if bind_runtime_image else None,
            supervisor_python=supervisor,
        )
