# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The MiMo backend repair: pinned images, the source's own grader, and typed refusal."""

import base64
import hashlib
import json
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.models import ScriptGrader

from experiments.post_training.task_curation.datasets.mimo.repair import (
    ImageBinding,
    RepairRejected,
    exclusions_from_document,
    load_bindings,
    parse_recipe,
    repair_population,
    repair_task,
    verifier_reward_execute,
)

DOCKERFILE = (
    "FROM docker.io/xiaomimimo/mimo-v2.6-rl-oss@sha256:" + "a" * 64 + "\n"
    "WORKDIR /testbed\n"
    "RUN install-tmux\n"
    "# ot-validation: repository-leak-repair-v1 (generated; replaced, never stacked)\n"
    "RUN repair-repository\n"
)

HEALTHCHECK = "git config --global --add safe.directory /testbed; cd /testbed && git rev-parse --verify HEAD"

VERIFIER_FILES = {
    "tests/reset_test_files.sh": ("0755", b"reset"),
    "tests/setup.sh": ("0755", b"setup"),
    "tests/test.patch": ("0644", b"patch"),
    "tests/test.sh": ("0755", b"test"),
    "tests/touched_paths.txt": ("0644", b"touched"),
}

TASK_TOML = (
    'version = "1.0"\n[agent]\ntimeout_sec = 1800\n[verifier]\ntimeout_sec = 1800\n'
    "[environment]\ncpus = 4\nmemory_mb = 4096\nallow_internet = true\n"
    f'[environment.healthcheck]\ncommand = "{HEALTHCHECK}"\n'
).encode()


def resource(path: str, content: bytes, mode: str = "0644") -> dict:
    return {
        "path": path,
        "source": {"kind": "inline_file", "content_base64": base64.b64encode(content).decode()},
        "mode": mode,
        "mtime_ns": 0,
    }


def retained_row(task_id: str) -> dict:
    verifier = [
        resource("environment/Dockerfile", DOCKERFILE.encode()),
        resource("instruction.md", b"work the repo"),
        resource("task.toml", TASK_TOML),
        *(resource(path, content, mode) for path, (mode, content) in VERIFIER_FILES.items()),
    ]
    task = {
        "id": task_id,
        "source": {
            "dataset": "open-athena/pin",
            "revision": "0" * 40,
            "row": "tasks.parquet:0",
            "importer_revision": "2026.11.13.2",
        },
        "context": {"events": [{"type": "message", "role": "user", "content": "work the repo"}]},
        "environment_requirements": {
            "capabilities": [],
            "command_semantics": "linux_process",
            "docker_image": None,
            "docker_build": {"files": [resource("Dockerfile", DOCKERFILE.encode())]},
            "working_directory": None,
            "setup_commands": [],
            "environment_variables": {},
            "tool_providers": {},
            "packages_lock": None,
            "package_spec": None,
        },
        "answer_type": "workspace_state",
        "answer_format": {"kind": "plain_text"},
        "grader": {"kind": "none", "reason": "The native Harbor Trial executes this package's declared verifier"},
        "resources": {"all": [], "worker": [], "oracle": [], "verifier": verifier},
        "tags": ["harbor"],
        "labels": {"native_import_version": "2026.11.13.2"},
        "schema_version": "0.30",
    }
    serialized = json.dumps(task)
    return {
        "id": task_id,
        "path": "mimo/tasks/task",
        "task_spec": serialized,
        "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "runtime": json.dumps(
            {"task_machine": {"cpus": 4, "memory_mb": 4096}, "verifier_machine": {"cpus": 4, "memory_mb": 4096}}
        ),
    }


def binding_for(row: dict, digest: str = "sha256:" + "b" * 64) -> ImageBinding:
    task = json.loads(row["task_spec"])
    dockerfile = base64.b64decode(
        task["environment_requirements"]["docker_build"]["files"][0]["source"]["content_base64"]
    )
    recipe = parse_recipe(dockerfile.decode())
    return ImageBinding(
        task_id=row["id"],
        image=f"ghcr.io/marin-community/rl-data-mimo-prepared@{digest}",
        digest=digest,
        base_digest=recipe.base_digest,
        recipe_sha256=hashlib.sha256(dockerfile).hexdigest(),
    )


def test_parse_recipe_accepts_the_audited_shape_and_refuses_other_directives():
    recipe = parse_recipe(DOCKERFILE)
    assert recipe.base_digest == "sha256:" + "a" * 64
    assert recipe.workdir == "/testbed"
    assert recipe.run_commands == ("install-tmux", "repair-repository")
    with pytest.raises(RepairRejected, match="does not pin the expected upstream base"):
        parse_recipe(DOCKERFILE.replace("docker.io/xiaomimimo/mimo-v2.6-rl-oss", "evil.example/repo"))
    with pytest.raises(RepairRejected, match="exactly one WORKDIR"):
        parse_recipe(DOCKERFILE.replace("WORKDIR /testbed\n", ""))
    with pytest.raises(RepairRejected, match="unmeasured directives"):
        parse_recipe(DOCKERFILE + "USER nobody\n")


def test_repair_pins_the_image_and_declares_the_source_grader():
    row = retained_row("fweindel-validated-tasks-mimo-v1-" + "c" * 64)
    binding = binding_for(row)
    repaired, ledger = repair_task(row, binding)
    environment = repaired.environment_requirements
    assert environment.docker_build is None
    assert environment.docker_image == binding.image
    assert environment.working_directory == "/testbed"
    assert len(environment.setup_commands) == 1 and HEALTHCHECK in environment.setup_commands[0]
    grader = repaired.grader
    assert isinstance(grader, ScriptGrader)
    assert grader.environment.docker_image == binding.image and grader.environment.docker_build is None
    assert grader.environment.working_directory == "/"
    assert grader.timeout == 1800.0
    assert grader.reward.files[0].path == "/logs/verifier/reward.txt"
    # quiesce, archive delivery chunks, restore, then the source's own pipeline
    assert grader.collect[0].argv[0] == "bash"
    assert grader.collect[-1].argv[:2] == ("bash", "-c")
    execute = grader.collect[-1].argv[2]
    assert "bash /tests/setup.sh" in execute and "bash /tests/test.sh" in execute
    assert "produced no reward file" in execute and "exit 7" in execute
    contract = repaired.integrity_contract
    assert contract.dependency_state.value == "agent_mutable"
    assert {file.name for file in contract.verifier_files} == {
        "reset_test_files.sh",
        "setup.sh",
        "test.patch",
        "test.sh",
        "touched_paths.txt",
    }
    for file in contract.verifier_files:
        content = dict((path, body) for path, (_, body) in VERIFIER_FILES.items())["tests/" + file.name]
        assert file.sha256 == hashlib.sha256(content).hexdigest()
    assert repaired.resources.verifier[0].path == "source_archive/environment/Dockerfile"
    assert repaired.resources.worker == ()
    assert repaired.labels["mimo_repair_version"] and repaired.labels["mimo_pinned_digest"] == binding.digest
    assert ledger["changed_fields"] == sorted(
        {
            "environment_requirements",
            "grader",
            "integrity_contract",
            "resources",
            "tags",
            "labels",
        }
    )


def test_repair_refuses_a_binding_built_from_other_bytes():
    row = retained_row("fweindel-validated-tasks-mimo-v1-" + "d" * 64)
    binding = binding_for(row)
    other = ImageBinding(
        task_id=binding.task_id,
        image=binding.image,
        digest=binding.digest,
        base_digest="sha256:" + "e" * 64,
        recipe_sha256=binding.recipe_sha256,
    )
    with pytest.raises(RepairRejected, match="own base digest"):
        repair_task(row, other)
    wrong_recipe = ImageBinding(
        task_id=binding.task_id,
        image=binding.image,
        digest=binding.digest,
        base_digest=binding.base_digest,
        recipe_sha256="f" * 64,
    )
    with pytest.raises(RepairRejected, match="own Dockerfile bytes"):
        repair_task(row, wrong_recipe)


def test_repair_refuses_tasks_that_do_not_carry_the_unresolved_context():
    row = retained_row("fweindel-validated-tasks-mimo-v1-" + "5" * 64)
    binding = binding_for(row)
    task = json.loads(row["task_spec"])
    task["environment_requirements"]["docker_build"] = None
    task["environment_requirements"]["docker_image"] = "ghcr.io/marin-community/some@sha256:" + "1" * 64
    row["task_spec"] = json.dumps(task)
    with pytest.raises(RepairRejected, match="expected unresolved Docker build context"):
        repair_task(row, binding)


def test_bindings_document_is_validated(tmp_path):
    row = retained_row("fweindel-validated-tasks-mimo-v1-" + "6" * 64)
    binding = binding_for(row)
    document = {"images": [binding.__dict__]}
    path = tmp_path / "bindings.json"
    path.write_text(json.dumps(document))
    assert load_bindings(path)[binding.task_id] == binding
    bad = dict(binding.__dict__, image="docker.io/evil/repo@" + binding.digest)
    path.write_text(json.dumps({"images": [bad]}))
    with pytest.raises(RepairRejected, match="not digest-pinned"):
        load_bindings(path)
    unprefixed = dict(binding.__dict__, digest="b" * 64)
    path.write_text(json.dumps({"images": [unprefixed]}))
    with pytest.raises(RepairRejected, match="not digest-pinned"):
        load_bindings(path)


def test_population_repair_covers_every_task_and_writes_the_ledger(tmp_path):
    rows = [retained_row("fweindel-validated-tasks-mimo-v1-" + f"{index:064x}") for index in range(3)]
    input_parquet = tmp_path / "input.parquet"
    table = pa.Table.from_pylist(rows)
    with input_parquet.open("wb") as stream:
        with pq.ParquetWriter(stream, table.schema) as writer:
            writer.write_table(table)
    bindings = {row["id"]: binding_for(row) for row in rows}
    summary = repair_population(input_parquet, bindings, tmp_path / "out", revision="landing-rev")
    assert summary["tasks"] == 3
    ledger = json.loads((tmp_path / "out" / "repair-ledger.json").read_text())
    assert len(ledger["tasks"]) == 3
    with pytest.raises(RepairRejected, match="omit task"):
        repair_population(
            input_parquet, {rows[0]["id"]: bindings[rows[0]["id"]]}, tmp_path / "out2", revision="landing-rev"
        )


def test_population_repair_excludes_typed_tasks(tmp_path):
    rows = [retained_row("fweindel-validated-tasks-mimo-v1-" + f"{index:064x}") for index in range(3)]
    input_parquet = tmp_path / "input.parquet"
    table = pa.Table.from_pylist(rows)
    with input_parquet.open("wb") as stream:
        with pq.ParquetWriter(stream, table.schema) as writer:
            writer.write_table(table)
    bindings = {row["id"]: binding_for(row) for row in rows}
    exclusions = exclusions_from_document(
        {
            "tasks": [
                {
                    "task_id": rows[0]["id"],
                    "reason": "backend_contract_unsupported",
                    "evidence": "the declared memory exceeds every production rollout backend",
                }
            ]
        }
    )
    summary = repair_population(input_parquet, bindings, tmp_path / "out", exclusions, revision="landing-rev")
    assert summary["tasks"] == 2 and summary["excluded_tasks"] == 1
    ledger = json.loads((tmp_path / "out" / "repair-ledger.json").read_text())
    assert ledger["excluded_tasks"] == [
        {
            "task_id": rows[0]["id"],
            "reason": "backend_contract_unsupported",
            "evidence": "the declared memory exceeds every production rollout backend",
        }
    ]
    output_rows = pq.ParquetFile(tmp_path / "out" / "tasks.parquet").metadata.num_rows
    assert output_rows == 2
    with pytest.raises(RepairRejected, match="outside the population"):
        repair_population(
            input_parquet,
            bindings,
            tmp_path / "out2",
            exclusions_from_document({"tasks": [{"task_id": "missing", "reason": "r", "evidence": "e"}]}),
            revision="landing-rev",
        )


def subprocess_completed(script: str) -> int:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode


def test_execute_command_fails_closed_without_a_reward_file(tmp_path):
    """The graded pipeline keeps upstream's explicit zero and fails when no reward exists."""
    script = verifier_reward_execute().argv[2]
    sandbox = tmp_path / "root"
    (sandbox / "tests").mkdir(parents=True)
    (sandbox / "logs" / "verifier").mkdir(parents=True)
    # The command uses absolute paths; run it through a bash wrapper that redirects them.
    wrapper = script.replace("/tests", f"{sandbox}/tests").replace("/logs/verifier", f"{sandbox}/logs/verifier")
    setup = sandbox / "tests" / "setup.sh"
    test_script = sandbox / "tests" / "test.sh"

    # Upstream's fail() writes its own explicit zero when the testbed cannot be reset or
    # patched; that zero is retained and the grading command succeeds.
    setup.write_text("echo reset-failed >&2\nprintf '0\\n' > " + f"{sandbox}/logs/verifier/reward.txt\nexit 3\n")
    assert subprocess_completed(wrapper) == 0
    assert (sandbox / "logs" / "verifier" / "reward.txt").read_text() == "0\n"

    # A setup failure without the explicit zero fails the grading command.
    setup.write_text("echo reset-failed >&2\nexit 3\n")
    (sandbox / "logs" / "verifier" / "reward.txt").unlink()
    assert subprocess_completed(wrapper) == 3

    # A passing pipeline derives the numeric reward from the upstream reward.json.
    setup.write_text("exit 0\n")
    test_script.write_text("printf '{\"reward\": 1.0}\\n' > " + f"{sandbox}/logs/verifier/reward.json\n")
    assert subprocess_completed(wrapper) == 0
    assert (sandbox / "logs" / "verifier" / "reward.txt").read_text() == "1.0\n"

    # A verifier that produces nothing fails closed.
    test_script.write_text("exit 0\n")
    (sandbox / "logs" / "verifier" / "reward.json").unlink(missing_ok=True)
    assert subprocess_completed(wrapper) == 7
    assert not (sandbox / "logs" / "verifier" / "reward.txt").exists()
