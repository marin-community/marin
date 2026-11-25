# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path

import pytest
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import NetworkPolicy
from taskcompendium.evaluation.corpus import evaluation_cases, evaluation_protocol
from taskcompendium.evaluation.spec import AgentEpisode, EpisodeInputs, PiAgent, TerminusAgent
from taskcompendium.importers.harbor.package import JobSettings
from taskcompendium.models import AgentMachine, ScriptGrader, Source, TaskSpec, VerifyitGrader
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow

import experiments.evaluation.benchmarks.harbor as harbor_module
import experiments.post_training.task_curation.harbor_packages as packages_module
from experiments.evaluation.benchmarks.agentic import BIXBENCH, FINANCEAGENT
from experiments.evaluation.benchmarks.harbor import (
    NODE_VERSION,
    HarborConversion,
    HarborDataset,
    PackageEdit,
    edit_package,
    harbor_source,
    huggingface_packages,
    image_lock,
    package_tests,
    pi_install,
)
from experiments.post_training.task_curation.harbor_packages import (
    extract_package,
    package_archive,
    stage_archive_packages,
    stage_repository_packages,
)

IMAGE = "alexgshaw/fixture:20251031"
DIGEST = "sha256:" + "1" * 64
REVISION = "a" * 40
BIXBENCH_PACKAGE = Path(__file__).parent / "fixtures" / "harbor" / "bixbench-bix-1-q1"
# A FinanceAgent package of DCAgent/financeagent_terminal@86f788bc, with its environment's other tools left out.
FINANCEAGENT_PACKAGE = Path(__file__).parent / "fixtures" / "harbor" / "financeagent-0"


def write_package(directory: Path, config: str) -> None:
    (directory / "tests").mkdir(parents=True)
    (directory / "task.toml").write_text(config)
    (directory / "instruction.md").write_text(f"Solve {directory.name}.\n")
    (directory / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    (directory / "tests" / "test.sh").chmod(0o775)


def test_package_archives_depend_only_on_names_contents_and_the_executable_bit(tmp_path):
    first, second = tmp_path / "first" / "task", tmp_path / "second" / "task"
    for directory in (first, second):
        write_package(directory, f'[environment]\ndocker_image = "{IMAGE}"\n')
    (second / "tests" / "test.sh").chmod(0o755)
    (second / "instruction.md").touch()

    archive = package_archive(first)
    extract_package(archive, tmp_path / "extracted")

    assert package_archive(second) == archive
    script = tmp_path / "extracted" / "tests" / "test.sh"
    assert (script.read_text(), script.stat().st_mode & 0o777) == ("echo 1 > /logs/verifier/reward.txt\n", 0o755)


def repository_archive(repository: Path, archive: Path) -> Path:
    """Write ``repository`` as a GitHub commit archive does: under one top-level directory."""
    with tarfile.open(archive, "w:gz") as tarball:
        tarball.add(repository, arcname=f"harbor-datasets-{REVISION}")
    return archive


def stage_archive(archive: Path, root: str, output: Path, selected: frozenset[str] | None) -> None:
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    with archive.open("rb") as stream:
        stage_archive_packages(stream, sha256, root, StoragePath(str(output)), selected)


@pytest.mark.parametrize("root", ["", "datasets/aime"])
def test_a_repository_archive_stages_each_package_directly_under_its_root(tmp_path, root):
    repository = tmp_path / "repository"
    packages = repository / root
    for name in ("aime_60", "aime_61"):
        write_package(packages / name, f'[environment]\ndocker_image = "{IMAGE}"\n')
    (packages / "aime_60" / "solution").mkdir()
    (packages / "aime_60" / "solution" / "solve.sh").write_text("echo 42 > /app/answer.txt\n")
    (packages / "README.md").write_text("Not a package.\n")
    write_package(repository / "elsewhere" / "nested" / "other", "")
    output = tmp_path / "staged"

    stage_archive(repository_archive(repository, tmp_path / "archive.tar.gz"), root, output, None)

    staged = {path.name: path.read_bytes() for path in (output / "tasks").iterdir()}
    assert staged == {f"{name}.tar.gz": package_archive(packages / name) for name in ("aime_60", "aime_61")}


def test_a_registry_subset_stages_only_its_selected_packages(tmp_path):
    repository = tmp_path / "repository"
    for name in ("3", "6", "11"):
        write_package(repository / "datasets" / "ds1000" / name, f'[environment]\ndocker_image = "{IMAGE}"\n')
    archive = repository_archive(repository, tmp_path / "archive.tar.gz")
    output = tmp_path / "staged"

    stage_archive(archive, "datasets/ds1000", output, frozenset({"3", "11"}))

    assert sorted(path.name for path in (output / "tasks").iterdir()) == ["11.tar.gz", "3.tar.gz"]
    with pytest.raises(ValueError, match="'12'"):
        stage_archive(archive, "datasets/ds1000", tmp_path / "other", frozenset({"3", "12"}))


def test_a_repository_archive_with_another_digest_stages_nothing(tmp_path):
    repository = tmp_path / "repository"
    write_package(repository / "aime_60", f'[environment]\ndocker_image = "{IMAGE}"\n')
    archive = repository_archive(repository, tmp_path / "archive.tar.gz")
    output = tmp_path / "staged"

    with archive.open("rb") as stream, pytest.raises(ValueError, match=r"expected 0{64}$"):
        stage_archive_packages(stream, "0" * 64, "", StoragePath(str(output)), None)

    assert not output.exists()


def test_nested_huggingface_repository_stages_packages_under_root(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    for name in ("tasks/task_001", "tasks/task_002", "other/task_003"):
        directory = repository / name
        directory.mkdir(parents=True)
        (directory / "task.toml").write_text(name)
    monkeypatch.setattr(packages_module, "snapshot_download", lambda *_args, **_kwargs: str(repository))
    output = tmp_path / "staged"
    stage_repository_packages(
        "org/fixture",
        REVISION,
        StoragePath(str(output)),
        root="tasks",
    )
    assert sorted(path.name for path in (output / "tasks").iterdir()) == ["task_001.tar.gz", "task_002.tar.gz"]


def test_a_harbor_repository_publishes_one_agent_item_per_package(tmp_path, local_graph, monkeypatch):
    repository = tmp_path / "repository"
    write_package(
        repository / "count-lines",
        "[agent]\ntimeout_sec = 900.0\n[verifier]\ntimeout_sec = 120.0\n"
        f'[environment]\ndocker_image = "{IMAGE}"\ncpus = 2\nmemory_mb = 4096\nallow_internet = false\n',
    )
    (repository / "README.md").write_text("Not a package.\n")

    def snapshot(repo, *, repo_type, revision, local_dir):
        assert (repo, repo_type, revision) == ("org/fixture", "dataset", REVISION)
        shutil.copytree(repository, local_dir, dirs_exist_ok=True)
        return local_dir

    monkeypatch.setattr(packages_module, "snapshot_download", snapshot)
    root, options = local_graph
    stage_repository_packages("org/fixture", REVISION, StoragePath(str(root)))
    assert [path.name for path in (root / "tasks").iterdir()] == ["count-lines.tar.gz"]

    monkeypatch.setattr(harbor_module, "IMAGE_LOCKS", tmp_path / "locks")
    (tmp_path / "locks").mkdir()
    (tmp_path / "locks" / "fixture.json").write_text(json.dumps({IMAGE: DIGEST}))
    dataset = HarborDataset(
        recipe="fixture",
        benchmark="fixture-bench",
        title="Fixture",
        source=huggingface_packages("org/fixture", REVISION),
        version="1",
    )
    source = harbor_source(dataset, TerminusAgent())

    artifact = run(source.pipeline(source, options), max_concurrent=1)[0]

    protocol, _ = evaluation_protocol(artifact.path)
    assert (protocol.benchmark, protocol.primary_metric) == ("fixture-bench", "reward")
    [case] = evaluation_cases(artifact.path)
    assert isinstance(case.item.operation, AgentEpisode)
    assert case.item.operation.harness == TerminusAgent()
    assert case.item.episode == EpisodeInputs(
        agent_timeout=900, verifier_timeout=120, cpus=2, memory_mb=4096, network=NetworkPolicy.DENY
    )
    assert case.task.environment_requirements.docker_image == f"{IMAGE}@{DIGEST}"
    assert isinstance(case.task.grader, ScriptGrader) and case.task.grader.placement == AgentMachine()


def test_job_settings_publish_three_trials_with_replaced_timeouts_and_pi_in_the_image(
    tmp_path, local_graph, monkeypatch
):
    repository = tmp_path / "repository"
    write_package(
        repository / "function-call",
        f'[agent]\ntimeout_sec = 3000.0\n[verifier]\ntimeout_sec = 120.0\n[environment]\ndocker_image = "{IMAGE}"\n',
    )

    def snapshot(repo, *, repo_type, revision, local_dir):
        shutil.copytree(repository, local_dir, dirs_exist_ok=True)
        return local_dir

    monkeypatch.setattr(packages_module, "snapshot_download", snapshot)
    root, options = local_graph
    stage_repository_packages("org/fixture", REVISION, StoragePath(str(root)))
    monkeypatch.setattr(harbor_module, "IMAGE_LOCKS", tmp_path / "locks")
    (tmp_path / "locks").mkdir()
    (tmp_path / "locks" / "fixture.json").write_text(json.dumps({IMAGE: DIGEST}))
    dataset = HarborDataset(
        recipe="fixture",
        benchmark="fixture-bench",
        title="Fixture",
        source=huggingface_packages("org/fixture", REVISION),
        version="1",
    )
    pi = PiAgent(version="0.87.0")
    job = JobSettings(agent_timeout=1800, verifier_timeout=300, agent_install=pi_install(pi))
    source = harbor_source(dataset, pi, trials=3, job=job)

    artifact = run(source.pipeline(source, options), max_concurrent=1)[0]

    protocol, _ = evaluation_protocol(artifact.path)
    assert protocol.trials.count == 3
    [case] = evaluation_cases(artifact.path)
    assert isinstance(case.item.operation, AgentEpisode) and case.item.episode is not None
    limits = case.item.operation.limits_for(case.item.episode)
    assert (limits.total_turn_timeout, limits.verifier_timeout) == (1800, 300)
    build = case.task.environment_requirements.docker_build
    assert build is not None
    [dockerfile] = build.files
    assert base64.b64decode(dockerfile.source.content_base64).decode() == f"FROM {IMAGE}@{DIGEST}\n{pi_install(pi)}"


@pytest.mark.docker
@pytest.mark.timeout(600)
@pytest.mark.parametrize(("base", "node"), [("node:18-slim", "v18."), ("debian:bookworm-slim", f"v{NODE_VERSION}")])
def test_pi_install_provides_the_pinned_pi_and_a_node_that_prefers_the_task_node(tmp_path, base, node):
    """Pi's version check passes in the built image, whose ``node`` is the task's own, else Pi's."""
    pi = PiAgent(version="0.87.0")
    (tmp_path / "Dockerfile").write_text(f"FROM {base}\n{pi_install(pi)}")
    image = f"marin-pi-install-test:{uuid.uuid4().hex}"
    subprocess.run(["docker", "build", "--tag", image, str(tmp_path)], check=True)
    try:
        versions = subprocess.run(
            ["docker", "run", "--rm", image, "/bin/sh", "-c", "pi --version && node --version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    finally:
        subprocess.run(["docker", "image", "rm", image], check=True)
    assert versions[0] == pi.version
    assert versions[1].startswith(node)


def test_a_package_the_schema_cannot_represent_is_an_explicit_rejection(tmp_path):
    write_package(tmp_path / "needs-gpu", f'[environment]\ndocker_image = "{IMAGE}"\ngpus = 1\n')
    row = RawRow(
        "tasks/needs-gpu.tar.gz:0",
        Source(dataset="org/fixture", revision=REVISION, row="tasks/needs-gpu.tar.gz:0", importer_revision="1"),
        {"name": "needs-gpu", "archive": package_archive(tmp_path / "needs-gpu")},
    )

    rejection = HarborConversion(package_tests, {IMAGE: DIGEST})(row, ConversionContext({}, None))

    assert isinstance(rejection, ImportRejection)
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.UNSUPPORTED, "harbor_feature")
    assert "GPUs" in rejection.detail


def test_bixbench_grades_its_host_judged_packages_on_the_host_with_the_locked_image():
    """The package's shared verifier reads the host's judge key, which only a host-side rule can supply."""
    row = RawRow(
        "tasks/bix-1-q1.tar.gz:0",
        Source(dataset="bixbench", revision=REVISION, row="tasks/bix-1-q1.tar.gz:0", importer_revision="1"),
        {"name": "bix-1-q1", "archive": package_archive(BIXBENCH_PACKAGE)},
    )
    lock = image_lock(BIXBENCH.recipe)

    rejection = HarborConversion(package_tests, lock)(row, ConversionContext({}, None))
    task = HarborConversion(BIXBENCH.rule, lock)(row, ConversionContext({}, None))

    assert isinstance(rejection, ImportRejection) and "OPENAI_API_KEY" in rejection.detail
    assert isinstance(task, TaskSpec)
    assert isinstance(task.grader, VerifyitGrader) and task.grader.placement is None
    assert task.output_paths == ("/workspace/answer.txt",)


def test_financeagent_images_install_the_openai_sdk_in_place_of_litellm():
    row = RawRow(
        "financeagent-0",
        Source(dataset="financeagent", revision=REVISION, row="financeagent-0", importer_revision="1"),
        {"name": "financeagent-0", "archive": package_archive(FINANCEAGENT_PACKAGE)},
    )
    convert = harbor_source(FINANCEAGENT, TerminusAgent()).config.recipe.convert

    task = convert(row, ConversionContext({}, None))

    assert isinstance(task, TaskSpec)
    build = task.environment_requirements.docker_build
    assert build is not None
    files = {resource.path: base64.b64decode(resource.source.content_base64).decode() for resource in build.files}
    assert [path for path, text in files.items() if "litellm" in text.lower()] == []
    assert "openai==" in files["Dockerfile"]
    assert "OpenAI(api_key=api_key).chat.completions.create(" in files["retrieve_information.py"]


@pytest.mark.parametrize("text", ["absent", "twice"])
def test_a_package_edit_must_match_its_file_exactly_once(tmp_path, text):
    (tmp_path / "tool.py").write_text("twice\ntwice\n")

    with pytest.raises(ValueError, match="exactly once"):
        edit_package(tmp_path, (PackageEdit("tool.py", text, "new"),))
