# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct and qualify staged SWE-Smith tasks with local native Docker controls.

This is a campaign authoring tool, not a training launcher. Source archives and
prior split manifests remain immutable. Only control-validated variants are
marked admitted. Gold content and verifier contracts stay outside actor images.
"""

import argparse
import contextlib
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

VERSION = "swesmith-staged-v1"
WORKSPACE = "/testbed"
SOURCE_PREFIX = "open-athena/task-trove:laion__swesmith-oracle-filtered-v2/"
SOURCE_RELEASE = "swesmith-release-v1-20261016"
SOURCE_SPLIT = "swesmith-split-v1-20261016"
BASE_RECIPE = (
    "FROM python:3.10-bookworm\n"
    "RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libffi-dev libssl-dev "
    "&& rm -rf /var/lib/apt/lists/*\n"
    "RUN python -m pip install 'pytest<9' pytest-json-report pytest-cov tomli\n"
    "RUN mkdir /testbed\nWORKDIR /testbed\n"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def archive_files(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        result = {}
        for member in archive:
            if not member.isfile():
                raise ValueError(f"Unsupported source archive member: {member.name}")
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            stream = archive.extractfile(member)
            assert stream is not None
            result[member.name.removeprefix("./")] = stream.read()
        return result


@dataclass(frozen=True)
class SourceTask:
    parent_id: str
    split: str
    origin: str
    branch: str
    setup: str
    problem: str
    task_hash: str
    solution_hash: str
    files: dict[str, bytes]
    solution: dict[str, bytes]


def source_tasks(data: Path) -> tuple[list[SourceTask], dict]:
    release = data / SOURCE_RELEASE
    manifest = json.loads((release / "manifest.json").read_text())
    for name, expected in manifest["outputs_sha256"].items():
        if sha256((release / name).read_bytes()) != expected:
            raise ValueError(f"Qualified source changed: {name}")
    split_dir = data / SOURCE_SPLIT
    split_manifest = json.loads((split_dir / "manifest.json").read_text())
    assignments = {}
    for split, info in split_manifest["splits"].items():
        if sha256((split_dir / f"{split}.parquet").read_bytes()) != info["sha256"]:
            raise ValueError(f"Prior split changed: {split}")
        for identity in info["task_ids"]:
            if identity in assignments:
                raise ValueError(f"Repeated parent assignment: {identity}")
            assignments[identity] = split
    tasks = []
    exclusions = []
    for batch in pq.ParquetFile(release / "tasks.parquet").iter_batches(batch_size=64):
        for row in batch.to_pylist():
            identity = SOURCE_PREFIX + row["path"]
            files = archive_files(row["task_binary"])
            instruction = files["instruction.md"].decode()
            match = re.search(r"```bash\n(.*?)\n```", instruction, re.S)
            clone = re.search(r"^git clone (https://github.com/swesmith/\S+) \.$", instruction, re.M)
            branch = re.search(r"^git checkout (\S+)$", instruction, re.M)
            if not (match and clone and branch and identity in assignments):
                exclusions.append({"parent_id": identity, "reason": "unrecognized_setup_or_missing_split"})
                continue
            tasks.append(
                SourceTask(
                    identity,
                    assignments[identity],
                    clone[1],
                    branch[1],
                    match[1],
                    instruction[match.end() :].lstrip("\n- "),
                    sha256(row["task_binary"]),
                    sha256(row["solution_binary"]),
                    files,
                    archive_files(row["solution_binary"]),
                )
            )
    groups = defaultdict(Counter)
    for task in tasks:
        groups[task.origin][task.split] += 1
    inventory = {
        "version": VERSION,
        "qualified_source_rows": manifest["retained_rows"],
        "source_revision": manifest["source_revision"],
        "source_manifest_sha256": sha256((release / "manifest.json").read_bytes()),
        "split_manifest_sha256": sha256((split_dir / "manifest.json").read_bytes()),
        "eligible_setup_recipes": len(tasks),
        "split_counts": dict(Counter(t.split for t in tasks)),
        "repository_groups": dict(groups),
        "cross_split_repositories": sorted(key for key, value in groups.items() if len(value) > 1),
        "split_policy": (
            "Preserve prior parent-task assignments; all variants inherit parent split. "
            "Repository overlap is retained and disclosed."
        ),
        "exclusions": exclusions,
        "qualification_limit": "Prior oracle success is source evidence, not admission of new stage variants.",
    }
    return tasks, inventory


class DockerPilot:
    """Run bounded authoring commands in disposable native Linux containers."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.sequence = 0

    def command(
        self, args: list[str], *, stdin: bytes = b"", check: bool = True, timeout: int = 900
    ) -> subprocess.CompletedProcess:
        self.sequence += 1
        completed = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout)
        log = self.directory / "logs" / f"{self.sequence:04d}.json"
        binary = b"\0" in completed.stdout
        write_json(
            log,
            {
                "argv": args,
                "returncode": completed.returncode,
                "stdout": None if binary else completed.stdout.decode(errors="replace"),
                "stdout_sha256": sha256(completed.stdout),
                "stdout_bytes": len(completed.stdout),
                "stderr": completed.stderr.decode(errors="replace"),
            },
        )
        if check and completed.returncode:
            raise RuntimeError(f"Command failed ({completed.returncode}); see {log}")
        return completed

    @contextlib.contextmanager
    def container(self, image: str):
        name = (
            self.command(
                [
                    "docker",
                    "run",
                    "-d",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--memory=3g",
                    "--cpus=2",
                    "--pids-limit=512",
                    image,
                    "sleep",
                    "infinity",
                ]
            )
            .stdout.decode()
            .strip()
        )
        try:
            yield name
        finally:
            self.command(["docker", "rm", "-f", name], check=False)

    def shell(self, container: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess:
        return self.command(
            ["docker", "exec", "-i", "-w", WORKSPACE, container, "/bin/bash", "-seu"],
            stdin=command.encode(),
            check=check,
        )

    def python(self, container: str, program: str, *, check: bool = True) -> subprocess.CompletedProcess:
        return self.command(
            ["docker", "exec", "-i", "-w", "/tmp", container, "/usr/local/bin/python", "-"],
            stdin=program.encode(),
            check=check,
        )

    def snapshot(self, container: str, label: str) -> str:
        image = self.command(["docker", "commit", container, f"{VERSION}-{label}"]).stdout.decode().strip()
        return image

    def upload(self, container: str, files: dict[str, bytes], root: str = WORKSPACE) -> None:
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            for path, content in sorted(files.items()):
                member = tarfile.TarInfo(path)
                member.size = len(content)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(content))
        self.command(["docker", "exec", "-i", container, "tar", "-xf", "-", "-C", root], stdin=payload.getvalue())

    def files(self, container: str, paths: list[str]) -> dict[str, bytes]:
        result = self.command(["docker", "exec", container, "tar", "-cf", "-", "-C", WORKSPACE, "--", *paths])
        return archive_files(result.stdout)


def git_value(pilot: DockerPilot, container: str, *args: str) -> str:
    return pilot.command(["docker", "exec", "-w", WORKSPACE, container, "/usr/bin/git", *args]).stdout.decode().strip()


def stage1_contract(pilot: DockerPilot, container: str, task: SourceTask, import_name: str) -> dict:
    entries = git_value(pilot, container, "ls-files", "--stage").splitlines()
    links = {
        line.split("\t", 1)[1]: git_value(pilot, container, "show", "HEAD:" + line.split("\t", 1)[1])
        for line in entries
        if line.startswith("120000")
    }
    tracked = [line.split("\t", 1)[1] for line in entries if not line.startswith("120000")]
    files = pilot.files(container, tracked)
    installed = pilot.command(
        ["docker", "exec", container, "python", "-m", "pip", "list", "--editable", "--format=json"]
    )
    distributions = [
        item["name"] for item in json.loads(installed.stdout) if item.get("editable_project_location") == WORKSPACE
    ]
    if len(distributions) != 1:
        raise ValueError("Cannot identify the checkout's installed distribution for controls")
    return {
        "origin": task.origin,
        "branch": task.branch,
        "commit": git_value(pilot, container, "rev-parse", "HEAD"),
        "workspace": WORKSPACE,
        "import_name": import_name,
        "distribution_name": distributions[0],
        "symlink_targets": links,
        "tracked_sha256": {key: sha256(value) for key, value in files.items()},
    }


def stage1_program(contract: dict) -> str:
    # Do not trust agent-written manifests or completion prose. Supply this program
    # after the actor has stopped, through stdin, with the private contract embedded.
    return """import hashlib, importlib, importlib.metadata, json, os, pathlib, subprocess
c = CONTRACT
root = pathlib.Path(c['workspace'])
def git(*args):
    p = subprocess.run(['/usr/bin/git', '-C', str(root), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None
def repository_url(value):
    return None if value is None else value.rstrip('/').removesuffix('.git')
checks = {
    'workspace': root.is_dir() and not root.is_symlink(),
    'origin': repository_url(git('config', '--get', 'remote.origin.url')) == repository_url(c['origin']),
    'branch': git('symbolic-ref', '--short', 'HEAD') == c['branch'],
    'commit': git('rev-parse', 'HEAD') == c['commit'],
    'pristine_tracked_files': all(
        (root / p).is_file() and not (root / p).is_symlink()
        and hashlib.sha256((root / p).read_bytes()).hexdigest() == h
        for p,h in c['tracked_sha256'].items()),
    'symlink_targets': all((root/p).is_symlink() and os.readlink(root/p) == target
                          for p,target in c.get('symlink_targets',{}).items()),
}
try:
    module = importlib.import_module(c['import_name'])
    checks['installed_import'] = pathlib.Path(module.__file__).resolve().is_relative_to(root)
except Exception as exc:
    checks['installed_import'] = False
    checks['import_error'] = type(exc).__name__ + ': ' + str(exc)
print(json.dumps({'passed': all(v is True for v in checks.values()), 'checks': checks}))
""".replace(
        "CONTRACT", repr(contract)
    )


def stage1_result(pilot: DockerPilot, container: str, contract: dict) -> dict:
    result = pilot.python(container, stage1_program(contract), check=False)
    if result.returncode:
        return {"passed": False, "inspection_error": result.stderr.decode(errors="replace")}
    return json.loads(result.stdout)


def pytest_result(pilot: DockerPilot, container: str, verifier: dict, report: str) -> dict:
    paths = verifier["paths"]
    args = [
        "docker",
        "exec",
        "-w",
        WORKSPACE,
        "-e",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        container,
        "/bin/bash",
        "-c",
        '"$@"; status=$?; exit "$status"',
        "pytest-parent",
        "/usr/local/bin/python",
        "-m",
        "pytest",
        "-p",
        "pytest_jsonreport.plugin",
        "-p",
        "pytest_cov.plugin",
        "--json-report",
        f"--json-report-file=/tmp/{report}.json",
        *verifier.get("args", []),
        *paths,
    ]
    result = pilot.command(args, check=False)
    read = pilot.command(["docker", "exec", container, "cat", f"/tmp/{report}.json"], check=False)
    if read.returncode:
        return {"passed": False, "returncode": result.returncode, "error": "missing_pytest_report"}
    payload = json.loads(read.stdout)
    outcomes = {test["nodeid"]: test["outcome"] for test in payload.get("tests", [])}
    required = verifier.get("must_pass", []) + verifier.get("must_not_break", [])
    failed = {test: outcomes.get(test, "missing") for test in required if outcomes.get(test) != "passed"}
    return {
        "passed": result.returncode == 0 and bool(required) and not failed,
        "returncode": result.returncode,
        "required": len(required),
        "collected": len(outcomes),
        "failed_required": failed,
        "summary": payload.get("summary", {}),
    }


def partial_hunks(patch: str) -> list[str]:
    """Keep proper, nonempty prefixes of original repair hunks without inventing code."""
    chunks = re.split(r"(?=^diff --git )", patch, flags=re.M)
    units = []
    for chunk in chunks:
        if not chunk:
            continue
        pieces = re.split(r"(?=^@@ )", chunk, flags=re.M)
        if len(pieces) < 2:
            raise ValueError("Repair contains a non-textual or whole-file operation")
        header = pieces[0]
        units.extend(header + hunk for hunk in pieces[1:])
    return ["".join(units[:count]) for count in range(1, len(units))]


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = parts[-1]
    return any(part in {"test", "tests", "testing"} for part in parts[:-1]) or (
        name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
    )


def candidate_integrity(files: dict[str, bytes], baseline: dict[str, bytes], allowed: set[str]) -> list[str]:
    """Reject missing or weakened tests and any tracked changes outside repair source."""
    return sorted(
        path
        for path, content in baseline.items()
        if path not in files or (path not in allowed and files[path] != content)
    )


def pilot_task(task: SourceTask, output: Path, base_image: str, import_name: str, dependency_command: str) -> dict:
    parent = output / "pilots" / task.parent_id.rsplit("/", 1)[1]
    directory = parent / f"attempt-{len(list(parent.glob('attempt-*'))) + 1:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    pilot = DockerPilot(directory)
    verifier = tomllib.loads(task.files["tests/verifier.toml"].decode())
    result = {
        "parent_id": task.parent_id,
        "split": task.split,
        "task_sha256": task.task_hash,
        "solution_sha256": task.solution_hash,
        "base_image": base_image,
        "controls": {},
        "exclusions": [],
    }
    try:
        with pilot.container(base_image) as container:
            pilot.shell(container, task.setup)
            # Explicit dependency additions are authored and recorded per pilot.
            if dependency_command:
                pilot.shell(container, dependency_command)
            contract = stage1_contract(pilot, container, task, import_name)
            write_json(directory / "private" / "stage1-contract.json", contract)
            result["controls"]["stage1_ready"] = stage1_result(pilot, container, contract)
            if not result["controls"]["stage1_ready"]["passed"]:
                raise ValueError("Stage 1 positive control failed")
            baseline = pilot.files(container, list(contract["tracked_sha256"]))
            mutated = pilot.snapshot(container, task.parent_id.rsplit("/", 1)[1] + "-mutated")
            fixed_commit = git_value(pilot, container, "rev-parse", "origin/HEAD")
            all_changed = git_value(pilot, container, "diff", "--name-only", "HEAD", fixed_commit).splitlines()
            hidden_paths = [path for path in all_changed if is_test_path(path)]
            changed = [path for path in all_changed if path not in hidden_paths]
            if any(path in verifier["paths"] for path in changed):
                raise ValueError("Unsupported production-source file also contains original verifier tests")
            if not changed or any(not path.endswith(".py") for path in changed):
                raise ValueError("Gold diff includes unsupported non-source changes")
            gold_patch = git_value(pilot, container, "diff", "HEAD", fixed_commit, "--", *changed) + "\n"
            hidden = {}
            if hidden_paths:
                hidden_archive = pilot.command(
                    ["docker", "exec", "-w", WORKSPACE, container, "git", "archive", fixed_commit, "--", *hidden_paths]
                )
                with tarfile.open(fileobj=io.BytesIO(hidden_archive.stdout)) as archive:
                    hidden = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
                pilot.upload(container, hidden)
            verifier_image = pilot.snapshot(container, task.parent_id.rsplit("/", 1)[1] + "-verifier")
            result.update(
                {
                    "mutation_commit": contract["commit"],
                    "fixed_commit": fixed_commit,
                    "changed_source_paths": changed,
                    "dependency_command": dependency_command,
                    "gold_patch_sha256": sha256(gold_patch.encode()),
                    "mutated_image": mutated,
                    "verifier_image": verifier_image,
                    "hidden_restored_paths": sorted(hidden),
                }
            )
            (directory / "private" / "gold.patch").write_text(gold_patch)
            pilot.shell(container, f"git checkout {shlex.quote(fixed_commit)} -- " + " ".join(map(shlex.quote, changed)))
            gold = pilot.files(container, changed)
            result["controls"]["stage3_complete"] = pytest_result(pilot, container, verifier, "complete")
            if not result["controls"]["stage3_complete"]["passed"]:
                raise ValueError("Complete handoff failed original correctness tests")
        with pilot.container(mutated) as container:
            pilot.upload(container, gold)
            pilot.shell(container, "rm -rf /testbed/.git")
            result["complete_image"] = pilot.snapshot(container, task.parent_id.rsplit("/", 1)[1] + "-complete")
        with pilot.container(base_image) as empty:
            result["controls"]["stage1_empty_false_submission"] = stage1_result(pilot, empty, contract)
        negative_commands = {
            "stage1_wrong_origin": "git remote set-url origin https://github.com/example/wrong",
            "stage1_wrong_branch": "git checkout -b wrong-branch",
            "stage1_wrong_workspace": "mv /testbed /wrongbed; mkdir /testbed",
            "stage1_missing_install": f"python -m pip uninstall -y {shlex.quote(contract['distribution_name'])}",
            "stage1_healthy_default": f"git checkout {shlex.quote(fixed_commit)}",
            "stage1_spoofed_metadata": (
                f"git checkout {shlex.quote(fixed_commit)} -- " + " ".join(map(shlex.quote, changed))
            ),
        }
        for name, command in negative_commands.items():
            with pilot.container(mutated) as container:
                pilot.shell(container, command)
                result["controls"][name] = stage1_result(pilot, container, contract)
        with pilot.container(verifier_image) as container:
            result["controls"]["stage3_unmodified_mutation"] = pytest_result(pilot, container, verifier, "mutation")
        test_path = next((path for path in verifier["paths"] if path in baseline), None)
        if test_path is not None:
            deletion = dict(baseline)
            del deletion[test_path]
            weakened = {**baseline, test_path: b"def test_ok():\n    assert True\n"}
            result["controls"]["stage3_deleted_test"] = {
                "passed": not candidate_integrity(deletion, baseline, set(changed))
            }
            result["controls"]["stage3_weakened_test"] = {
                "passed": not candidate_integrity(weakened, baseline, set(changed))
            }
        else:
            result["control_limits"] = [
                "No supplied original test file; hidden-test deletion is not an available actor action"
            ]
        partials = partial_hunks(gold_patch)
        for index, patch in enumerate(partials[:8]):
            with pilot.container(verifier_image) as container:
                pilot.command(
                    ["docker", "exec", "-i", "-w", WORKSPACE, container, "git", "apply", "--recount", "-"],
                    stdin=patch.encode(),
                )
                partial = pytest_result(pilot, container, verifier, "partial")
                if partial["passed"] or not any(
                    value == "failed" for value in partial.get("failed_required", {}).values()
                ):
                    result["exclusions"].append(
                        {
                            "recipe": f"hunk-prefix-{index+1}",
                            "reason": "no_demonstrated_relevant_test_failure",
                            "result": partial,
                        }
                    )
                    continue
                (directory / "private" / "partial.patch").write_text(patch)
                result["controls"]["stage3_partial_unmodified_false_submission"] = partial
                pilot.upload(container, gold)
                result["controls"]["stage3_partial_repaired"] = pytest_result(
                    pilot, container, verifier, "partial-repaired"
                )
                result["partial_recipe"] = f"first-{index+1}-original-repair-hunks"
                break
        if "partial_recipe" in result:
            with pilot.container(mutated) as container:
                pilot.command(
                    ["docker", "exec", "-i", "-w", WORKSPACE, container, "git", "apply", "--recount", "-"],
                    stdin=(directory / "private" / "partial.patch").read_bytes(),
                )
                pilot.shell(container, "rm -rf /testbed/.git")
                result["partial_image"] = pilot.snapshot(container, task.parent_id.rsplit("/", 1)[1] + "-partial")
        negative = [
            value
            for key, value in result["controls"].items()
            if key not in {"stage1_ready", "stage3_complete", "stage3_partial_repaired"}
        ]
        if any(value["passed"] for value in negative):
            raise ValueError("A negative control passed")
        if not result["controls"].get("stage3_partial_repaired", {}).get("passed"):
            result["exclusions"].append({"stage": "stage3_partial", "reason": "no_qualified_partial_handoff"})
        result["status"] = "controls_validated"
        result["test_isolation_policy"] = "nested-and-top-level-tests-v2"
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        result["status"] = "excluded"
        result["error"] = str(exc)
    write_json(directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="append", default=[], help="task-name:import-name:dependency-command")
    args = parser.parse_args()
    tasks, inventory = source_tasks(args.data)
    write_json(args.output / "source-inventory.json", inventory)
    if not args.pilot:
        print(
            json.dumps(
                {key: inventory[key] for key in ("qualified_source_rows", "eligible_setup_recipes", "split_counts")}
            )
        )
        return
    build = args.output / "base-image"
    build.mkdir(exist_ok=True)
    (build / "Dockerfile").write_text(BASE_RECIPE)
    runner = DockerPilot(build)
    resolved = build / "resolved.json"
    if resolved.exists():
        image_record = json.loads(resolved.read_text())
        if image_record["recipe_sha256"] != sha256(BASE_RECIPE.encode()):
            raise ValueError("Base recipe changed; use a new versioned output directory")
        image = image_record["image"]
        runner.command(["docker", "image", "inspect", image])
    else:
        runner.command(["docker", "build", "-t", VERSION + "-base", str(build)])
        image = (
            runner.command(["docker", "image", "inspect", VERSION + "-base", "--format", "{{.Id}}"])
            .stdout.decode()
            .strip()
        )
        tag = VERSION + "-base-" + image.split(":")[1][:16]
        runner.command(["docker", "tag", image, tag])
        write_json(resolved, {"image": image, "immutable_local_tag": tag, "recipe_sha256": sha256(BASE_RECIPE.encode())})
    reports = []
    for selection in args.pilot:
        name, module, dependencies = selection.split(":", 2)
        task = next(task for task in tasks if task.parent_id == SOURCE_PREFIX + name)
        reports.append(pilot_task(task, args.output, image, module, dependencies))
    write_json(
        args.output / "pilot-summary.json",
        {"base_image": image, "builder_sha256": sha256(Path(__file__).read_bytes()), "pilots": reports},
    )
    print(json.dumps({row["parent_id"]: row["status"] for row in reports}))


if __name__ == "__main__":
    main()
