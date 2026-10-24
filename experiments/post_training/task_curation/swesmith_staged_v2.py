# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Author shared-image Stage3 handoffs with per-parent correctness evidence."""

import argparse
import base64
import json
import shlex
import time
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from experiments.post_training.task_curation.swesmith_staged import (
    WORKSPACE,
    DockerPilot,
    SourceTask,
    git_value,
    is_test_path,
    partial_hunks,
    pytest_result,
    sha256,
    source_tasks,
    write_json,
)

REPOSITORIES = ("textdistance", "marshmallow", "glom", "parso", "bleach", "apispec")


def encoded_files(files: dict[str, bytes]) -> dict[str, str]:
    return {path: base64.b64encode(content).decode() for path, content in files.items()}


def decoded_files(files: dict[str, str]) -> dict[str, bytes]:
    return {path: base64.b64decode(content) for path, content in files.items()}


@dataclass(frozen=True)
class RepositoryGroup:
    name: str
    seed: dict
    tasks: list[SourceTask]


def source_groups(data: Path, previous: Path) -> list[RepositoryGroup]:
    tasks, _ = source_tasks(data)
    sources = {task.parent_id: task for task in tasks}
    eligible = {
        row["parent_id"]
        for row in json.loads((previous / "cohort-stage1-decisions.json").read_text())["decisions"]
        if row["status"] == "stage1_controls_validated"
    }
    seeds = {}
    for path in sorted((previous / "pilots").glob("*/attempt-*/result.json")):
        row = json.loads(path.read_text())
        if row["status"] == "controls_validated":
            source = sources[row["parent_id"]]
            seeds[source.origin] = row
    return [
        RepositoryGroup(
            origin.rsplit("/", 1)[1],
            seed,
            [task for task in tasks if task.origin == origin and task.parent_id in eligible],
        )
        for origin, seed in sorted(seeds.items())
        if any(f"__{name}." in origin for name in REPOSITORIES)
    ]


def prepare_repository(group: RepositoryGroup, output: Path, per_repository: int) -> dict:
    name, seed, tasks = group.name, group.seed, group.tasks
    directory = output / "repositories" / name
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    runner = DockerPilot(directory)
    with runner.container(seed["mutated_image"]) as container:
        fixed = seed["fixed_commit"]
        runner.shell(container, f"git checkout --detach {shlex.quote(fixed)}\ngit reset --hard {shlex.quote(fixed)}")
        payload = [{"parent_id": task.parent_id, "branch": task.branch} for task in tasks]
        program = """
import json,subprocess
def git(*args):
    return subprocess.check_output(['git','-C',WORKSPACE,*args],text=True)
for task in TASKS:
    commit=git('rev-parse','origin/'+task['branch']).strip()
    paths=git('diff','--name-only',commit,'HEAD').splitlines()
    print(json.dumps(dict(task,commit=commit,paths=paths,
                         patch=git('diff',commit,'HEAD','--',*[p for p in paths if p.endswith('.py')]))))
""".replace(
            "WORKSPACE", repr(WORKSPACE)
        ).replace(
            "TASKS", repr(payload)
        )
        raw = runner.python(container, program)
        inventory = [json.loads(line) for line in raw.stdout.decode().splitlines()]
        source_map = {task.parent_id: task for task in tasks}
        candidates = []
        for row in inventory:
            task = source_map[row["parent_id"]]
            verifier = tomllib.loads(task.files["tests/verifier.toml"].decode())
            changed = [p for p in row["paths"] if not is_test_path(p)]
            if not changed or any(not p.endswith(".py") or p in verifier["paths"] for p in changed):
                row["status"] = "unsupported_repair_scope"
                continue
            patch = git_value(runner, container, "diff", row["commit"], fixed, "--", *changed) + "\n"
            try:
                prefixes = partial_hunks(patch)
            except ValueError:
                row["status"] = "unsupported_nontextual_patch"
                continue
            row.update(changed_source_paths=changed, patch=patch, partial_prefixes=len(prefixes))
            candidates.append(row)
        candidates.sort(
            key=lambda row: (
                source_map[row["parent_id"]].split != "validation",
                row["partial_prefixes"] == 0,
                sha256(row["parent_id"].encode()),
            )
        )
        selected = candidates[:per_repository]
        write_json(directory / "private/inventory.json", inventory)
        tracked = git_value(runner, container, "ls-files").splitlines()
        hidden = [p for p in tracked if is_test_path(p)]
        regular = json.loads(
            runner.python(
                container,
                "import json,pathlib\nprint(json.dumps([p for p in "
                + repr(tracked)
                + f" if not pathlib.Path({WORKSPACE!r},p).is_symlink()]))",
            ).stdout
        )
        actor_paths = [p for p in regular if p not in hidden]
        baseline = runner.files(container, actor_paths)
        baseline_hashes = {p: sha256(content) for p, content in baseline.items()}
        verifier_image = runner.snapshot(container, "v2-" + name + "-verifier")
        reports = {}
        for row in selected:
            task = source_map[row["parent_id"]]
            verifier = tomllib.loads(task.files["tests/verifier.toml"].decode())
            key = sha256(
                json.dumps({"paths": verifier["paths"], "args": verifier.get("args", [])}, sort_keys=True).encode()
            )
            if key not in reports:
                report_name = "fixed-" + key[:12]
                check = pytest_result(runner, container, verifier, report_name)
                report = json.loads(
                    runner.command(["docker", "exec", container, "cat", f"/tmp/{report_name}.json"]).stdout
                )
                reports[key] = {
                    "check": check,
                    "outcomes": {x["nodeid"]: x["outcome"] for x in report.get("tests", [])},
                    "fixed_commit": fixed,
                    "dependency_image": seed["mutated_image"],
                    "paths": verifier["paths"],
                    "args": verifier.get("args", []),
                }
                write_json(directory / "private" / ("report-" + key + ".json"), reports[key])
            report = reports[key]
            required = verifier["must_pass"] + verifier.get("must_not_break", [])
            passed = (
                report["check"]["returncode"] == 0
                and bool(required)
                and all(report["outcomes"].get(node) == "passed" for node in required)
            )
            if not passed:
                row.update(status="fixed_reference_not_passing", report_key=key)
                continue
            parent = output / "parents" / row["parent_id"].rsplit("/", 1)[1]
            gold = runner.files(container, row["changed_source_paths"])
            mutation = {
                p: (
                    runner.command(
                        ["docker", "exec", "-w", WORKSPACE, container, "git", "show", row["commit"] + ":" + p]
                    ).stdout
                )
                for p in gold
            }
            write_json(parent / "private/gold-files.json", encoded_files(gold))
            write_json(parent / "private/mutation-files.json", encoded_files(mutation))
            (parent / "private/gold.patch").write_text(row["patch"])
            row.update(status="complete_qualified", report_key=key)
            write_json(
                parent / "result.json",
                {
                    "parent_id": task.parent_id,
                    "split": task.split,
                    "origin": task.origin,
                    "source_task_sha256": task.task_hash,
                    "source_solution_sha256": task.solution_hash,
                    "mutation_commit": row["commit"],
                    "fixed_commit": fixed,
                    "repository": name,
                    "changed_source_paths": row["changed_source_paths"],
                    "verifier": verifier,
                    "complete_report": str((directory / "private" / ("report-" + key + ".json")).relative_to(output)),
                    "complete_passed": True,
                    "partial_prefixes": row["partial_prefixes"],
                    "partial_status": "not_attempted",
                    "problem": task.problem,
                },
            )
        cleanup = """
import pathlib,shutil
root=pathlib.Path(WORKSPACE)
for p in HIDDEN:
    path=root/p
    if path.is_symlink() or path.is_file():path.unlink()
for path in root.rglob('*.pyc'):path.unlink()
shutil.rmtree(root/'.git')
shutil.rmtree(root/'.pytest_cache',ignore_errors=True)
""".replace(
            "HIDDEN", repr(hidden)
        ).replace(
            "WORKSPACE", repr(WORKSPACE)
        )
        with runner.container(verifier_image) as actor:
            runner.python(actor, cleanup)
            actor_image = runner.snapshot(actor, "v2-" + name + "-actor")
    result = {
        "repository": name,
        "actor_image": actor_image,
        "verifier_image": verifier_image,
        "fixed_commit": fixed,
        "dependency_image": seed["mutated_image"],
        "tracked_sha256": baseline_hashes,
        "hidden_paths": hidden,
        "selected": [{k: v for k, v in row.items() if k != "patch"} for row in selected],
    }
    write_json(directory / "result.json", result)
    return result


def qualify_partial(parent_path: Path, output: Path) -> dict:
    result = json.loads(parent_path.read_text())
    control_version = "fresh-bytecode-v2"
    if result.get("partial_control_version") == control_version:
        return result
    parent = parent_path.parent
    repo = json.loads((output / "repositories" / result["repository"] / "result.json").read_text())
    attempt = str(time.time_ns())
    write_json(parent / "partial-history" / (attempt + ".json"), result)
    runner = DockerPilot(parent / "partial-controls" / attempt)
    mutation = decoded_files(json.loads((parent / "private/mutation-files.json").read_text()))
    gold = decoded_files(json.loads((parent / "private/gold-files.json").read_text()))
    prefixes = partial_hunks((parent / "private/gold.patch").read_text())
    result["partial_status"] = "no_witnessed_partial"
    result["partial_control_version"] = control_version
    for index, patch in enumerate(prefixes[:3]):
        with runner.container(repo["verifier_image"]) as container:
            runner.upload(container, mutation)
            applied = runner.command(
                ["docker", "exec", "-i", "-w", WORKSPACE, container, "git", "apply", "--recount", "-"],
                stdin=patch.encode(),
                check=False,
            )
            if applied.returncode:
                continue
            handoff = runner.files(container, result["changed_source_paths"])
            if handoff == mutation or handoff == gold:
                continue
            # Original must-pass nodes provide a bounded witness; no invented tests.
            relevant = {**result["verifier"], "paths": result["verifier"]["must_pass"], "must_not_break": []}
            runner.python(container, f"import pathlib\nfor p in pathlib.Path({WORKSPACE!r}).rglob('*.pyc'):p.unlink()")
            before = pytest_result(runner, container, relevant, "partial-before")
            if not any(outcome == "failed" for outcome in before.get("failed_required", {}).values()):
                continue
            runner.upload(container, gold)
            runner.python(container, f"import pathlib\nfor p in pathlib.Path({WORKSPACE!r}).rglob('*.pyc'):p.unlink()")
            after = pytest_result(runner, container, relevant, "partial-after")
            if not after["passed"]:
                result["partial_status"] = "repair_control_failed"
                break
            write_json(parent / "handoff-files.json", encoded_files(handoff))
            result.update(
                partial_status="qualified",
                partial_prefix_index=index,
                partial_before=before,
                partial_after=after,
                handoff_sha256=sha256((parent / "handoff-files.json").read_bytes()),
            )
            break
    write_json(parent_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "partials"), required=True)
    parser.add_argument("--per-repository", type=int, default=100)
    parser.add_argument("--partial-limit", type=int, default=150)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    if args.phase == "prepare":
        groups = source_groups(args.data, args.previous)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda group: prepare_repository(group, args.output, args.per_repository), groups))
        print(json.dumps({"repositories": len(results), "selected": sum(len(row["selected"]) for row in results)}))
    else:
        paths = sorted(
            (args.output / "parents").glob("*/result.json"), key=lambda path: sha256(path.parent.name.encode())
        )
        paths = [path for path in paths if json.loads(path.read_text())["partial_prefixes"] > 0][: args.partial_limit]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda path: qualify_partial(path, args.output), paths))
        print(json.dumps(dict(Counter(row["partial_status"] for row in results))))
    write_json(
        args.output / (args.phase + "-execution.json"),
        {
            "elapsed": time.monotonic() - started,
            "code_sha256": sha256(Path(__file__).read_bytes()),
        },
    )


if __name__ == "__main__":
    main()
