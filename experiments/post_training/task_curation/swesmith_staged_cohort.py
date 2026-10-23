# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify every setup branch in pilot-supported repositories with shared private caches.

The cache is used only by the authoring controls. Exported Stage 1 actors still
start with the original empty /testbed initialization and perform setup themselves.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from experiments.post_training.task_curation.swesmith_staged import (
    WORKSPACE,
    DockerPilot,
    sha256,
    source_tasks,
    write_json,
)

CONTROL_PROGRAM = r"""
import hashlib, json, pathlib, subprocess
tasks = TASKS
root = pathlib.Path(WORKSPACE)
def git(*args):
    return subprocess.check_output(['/usr/bin/git','-C',str(root),*args],text=True).strip()
def packaging():
    return {p: git('hash-object',p) for p in git('ls-files').splitlines()
            if pathlib.Path(p).name in ('setup.py','setup.cfg','pyproject.toml') or
               (pathlib.Path(p).name.startswith('requirements') and p.endswith(('.txt','.in')))}
def untracked():
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest()
            for p in git('ls-files','--others').splitlines()
            if not p.endswith('.pyc') and (root/p).is_file()}
seed_packaging = packaging()
seed_untracked = untracked()
origin = git('config','--get','remote.origin.url')
for task in tasks:
    result = {'parent_id': task['parent_id']}
    if not task['setup_matches_seed']:
        result.update(status='needs_dependency_requalification',reason='setup recipe differs')
        print(json.dumps(result),flush=True)
        continue
    try:
        commit = git('rev-parse', 'origin/' + task['branch'])
        subprocess.run(['/usr/bin/git','-C',str(root),'checkout','-B',task['branch'],commit],check=True,capture_output=True)
        if packaging() != seed_packaging:
            result.update(status='needs_dependency_requalification',
                          reason='packaging differs from qualified repository seed')
        else:
            paths = git('ls-files').splitlines()
            links = {p: git('show','HEAD:' + p) for p in paths if (root / p).is_symlink()}
            # A fresh process and no bytecode prevent source surviving a branch checkout.
            for cached in root.rglob('*.pyc'):
                cached.unlink()
            code = ('import importlib,pathlib; m=importlib.import_module(' + repr(task['import_name'])
                    + '); assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path('
                    + repr(str(root)) + '))')
            checked = subprocess.run(['/usr/local/bin/python','-B','-c',code],
                                     cwd='/tmp',capture_output=True,text=True)
            if untracked() != seed_untracked:
                result.update(status='control_error',reason='untracked state changed during shared-cache qualification')
            elif checked.returncode:
                result.update(status='not_ready_without_repair',reason=checked.stderr[-4000:])
            elif (origin != task['origin'] or git('rev-parse','HEAD') != commit
                  or git('symbolic-ref','--short','HEAD') != task['branch'] or git('diff','--name-only','HEAD')):
                result.update(status='control_error',reason='checkout is not pristine at required branch')
            else:
                contract = {'origin':origin,'branch':task['branch'],'commit':commit,'workspace':'/testbed',
                            'import_name':task['import_name'], 'symlink_targets': links,
                            'tracked_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest()
                                              for p in paths if p not in links}}
                result.update(status='stage1_controls_validated',contract=contract)
    except (subprocess.CalledProcessError,OSError) as exc:
        result.update(status='control_error',reason=str(exc))
    print(json.dumps(result),flush=True)
"""


def qualify(data: Path, output: Path, max_repositories: int | None) -> dict:
    started = time.monotonic()
    tasks, inventory = source_tasks(data)
    sources = {task.parent_id: task for task in tasks}
    seeds = {}
    for path in sorted((output / "pilots").glob("*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] != "controls_validated":
            continue
        source = sources[result["parent_id"]]
        contract = json.loads((path.parent / "private/stage1-contract.json").read_text())
        seeds[source.origin] = (path, result, contract)
    decisions = []
    for index, (origin, (path, seed, contract)) in enumerate(sorted(seeds.items())):
        if max_repositories is not None and index >= max_repositories:
            break
        group = [task for task in tasks if task.origin == origin]
        group_root = output / "cohort-cache" / origin.rsplit("/", 1)[1]
        reuse_key = {
            "control_program_sha256": sha256(CONTROL_PROGRAM.encode()),
            "seed_image": seed["mutated_image"],
            "source_tasks": {task.parent_id: task.task_hash for task in group},
        }
        qualification_key = sha256(json.dumps(reuse_key, sort_keys=True).encode())
        cache_key = group_root / "reuse-key.json"
        if (
            (group_root / "decisions.json").exists()
            and cache_key.exists()
            and json.loads(cache_key.read_text()) == reuse_key
        ):
            decisions.extend(json.loads((group_root / "decisions.json").read_text()))
            continue
        runner = DockerPilot(group_root)
        payload = [
            {
                "parent_id": task.parent_id,
                "branch": task.branch,
                "origin": origin,
                "import_name": contract["import_name"],
                "setup_matches_seed": (
                    task.setup.replace(task.branch, "<BRANCH>")
                    == sources[seed["parent_id"]].setup.replace(sources[seed["parent_id"]].branch, "<BRANCH>")
                ),
            }
            for task in group
        ]
        with runner.container(seed["mutated_image"]) as container:
            response = runner.python(
                container, CONTROL_PROGRAM.replace("WORKSPACE", repr(WORKSPACE)).replace("TASKS", repr(payload))
            )
        rows = [json.loads(line) for line in response.stdout.decode().splitlines()]
        assert {row["parent_id"] for row in rows} == {task.parent_id for task in group}
        for row in rows:
            row["qualification_key"] = qualification_key
            source = sources[row["parent_id"]]
            if row["status"] == "stage1_controls_validated":
                root = output / "cohort-stage1" / row["parent_id"].rsplit("/", 1)[1]
                write_json(root / "private/stage1-contract.json", row.pop("contract"))
                stage_contract = json.loads((root / "private/stage1-contract.json").read_text())
                write_json(
                    root / "result.json",
                    {
                        **row,
                        "split": source.split,
                        "task_sha256": source.task_hash,
                        "solution_sha256": source.solution_hash,
                        "mutation_commit": stage_contract["commit"],
                        "base_image": seed["base_image"],
                        "dependency_command": seed["dependency_command"],
                        "repository_seed": str(path.relative_to(output)),
                        "qualification_method": (
                            "shared private dependency image; unchanged packaging hashes; fresh branch checkout; "
                            "pristine tracked tree; fresh outside-cwd source-bound import"
                        ),
                        "actor_initialization": (
                            "empty /testbed; original clone, checkout, installation instructions retained"
                        ),
                        "controls": {"stage1_ready": {"passed": True}},
                    },
                )
        write_json(group_root / "decisions.json", rows)
        write_json(cache_key, reuse_key)
        decisions.extend(rows)
    covered = {row["parent_id"] for row in decisions}
    decisions.extend(
        {
            "parent_id": task.parent_id,
            "status": "pending_repository_qualification",
            "reason": "repository has no validated seed in this bounded build",
        }
        for task in tasks
        if task.parent_id not in covered
    )
    summary = {
        "source_rows": inventory["qualified_source_rows"],
        "decisions": decisions,
        "elapsed": time.monotonic() - started,
        "scope": "All eligible source parents have an explicit status; unqualified repositories are not admitted.",
        "cache_policy": (
            "Private authoring cache only; no completed workspace or git cache is provided to Stage 1 actors."
        ),
    }
    write_json(output / "cohort-stage1-decisions.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-repositories", type=int)
    args = parser.parse_args()
    result = qualify(args.data, args.output, args.max_repositories)
    print(json.dumps(dict(Counter(row["status"] for row in result["decisions"]))))
