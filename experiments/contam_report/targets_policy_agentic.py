# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build contamination-scan targets for the agentic (Harbor) evals of the post-training eval policy.

Covers the in-distribution Harbor evals of the eval policy (issue #9566) except Terminal-Bench 2.1,
plus the out-of-distribution DeepSWE 1.1. An item is one Harbor task. Each task yields one full
record ``<eval>-<task_id>`` and one record per view ``<eval>-<task_id>-<view>``, where a view is the
instruction the agent sees, the reference solution, the task-specific verifier code, or (SOTOPIA)
the scenario with goals and character profiles.

Task trees come from the dataset registry pinned by the campaign Harbor configs: sparse, shallow
git checkouts of the pinned commits, a pinned GitHub zip for OpenThoughts-TBLite, and a pinned
Hugging Face revision for SOTOPIA-hard. Lines that recur in at least half of an eval's tasks
(a quarter for OpenThoughts-TBLite) for the same source file are template boilerplate and are dropped
before indexing.

Usage: uv run python experiments/contam_report/targets_policy_agentic.py OUT_DIR SRC_DIR
"""

import dataclasses
import io
import json
import math
import re
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download

from experiments.contam_report.prepare_targets import write

REGISTRY_URL = (
    "https://raw.githubusercontent.com/marin-community/harbor/38742b0f9f411c95e0c01468785cac7bb77c9894/registry.json"
)
OT_TBLITE_ZIP = "https://github.com/open-thoughts/OpenThoughts-TBLite/archive/{commit}.zip"
SOTOPIA_REPO = "open-athena/sotopia-hard-harbor"
SOTOPIA_REVISION = "b388b6634d82b25811afa431997b3dbfee9c17cf"
SOTOPIA_ITEMS = 100

MAX_VIEW_CHARS = 200_000
MIN_LETTER_WORDS = 13
BOILERPLATE_TASK_FRACTION = 0.5
# About 40 of the 100 OpenThoughts-TBLite tasks share a grader wrapper and test.sh template.
OT_TBLITE_BOILERPLATE_FRACTION = 0.25
BOILERPLATE_MIN_TASKS = 5
CODE_SUFFIXES = frozenset({".py", ".sh", ".sql", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".java", ".r", ".rb"})
LETTER = re.compile(r"[A-Za-z]")


@dataclasses.dataclass(frozen=True)
class RegistryDataset:
    git_url: str
    commit: str
    parent: str
    task_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Task:
    """One Harbor task: ``components`` maps ``view:source`` keys to raw text."""

    task_id: str
    components: dict[str, str]


@dataclasses.dataclass(frozen=True)
class EvalSpec:
    name: str
    subdir: str
    registry_name: str | None
    registry_version: str | None
    notes: str
    boilerplate_fraction: float = BOILERPLATE_TASK_FRACTION


def registry_dataset(registry: list[dict], name: str, version: str) -> RegistryDataset:
    (entry,) = [d for d in registry if d["name"] == name and d["version"] == version]
    tasks = entry["tasks"]
    sources = {(t["git_url"], t["git_commit_id"], str(Path(t["path"]).parent)) for t in tasks}
    if len(sources) != 1:
        raise ValueError(f"{name}@{version}: expected one git source, got {sources}")
    ((git_url, commit, parent),) = sources
    for t in tasks:
        if Path(t["path"]).name != t["name"]:
            raise ValueError(f"{name}@{version}: task {t['name']} lives at {t['path']}")
    return RegistryDataset(git_url, commit, "" if parent == "." else parent, tuple(t["name"] for t in tasks))


def sparse_checkout(git_url: str, commit: str, path: str, dest: Path) -> Path:
    """Shallow, blob-filtered checkout of one directory at a pinned commit; reuses an existing one."""
    if not (dest / path).is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        git = ["git", "-C", str(dest)]
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "remote", "add", "origin", git_url], check=True)
        subprocess.run([*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], check=True)
        subprocess.run([*git, "sparse-checkout", "set", "--cone", path], check=True)
        subprocess.run([*git, "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    if head.stdout.strip() != commit:
        raise ValueError(f"{dest}: HEAD {head.stdout.strip()} != {commit}")
    return dest / path


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            dest.write_bytes(response.read())
    return dest


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def code_files(task_dir: Path, subdir: str) -> dict[str, str]:
    """Text code files under ``task_dir/subdir`` keyed by relative path; data files are skipped."""
    root = task_dir / subdir
    if not root.is_dir():
        return {}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in CODE_SUFFIXES)
    return {str(p.relative_to(task_dir)): read(p) for p in files}


def pretty_json(value: object) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False)


def swebench_task(task_dir: Path) -> Task:
    config = json.loads(read(task_dir / "tests/config.json"))
    return Task(
        config["instance_id"],
        {
            "instruction:instruction.md": read(task_dir / "instruction.md"),
            "solution:patch": config["patch"],
            "tests:test_patch": config["test_patch"],
        },
    )


def ot_tblite_task(task_dir: Path) -> Task:
    components = {"instruction:instruction.md": read(task_dir / "instruction.md")}
    components.update({f"solution:{k}": v for k, v in code_files(task_dir, "solution").items()})
    components.update({f"tests:{k}": v for k, v in code_files(task_dir, "tests").items()})
    return Task(task_dir.name, components)


def ds1000_task(task_dir: Path) -> Task:
    config = json.loads(read(task_dir / "tests/config.json"))
    return Task(
        task_dir.name,
        {
            "instruction:instruction.md": read(task_dir / "instruction.md"),
            "solution:solution/solve.sh": read(task_dir / "solution/solve.sh"),
            "tests:code_context": config["code_context"],
        },
    )


def bfcl_task(task_dir: Path) -> Task:
    return Task(
        task_dir.name,
        {
            "instruction:instruction.md": read(task_dir / "instruction.md"),
            "solution:solution/solve.sh": read(task_dir / "solution/solve.sh"),
            "tests:tests/evaluate.py": read(task_dir / "tests/evaluate.py"),
        },
    )


def bixbench_task(task_dir: Path) -> Task:
    truth = json.loads(read(task_dir / "tests/ground_truth.json"))
    if truth["question_id"] != task_dir.name:
        raise ValueError(f"{task_dir}: ground truth is for {truth['question_id']}")
    return Task(
        task_dir.name,
        {
            "instruction:instruction.md": read(task_dir / "instruction.md"),
            "solution:ideal_answer": truth["ideal_answer"],
        },
    )


def tau3_task(task_dir: Path) -> Task:
    task = json.loads(read(task_dir / "tests/config.json"))["task"]
    return Task(
        task_dir.name.removeprefix("tau3-"),
        {
            "instruction:user_scenario": pretty_json(
                {"description": task["description"], "user_scenario": task["user_scenario"]}
            ),
            "solution:evaluation_criteria": pretty_json(task["evaluation_criteria"]),
        },
    )


def sotopia_task(task_dir: Path) -> Task:
    config = json.loads(read(task_dir / "tests/task_config.json"))
    return Task(
        task_dir.name.removeprefix("sotopia-"),
        {
            "instruction:instruction.md": read(task_dir / "instruction.md"),
            "scenario:task_config": pretty_json({"environment": config["environment"], "agents": config["agents"]}),
        },
    )


def deepswe_task(task_dir: Path) -> Task:
    return Task(
        task_dir.name,
        {
            "instruction:instruction.md": read(task_dir / "instruction.md"),
            "solution:solution/solution.patch": read(task_dir / "solution/solution.patch"),
            "tests:tests/test.patch": read(task_dir / "tests/test.patch"),
            "tests:tests/test.sh": read(task_dir / "tests/test.sh"),
        },
    )


def strip_boilerplate(tasks: list[Task], fraction: float) -> tuple[list[Task], dict[str, int]]:
    """Drop lines that recur in ``fraction`` of tasks for the same component; returns dropped-line counts."""
    threshold = max(BOILERPLATE_MIN_TASKS, math.ceil(fraction * len(tasks)))
    counts: dict[str, Counter[str]] = {}
    for task in tasks:
        for key, text in task.components.items():
            counts.setdefault(key, Counter()).update({line.strip() for line in text.splitlines() if line.strip()})
    common = {key: {line for line, n in c.items() if n >= threshold} for key, c in counts.items()}
    dropped: Counter[str] = Counter()
    stripped = []
    for task in tasks:
        components = {}
        for key, text in task.components.items():
            kept = [line for line in text.splitlines() if line.strip() not in common[key]]
            dropped[key] += sum(1 for line in text.splitlines() if line.strip() in common[key])
            body = "\n".join(kept).strip()
            if body:
                components[key] = body
        stripped.append(Task(task.task_id, components))
    return stripped, dict(dropped)


def letter_words(text: str) -> int:
    return sum(1 for word in text.split() if LETTER.search(word))


def capped(text: str, record_id: str, truncated: list[str]) -> str:
    if len(text) <= MAX_VIEW_CHARS:
        return text
    truncated.append(record_id)
    return text[:MAX_VIEW_CHARS]


def records(name: str, tasks: list[Task]) -> tuple[list[dict], dict]:
    """Full and per-view records for ``tasks``, plus counters for the manifest."""
    rows: list[dict] = []
    truncated: list[str] = []
    short: list[str] = []
    view_counts: Counter[str] = Counter()
    for task in tasks:
        views: dict[str, list[str]] = {}
        for key, text in task.components.items():
            views.setdefault(key.split(":", 1)[0], []).append(text)
        texts = {view: "\n\n".join(parts) for view, parts in views.items()}
        candidates = [(f"{name}-{task.task_id}", "\n\n".join(texts.values()))]
        candidates += [(f"{name}-{task.task_id}-{view}", text) for view, text in texts.items()]
        for record_id, text in candidates:
            if letter_words(text) < MIN_LETTER_WORDS:
                short.append(record_id)
                continue
            rows.append({"id": record_id, "text": capped(text, record_id, truncated)})
            view_counts[record_id.removeprefix(f"{name}-{task.task_id}").lstrip("-") or "full"] += 1
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate record ids")
    return rows, {"view_records": dict(view_counts), "truncated": truncated, "skipped_short": short}


def harbor_tasks(
    registry: list[dict], spec: EvalSpec, src_dir: Path, parse: Callable[[Path], Task]
) -> tuple[list[Task], RegistryDataset]:
    dataset = registry_dataset(registry, spec.registry_name, spec.registry_version)
    root = sparse_checkout(dataset.git_url, dataset.commit, dataset.parent, src_dir / f"{spec.subdir}_{dataset.commit}")
    return [parse(root / task) for task in dataset.task_names], dataset


SPECS = {
    "swebench_verified": EvalSpec(
        "SWE-bench Verified",
        "swebench_verified",
        "swebench-verified",
        "1.0",
        "Views: instruction.md (issue text), gold patch, and test_patch from tests/config.json. "
        "Skipped: hints_text, FAIL_TO_PASS/PASS_TO_PASS lists, tests/test.sh (re-embeds test_patch in a "
        "shared runner template), solve.sh (re-embeds the gold patch), Dockerfiles, task.toml.",
    ),
    "ot_tblite": EvalSpec(
        "OpenThoughts-TBLite 2.0",
        "ot_tblite",
        "openthoughts-tblite",
        "2.0",
        "The campaign config resolves openthoughts-tblite@2.0 through the Harbor hub; the hub and the pinned "
        "marin-community registry list the same 100 tasks at the same commit. Views: instruction.md, code files "
        "under solution/ (solve.sh and helpers), and code files under tests/ (.py, .sh, .sql). Skipped: environment/ "
        "(Dockerfiles, data, seed code), expected-output data files under tests/ (json, csv, txt, yaml), pyc, "
        "task.toml. Lines shared by at least 25% of tasks are dropped as template boilerplate "
        "(grader wrapper, test.sh).",
        boilerplate_fraction=OT_TBLITE_BOILERPLATE_FRACTION,
    ),
    "ds1000": EvalSpec(
        "DS-1000",
        "ds1000",
        "ds-1000",
        "mini-200",
        "All 1,000 DS-1000 problems from the task tree at the mini-200 commit; mini_200 lists the 200 policy tasks "
        "(seed-42 subset). Views: instruction.md (problem with code context), reference solution from solve.sh, "
        "and the test code_context from tests/config.json. Skipped: shared tests/run_test.py and tests/test.sh, "
        "the empty solution/solution.py scaffold, Dockerfiles, task.toml.",
    ),
    "bfcl_parity": EvalSpec(
        "BFCL-Parity",
        "bfcl_parity",
        "bfcl_parity",
        "1.0",
        "Views: instruction.md (question and function docs), ground-truth calls from solve.sh, and the "
        "task-specific remainder of tests/evaluate.py (ground truth and checks). Skipped: shared tests/test.sh, "
        "Dockerfiles, task.toml. Irrelevance tasks have an empty ground truth, so their solution view is short.",
    ),
    "bixbench": EvalSpec(
        "BixBench CLI corrections",
        "bixbench",
        "bixbench-cli-marin-corrections",
        "3",
        "Views: question from instruction.md and ideal_answer from tests/ground_truth.json. Ideal answers are "
        "short, so most solution views fall below 13 words and only appear in the full record. Not indexed: the "
        "capsule data and analysis notebooks, which tasks download from futurehouse/BixBench at build time and are "
        "not part of the task tree. Skipped: shared llm_judge.py, test.sh, download_capsule.py, Dockerfiles.",
    ),
    "tau3": EvalSpec(
        "tau3-bench",
        "tau3",
        "tau3-bench",
        "1.0.1",
        "All four domains (airline, retail, telecom, banking_knowledge); banking is included because indexing "
        "all domains together is simpler. Task ids drop the Harbor 'tau3-' prefix. Views: task description plus "
        "user scenario (JSON) and evaluation criteria (JSON) from tests/config.json. Skipped: the per-domain agent "
        "policy in instruction.md (identical across a domain's tasks), initial_state device and database data, "
        "runtime server, evaluate.py, solve.sh (oracle replay of the evaluation actions), Dockerfiles.",
    ),
    "sotopia_hard": EvalSpec(
        "SOTOPIA-hard",
        "sotopia_hard",
        None,
        None,
        "Task ids drop the Harbor 'sotopia-' prefix. Views: instruction.md (scenario and private goal for the "
        "evaluated agent) and scenario (environment, both agent goals and both character profiles from "
        "tests/task_config.json). No reference solution exists. Skipped: shared verify.py, test.sh, Dockerfile, "
        "task.toml.",
    ),
    "deepswe": EvalSpec(
        "DeepSWE 1.1",
        "deepswe",
        "deep-swe-litecontainer",
        "1.1",
        "Views: instruction.md, solution/solution.patch, and tests (tests/test.patch plus the task-specific part "
        "of tests/test.sh). Skipped: shared tests/grader.py and solve.sh, tests/config.json (test node id lists), "
        "Dockerfiles, task.toml.",
    ),
}

HARBOR_PARSERS: dict[str, Callable[[Path], Task]] = {
    "swebench_verified": swebench_task,
    "bfcl_parity": bfcl_task,
    "bixbench": bixbench_task,
    "tau3": tau3_task,
    "deepswe": deepswe_task,
}


def ot_tblite(registry: list[dict], src_dir: Path) -> tuple[list[Task], str, str]:
    spec = SPECS["ot_tblite"]
    dataset = registry_dataset(registry, spec.registry_name, spec.registry_version)
    archive = download(OT_TBLITE_ZIP.format(commit=dataset.commit), src_dir / f"ot_tblite_{dataset.commit}.zip")
    extract_dir = src_dir / f"ot_tblite_{dataset.commit}"
    if not extract_dir.is_dir():
        with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as source_zip:
            source_zip.extractall(extract_dir)
    (root,) = extract_dir.iterdir()
    tasks = [ot_tblite_task(root / name) for name in dataset.task_names]
    return tasks, dataset.git_url.removesuffix(".git"), dataset.commit


def ds1000(registry: list[dict], src_dir: Path) -> tuple[list[Task], RegistryDataset, list[str]]:
    spec = SPECS["ds1000"]
    mini = registry_dataset(registry, spec.registry_name, spec.registry_version)
    root = sparse_checkout(mini.git_url, mini.commit, mini.parent, src_dir / f"ds1000_{mini.commit}")
    names = sorted((p.name for p in root.iterdir() if p.is_dir()), key=int)
    if not set(mini.task_names) <= set(names):
        raise ValueError("ds1000: mini-200 tasks missing from the task tree")
    return [ds1000_task(root / name) for name in names], mini, sorted(mini.task_names, key=int)


def sotopia(src_dir: Path) -> list[Task]:
    local = src_dir / f"sotopia_hard_{SOTOPIA_REVISION}"
    snapshot_download(
        SOTOPIA_REPO,
        repo_type="dataset",
        revision=SOTOPIA_REVISION,
        local_dir=local,
        allow_patterns=["*/instruction.md", "*/tests/task_config.json"],
    )
    return [sotopia_task(p) for p in sorted(local.glob("sotopia-*")) if p.is_dir()]


def main() -> None:
    out_dir, src_dir = Path(sys.argv[1]), Path(sys.argv[2])
    registry = json.loads(read(download(REGISTRY_URL, src_dir / "registry_38742b0f.json")))

    loaded: dict[str, tuple[list[Task], str, str, int, dict]] = {}
    for subdir, parse in HARBOR_PARSERS.items():
        tasks, dataset = harbor_tasks(registry, SPECS[subdir], src_dir, parse)
        loaded[subdir] = (tasks, dataset.git_url.removesuffix(".git"), dataset.commit, len(dataset.task_names), {})
    tasks, source, revision = ot_tblite(registry, src_dir)
    loaded["ot_tblite"] = (tasks, source, revision, len(tasks), {})
    tasks, mini, mini_ids = ds1000(registry, src_dir)
    loaded["ds1000"] = (tasks, mini.git_url.removesuffix(".git"), mini.commit, 1000, {"mini_200": mini_ids})
    loaded["sotopia_hard"] = (sotopia(src_dir), f"hf://datasets/{SOTOPIA_REPO}", SOTOPIA_REVISION, SOTOPIA_ITEMS, {})

    manifest = []
    for subdir, spec in SPECS.items():
        tasks, source, revision, expected, extra = loaded[subdir]
        if len(tasks) != expected:
            raise ValueError(f"{subdir}: {len(tasks)} tasks, expected {expected}")
        tasks, dropped = strip_boilerplate(tasks, spec.boilerplate_fraction)
        rows, stats = records(subdir, tasks)
        write(out_dir, subdir, rows)
        manifest.append(
            {
                "name": spec.name,
                "dir": subdir,
                "source": source,
                "revision": revision,
                "registry": f"{spec.registry_name}@{spec.registry_version}" if spec.registry_name else None,
                "items": len(tasks),
                "records": len(rows),
                "views": stats["view_records"],
                "notes": spec.notes,
                "boilerplate_lines_dropped": dropped,
                "truncated_records": stats["truncated"],
                "skipped_short_records": stats["skipped_short"],
                **extra,
            }
        )
    manifest_path = out_dir / ".manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
