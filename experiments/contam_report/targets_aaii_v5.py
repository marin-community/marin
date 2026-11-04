# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build contamination-scan targets for the public parts of AAII v5.0.

Usage: ``targets_aaii_v5.py OUT_DIR CACHE_DIR``. Writes ``OUT_DIR/<eval>/eval.parquet``
(columns ``id`` and ``text``) and ``OUT_DIR/.manifest.json``. Every source is pinned to
a commit. Downloads land in CACHE_DIR; only text files are fetched, never images,
data blobs or Docker images.

QA evals get one record per item. Task evals get one full record per task
(``<eval>-<item>``) plus one record per view (``<eval>-<item>-<view>``): the
instruction the agent sees, the reference solution or grading rubric, and the
task-specific tests. Records with fewer than MIN_WORDS words containing letters
are dropped, and each record is cut to MAX_CHARS characters.
"""

import ast
import csv
import dataclasses
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

from experiments.contam_report.prepare_targets import write

MAX_CHARS = 200_000
MIN_WORDS = 13
# A file whose exact content appears in this many tasks of one eval is harness
# boilerplate (shared test runners, helper scripts), not task content.
BOILERPLATE_MIN_TASKS = 3
# Larger files under solution/ or tests/ are generated data, not code.
MAX_FILE_BYTES = 500_000
FETCH_THREADS = 8

# Extensions of code and prose under solution/ and tests/. Data (json, csv, txt,
# npz, parquet), images, binaries and Dockerfiles are not indexed.
TEXT_EXTENSIONS = frozenset(
    {
        "bas", "c", "cc", "cjs", "cls", "cpp", "css", "cu", "cuh", "diff", "f90", "frm", "go", "h", "hpp",
        "html", "java", "jl", "js", "jsx", "lean", "lua", "m", "md", "mjs", "mts", "patch", "pl", "pm", "py",
        "r", "rb", "rq", "rs", "s", "sage", "sbt", "scala", "sh", "sql", "tex", "toml", "ts", "tsx", "v",
        "yaml", "yml",
    }
)  # fmt: skip
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/"
TERMINAL_BENCH_VIEWS = ("instruction", "solution", "tests")

TERMINAL_BENCH_4_REPO = "harbor-framework/terminal-bench"
TERMINAL_BENCH_4_REVISION = "452bf305c6daa62fc59061d22133a7cbc7c1572e"  # tag v4.0.0
TERMINAL_BENCH_SCIENCE_REPO = "harbor-framework/terminal-bench-science"
TERMINAL_BENCH_SCIENCE_REVISION = "f81afac4f11048e77a15dfc8fb1dbfb897fea0ce"  # tag v0.1.0
AUTOMATIONBENCH_REPO = "zapier/AutomationBench"
AUTOMATIONBENCH_REVISION = "6d210543b7a046f0f451c828cd2dadef774276eb"  # "1.0.6 release" commit
AUTOMATIONBENCH_DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")
BRIEFCASE_REPO = "ArtificialAnalysis/AA-Briefcase-Lite"
BRIEFCASE_REVISION = "4dec557b47d43867a1648c0974db1d8208c8b677"
OMNISCIENCE_REPO = "ArtificialAnalysis/AA-Omniscience-Public-v1.1"
OMNISCIENCE_REVISION = "1f9cbcec472000b28f9782205f1ef9498b45bd21"
HLE_DIAMOND_REPO = "cais/hle-diamond"
HLE_DIAMOND_REVISION = "04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23"
GDP_PDF_REPO = "surgeai/GDP.pdf"


@dataclasses.dataclass(frozen=True)
class EvalTargets:
    name: str
    source: str
    revision: str | None
    items: int
    rows: list[dict[str, str]]
    views: tuple[str, ...]
    notes: str


def _letter_words(text: str) -> int:
    return sum(1 for word in text.split() if any(ch.isalpha() for ch in word))


def _record(record_id: str, text: str) -> dict[str, str] | None:
    text = text.strip()[:MAX_CHARS]
    if _letter_words(text) < MIN_WORDS:
        return None
    return {"id": record_id, "text": text}


def _task_rows(name: str, item_id: str, views: dict[str, str], full_only: tuple[str, ...] = ()) -> list[dict]:
    """Full record joining every view (plus ``full_only`` parts), then one record per view."""
    full = "\n\n".join(part for part in (*views.values(), *full_only) if part.strip())
    records = [_record(f"{name}-{item_id}", full)]
    records += [_record(f"{name}-{item_id}-{view}", text) for view, text in views.items()]
    return [r for r in records if r is not None]


def _fetch(url: str, path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        path.write_bytes(data)
    return path.read_bytes()


def _github_tree(repo: str, revision: str, cache_dir: Path) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"
    tree = json.loads(_fetch(url, cache_dir / repo / revision / "_tree.json"))
    assert not tree["truncated"], f"{repo}@{revision}: git tree listing is truncated"
    return tree["tree"]


def _github_files(repo: str, revision: str, paths: list[str], cache_dir: Path) -> dict[str, str]:
    def fetch(path: str) -> str:
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
        return _fetch(url, cache_dir / repo / revision / path).decode("utf-8")

    with ThreadPoolExecutor(FETCH_THREADS) as pool:
        return dict(zip(paths, pool.map(fetch, paths), strict=True))


def _is_indexed_file(relative_path: str, size: int) -> bool:
    if relative_path == "instruction.md":
        return True
    view, _, rest = relative_path.partition("/")
    if view not in ("solution", "tests") or not rest:
        return False
    filename = rest.rsplit("/", 1)[-1]
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return extension in TEXT_EXTENSIONS and "lock" not in filename.lower() and size <= MAX_FILE_BYTES


def terminal_bench(name: str, repo: str, revision: str, task_depth: int, cache_dir: Path) -> EvalTargets:
    """Index a Harbor task tree: ``tasks/<task_depth path parts>/{instruction.md,solution/,tests/}``.

    The item id is the task directory name, which must be unique.
    """
    task_dirs: dict[str, str] = {}  # task name -> task directory
    selected: dict[str, list[str]] = defaultdict(list)
    not_indexed = Counter()
    for entry in _github_tree(repo, revision, cache_dir):
        parts = entry["path"].split("/")
        if entry["type"] != "blob" or parts[0] != "tasks" or len(parts) <= task_depth + 1:
            continue
        task_dir, relative_path = "/".join(parts[: task_depth + 1]), "/".join(parts[task_depth + 1 :])
        if relative_path == "instruction.md":
            assert parts[task_depth] not in task_dirs, f"{name}: duplicate task name {parts[task_depth]}"
            task_dirs[parts[task_depth]] = task_dir
        if _is_indexed_file(relative_path, entry["size"]):
            selected[task_dir].append(relative_path)
        elif relative_path.split("/", 1)[0] in ("solution", "tests"):
            not_indexed[relative_path.split("/", 1)[0]] += 1

    paths = sorted(f"{d}/{p}" for d in task_dirs.values() for p in selected[d])
    contents = _github_files(repo, revision, paths, cache_dir)
    files = {task: {p: contents[f"{d}/{p}"] for p in sorted(selected[d])} for task, d in sorted(task_dirs.items())}
    tasks_by_digest: dict[str, set[str]] = defaultdict(set)
    for task, task_files in files.items():
        for text in task_files.values():
            tasks_by_digest[hashlib.sha256(text.encode()).hexdigest()].add(task)

    rows: list[dict] = []
    dropped = Counter()
    for task, task_files in files.items():
        views: dict[str, list[str]] = {view: [] for view in TERMINAL_BENCH_VIEWS}
        for relative_path, text in task_files.items():
            view = relative_path.split("/", 1)[0].removesuffix(".md")
            if text.startswith(LFS_POINTER_PREFIX):
                dropped["lfs pointer"] += 1
            elif view != "instruction" and len(tasks_by_digest[hashlib.sha256(text.encode()).hexdigest()]) >= (
                BOILERPLATE_MIN_TASKS
            ):
                dropped[f"{view} boilerplate"] += 1
            else:
                views[view].append(text)
        assert views["instruction"], f"{name}: task {task} has no instruction"
        rows += _task_rows(name, task, {view: "\n\n".join(texts) for view, texts in views.items()})

    notes = (
        f"{len(task_dirs)} tasks at {repo}@{revision}. Indexed instruction.md plus code and prose files "
        f"({len(paths) - len(task_dirs)} files) under solution/ and tests/. Not indexed: "
        f"{sum(not_indexed.values())} data, binary, lock, Dockerfile or >{MAX_FILE_BYTES} byte files "
        f"({dict(not_indexed)}); files identical in >= {BOILERPLATE_MIN_TASKS} tasks ({dict(dropped)}); "
        "environment/, task.toml, README.md and cheat/ solutions."
    )
    return EvalTargets(name, f"https://github.com/{repo}", revision, len(task_dirs), rows, TERMINAL_BENCH_VIEWS, notes)


class _Literal:
    """Evaluate a Python expression made of literals, names bound to literals, ``+``/``*`` and f-strings.

    AutomationBench defines each task as a function returning a dict literal. Evaluating
    the AST avoids importing and running the benchmark package.
    """

    def __init__(self, env: dict):
        self.env = env

    def __call__(self, node: ast.expr):
        match node:
            case ast.Constant(value=value):
                return value
            case ast.Dict(keys=keys, values=values):
                return {self(k): self(v) for k, v in zip(keys, values, strict=True)}
            case ast.List(elts=elts):
                return [self(e) for e in elts]
            case ast.Tuple(elts=elts):
                return tuple(self(e) for e in elts)
            case ast.Name(id=name):
                return self.env[name]
            case ast.BinOp(left=left, op=ast.Add(), right=right):
                return self(left) + self(right)
            case ast.BinOp(left=left, op=ast.Mult(), right=right):
                return self(left) * self(right)
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return -self(operand)
            case ast.JoinedStr(values=values):
                return "".join(v.value if isinstance(v, ast.Constant) else self._formatted(v) for v in values)
        raise ValueError(f"unsupported expression {ast.dump(node)[:200]}")

    def _formatted(self, node: ast.expr) -> str:
        assert isinstance(node, ast.FormattedValue) and node.conversion == -1 and node.format_spec is None
        return str(self(node.value))


def _automationbench_tasks(source: str, domain: str) -> list[dict]:
    """Statically evaluate the task functions listed in ``get_<domain>_dataset``."""
    module = ast.parse(source)
    env: dict = {}
    functions: dict[str, ast.FunctionDef] = {}
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = _Literal(env)(node.value)
            except (ValueError, KeyError):
                pass  # module-level helpers that are not literals; a task that needs one fails below
    (task_list,) = [
        node.value
        for node in ast.walk(functions[f"get_{domain}_dataset"])
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("task_fns", "tasks")
        and isinstance(node.value, ast.List)
    ]
    names = [e.func.id if isinstance(e, ast.Call) else e.id for e in task_list.elts]
    tasks = []
    for name in names:
        local = dict(env)
        for statement in functions[name].body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue
            if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
                local[statement.targets[0].id] = _Literal(local)(statement.value)
            elif isinstance(statement, ast.Return) and statement.value is not None:
                tasks.append(_Literal(local)(statement.value))
                break
            else:
                raise ValueError(f"{domain}.{name}: unsupported statement {type(statement).__name__}")
    return tasks


def automationbench(cache_dir: Path) -> EvalTargets:
    name = "automationbench"
    url = f"https://github.com/{AUTOMATIONBENCH_REPO}/archive/{AUTOMATIONBENCH_REVISION}.zip"
    archive = _fetch(url, cache_dir / AUTOMATIONBENCH_REPO / f"{AUTOMATIONBENCH_REVISION}.zip")
    root = f"AutomationBench-{AUTOMATIONBENCH_REVISION}/automationbench/domains"
    rows: list[dict] = []
    seen: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(archive)) as source_zip:
        for domain in AUTOMATIONBENCH_DOMAINS:
            source = source_zip.read(f"{root}/{domain}/tasks.py").decode("utf-8")
            for task in _automationbench_tasks(source, domain):
                task_id = task["task"]
                assert task_id not in seen, f"{name}: duplicate task {task_id}"
                seen.add(task_id)
                user_turns = [m["content"] for m in task["prompt"] if m["role"] == "user"]
                assert user_turns, f"{name}: task {task_id} has no user prompt"
                info = task["info"]
                views = {
                    "instruction": "\n\n".join(user_turns),
                    "rubric": json.dumps(info["assertions"], ensure_ascii=False, indent=1),
                }
                state = json.dumps(info["initial_state"], ensure_ascii=False, indent=1)
                rows += _task_rows(name, task_id, views, full_only=(state,))
    notes = (
        f"Public tasks of the six scored domains ({', '.join(AUTOMATIONBENCH_DOMAINS)}; 100 each) at version 1.0.6. "
        "AA scores a private 657-task held-out split that is not released. Instruction is the user turn; the "
        "per-domain system prompt is boilerplate and not indexed. Rubric is the assertion list. The full record "
        "also holds the task's initial_state (seeded app data the agent reads through tools). Tasks are read by "
        "statically evaluating automationbench/domains/<domain>/tasks.py; randomized background noise records "
        "(_noise.py, shared across tasks) and the unscored 'simple' domain are not indexed."
    )
    return EvalTargets(
        name, f"https://github.com/{AUTOMATIONBENCH_REPO}", AUTOMATIONBENCH_REVISION, len(seen), rows,
        ("instruction", "rubric"), notes,
    )  # fmt: skip


def _hf_file(repo: str, revision: str, filename: str, cache_dir: Path) -> Path:
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir / "hf")))


def aa_briefcase_lite(cache_dir: Path) -> EvalTargets:
    name = "aa_briefcase_lite"

    def read(filename: str) -> str:
        return _hf_file(BRIEFCASE_REPO, BRIEFCASE_REVISION, filename, cache_dir).read_text(encoding="utf-8")

    tasks = [json.loads(line) for line in read("tasks.jsonl").splitlines() if line.strip()]
    checks = json.loads(read("grader/grading_rubric_w1.json"))["checks"]
    traceability = {e["check_id"]: e for e in json.loads(read("grader/traceability_w1.json"))["entries"]}
    rows: list[dict] = []
    for task in tasks:
        task_checks = [c for c in checks if c["task_ref"] == task["task_id"]]
        assert task_checks, f"{name}: task {task['task_id']} has no checks"
        rubric = []
        for check in task_checks:
            rubric += [check["check_description"], check["score_1_criteria"], check["score_0_criteria"]]
            rubric.append(check["specific_file_locations"])
            evidence = traceability[check["check_id"]]
            rubric += [source["justification"] for source in evidence["sources"]]
            rubric.append(evidence["reasoning"])
        views = {"instruction": read(task["task_md_path"]), "rubric": "\n\n".join(rubric)}
        rows += _task_rows(name, task["task_id"], views)
    notes = (
        f"The public example week ({len(tasks)} tasks, {len(checks)} checks); the 91 scored AA-Briefcase tasks are "
        "private. Instruction is the task brief (tasks/<id>.md). Rubric joins, per check, the description, pass and "
        "fail criteria, cited source excerpts and the traceability evidence. Not indexed: the scenario and week "
        "overviews and the agent and judge prompt templates (shared by all tasks), the source pool (PDF, DOCX, "
        "XLSX, EML, Slack JSON; data), and model submissions in submissions/ (not reference solutions)."
    )
    return EvalTargets(
        name, f"https://huggingface.co/datasets/{BRIEFCASE_REPO}", BRIEFCASE_REVISION, len(tasks), rows,
        ("instruction", "rubric"), notes,
    )  # fmt: skip


def aa_omniscience_v1_1(cache_dir: Path) -> EvalTargets:
    name = "aa_omniscience_v1_1"
    path = _hf_file(OMNISCIENCE_REPO, OMNISCIENCE_REVISION, "AA-Omniscience_dataset_public.csv", cache_dir)
    with path.open(encoding="utf-8", newline="") as f:
        items = list(csv.DictReader(f))
    records = (_record(f"{name}-{r['question_id']}", f"{r['question']}\n\n{r['answer']}") for r in items)
    rows = [r for r in records if r is not None]
    notes = (
        f"The public 10% subset ({len(items)} of 3,000 questions); the rest is private. Record is question and "
        f"answer; {len(items) - len(rows)} records under {MIN_WORDS} words were dropped."
    )
    source = f"https://huggingface.co/datasets/{OMNISCIENCE_REPO}"
    return EvalTargets(name, source, OMNISCIENCE_REVISION, len(items), rows, (), notes)


def hle_diamond() -> EvalTargets:
    name = "hle_diamond"
    path = f"datasets/{HLE_DIAMOND_REPO}@{HLE_DIAMOND_REVISION}/data/test-00000-of-00001.parquet"
    # Read only the text columns; the image columns hold most of the file's bytes.
    with HfFileSystem().open(path) as f:
        items = pq.read_table(f, columns=["id", "question", "answer"]).to_pylist()
    records = (_record(f"{name}-{r['id']}", f"{r['question']}\n\n{r['answer']}") for r in items)
    rows = [r for r in records if r is not None]
    notes = (
        f"All {len(items)} questions of the test split. Record is question and answer; image bytes and the "
        f"rationale are not indexed. {len(items) - len(rows)} records under {MIN_WORDS} words were dropped."
    )
    source = f"https://huggingface.co/datasets/{HLE_DIAMOND_REPO}"
    return EvalTargets(name, source, HLE_DIAMOND_REVISION, len(items), rows, (), notes)


GDP_PDF_UNAVAILABLE = EvalTargets(
    name="gdp_pdf",
    source=f"https://huggingface.co/datasets/{GDP_PDF_REPO}",
    revision=None,
    items=0,
    rows=[],
    views=(),
    notes=(
        "Not indexed. The Hugging Face dataset named by AA and by the harness (github.com/surge-ai/gdp-pdf) returns "
        "404 to an authenticated read token (private or withdrawn); the harness repo ships only two synthetic "
        "placeholder tasks."
    ),
)


def main() -> None:
    out_dir, cache_dir = Path(sys.argv[1]), Path(sys.argv[2])
    evals = [
        GDP_PDF_UNAVAILABLE,
        terminal_bench("terminal_bench_4", TERMINAL_BENCH_4_REPO, TERMINAL_BENCH_4_REVISION, 1, cache_dir),
        terminal_bench(
            "terminal_bench_science", TERMINAL_BENCH_SCIENCE_REPO, TERMINAL_BENCH_SCIENCE_REVISION, 3, cache_dir
        ),
        aa_briefcase_lite(cache_dir),
        automationbench(cache_dir),
        aa_omniscience_v1_1(cache_dir),
        hle_diamond(),
    ]
    manifest = []
    for targets in evals:
        ids = [row["id"] for row in targets.rows]
        assert len(ids) == len(set(ids)), f"{targets.name}: duplicate record ids"
        if targets.rows:
            write(out_dir, targets.name, targets.rows)
        truncated = sum(len(row["text"]) == MAX_CHARS for row in targets.rows)
        manifest.append(
            {
                "name": targets.name,
                "source": targets.source,
                "revision": targets.revision,
                "items": targets.items,
                "records": len(targets.rows),
                "views": list(targets.views),
                "notes": f"{targets.notes} {truncated} records were cut to {MAX_CHARS} characters.",
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
