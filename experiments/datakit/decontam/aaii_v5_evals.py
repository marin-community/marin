# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontamination records for the public parts of the Artificial Analysis Intelligence Index v5.0.

AA scores private test sets for AA-Briefcase, AutomationBench, FrontierCode, CritPt
and most of AA-Omniscience; only their public releases are indexed here, and
FrontierCode and CritPt v2 are not indexed at all. Every source is pinned to a commit
or dataset revision. Only text files are fetched, never images, PDFs, data blobs or
Docker images.

QA evals get one record per item. Task evals get one full record per task
(``<eval>-<item>``) plus one record per view (``<eval>-<item>-<view>``): the
instruction the agent sees, the reference solution or grading rubric, and the
task-specific tests. Records with fewer than MIN_LETTER_WORDS words containing letters
are dropped, and each record is cut to MAX_CHARS characters.
"""

import ast
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

from experiments.datakit.decontam.pinned_evals import MIN_LETTER_WORDS, PinnedEval, fetch_cached, letter_words

MAX_CHARS = 200_000
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
# Surge AI's own repository (surgeai/GDP.pdf) is no longer public; this mirror holds its
# README, canary and the 100 prompts with grading rubrics. The PDFs are public source
# documents, not answers, and are not indexed.
GDP_PDF_REPO = "Oitaaa/GDP.pdf"
GDP_PDF_REVISION = "7f7657830967e418091b010bc4ba0550a87d5e09"
GDP_PDF_RUBRIC_CRITERIA = 30
GDP_PDF_EMPTY_CRITERION = "None"


def _record(record_id: str, text: str) -> dict[str, str] | None:
    text = text.strip()[:MAX_CHARS]
    if letter_words(text) < MIN_LETTER_WORDS:
        return None
    return {"id": record_id, "text": text}


def _task_rows(name: str, item_id: str, views: dict[str, str], full_only: tuple[str, ...] = ()) -> list[dict]:
    """Full record joining every view (plus ``full_only`` parts), then one record per view."""
    full = "\n\n".join(part for part in (*views.values(), *full_only) if part.strip())
    records = [_record(f"{name}-{item_id}", full)]
    records += [_record(f"{name}-{item_id}-{view}", text) for view, text in views.items()]
    return [r for r in records if r is not None]


def _github_tree(repo: str, revision: str, cache_dir: Path) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"
    tree = json.loads(fetch_cached(url, cache_dir / repo / revision / "_tree.json"))
    assert not tree["truncated"], f"{repo}@{revision}: git tree listing is truncated"
    return tree["tree"]


def _github_files(repo: str, revision: str, paths: list[str], cache_dir: Path) -> dict[str, str]:
    def fetch(path: str) -> str:
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
        return fetch_cached(url, cache_dir / repo / revision / path).decode("utf-8")

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


def terminal_bench(name: str, repo: str, revision: str, task_depth: int, cache_dir: Path) -> list[dict[str, str]]:
    """Index a Harbor task tree: ``tasks/<task_depth path parts>/{instruction.md,solution/,tests/}``.

    Indexes instruction.md plus code and prose files under solution/ and tests/. Data,
    binary, lock and Dockerfiles, files over MAX_FILE_BYTES, files identical in at least
    BOILERPLATE_MIN_TASKS tasks, LFS pointers, environment/, task.toml, README.md and
    cheat/ solutions are not indexed. The item id is the task directory name, which must
    be unique.
    """
    task_dirs: dict[str, str] = {}  # task name -> task directory
    selected: dict[str, list[str]] = defaultdict(list)
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

    paths = sorted(f"{d}/{p}" for d in task_dirs.values() for p in selected[d])
    contents = _github_files(repo, revision, paths, cache_dir)
    files = {task: {p: contents[f"{d}/{p}"] for p in sorted(selected[d])} for task, d in sorted(task_dirs.items())}
    tasks_by_digest: dict[str, set[str]] = defaultdict(set)
    for task, task_files in files.items():
        for text in task_files.values():
            tasks_by_digest[hashlib.sha256(text.encode()).hexdigest()].add(task)

    rows: list[dict] = []
    for task, task_files in files.items():
        views: dict[str, list[str]] = {view: [] for view in TERMINAL_BENCH_VIEWS}
        for relative_path, text in task_files.items():
            view = relative_path.split("/", 1)[0].removesuffix(".md")
            if text.startswith(LFS_POINTER_PREFIX):
                continue
            boilerplate = len(tasks_by_digest[hashlib.sha256(text.encode()).hexdigest()]) >= BOILERPLATE_MIN_TASKS
            if view == "instruction" or not boilerplate:
                views[view].append(text)
        assert views["instruction"], f"{name}: task {task} has no instruction"
        rows += _task_rows(name, task, {view: "\n\n".join(texts) for view, texts in views.items()})
    return rows


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


def automationbench(cache_dir: Path) -> list[dict[str, str]]:
    """Public tasks of the six scored domains, 100 each, at version 1.0.6.

    AA scores a private 657-task held-out split. Instruction is the user turn (the
    per-domain system prompt is boilerplate); rubric is the assertion list; the full
    record also holds the task's initial_state. Tasks are read by statically evaluating
    ``automationbench/domains/<domain>/tasks.py``; the shared randomized noise records
    (_noise.py) and the unscored 'simple' domain are not indexed.
    """
    name = "automationbench"
    url = f"https://github.com/{AUTOMATIONBENCH_REPO}/archive/{AUTOMATIONBENCH_REVISION}.zip"
    archive = fetch_cached(url, cache_dir / AUTOMATIONBENCH_REPO / f"{AUTOMATIONBENCH_REVISION}.zip")
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
    return rows


def _hf_file(repo: str, revision: str, filename: str, cache_dir: Path) -> Path:
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir / "hf")))


def aa_briefcase_lite(cache_dir: Path) -> list[dict[str, str]]:
    """The public example week of AA-Briefcase; the scored tasks are private.

    Instruction is the task brief (tasks/<id>.md). Rubric joins, per check, the
    description, pass and fail criteria, cited source excerpts and traceability evidence.
    Not indexed: the scenario and week overviews and the agent and judge prompt
    templates (shared by all tasks), the source pool (PDF, DOCX, XLSX, EML, Slack JSON)
    and model submissions.
    """
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
    return rows


def aa_omniscience_v1_1(cache_dir: Path) -> list[dict[str, str]]:
    """The public 10% subset of AA-Omniscience v1.1; record is question and answer."""
    name = "aa_omniscience_v1_1"
    path = _hf_file(OMNISCIENCE_REPO, OMNISCIENCE_REVISION, "AA-Omniscience_dataset_public.csv", cache_dir)
    with path.open(encoding="utf-8", newline="") as f:
        items = list(csv.DictReader(f))
    records = (_record(f"{name}-{r['question_id']}", f"{r['question']}\n\n{r['answer']}") for r in items)
    return [r for r in records if r is not None]


def hle_diamond(cache_dir: Path) -> list[dict[str, str]]:
    """All test questions; record is question and answer. Images and rationales are not indexed."""
    name = "hle_diamond"
    path = f"datasets/{HLE_DIAMOND_REPO}@{HLE_DIAMOND_REVISION}/data/test-00000-of-00001.parquet"
    # Read only the text columns; the image columns hold most of the file's bytes.
    with HfFileSystem().open(path) as f:
        items = pq.read_table(f, columns=["id", "question", "answer"]).to_pylist()
    records = (_record(f"{name}-{r['id']}", f"{r['question']}\n\n{r['answer']}") for r in items)
    return [r for r in records if r is not None]


def gdp_pdf_rows(items: list[dict]) -> list[dict[str, str]]:
    """Full record (prompt and rubric criteria) plus ``instruction`` and ``rubric`` views per task."""
    rows = []
    for item in items:
        criteria = [item[f"rubric - {n}. criterion"] for n in range(1, GDP_PDF_RUBRIC_CRITERIA + 1)]
        criteria = [c.strip() for c in criteria if c and c.strip() != GDP_PDF_EMPTY_CRITERION]
        assert criteria, f"gdp_pdf: task {item['task_id']} has no rubric criteria"
        views = {"instruction": item["prompt"], "rubric": "\n".join(criteria)}
        rows += _task_rows("gdp_pdf", item["task_id"], views)
    return rows


def gdp_pdf(cache_dir: Path) -> list[dict[str, str]]:
    path = _hf_file(GDP_PDF_REPO, GDP_PDF_REVISION, "data.parquet", cache_dir)
    return gdp_pdf_rows(pq.read_table(path).to_pylist())


AAII_V5_EVALS: tuple[PinnedEval, ...] = (
    PinnedEval("GDP.pdf", "gdp_pdf", f"https://huggingface.co/datasets/{GDP_PDF_REPO}", GDP_PDF_REVISION, 299, gdp_pdf),
    PinnedEval(
        "Terminal-Bench 4.0",
        "terminal_bench_4",
        f"https://github.com/{TERMINAL_BENCH_4_REPO}",
        TERMINAL_BENCH_4_REVISION,
        264,
        lambda cache_dir: terminal_bench(
            "terminal_bench_4", TERMINAL_BENCH_4_REPO, TERMINAL_BENCH_4_REVISION, 1, cache_dir
        ),
    ),
    PinnedEval(
        "Terminal-Bench-Science 0.1",
        "terminal_bench_science",
        f"https://github.com/{TERMINAL_BENCH_SCIENCE_REPO}",
        TERMINAL_BENCH_SCIENCE_REVISION,
        280,
        lambda cache_dir: terminal_bench(
            "terminal_bench_science", TERMINAL_BENCH_SCIENCE_REPO, TERMINAL_BENCH_SCIENCE_REVISION, 3, cache_dir
        ),
    ),
    PinnedEval(
        "AA-Briefcase-Lite",
        "aa_briefcase_lite",
        f"https://huggingface.co/datasets/{BRIEFCASE_REPO}",
        BRIEFCASE_REVISION,
        12,
        aa_briefcase_lite,
    ),
    PinnedEval(
        "AutomationBench (public)",
        "automationbench",
        f"https://github.com/{AUTOMATIONBENCH_REPO}",
        AUTOMATIONBENCH_REVISION,
        1798,
        automationbench,
    ),
    PinnedEval(
        "AA-Omniscience v1.1 (public)",
        "aa_omniscience_v1_1",
        f"https://huggingface.co/datasets/{OMNISCIENCE_REPO}",
        OMNISCIENCE_REVISION,
        295,
        aa_omniscience_v1_1,
    ),
    PinnedEval(
        "HLE-Diamond",
        "hle_diamond",
        f"https://huggingface.co/datasets/{HLE_DIAMOND_REPO}",
        HLE_DIAMOND_REVISION,
        970,
        hle_diamond,
    ),
)
