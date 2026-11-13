# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen and sample the TMax-15K Harbor reserve (``TMaxxx/TMax-15K-Harbor``) for a generation pilot.

``select`` screens out, without running anything:
- hidden verifier assets, by two rules (a task is excluded when either fires):
  - ``image_path``: test_final_state.py reads an absolute path the image provides (a shipped fixture, or a path in
    the Dockerfile/install scripts) that the instruction names neither by path nor by file name;
  - ``answer_name``: pool P's tmax_leaks rule (truth/expected/oracle/solution/verifier/answer/reference paths created
    by the build and read by the test), applied to the Harbor layout;
- every task shipping an ``_fixtures/.../oracle`` file;
- exact overlap (normalized instruction or test_final_state.py hash) with the TMax SFT environments.
- verifiers that import a third-party module the task's setup never installs (``missing_test_imports``): the
  pilot sandbox has no egress, so such a verifier cannot pass;
- instructions that tell the agent to install Python packages the setup never installs
  (``missing_instruction_packages``, a heuristic), for the same reason.

It then runs pool P's DataKit decon (13-gram, overlap 0.5, >= 2 features) of the screened tasks against SWE-bench
Verified, Terminal-Bench 2.1, TBLite 2.0 and, as a near-duplicate check, the TMax SFT environments.

``select`` then takes a seeded sample stratified by ``metadata.domain`` from the screened, decon-clean tasks.

``rescreen`` re-runs the screen with the current rules on a previous ``select`` output, reuses its decon marks, and
replaces each drawn task the new rules exclude with a seeded pick from the same domain.

``package`` writes the sampled tasks as local Harbor task folders. Only the Dockerfile changes: ``TMUX_BAKE`` is
appended, and JSON-form ``COPY`` lines are rewritten to shell form (``shell_form_copy``).
"""

import argparse
import ast
import collections
import hashlib
import json
import random
import re
import sys
import tempfile
import tomllib
import urllib.request
import warnings
import zipfile
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.tmax_reserve.decon import decon_marks, eval_documents

RESERVE = ("TMaxxx/TMax-15K-Harbor", "48a77eb0")
SFT_ENVS = ("allenai/TMax-SFT-16.5K", "848e218270672080a92a94791c028ee43ae33387")
ABSOLUTE_PATH = re.compile(r"(/(?:app|opt|home|root|srv|var|usr/local|etc|data|tmp)/[A-Za-z0-9_./-]+)")
ANSWER_PATH = re.compile(
    r"/[A-Za-z0-9_./-]*(?:truth|expected|oracle|solution|verifier|answer|reference)[A-Za-z0-9_./-]*", re.IGNORECASE
)
BUILD_FILES = ("environment/Dockerfile", "environment/post_install.sh", "environment/base_install.sh")
ORACLE_FIXTURE = "fixtures/oracle"
# The only change to a task: Terminus-2 installs tmux at agent setup when the image lacks it (no reserve image has
# it), and the pilot's sandboxes have no general egress. Eval sandboxes install tmux at setup instead.
JSON_COPY = re.compile(r"^([ \t]*COPY[ \t]+)(\[.*\])[ \t]*$", re.MULTILINE | re.IGNORECASE)
TMUX_BAKE = "RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*"


def hub_zip(repo: str, revision: str, dest: Path) -> zipfile.ZipFile:
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/tasks.zip"
    urllib.request.urlretrieve(url, dest)
    return zipfile.ZipFile(dest)


def _normalized_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()


def _names_path(path: str, instruction: str) -> bool:
    return path in instruction or path.rstrip("/").rsplit("/", 1)[-1] in instruction


def task_files(archive: zipfile.ZipFile) -> dict[str, set[str]]:
    files: dict[str, set[str]] = collections.defaultdict(set)
    for name in archive.namelist():
        if not name.endswith("/"):
            task, rel = name.split("/", 1)
            files[task].add(rel)
    return files


def hidden_assets(task: str, files: set[str], archive: zipfile.ZipFile) -> dict[str, list[str]]:
    """The hidden paths each rule finds for one Harbor-layout task."""

    def read(rel: str) -> str:
        return archive.read(f"{task}/{rel}").decode(errors="ignore") if rel in files else ""

    tests, instruction = read("tests/test_final_state.py"), read("instruction.md")
    build = "".join(read(rel) for rel in BUILD_FILES)
    fixtures = {"/" + rel.split("_fixtures/", 1)[1] for rel in files if "_fixtures/" in rel}
    provided = fixtures | set(ABSOLUTE_PATH.findall(build))
    image_path = sorted(
        path
        for path in set(ABSOLUTE_PATH.findall(tests))
        if any(path == p or p.startswith(path.rstrip("/") + "/") for p in provided)
        and not _names_path(path, instruction)
    )
    created = {path.rstrip("/.") for path in ANSWER_PATH.findall(build + "\n" + "\n".join(fixtures))}
    answer_name = sorted(
        path for path in created if len(path) > 6 and path in tests and not _names_path(path, instruction)
    )
    return {"image_path": image_path, "answer_name": answer_name}


# Import name -> PyPI/apt distribution names, for imports whose distribution is named differently.
DISTRIBUTIONS = {
    "yaml": ("pyyaml",),
    "jwt": ("pyjwt",),
    "sklearn": ("scikit-learn",),
    "skimage": ("scikit-image",),
    "bs4": ("beautifulsoup4",),
    "cv2": ("opencv-python", "opencv-python-headless", "opencv"),
    "PIL": ("pillow", "pil"),
    "dateutil": ("python-dateutil",),
    "Crypto": ("pycryptodome", "pycrypto"),
    "dotenv": ("python-dotenv",),
    "grpc_tools": ("grpcio-tools",),
    "attr": ("attrs",),
    "OpenSSL": ("pyopenssl", "openssl"),
    "magic": ("python-magic",),
    "serial": ("pyserial",),
    "docx": ("python-docx",),
    "fitz": ("pymupdf",),
    "google": ("protobuf", "google-cloud-storage"),
    "grpc": ("grpcio",),
    "psycopg2": ("psycopg2-binary", "psycopg2"),
    "MySQLdb": ("mysqlclient",),
    "zmq": ("pyzmq",),
    "websocket": ("websocket-client",),
    "Levenshtein": ("python-levenshtein", "levenshtein"),
}
TEST_RUNNER_MODULES = {"pytest", "_pytest"}
# Standard library in the images' Python 3.10 (ubuntu:22.04) but removed from newer Pythons.
INSTALL_SENTENCE = re.compile(
    r"[^\n]*(?:\bpip3? install\b|virtual environment|\bvenv\b|Python (?:package|librar)"
    r"|\binstall\b[^\n]*\b(?:package|librar)(?:y|ies|s)?\b)[^\n]*",
    re.IGNORECASE,
)
# A back-quoted package, optionally with `pip install`, extras or a version pin.
PACKAGE_NAME = re.compile(r"`(?:pip3? install )?([A-Za-z][A-Za-z0-9_-]*)(?:\[[^\]]*\])?(?:[=<>~!]=?[0-9][^`]*)?`")
NOT_PACKAGES = {"pip", "pip3", "python", "python3", "venv", "virtualenv", "pytest", "activate", "install", "nan", "none"}
# Protobuf/gRPC modules the agent generates with protoc.
GENERATED_SUFFIXES = ("_pb2", "_pb2_grpc")
PYTHON310_STDLIB = {"asynchat", "asyncore", "distutils", "imp", "smtpd"}
# Import names a distribution installs as its own dependencies (pytest on Python 3.10 pulls tomli).
DEPENDENCIES = {
    "pytest": {"tomli", "exceptiongroup", "iniconfig", "pluggy", "packaging"},
    "requests": {"urllib3", "idna", "certifi", "charset_normalizer"},
    "flask": {"werkzeug", "jinja2", "markupsafe", "itsdangerous", "click", "blinker"},
    "pandas": {"numpy", "dateutil", "pytz"},
    "scipy": {"numpy"},
    "mlflow": {"yaml", "numpy", "pandas", "sqlalchemy", "flask", "requests"},
    "grpcio-tools": {"grpc", "google"},
    "scikit-learn": {"numpy", "scipy", "joblib", "threadpoolctl"},
    "matplotlib": {"numpy", "PIL", "dateutil", "pyparsing", "cycler", "kiwisolver"},
}


def _imported_modules(source: str) -> set[str]:
    """Top-level modules imported by ``source``, except fallbacks imported in an ``except`` clause."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # invalid escapes in the task's own test code
            tree = ast.parse(source)
    except SyntaxError:
        return set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.MULTILINE))
    fallbacks = {
        id(node) for handler in ast.walk(tree) if isinstance(handler, ast.ExceptHandler) for node in ast.walk(handler)
    }
    modules = set()
    for node in ast.walk(tree):
        if id(node) in fallbacks:
            continue
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _local_module_pattern(module: str) -> str:
    """A file or folder of that name, an extension's ``PyInit_``, or the name as a quoted string (setup.py, Cargo)."""
    name = re.escape(module)
    return rf"(?<![\w.-]){name}\.(?:py|c|pyx|rs)\b|/{name}/|PyInit_{name}\b|[\"']{name}[\"']"


def _installed_by(setup: str, module: str) -> bool:
    names = {module, module.lower(), *DISTRIBUTIONS.get(module, ())}
    for name in names:
        word = re.escape(name)
        pip = rf"pip3?\s+install\b[^\n]*?(?<![\w.-]){word}(?![\w.-])"
        apt = rf"(?:apt-get|apt)\s+install\b[^\n]*?\bpython3-{word}(?![\w.-])"
        if re.search(pip, setup, re.IGNORECASE) or re.search(apt, setup, re.IGNORECASE):
            return True
    return False


def missing_test_imports(task: str, files: set[str], archive: zipfile.ZipFile) -> list[str]:
    """Third-party modules the verifier imports that the task's own setup never installs.

    With no egress in the sandbox, such a verifier fails on import whatever the agent does. Standard-library
    modules, pytest, and modules the task itself provides (named as a file, folder, ``PyInit_`` or quoted string in the
    instruction or setup, e.g. an extension the agent must build) are exempt, as are fallback imports in an ``except``.
    """

    instruction, setup = _instruction_and_setup(task, files, archive)
    tests = _read(archive, task, files, "tests/test_final_state.py")
    context = instruction + setup  # not the tests: they quote the modules they check
    candidates = _imported_modules(tests) - sys.stdlib_module_names - PYTHON310_STDLIB - TEST_RUNNER_MODULES
    missing = []
    for module in sorted(candidates - _implied(setup)):
        local = module.endswith(GENERATED_SUFFIXES) or re.search(_local_module_pattern(module), context)
        if not local and not _installed_by(setup, module):
            missing.append(module)
    return missing


def missing_instruction_packages(task: str, files: set[str], archive: zipfile.ZipFile) -> list[str]:
    """Back-quoted names in the instruction's Python-install sentences that the setup never installs.

    Such a task asks the agent to fetch packages (e.g. numpy and scipy into a venv), which a sandbox with no egress
    cannot do. A heuristic: it also flags tools the agent may use but does not need, so it trades a little yield for
    not spending sandbox time on tasks that cannot pass.
    """
    instruction, setup = _instruction_and_setup(task, files, archive)
    implied = _implied(setup)
    missing = set()
    for line in INSTALL_SENTENCE.findall(instruction):
        for name in PACKAGE_NAME.findall(line):
            module = name.replace("-", "_")
            if (
                name.lower() in NOT_PACKAGES
                or name.isupper()
                or module in sys.stdlib_module_names
                or {name, name.lower(), module} & implied
                or re.search(_local_module_pattern(name), setup)
                or _installed_by(setup, name)
                or _installed_by(setup, module)
                or re.search(rf"\b{re.escape(name)}\b", setup)  # installed some other way (apt, a source build)
            ):
                continue
            missing.add(name)
    return sorted(missing)


def _read(archive: zipfile.ZipFile, task: str, files: set[str], rel: str) -> str:
    return archive.read(f"{task}/{rel}").decode(errors="ignore") if rel in files else ""


def _instruction_and_setup(task: str, files: set[str], archive: zipfile.ZipFile) -> tuple[str, str]:
    """The instruction and the build text: the Dockerfile plus the install scripts it runs, continued lines joined."""
    dockerfile = _read(archive, task, files, "environment/Dockerfile")
    # A script counts only when the Dockerfile runs it (many tasks ship an unused base_install.sh).
    scripts = [rel for rel in BUILD_FILES[1:] if rel.rsplit("/", 1)[-1] in dockerfile]
    setup = re.sub(r"\\\n", " ", dockerfile + "".join(_read(archive, task, files, rel) for rel in scripts))
    return _read(archive, task, files, "instruction.md"), setup


def _implied(setup: str) -> set[str]:
    return set().union(*(deps for dist, deps in DEPENDENCIES.items() if _installed_by(setup, dist)))


def sft_hashes(archive: zipfile.ZipFile) -> set[str]:
    hashes = set()
    for name in archive.namelist():
        if name.endswith("/task.json"):
            hashes.add(_normalized_hash(json.loads(archive.read(name))["description"]))
        elif name.endswith("/test_final_state.py") and name.count("/") == 1:
            hashes.add(hashlib.sha256(archive.read(name)).hexdigest())
    return hashes


def screen(reserve: zipfile.ZipFile, sft: zipfile.ZipFile) -> list[dict]:
    known = sft_hashes(sft)
    rows = []
    for task, files in sorted(task_files(reserve).items()):
        instruction = reserve.read(f"{task}/instruction.md").decode(errors="ignore")
        tests = reserve.read(f"{task}/tests/test_final_state.py")
        metadata = tomllib.loads(reserve.read(f"{task}/task.toml").decode())["metadata"]
        hidden = hidden_assets(task, files, reserve)
        reasons = [rule for rule, paths in hidden.items() if paths]
        if any(rel.endswith(ORACLE_FIXTURE) for rel in files):
            reasons.append("oracle_fixture")
        if _normalized_hash(instruction) in known or hashlib.sha256(tests).hexdigest() in known:
            reasons.append("sft_overlap")
        missing = missing_test_imports(task, files, reserve)
        if missing:
            reasons.append("test_import_not_installed")
        packages = missing_instruction_packages(task, files, reserve)
        if packages:
            reasons.append("instruction_package_not_installed")
        rows.append(
            {
                "task": task,
                "domain": metadata["domain"],
                "excluded": reasons,
                "missing_test_imports": missing,
                "missing_instruction_packages": packages,
                **{f"hidden_{k}": v for k, v in hidden.items()},
            }
        )
    return rows


def tmax_docs(archive: zipfile.ZipFile, tasks: list[str], layout: str) -> list[dict]:
    """Pool P's TMax decon document: the instruction plus test_final_state.py."""
    if layout == "harbor":
        return [
            {
                "id": task,
                "text": (
                    archive.read(f"{task}/instruction.md").decode(errors="ignore")
                    + "\n\n"
                    + archive.read(f"{task}/tests/test_final_state.py").decode(errors="ignore")
                ),
            }
            for task in tasks
        ]
    return [
        {
            "id": f"tmax-sft/{task}",
            "text": (
                json.loads(archive.read(f"{task}/task.json"))["description"]
                + "\n\n"
                + archive.read(f"{task}/test_final_state.py").decode(errors="ignore")
            ),
        }
        for task in tasks
    ]


def draw(rows: list[dict], decon: dict[str, dict], size: int, seed: int) -> list[str]:
    """Seeded sample stratified by domain (largest remainder) from the screened, decon-clean tasks."""
    eligible = eligible_tasks(rows, decon)
    total = sum(len(v) for v in eligible.values())
    quotas = {d: size * len(v) / total for d, v in eligible.items()}
    counts = {d: int(q) for d, q in quotas.items()}
    for d in sorted(quotas, key=lambda d: (counts[d] - quotas[d], d))[: size - sum(counts.values())]:
        counts[d] += 1
    rng = random.Random(seed)
    picked = []
    for domain in sorted(eligible):
        picked += rng.sample(sorted(eligible[domain]), counts[domain])
    return sorted(picked)


def eligible_tasks(rows: list[dict], decon: dict[str, dict]) -> dict[str, list[str]]:
    eligible: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        if not row["excluded"] and not decon[row["task"]]["contaminated"]:
            eligible[row["domain"]].append(row["task"])
    return eligible


def replace_ineligible(rows: list[dict], decon: dict[str, dict], previous: list[str], seed: int) -> dict[str, str]:
    """Map each previously drawn task that is no longer eligible to a seeded replacement from its own domain.

    The rest of the previous draw stays, so only the replacements need new checks, and domain counts are unchanged.
    """
    eligible = eligible_tasks(rows, decon)
    domain = {row["task"]: row["domain"] for row in rows}
    keep = set(previous)
    rng = random.Random(seed)
    replacements = {}
    for task in sorted(t for t in previous if t not in eligible[domain[t]]):
        pool = sorted(set(eligible[domain[task]]) - keep)
        replacements[task] = rng.choice(pool)
        keep.add(replacements[task])
    return replacements


def rescreen(previous: str, output: str, seed: int) -> None:
    """Re-run the screen with the current rules, reusing ``previous``'s decon, and replace newly excluded draws."""
    prev = StoragePath(previous)
    decon = json.loads((prev / "decon.json").read_text())
    old = json.loads((prev / "draw.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rows = screen(hub_zip(*RESERVE, work / "reserve.zip"), hub_zip(*SFT_ENVS, work / "sft.zip"))
    # The new rules only add exclusions, so every task they leave in was screened clean before and has decon marks.
    missing = {row["task"] for row in rows if not row["excluded"]} - set(decon)
    if missing:
        raise ValueError(f"{len(missing)} screened tasks have no decon marks; run select instead")
    replacements = replace_ineligible(rows, decon, old["tasks"], seed)
    picked = sorted((set(old["tasks"]) - set(replacements)) | set(replacements.values()))
    reasons = collections.Counter(reason for row in rows for reason in row["excluded"])
    eligible = sum(len(v) for v in eligible_tasks(rows, decon).values())
    summary = {
        "reserve": f"{RESERVE[0]}@{RESERVE[1]}",
        "previous": previous,
        "reserve_tasks": len(rows),
        "excluded_any": sum(bool(row["excluded"]) for row in rows),
        "excluded_by_reason": dict(reasons),
        "eligible": eligible,
        "seed": seed,
        "replacements": replacements,
        "drawn": len(picked),
    }
    out = StoragePath(output)
    (out / "screen.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "decon.json").write_text(json.dumps(decon, indent=1))
    (out / "draw.json").write_text(json.dumps({"tasks": picked, **summary}, indent=1))
    print(json.dumps(summary, indent=1))


def select(output: str, size: int, seed: int) -> None:
    out = StoragePath(output)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        reserve = hub_zip(*RESERVE, work / "reserve.zip")
        sft = hub_zip(*SFT_ENVS, work / "sft.zip")
        rows = screen(reserve, sft)
        clean = [row["task"] for row in rows if not row["excluded"]]
        evals = eval_documents(work)
        sft_tasks = sorted({name.split("/")[0] for name in sft.namelist() if name.endswith("/task.json")})
        evals["tmax-sft-2200"] = tmax_docs(sft, sft_tasks, "sft")
        decon = decon_marks(tmax_docs(reserve, clean, "harbor"), evals, work / "decon")
    picked = draw(rows, decon, size, seed)
    reasons = collections.Counter(reason for row in rows for reason in row["excluded"])
    summary = {
        "reserve": f"{RESERVE[0]}@{RESERVE[1]}",
        "sft_envs": f"{SFT_ENVS[0]}@{SFT_ENVS[1]}",
        "reserve_tasks": len(rows),
        "excluded_any": sum(bool(row["excluded"]) for row in rows),
        "excluded_by_reason": dict(reasons),
        "screened": len(clean),
        "decon_contaminated": sorted(t for t, d in decon.items() if d["contaminated"]),
        "decon_near": {t: d["max_overlap"] for t, d in decon.items() if d["max_overlap"] >= 0.2},
        "eval_documents": {name: len(docs) for name, docs in evals.items()},
        "seed": seed,
        "drawn": len(picked),
    }
    (out / "screen.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "decon.json").write_text(json.dumps(decon, indent=1))
    (out / "draw.json").write_text(json.dumps({"tasks": picked, **summary}, indent=1))
    print(json.dumps(summary, indent=1))


def shell_form_copy(dockerfile: str) -> str:
    """Rewrite ``COPY ["src", "dst"]`` as ``COPY src dst``; Docker treats the two the same when no path has a space.

    The Daytona SDK's build-context parser (``daytona.common.image._parse_copy_command``) leaves the comma on each
    JSON-form source path, so the snapshot build fails with "Path does not exist".
    """

    def rewrite(match: re.Match) -> str:
        paths = json.loads(match.group(2))
        if any(not isinstance(path, str) or re.search(r"\s", path) for path in paths):
            raise ValueError(f"cannot rewrite COPY with a space in a path: {match.group(0)!r}")
        return match.group(1) + " ".join(paths)

    return JSON_COPY.sub(rewrite, dockerfile)


def package(reserve: zipfile.ZipFile, tasks: list[str], dest: Path) -> dict:
    """Write the drawn tasks as Harbor task folders: tmux baked into each image and shell-form COPY lines."""
    dest.mkdir(parents=True)
    files = task_files(reserve)
    digests = {}
    for task in tasks:
        digest = hashlib.sha256()
        for rel in sorted(files[task]):
            data = reserve.read(f"{task}/{rel}")
            if rel == "environment/Dockerfile":
                data = shell_form_copy(data.decode()).rstrip("\n").encode() + b"\n" + TMUX_BAKE.encode() + b"\n"
            path = dest / task / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            digest.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
        digests[task] = digest.hexdigest()
    manifest = {
        "reserve": f"{RESERVE[0]}@{RESERVE[1]}",
        "environment_changes": [TMUX_BAKE, "JSON-form COPY rewritten to shell form"],
        "tasks": digests,
        "sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
    }
    (dest.parent / f"{dest.name}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select", help="Screen, decon and draw; write outputs to --output.")
    select_parser.add_argument("--output", required=True, help="Directory (local or remote) for the outputs.")
    select_parser.add_argument("--size", type=int, default=200)
    select_parser.add_argument("--seed", type=int, default=20261113)
    rescreen_parser = commands.add_parser(
        "rescreen", help="Re-screen with the current rules and replace newly excluded draws from the same domain."
    )
    rescreen_parser.add_argument("--previous", required=True, help="Directory a previous select wrote.")
    rescreen_parser.add_argument("--output", required=True, help="Directory (local or remote) for the outputs.")
    rescreen_parser.add_argument("--seed", type=int, default=20261114)
    package_parser = commands.add_parser("package", help="Write the drawn tasks as local Harbor task folders.")
    package_parser.add_argument("--draw", required=True, help="draw.json written by select (local or remote).")
    package_parser.add_argument("--dest", required=True, type=Path, help="New directory for the task folders.")
    args = parser.parse_args()
    if args.command == "select":
        select(args.output, args.size, args.seed)
        return
    if args.command == "rescreen":
        rescreen(args.previous, args.output, args.seed)
        return
    tasks = json.loads(StoragePath(args.draw).read_text())["tasks"]
    with tempfile.TemporaryDirectory() as tmp:
        manifest = package(hub_zip(*RESERVE, Path(tmp) / "reserve.zip"), tasks, args.dest)
    print(json.dumps({"tasks": len(manifest["tasks"]), "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
