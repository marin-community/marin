# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage SkyRL source and active work without sharing Marin's dependency resolution.

Run from the Marin root with ``uv run --no-project python -m scripts.skyrl_migration --help``.
Only the two publication commands, with --execute, write to GitHub.
"""

import argparse
import ast
import base64
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from infra.ci.select_tests import path_to_module, resolve

SOURCE = "marin-community/MarinSkyRL"
TARGET = "marin-community/marin"
VENDOR = "vendor/skyrl"
BUNDLE_MANIFEST = "cloud/iris/runtime_bundle_files.txt"
PROVENANCE = "marinskyrl/_vendor_provenance.json"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/skyrl-migration.toml"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_CONTENT_SHA256 = "0a7389c2f2827aa8a29e82dd296afb85ce0ac0c3a02541ac975f00c3c57f0f5b"


def run(
    *args: str, cwd: Path | None = None, data: bytes | None = None, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(args, cwd=cwd, input=data, capture_output=True, check=check)
    except subprocess.CalledProcessError as error:
        error.add_note((error.stderr or error.stdout).decode(errors="replace"))
        raise


def git(root: Path, *args: str) -> str:
    return run("git", "-C", str(root), *args).stdout.decode().strip()


def gh(endpoint: str, *, method: str = "GET", payload: dict | None = None) -> Any:
    args = ["gh", "api", "--method", method, endpoint]
    if payload is not None:
        args += ["--input", "-"]
    result = json.loads(run(*args, data=json.dumps(payload).encode() if payload is not None else None).stdout)
    if isinstance(result, dict) and result.get("errors"):
        raise RuntimeError(result["errors"])
    return result


def pages(endpoint: str) -> list[dict]:
    return [item for page in json.loads(run("gh", "api", "--paginate", "--slurp", endpoint).stdout) for item in page]


def graphql(query: str, **variables: str) -> dict:
    return gh("graphql", method="POST", payload={"query": query, "variables": variables})["data"]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts or not path.parts:
        raise ValueError(f"Path must stay inside the candidate tree: {value}")
    return path.as_posix()


def file_digest(path: Path) -> str:
    data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def tree_paths(objects: Path, revision: str) -> set[str]:
    names = run("git", "-C", str(objects), "ls-tree", "-r", "--name-only", "-z", revision).stdout
    return {name.decode() for name in names.split(b"\0") if name}


def verify_source(source: Path, objects: Path, revision: str, paths: dict[str, list[str]]) -> None:
    entries = run("git", "-C", str(objects), "ls-tree", "-r", "-z", revision).stdout
    expected = {}
    for entry in entries.split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, _, blob = metadata.decode().split()
            expected[name.decode()] = (mode, blob)
    for name in paths:
        path = source / relative_path(name)
        data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        digest = run("git", "-C", str(objects), "hash-object", "--stdin", data=data).stdout.decode().strip()
        mode = "120000" if path.is_symlink() else "100755" if path.stat().st_mode & 0o111 else "100644"
        if expected[name] != (mode, digest):
            raise ValueError(f"Snapshot source changed after capture: {name}")


def extract_source(objects: Path, revision: str, destination: Path) -> None:
    destination.mkdir()
    archive = run("git", "-C", str(objects), "archive", revision).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")
        # Python's data filter normalizes link text, including trailing slashes.
        # Restore the Git bytes after containment validation so provenance stays exact.
        for member in tar.getmembers():
            if member.issym():
                path = destination / relative_path(member.name)
                (path.parent / member.linkname).resolve().relative_to(destination.resolve())
                path.unlink()
                path.symlink_to(member.linkname)


def snapshot(output: Path, ref: str, days: int, now: datetime) -> None:
    """Capture immutable Git objects and paginated open work; never write to GitHub."""
    output.mkdir(parents=True, exist_ok=False)
    cutoff_time = now - timedelta(days=days)
    cutoff = cutoff_time.isoformat().replace("+00:00", "Z")
    repository = gh(f"repos/{SOURCE}")
    source_sha = gh(f"repos/{SOURCE}/commits/{quote(ref, safe='')}")["sha"]
    pulls = pages(f"repos/{SOURCE}/pulls?state=open&per_page=100")
    issues = [
        issue
        for issue in pages(f"repos/{SOURCE}/issues?state=open&since={quote(cutoff)}&per_page=100")
        if "pull_request" not in issue and datetime.fromisoformat(issue["updated_at"]) >= cutoff_time
    ]

    def capture_pull(pull: dict) -> dict:
        number = pull["number"]
        base, head = pull["base"]["sha"], pull["head"]["sha"]
        comparison = gh(f"repos/{SOURCE}/compare/{base}...{head}")
        return pull | {
            "merge_base": comparison["merge_base_commit"]["sha"],
            "comments": pages(f"repos/{SOURCE}/issues/{number}/comments?per_page=100"),
            "reviews": pages(f"repos/{SOURCE}/pulls/{number}/reviews?per_page=100"),
            "review_comments": pages(f"repos/{SOURCE}/pulls/{number}/comments?per_page=100"),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        pulls = list(pool.map(capture_pull, pulls))
        comments = list(
            pool.map(lambda issue: pages(f"repos/{SOURCE}/issues/{issue['number']}/comments?per_page=100"), issues)
        )
    issues = [issue | {"comments": comment} for issue, comment in zip(issues, comments, strict=True)]
    objects = output / "source.git"
    run("git", "init", "--bare", "--quiet", str(objects))
    refs = [f"{source_sha}:refs/snapshot/main"]
    for pull in pulls:
        number = pull["number"]
        refs += [
            f"{pull['head']['sha']}:refs/snapshot/pr-{number}-head",
            f"{pull['merge_base']}:refs/snapshot/pr-{number}-base",
        ]
    git(objects, "fetch", "--quiet", "--no-tags", "--depth=1", f"https://github.com/{SOURCE}.git", *refs)
    source = output / "source"
    extract_source(objects, source_sha, source)
    for pull in pulls:
        base, head = pull["merge_base"], pull["head"]["sha"]
        changed = run("git", "-C", str(objects), "diff", "--name-only", "--no-renames", "-z", base, head).stdout
        pull["paths"] = [relative_path(path.decode()) for path in changed.split(b"\0") if path]
        patch = run("git", "-C", str(objects), "diff", "--binary", "--full-index", "-M", base, head).stdout
        (output / f"pr-{pull['number']}.patch").write_bytes(patch)
        pull["patch_sha256"] = hashlib.sha256(patch).hexdigest()
    write_json(
        output / "snapshot.json",
        {
            "schema": 1,
            "source": SOURCE,
            "target": TARGET,
            "source_sha": source_sha,
            "default_branch": repository["default_branch"],
            "as_of": now.isoformat(),
            "cutoff": cutoff,
            "pulls": pulls,
            "issues": issues,
        },
    )


@dataclass
class Selection:
    retained: dict[str, list[str]]
    discarded: list[str]
    dynamic_imports: list[str]
    missing_imports: list[str]


def module_imports(path: Path, module: str) -> tuple[set[str], list[str], set[str]]:
    """Include nested optional imports and literal dynamic imports in the closure."""
    names: set[str] = set()
    required: set[str] = set()
    dynamic: list[str] = []
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
            required.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
            if node.module:
                required.add(base)
        elif isinstance(node, ast.Call):
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr if isinstance(function, ast.Attribute) else ""
            )
            if name not in ("import_module", "__import__"):
                continue
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                name = node.args[0].value
                if name.startswith("."):
                    name = resolve_name(name, package)
                names.add(name)
                required.add(name)
            else:
                dynamic.append(f"{module}:{node.lineno}")
    return names, dynamic, required


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def selection(source: Path, manifest: dict, marin: Path | None) -> Selection:
    """Mark files from explicit roots, runtime assets, and transitive Python imports."""
    files = {
        path.relative_to(source).as_posix(): path for path in source.rglob("*") if path.is_file() or path.is_symlink()
    }
    modules: dict[str, Path] = {}
    for package in manifest["packages"]:
        for path in (source / package["directory"]).rglob("*.py"):
            module = path_to_module(path, source / package["import_root"])
            if module:
                if module in modules:
                    raise ValueError(f"Ambiguous module: {module}")
                modules[module] = path
    by_path = {path.relative_to(source).as_posix(): module for module, path in modules.items()}
    retained: dict[str, list[str]] = {}
    pending: list[str] = []

    def mark(path: str, reason: str) -> None:
        path = relative_path(path)
        if path not in files:
            raise FileNotFoundError(f"Retention root does not exist: {path}")
        if matches(path, manifest["discard"]):
            raise ValueError(f"Retained dependency contradicts discard policy: {path}: {reason}")
        if path not in retained:
            retained[path] = []
            pending.append(path)
        if reason not in retained[path]:
            retained[path].append(reason)

    for rule in manifest["keep"]:
        matched = [path for path in files if fnmatch.fnmatchcase(path, rule["glob"])]
        if not matched:
            raise FileNotFoundError(f"Retention glob matched no files: {rule['glob']}")
        for path in matched:
            mark(path, rule["reason"])
    for root in manifest["roots"]:
        mark(modules[root].relative_to(source).as_posix(), f"supported entrypoint: {root}")
    for line in (source / BUNDLE_MANIFEST).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            mark(line.strip(), "launcher runtime bundle")
    known = set(modules)
    namespaces = {".".join(name.split(".")[:length]) for name in known for length in range(1, len(name.split(".")))}
    if marin is not None:
        found = run(
            "rg",
            "-l",
            "--glob",
            "*.py",
            r"(^|\s)(from|import) (skyrl_train|skyrl_gym|marinskyrl|cloud\.iris)",
            "experiments",
            "lib",
            "tests",
            cwd=marin,
            check=False,
        )
        if found.returncode not in (0, 1):
            found.check_returncode()
        for filename in found.stdout.decode().splitlines():
            names, _, _ = module_imports(marin / filename, path_to_module(marin / filename, marin) or "")
            for name in resolve(names, known):
                mark(modules[name].relative_to(source).as_posix(), f"Marin caller: {filename}")
    dynamic: set[str] = set()
    missing: set[str] = set()
    local_packages = {module.split(".")[0] for module in known}
    visited: set[str] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for parent in PurePosixPath(path).parents:
            for helper in (parent / "__init__.py", parent / "conftest.py"):
                if helper.as_posix() in files:
                    mark(helper.as_posix(), f"package/test setup for {path}")
        if path not in by_path or files[path].is_symlink():
            continue
        names, unresolved, required = module_imports(files[path], by_path[path])
        dynamic.update(unresolved)
        missing.update(
            f"{path}: {name}"
            for name in required
            if name.split(".")[0] in local_packages and name not in known and name not in namespaces
        )
        for name in resolve(names, known):
            mark(modules[name].relative_to(source).as_posix(), f"imported by {path}")
    return Selection(
        dict(sorted(retained.items())), sorted(set(files) - retained.keys()), sorted(dynamic), sorted(missing)
    )


def plan(output: Path, manifest_path: Path, marin: Path | None) -> None:
    manifest = tomllib.loads(manifest_path.read_text())
    chosen = selection(output / "source", manifest, marin)
    shutil.copyfile(manifest_path, output / "retention.toml")
    patches = []
    for name in manifest.get("patches", []):
        path = manifest_path.parent / relative_path(name)
        target = output / "migration-patches" / relative_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        patches.append({"path": name, "sha256": file_digest(target)})
    write_json(output / "selection.json", asdict(chosen) | {"patches": patches})


def parent_pulls(pulls: list[dict], default_branch: str) -> dict[int, int | None]:
    branches = {
        (pull["head"]["repo"]["full_name"] if pull["head"]["repo"] else SOURCE, pull["head"]["ref"]): pull["number"]
        for pull in pulls
    }
    return {
        pull["number"]: (
            branches.get((pull["base"]["repo"]["full_name"], pull["base"]["ref"]))
            if pull["base"]["ref"] != default_branch
            else None
        )
        for pull in pulls
    }


def ordered_pulls(pulls: list[dict], parents: dict[int, int | None]) -> list[dict]:
    ordered: list[dict] = []
    pending = {pull["number"]: pull for pull in pulls}
    while pending:
        ready = sorted(number for number in pending if parents[number] not in pending)
        if not ready:
            raise ValueError("PR base refs form a cycle")
        for number in ready:
            ordered.append(pending.pop(number))
    return ordered


def commit(root: Path, message: str) -> str:
    git(root, "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "--no-gpg-sign", "-m", message)
    return git(root, "rev-parse", "HEAD")


def stage(snapshot_dir: Path, output: Path) -> None:
    """Copy the candidate and test each complete Git delta on its staged parent."""
    snapshot_dir = snapshot_dir.resolve()
    record = read_json(snapshot_dir / "snapshot.json")
    chosen = read_json(snapshot_dir / "selection.json")
    manifest = tomllib.loads((snapshot_dir / "retention.toml").read_text())
    output.mkdir(parents=True, exist_ok=False)
    tree = output / "tree"
    destination = tree / VENDOR
    source = snapshot_dir / "source"
    objects = snapshot_dir / "source.git"
    verify_source(source, objects, record["source_sha"], chosen["retained"])
    for name in chosen["retained"]:
        original, target = source / relative_path(name), destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_symlink():
            original.resolve().relative_to(source)
            target.symlink_to(os.readlink(original))
        else:
            shutil.copy2(original, target)
    hashes = {name: file_digest(source / name) for name in chosen["retained"]}
    (destination / PROVENANCE).parent.mkdir(parents=True, exist_ok=True)
    write_json(destination / PROVENANCE, {"source": SOURCE, "source_sha": record["source_sha"], "files": hashes})
    run("git", "init", "--quiet", "--initial-branch=main", str(tree))
    git(tree, "add", ".")
    baseline = commit(tree, "Stage selected SkyRL sources")
    for patch in chosen["patches"]:
        path = snapshot_dir / "migration-patches" / patch["path"]
        if file_digest(path) != patch["sha256"]:
            raise ValueError(f"Migration patch changed after capture: {path}")
        run("git", "-C", str(tree), "apply", "--index", f"--directory={VENDOR}", str(path))
    if chosen["patches"]:
        hashes = {name: file_digest(destination / name) for name in hashes}
        for name in git(tree, "diff", "--cached", "--name-only").splitlines():
            relative = Path(name).relative_to(VENDOR).as_posix()
            hashes[relative] = file_digest(destination / relative)
        write_json(
            destination / PROVENANCE,
            {"source": SOURCE, "source_sha": record["source_sha"], "patches": chosen["patches"], "files": hashes},
        )
        git(tree, "add", ".")
        baseline = commit(tree, "Apply migration runtime fixes")
    git(tree, "fetch", "--quiet", "--no-tags", str(snapshot_dir / "source.git"), "+refs/snapshot/*:refs/source/*")
    parents = parent_pulls(record["pulls"], record["default_branch"])
    outcomes: dict[int, dict] = {}
    for pull in ordered_pulls(record["pulls"], parents):
        number, parent = pull["number"], parents[pull["number"]]
        branch = f"skyrl-migration/pr-{number}"
        outcome = {"number": number, "url": pull["html_url"], "parent": parent, "branch": branch}
        outcomes[number] = outcome
        if parent is not None and outcomes[parent]["status"] != "ready":
            outcome.update(status="blocked-parent")
            continue
        if parent is None and pull["base"]["ref"] != record["default_branch"]:
            outcome.update(status="unmapped-base", base=pull["base"]["ref"])
            continue
        selected, excluded = [], []
        before = tree_paths(objects, pull["merge_base"])
        after = tree_paths(objects, pull["head"]["sha"])
        for path in pull["paths"]:
            new = path not in before and path in after
            in_package = any(
                PurePosixPath(path).is_relative_to(package["directory"]) for package in manifest["packages"]
            )
            accepted = path in chosen["retained"] or (new and in_package and not matches(path, manifest["discard"]))
            (selected if accepted else excluded).append(path)
        outcome.update(selected=selected, excluded=excluded)
        if not selected:
            outcome.update(status="excluded")
            continue
        base = outcomes[parent]["commit"] if parent is not None else baseline
        git(tree, "checkout", "--quiet", "-b", branch, base)
        patch = run(
            "git",
            "-C",
            str(snapshot_dir / "source.git"),
            "diff",
            "--binary",
            "--full-index",
            "-M",
            pull["merge_base"],
            pull["head"]["sha"],
            "--",
            *selected,
        ).stdout
        patch_path = output / f"pr-{number}-retained.patch"
        patch_path.write_bytes(patch)
        applied = run(
            "git",
            "-C",
            str(tree),
            "apply",
            "--3way",
            "--index",
            f"--directory={VENDOR}",
            "--whitespace=nowarn",
            str(patch_path.resolve()),
            check=False,
        )
        if applied.returncode:
            outcome.update(status="conflict", diagnostic=applied.stderr.decode(errors="replace"))
            git(tree, "reset", "--hard", "--quiet", base)
            git(tree, "clean", "-fd", "--quiet")
            continue
        if not git(tree, "diff", "--cached", "--name-only"):
            outcome.update(status="already-applied")
            continue
        outcome.update(
            status="needs-decision" if excluded else "ready",
            commit=commit(tree, f"Transplant {SOURCE}#{number}\n\nSource-head: {pull['head']['sha']}"),
        )
    git(tree, "checkout", "--quiet", "main")
    report = {
        "snapshot": str(snapshot_dir),
        "source_sha": record["source_sha"],
        "patches": chosen["patches"],
        "retained": len(chosen["retained"]),
        "discarded": len(chosen["discarded"]),
        "pulls": list(outcomes.values()),
        "issues": [{"number": issue["number"], "url": issue["html_url"]} for issue in record["issues"]],
        "dynamic_imports": chosen["dynamic_imports"],
        "missing_imports": chosen["missing_imports"],
        "production_ready": False,
    }
    write_json(output / "report.json", report)
    rows = [
        "# SkyRL migration dry run",
        "",
        f"Source: `{record['source_sha']}`. Retained {report['retained']} files; discarded {report['discarded']}.",
        "",
        "This is a source candidate. Runtime bootstrap and wheel-based CI acceptance remain required.",
        "",
        "| PR | parent | outcome | excluded paths |",
        "| --- | --- | --- | --- |",
    ]
    rows += [
        f"| [#{item['number']}]({item['url']}) | {item['parent'] or ''} | "
        f"{item['status']} | {len(item.get('excluded', []))} |"
        for item in outcomes.values()
    ]
    rows += [
        "",
        f"Selected issues: {len(record['issues'])}. Complete source patches and review history remain in the snapshot.",
        "",
        "See docs/dev-guide/skyrl-migration.md for cutover and archive requirements.",
    ]
    (output / "report.md").write_text("\n".join(rows) + "\n")


def audit_wheel(wheel: Path) -> list[str]:
    """Check that every retained runtime file is packaged with its recorded bytes."""
    with zipfile.ZipFile(wheel) as archive:
        members = sorted(archive.namelist())
        if PROVENANCE not in members or BUNDLE_MANIFEST not in members:
            raise ValueError("Wheel is missing runtime provenance or the launcher bundle manifest")
        provenance = json.loads(archive.read(PROVENANCE))
        expected = {PROVENANCE}
        runtime_packages = {"cloud", "marinskyrl", "skyrl_train", "skyrl_gym", "chat_templates"}
        for name, digest in provenance["files"].items():
            packaged = name.removeprefix("skyrl-train/").removeprefix("skyrl-gym/")
            if packaged.split("/")[0] not in runtime_packages:
                continue
            if name.endswith(".pyc"):
                continue
            expected.add(packaged)
            if packaged not in members or hashlib.sha256(archive.read(packaged)).hexdigest() != digest:
                raise ValueError(f"Wheel member missing or changed: {packaged}")
        unexpected = {
            name for name in members if name.split("/")[0] in runtime_packages and not name.endswith("/")
        } - expected
        if unexpected:
            raise ValueError(f"Wheel contains runtime files outside staged provenance: {sorted(unexpected)}")
    return members


def verify_vendor(source: Path) -> None:
    """Reject source drift before building the candidate."""
    provenance = read_json(source / PROVENANCE)
    for name, digest in provenance["files"].items():
        path = source / relative_path(name)
        path.resolve().relative_to(source.resolve())
        if file_digest(path) != digest:
            raise ValueError(f"Vendored source changed after staging: {name}")


def check_vendor(source: Path) -> None:
    """Check the independent lock and build the wheel without installing SkyRL."""
    verify_vendor(source)
    for patch in read_json(source / PROVENANCE).get("patches", []):
        if file_digest(ROOT / "config" / relative_path(patch["path"])) != patch["sha256"]:
            raise ValueError(f"Migration patch differs from the staged candidate: {patch['path']}")
    run("uv", "lock", "--project", str(source), "--check")
    with tempfile.TemporaryDirectory(prefix="skyrl-wheel-") as temporary:
        run("uv", "build", "--wheel", "--project", str(source), "--out-dir", temporary)
        [wheel] = Path(temporary).glob("*.whl")
        members = audit_wheel(wheel)
        print(json.dumps({"wheel_sha256": file_digest(wheel), "members": len(members)}))


def build_wheel(candidate: Path) -> None:
    """Build and inventory a binary candidate without resolving or installing its dependencies."""
    source = candidate.resolve() / "tree" / VENDOR
    verify_vendor(source)
    wheels = candidate.resolve() / "wheels"
    wheels.mkdir(exist_ok=False)
    run("uv", "build", "--wheel", "--project", str(source), "--out-dir", str(wheels))
    [wheel] = wheels.glob("*.whl")
    members = audit_wheel(wheel)
    write_json(candidate / "wheel.json", {"path": str(wheel), "sha256": file_digest(wheel), "members": members})
    runtime_dependencies = candidate / "runtime-dependencies"
    runtime_dependencies.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(source / name, runtime_dependencies / name)
    external = candidate / "binary-consumer"
    external.mkdir()
    source_policy = tomllib.loads((source / "pyproject.toml").read_text())["tool"]["uv"]
    # Wheel metadata omits uv policy; preserve the qualified sibling package versions as well as overrides.
    frozen_base = run(
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-hashes",
        "--no-header",
        "--project",
        str(source),
    ).stdout.decode()
    constraints = [
        re.sub(r"^([A-Za-z0-9_.-]+)\[[^]]+\]", r"\1", line)
        for line in frozen_base.splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    policy = {
        "environments": ["sys_platform == 'linux' and platform_machine == 'x86_64'"],
        "required-environments": ["sys_platform == 'linux' and platform_machine == 'x86_64'"],
        "override-dependencies": source_policy.get("override-dependencies", []),
        "constraint-dependencies": source_policy.get("constraint-dependencies", []) + constraints,
    }
    policy_lines = "".join(f"{key} = {json.dumps(value)}\n" for key, value in policy.items())
    (external / "pyproject.toml").write_text(
        '[project]\nname = "skyrl-migration-binary-consumer"\nversion = "0"\nrequires-python = "==3.12.*"\n'
        'dependencies = ["marinskyrl"]\n\n[tool.uv]\npackage = false\nprerelease = "allow"\n'
        + policy_lines
        + '\n[tool.uv.sources]\nmarinskyrl = { path = "../wheels/'
        + wheel.name
        + '" }\n'
    )


def prepare_math500(candidate: Path, dataset: Path, output: Path) -> None:
    """Bundle only the candidate wheel, standalone dependencies, and Math500 smoke inputs."""
    rows = read_json(dataset)
    content = json.dumps(
        [[row["unique_id"], row["problem"], row["answer"]] for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(content).hexdigest() != MATH500_CONTENT_SHA256:
        raise ValueError("Dataset does not match the complete, ordered pinned Math500 test split")
    output.mkdir(parents=True, exist_ok=False)
    wheel_record = read_json(candidate / "wheel.json")
    wheel = Path(wheel_record["path"])
    if file_digest(wheel) != wheel_record["sha256"]:
        raise ValueError("Candidate wheel changed after its inventory audit")
    with zipfile.ZipFile(wheel) as archive:
        provenance = json.loads(archive.read(PROVENANCE))
    for name in ("pyproject.toml", "uv.lock"):
        if file_digest(candidate / "runtime-dependencies" / name) != provenance["files"][name]:
            raise ValueError(f"Runtime dependency manifest changed after staging: {name}")
    for name, digest in provenance["files"].items():
        if name.startswith("skyrl-train/tests/") and file_digest(candidate / "tree" / VENDOR / name) != digest:
            raise ValueError(f"Retained acceptance test changed after staging: {name}")
    shutil.copyfile(wheel, output / wheel.name)
    shutil.copytree(candidate / "runtime-dependencies", output / "dependencies")
    shutil.copyfile(dataset, output / "math500.json")
    for name in ("run.sh", "run.py", "check_dependencies.py"):
        shutil.copyfile(ROOT / "config/skyrl-migration/smoke" / name, output / name)
    shutil.copytree(candidate / "tree" / VENDOR / "skyrl-train/tests", output / "acceptance/tests")
    shutil.copyfile(ROOT / "config/skyrl-migration/smoke/conftest.py", output / "acceptance/conftest.py")
    (output / "pyproject.toml").write_text(
        '[project]\nname = "skyrl-math500-smoke"\nversion = "0"\nrequires-python = "==3.12.*"\n'
        "\n[tool.uv]\npackage = false\n"
    )
    write_json(
        output / "receipt.json",
        {
            "provenance": provenance,
            "wheel_sha256": wheel_record["sha256"],
            "dependency_lock_sha256": file_digest(output / "dependencies/uv.lock"),
            "runtime_extras": ["megatron", "vllm", "telemetry"],
            "grading_protocol": "skyrl-aime-strict-box",
            "dataset": "HuggingFaceH4/MATH-500",
            "dataset_revision": MATH500_REVISION,
            "dataset_content_sha256": MATH500_CONTENT_SHA256,
            "dataset_sha256": file_digest(output / "math500.json"),
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        },
    )


def validate_cutover(number: int, source_sha: str, patches: list[dict]) -> dict:
    pull = gh(f"repos/{TARGET}/pulls/{number}")
    if not pull["merged"] or pull["base"]["repo"]["full_name"] != TARGET:
        raise ValueError("Publication requires a merged Marin cutover PR")
    provenance = gh(f"repos/{TARGET}/contents/{VENDOR}/{PROVENANCE}?ref={pull['merge_commit_sha']}")
    identity = json.loads(base64.b64decode(provenance["content"]))
    if identity["source"] != SOURCE or identity["source_sha"] != source_sha:
        raise ValueError("Merged cutover provenance does not match the reviewed source snapshot")
    if identity.get("patches", []) != patches:
        raise ValueError("Merged cutover migration fixes do not match the reviewed candidate")
    return pull


def validate_fresh(record: dict) -> None:
    head = gh(f"repos/{SOURCE}/commits/{quote(record['default_branch'], safe='')}")["sha"]
    pulls = pages(f"repos/{SOURCE}/pulls?state=open&per_page=100")
    actual = {pull["number"]: (pull["head"]["sha"], pull["base"]["sha"], pull["base"]["ref"]) for pull in pulls}
    expected = {
        pull["number"]: (pull["head"]["sha"], pull["base"]["sha"], pull["base"]["ref"]) for pull in record["pulls"]
    }
    if head != record["source_sha"] or actual != expected:
        raise ValueError("Fork main or open PRs changed; resnapshot and stage before publication")


def source_references(body: str) -> str:
    return re.sub(
        r"(?<![\w/])(?:#|GH-)(\d+)\b",
        lambda match: f"https://github.com/{SOURCE}/issues/{match[1]}",
        body,
    )


def transfer_issues(snapshot_dir: Path, state_path: Path, cutover: int) -> None:
    """Transfer selected issues, recovering destination identity after an interrupted transfer."""
    record = read_json(snapshot_dir / "snapshot.json")
    validate_cutover(cutover, record["source_sha"], read_json(snapshot_dir / "selection.json")["patches"])
    validate_fresh(record)
    owner, name = TARGET.split("/")
    repository_id = graphql(
        "query($owner:String!,$name:String!){repository(owner:$owner,name:$name){id}}", owner=owner, name=name
    )["repository"]["id"]
    state = read_json(state_path) if state_path.exists() else {}
    for issue in record["issues"]:
        node = graphql(
            "query($id:ID!){node(id:$id){... on Issue{id number url body updatedAt state repository{nameWithOwner}}}}",
            id=issue["node_id"],
        )["node"]
        origin = issue["html_url"]
        body = source_references(issue["body"] or "")
        body += f"\n\nTransferred from {origin}.\n<!-- skyrl-migration: issue={issue['node_id']} -->\n"
        if node["repository"]["nameWithOwner"] == SOURCE:
            if node["updatedAt"] != issue["updated_at"] or node["state"] != "OPEN":
                raise ValueError(f"Issue changed since snapshot: {origin}")
            node = graphql(
                "mutation($id:ID!,$repository:ID!){transferIssue(input:{issueId:$id,repositoryId:$repository,"
                "createLabelsIfMissing:true}){issue{id number url body repository{nameWithOwner}}}}",
                id=issue["node_id"],
                repository=repository_id,
            )["transferIssue"]["issue"]
        if node["repository"]["nameWithOwner"] != TARGET:
            raise ValueError(f"Issue moved to an unexpected repository: {origin}")
        if node["body"] != body:
            if (node["body"] or "") != (issue["body"] or ""):
                raise ValueError(f"Transferred issue body changed; reconcile manually: {node['url']}")
            gh(f"repos/{TARGET}/issues/{node['number']}", method="PATCH", payload={"body": body})
        state[issue["node_id"]] = {"source": origin, "destination": node["url"]}
        write_json(state_path, state)


def publish_prs(candidate: Path, cutover: int) -> None:
    """Create draft PRs with preserved stack bases, without replacing existing branches."""
    report = read_json(candidate / "report.json")
    snapshot_dir = Path(report["snapshot"])
    record = read_json(snapshot_dir / "snapshot.json")
    cutover_pull = validate_cutover(cutover, record["source_sha"], report["patches"])
    validate_fresh(record)
    incomplete = [item["number"] for item in report["pulls"] if item["status"] != "ready"]
    target = gh(f"repos/{TARGET}")
    default = target["default_branch"]
    workspace = candidate / "publication"
    if not workspace.exists():
        run(
            "git",
            "clone",
            "--quiet",
            "--filter=blob:none",
            "--no-checkout",
            "--single-branch",
            f"https://github.com/{TARGET}.git",
            str(workspace),
        )
    git(workspace, "fetch", "--quiet", "origin", cutover_pull["merge_commit_sha"])
    target_base = git(workspace, "rev-parse", "FETCH_HEAD")
    git(workspace, "fetch", "--quiet", "--no-tags", str(snapshot_dir / "source.git"), "+refs/snapshot/*:refs/source/*")
    labels = pages(f"repos/{TARGET}/labels?per_page=100")
    if not any(label["name"] == "agent-generated" for label in labels):
        gh(f"repos/{TARGET}/labels", method="POST", payload={"name": "agent-generated", "color": "ededed"})
    state_path = candidate / "published-prs.json"
    state = read_json(state_path) if state_path.exists() else {}
    pulls = {pull["number"]: pull for pull in record["pulls"]}
    for outcome in report["pulls"]:
        if outcome["status"] != "ready":
            continue
        number, parent, branch = outcome["number"], outcome["parent"], outcome["branch"]
        pull = pulls[number]
        marker = f"<!-- skyrl-migration: source={SOURCE}#{number} head={pull['head']['sha']} -->"
        existing = pages(f"repos/{TARGET}/pulls?state=all&head=marin-community:{quote(branch, safe='')}&per_page=100")
        base_branch = f"skyrl-migration/pr-{parent}" if parent else default
        if existing:
            if len(existing) != 1 or marker not in existing[0]["body"] or existing[0]["base"]["ref"] != base_branch:
                raise ValueError(f"Destination PR identity mismatch: {branch}")
            destination = re.search(r"<!-- skyrl-destination: ([0-9a-f]{40}) -->", existing[0]["body"])
            if destination is None or existing[0]["head"]["sha"] != destination[1]:
                raise ValueError(f"Destination PR head changed: {branch}")
            git(workspace, "fetch", "--quiet", "origin", existing[0]["head"]["sha"])
            git(workspace, "branch", "-f", branch, existing[0]["head"]["sha"])
            gh(
                f"repos/{TARGET}/issues/{existing[0]['number']}/labels",
                method="POST",
                payload={"labels": ["agent-generated"]},
            )
            state[str(number)] = existing[0]["html_url"]
            write_json(state_path, state)
            continue
        base = f"skyrl-migration/pr-{parent}" if parent else target_base
        git(workspace, "checkout", "--quiet", "--detach", base)
        patch = candidate / f"pr-{number}-retained.patch"
        run("git", "-C", str(workspace), "apply", "--3way", "--index", f"--directory={VENDOR}", str(patch.resolve()))
        commit(workspace, f"Transplant {SOURCE}#{number}\n\nSource-head: {pull['head']['sha']}")
        intended = git(workspace, "rev-parse", "HEAD")
        remote = git(workspace, "ls-remote", "origin", f"refs/heads/{branch}")
        if remote:
            remote_sha = remote.split()[0]
            git(workspace, "fetch", "--quiet", "origin", remote_sha)
            if git(workspace, "rev-parse", "FETCH_HEAD^{tree}") != git(workspace, "rev-parse", "HEAD^{tree}"):
                raise ValueError(f"Existing destination branch differs: {branch}")
            intended = remote_sha
        else:
            git(workspace, "push", "origin", f"HEAD:refs/heads/{branch}")
        git(workspace, "branch", "-f", branch, intended)
        body = (
            source_references(pull["body"] or "")
            + f"\n\nTransplanted from {pull['html_url']} by @{pull['user']['login']}. "
            "Original discussion and reviews remain at that URL.\n\n"
            f"Source base: `{pull['merge_base']}`; source head: `{pull['head']['sha']}`.\n\n"
            f"{marker}\n<!-- skyrl-destination: {intended} -->\n"
        )
        created = gh(
            f"repos/{TARGET}/pulls",
            method="POST",
            payload={"title": pull["title"], "body": body, "head": branch, "base": base_branch, "draft": True},
        )
        gh(f"repos/{TARGET}/issues/{created['number']}/labels", method="POST", payload={"labels": ["agent-generated"]})
        state[str(number)] = created["html_url"]
        write_json(state_path, state)
    write_json(candidate / "publication-blockers.json", {"pending_prs": incomplete})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-vendor", help="Audit the checked-in source, independent lock, and built wheel")
    check.add_argument("--source", type=Path, default=ROOT / VENDOR)
    capture = commands.add_parser("plan", help="Snapshot live source/work and compute the retained file closure")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--ref", default="main")
    capture.add_argument("--days", type=int, default=10)
    capture.add_argument("--manifest", type=Path, default=MANIFEST)
    capture.add_argument("--marin", type=Path, default=ROOT)
    staging = commands.add_parser("stage", help="Offline candidate copy and stacked patch dry run")
    staging.add_argument("--snapshot", type=Path, required=True)
    staging.add_argument("--output", type=Path, required=True)
    wheel = commands.add_parser("build-wheel", help="Build a binary candidate without installing dependencies")
    wheel.add_argument("--candidate", type=Path, required=True)
    smoke = commands.add_parser("prepare-math500", help="Bundle a wheel-only Math500 smoke for an isolated Iris task")
    smoke.add_argument("--candidate", type=Path, required=True)
    smoke.add_argument("--dataset", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    for name in ("transfer-issues", "publish-prs"):
        command = commands.add_parser(name, help="Preview publication; add --execute to write to GitHub")
        command.add_argument("--candidate", type=Path, required=True)
        command.add_argument("--cutover-pr", type=int)
        command.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "check-vendor":
        check_vendor(args.source)
    elif args.command == "plan":
        snapshot(args.output, args.ref, args.days, datetime.now(UTC))
        plan(args.output, args.manifest, args.marin)
    elif args.command == "stage":
        stage(args.snapshot, args.output)
    elif args.command == "build-wheel":
        build_wheel(args.candidate)
    elif args.command == "prepare-math500":
        prepare_math500(args.candidate, args.dataset, args.output)
    elif not args.execute:
        report = read_json(args.candidate / "report.json")
        print(json.dumps(report["issues"] if args.command == "transfer-issues" else report["pulls"], indent=2))
    else:
        if args.cutover_pr is None:
            parser.error("--execute requires --cutover-pr")
        if args.command == "transfer-issues":
            report = read_json(args.candidate / "report.json")
            transfer_issues(Path(report["snapshot"]), args.candidate / "transferred-issues.json", args.cutover_pr)
        else:
            publish_prs(args.candidate, args.cutover_pr)


if __name__ == "__main__":
    main()
