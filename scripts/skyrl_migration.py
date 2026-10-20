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
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
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
AGENT_LABEL = "agent-generated"
PR_BRANCH_PREFIX = "skyrl-migration/pr-"
DESTINATION_MARKER = "skyrl-destination"
DESTINATION_PATTERN = re.compile(r"<!-- " + DESTINATION_MARKER + r": ([0-9a-f]{40}) -->")
LOCAL_CHANGES_PATCH = "skyrl-migration/local-changes.patch"
WHEEL_RUNTIME_ROOTS = {"cloud", "marinskyrl", "skyrl_train", "skyrl_gym", "chat_templates", "open_mopd_versions.py"}


def run(
    *args: str, cwd: Path | None = None, data: bytes | None = None, check: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(args, cwd=cwd, input=data, capture_output=True, check=check, env=env)
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


def github_items(endpoint: str) -> list[dict]:
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


def snapshot(output: Path, ref: str, days: int, now: datetime, objects_cache: Path | None = None) -> None:
    """Capture immutable Git objects and paginated open work; never write to GitHub."""
    output.mkdir(parents=True, exist_ok=False)
    cutoff_time = now - timedelta(days=days)
    cutoff = cutoff_time.isoformat().replace("+00:00", "Z")
    repository = gh(f"repos/{SOURCE}")
    source_sha = gh(f"repos/{SOURCE}/commits/{quote(ref, safe='')}")["sha"]
    pulls = github_items(f"repos/{SOURCE}/pulls?state=open&per_page=100")
    issues = [
        issue
        for issue in github_items(f"repos/{SOURCE}/issues?state=open&since={quote(cutoff)}&per_page=100")
        if "pull_request" not in issue and datetime.fromisoformat(issue["updated_at"]) >= cutoff_time
    ]

    def capture_pull(pull: dict) -> dict:
        number = pull["number"]
        base, head = pull["base"]["sha"], pull["head"]["sha"]
        comparison = gh(f"repos/{SOURCE}/compare/{base}...{head}")
        return pull | {
            "merge_base": comparison["merge_base_commit"]["sha"],
            "comments": github_items(f"repos/{SOURCE}/issues/{number}/comments?per_page=100"),
            "reviews": github_items(f"repos/{SOURCE}/pulls/{number}/reviews?per_page=100"),
            "review_comments": github_items(f"repos/{SOURCE}/pulls/{number}/comments?per_page=100"),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        pulls = list(pool.map(capture_pull, pulls))
        comments = list(
            pool.map(
                lambda issue: github_items(f"repos/{SOURCE}/issues/{issue['number']}/comments?per_page=100"), issues
            )
        )
    issues = [issue | {"comments": comment} for issue, comment in zip(issues, comments, strict=True)]
    objects = output / "source.git"
    if objects_cache is None:
        run("git", "init", "--bare", "--quiet", str(objects))
    else:
        run("git", "clone", "--mirror", "--local", "--quiet", str(objects_cache.resolve()), str(objects))
    refs = [f"+{source_sha}:refs/snapshot/main"]
    for pull in pulls:
        number = pull["number"]
        refs += [
            f"+{pull['head']['sha']}:refs/snapshot/pr-{number}-head",
            f"+{pull['merge_base']}:refs/snapshot/pr-{number}-base",
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


@dataclass
class ModuleImports:
    names: set[str]
    dynamic: list[str]
    required: set[str]


def module_imports(path: Path, module: str) -> ModuleImports:
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
    return ModuleImports(names, dynamic, required)


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
            imports = module_imports(marin / filename, path_to_module(marin / filename, marin) or "")
            for name in resolve(imports.names, known):
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
        imports = module_imports(files[path], by_path[path])
        dynamic.update(imports.dynamic)
        missing.update(
            f"{path}: {name}"
            for name in imports.required
            if name.split(".")[0] in local_packages and name not in known and name not in namespaces
        )
        for name in resolve(imports.names, known):
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


class PullStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    BLOCKED_PARENT = "blocked-parent"
    UNMAPPED_BASE = "unmapped-base"
    EXCLUDED = "excluded"
    CONFLICT = "conflict"
    ALREADY_APPLIED = "already-applied"
    NEEDS_DECISION = "needs-decision"


@dataclass
class PullOutcome:
    number: int
    url: str
    parent: int | None
    branch: str
    status: PullStatus = PullStatus.PENDING
    selected: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    commit: str = ""
    base: str = ""
    diagnostic: str = ""


@dataclass(frozen=True)
class IssueReference:
    number: int
    url: str


@dataclass(frozen=True)
class MigrationPatch:
    path: str
    sha256: str


@dataclass
class StageReport:
    snapshot: str
    source_sha: str
    patches: list[MigrationPatch]
    retained: int
    discarded: int
    pulls: list[PullOutcome]
    issues: list[IssueReference]
    dynamic_imports: list[str]
    missing_imports: list[str]


def read_stage_report(path: Path) -> StageReport:
    value = read_json(path)
    value["patches"] = [MigrationPatch(**item) for item in value["patches"]]
    value["pulls"] = [PullOutcome(**(item | {"status": PullStatus(item["status"])})) for item in value["pulls"]]
    value["issues"] = [IssueReference(**item) for item in value["issues"]]
    return StageReport(**value)


def copy_source_files(source: Path, destination: Path, names: list[str]) -> None:
    for name in names:
        original, target = source / relative_path(name), destination / name
        original.resolve().relative_to(source.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        if original.is_symlink():
            if target.exists():
                target.unlink()
            target.symlink_to(os.readlink(original))
        else:
            shutil.copy2(original, target)


def stage(snapshot_dir: Path, output: Path) -> None:
    """Copy the candidate and test each complete Git delta on its staged parent."""
    snapshot_dir = snapshot_dir.resolve()
    record = read_json(snapshot_dir / "snapshot.json")
    chosen = read_json(snapshot_dir / "selection.json")
    output.mkdir(parents=True, exist_ok=False)
    tree = output / "tree"
    destination = tree / VENDOR
    source = snapshot_dir / "source"
    objects = snapshot_dir / "source.git"
    verify_source(source, objects, record["source_sha"], chosen["retained"])
    copy_source_files(source, destination, list(chosen["retained"]))
    (destination / PROVENANCE).parent.mkdir(parents=True, exist_ok=True)
    write_json(destination / PROVENANCE, {"source": SOURCE, "source_sha": record["source_sha"]})
    run("git", "init", "--quiet", "--initial-branch=main", str(tree))
    git(tree, "add", ".")
    baseline = commit(tree, "Stage selected SkyRL sources")
    git(tree, "update-ref", "refs/migration/retained", baseline)
    git(tree, "fetch", "--quiet", "--no-tags", str(objects), "+refs/snapshot/*:refs/source/*")
    for patch in chosen["patches"]:
        path = snapshot_dir / "migration-patches" / patch["path"]
        if file_digest(path) != patch["sha256"]:
            raise ValueError(f"Migration patch changed after capture: {path}")
        run("git", "-C", str(tree), "apply", "--3way", "--index", "--unidiff-zero", f"--directory={VENDOR}", str(path))
    if chosen["patches"]:
        write_json(
            destination / PROVENANCE,
            {"source": SOURCE, "source_sha": record["source_sha"], "patches": chosen["patches"]},
        )
        git(tree, "add", ".")
        baseline = commit(tree, "Apply migration runtime fixes")
    pulls = transplant_pulls(snapshot_dir, output, baseline)
    report = StageReport(
        snapshot=str(snapshot_dir),
        source_sha=record["source_sha"],
        patches=[MigrationPatch(**patch) for patch in chosen["patches"]],
        retained=len(chosen["retained"]),
        discarded=len(chosen["discarded"]),
        pulls=pulls,
        issues=[IssueReference(issue["number"], issue["html_url"]) for issue in record["issues"]],
        dynamic_imports=chosen["dynamic_imports"],
        missing_imports=chosen["missing_imports"],
    )
    write_report(output, report)


class RefreshStatus(StrEnum):
    READY = "ready"
    CONFLICT = "conflict"
    FAILED = "failed"


@dataclass
class RefreshReport:
    status: RefreshStatus
    previous_sha: str
    source_sha: str
    candidate: str
    snapshot: str
    local_files: dict[str, str]
    manifest_sha256: str
    conflicts: list[str] = field(default_factory=list)
    diagnostic: str = ""


def read_refresh_report(path: Path) -> RefreshReport:
    value = read_json(path)
    return RefreshReport(**(value | {"status": RefreshStatus(value["status"])}))


def checkout_source_files(source: Path) -> dict[str, str]:
    """Inventory tracked and unignored local files, including deletions and modes."""
    source = source.resolve()
    root = Path(git(source, "rev-parse", "--show-toplevel"))
    prefix = source.resolve().relative_to(root)
    names = run(
        "git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", str(prefix)
    ).stdout
    files = {}
    for value in names.split(b"\0"):
        if not value:
            continue
        path = root / value.decode()
        name = path.relative_to(source).as_posix()
        if name == PROVENANCE or not (path.is_file() or path.is_symlink()):
            continue
        mode = "120000" if path.is_symlink() else "100755" if path.stat().st_mode & 0o111 else "100644"
        files[name] = f"{mode}:{file_digest(path)}"
    return files


def vendor_patch(tree: Path, *revisions: str) -> bytes:
    # These task-owned Git repositories need compact deltas for generated locks.
    # The installed vendor lock remains readable text; Git encodes its replay delta.
    attributes = tree / git(tree, "rev-parse", "--git-path", "info/attributes")
    attributes.parent.mkdir(parents=True, exist_ok=True)
    attributes.write_text(f"{VENDOR}/uv.lock -diff\n")
    return run(
        "git",
        "-C",
        str(tree),
        "diff",
        "--binary",
        "--full-index",
        # Empty context markers are trailing whitespace to repository formatters.
        # Full blob identities retain the base for three-way replay without context lines.
        "--unified=0",
        f"--relative={VENDOR}/",
        *revisions,
        "--",
        VENDOR,
        f":!{VENDOR}/{PROVENANCE}",
    ).stdout


def refresh_manifest(contents: str, patches: list[str]) -> str:
    """Replace migration patch inputs while preserving the retention policy."""
    contents = re.sub(r"(?m)^patches\s*=\s*\[[^\]]*\]\s*\n", "", contents)
    return f"patches = {json.dumps(patches)}\n" + contents


def select_local_patch(snapshot_dir: Path, patch_path: Path) -> None:
    """Record the replay delta in the snapshot's selection and retention policy."""
    chosen = read_json(snapshot_dir / "selection.json")
    chosen["patches"] = (
        [{"path": LOCAL_CHANGES_PATCH, "sha256": file_digest(patch_path)}] if patch_path.stat().st_size else []
    )
    write_json(snapshot_dir / "selection.json", chosen)
    policy = snapshot_dir / "retention.toml"
    policy.write_text(refresh_manifest(policy.read_text(), [patch["path"] for patch in chosen["patches"]]))


def refresh_candidate(previous: Path, current: Path, updated: Path, candidate: Path) -> RefreshReport:
    """Replay the complete local vendor delta onto a fresh upstream snapshot."""
    previous, current, updated, candidate = (path.resolve() for path in (previous, current, updated, candidate))
    old = read_json(previous / "snapshot.json")
    new = read_json(updated / "snapshot.json")
    if read_json(current / PROVENANCE)["source_sha"] != old["source_sha"]:
        raise ValueError("Previous snapshot does not match the checked-out SkyRL origin")
    local_files = checkout_source_files(current)
    report = RefreshReport(
        RefreshStatus.READY,
        old["source_sha"],
        new["source_sha"],
        str(candidate),
        str(updated),
        local_files,
        file_digest(updated / "retention.toml"),
    )
    overlay = candidate.with_name(candidate.name + "-overlay")
    retained = read_json(previous / "selection.json")["retained"]
    verify_source(previous / "source", previous / "source.git", old["source_sha"], retained)
    copy_source_files(previous / "source", overlay / VENDOR, list(retained))
    run("git", "init", "--quiet", "--initial-branch=main", str(overlay))
    git(overlay, "add", ".")
    commit(overlay, "Capture previous retained upstream sources")
    git(updated / "source.git", "fetch", "--quiet", str(overlay), "+HEAD:refs/snapshot/local-base")
    shutil.rmtree(overlay / VENDOR)
    copy_source_files(current, overlay / VENDOR, list(local_files))
    git(overlay, "add", "--all")
    patch_name = LOCAL_CHANGES_PATCH
    patch_path = updated / "migration-patches" / patch_name
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    local_patch = vendor_patch(overlay, "--cached")
    patch_path.write_bytes(local_patch)
    (candidate.parent / "local-changes-before-replay.patch").write_bytes(local_patch)
    shutil.rmtree(overlay)
    select_local_patch(updated, patch_path)
    try:
        stage(updated, candidate)
    except subprocess.CalledProcessError as error:
        report.conflicts = git(candidate / "tree", "diff", "--name-only", "--diff-filter=U").splitlines()
        report.status = RefreshStatus.CONFLICT if report.conflicts else RefreshStatus.FAILED
        report.diagnostic = (error.stderr or error.stdout).decode(errors="replace")
        write_json(candidate.parent / "refresh.json", asdict(report))
        return report
    finish_refresh(report)
    return report


def finish_refresh(report: RefreshReport) -> None:
    """Normalize a clean or manually resolved replay and rebuild its PR transplant report."""
    candidate, updated = Path(report.candidate), Path(report.snapshot)
    if git(candidate / "tree", "diff", "--name-only", "--diff-filter=U"):
        raise ValueError("Resolve the candidate conflicts and git add the resolved files before finishing")
    # Store the resolved delta against the new origin before recreating the candidate.
    # Future plan/stage runs then need only the new snapshot's Git objects.
    patch_path = updated / "migration-patches" / LOCAL_CHANGES_PATCH
    patch_path.write_bytes(vendor_patch(candidate / "tree", "refs/migration/retained"))
    select_local_patch(updated, patch_path)
    if report.status is RefreshStatus.READY:
        shutil.rmtree(candidate)
    else:
        candidate.rename(candidate.with_name(candidate.name + "-resolution"))
    stage(updated, candidate)
    report.status = RefreshStatus.READY
    report.conflicts = []
    report.diagnostic = ""
    write_json(candidate.parent / "refresh.json", asdict(report))


def apply_refresh(report: RefreshReport, current: Path, manifest: Path) -> None:
    """Install a reviewed clean candidate while preserving ignored local environments."""
    if report.status is not RefreshStatus.READY:
        raise ValueError("A conflicting or failed refresh cannot be applied")
    if checkout_source_files(current) != report.local_files:
        raise ValueError("SkyRL checkout changed during refresh; inspect the candidate before applying")
    if file_digest(manifest) != report.manifest_sha256:
        raise ValueError("Retention policy changed during refresh; inspect the candidate before applying")
    candidate = Path(report.candidate)
    tree = candidate / "tree"
    if git(tree, "status", "--porcelain") or git(tree, "rev-parse", "HEAD") != git(tree, "rev-parse", "main"):
        raise ValueError("Candidate changed after staging; finish its replay before applying")
    staged = read_stage_report(candidate / "report.json")
    previous_patches = tomllib.loads(manifest.read_text()).get("patches", [])
    source = candidate / "tree" / VENDOR
    names = [
        name.removeprefix(VENDOR + "/")
        for name in tree_paths(candidate / "tree", "main")
        if name.startswith(VENDOR + "/")
    ]
    for name in report.local_files.keys() - set(names):
        (current / relative_path(name)).unlink()
    copy_source_files(source, current, names)
    snapshot_dir = Path(staged.snapshot)
    for patch in staged.patches:
        target = manifest.parent / relative_path(patch.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot_dir / "migration-patches" / patch.path, target)
    manifest.write_text((snapshot_dir / "retention.toml").read_text())
    for name in set(previous_patches) - {patch.path for patch in staged.patches}:
        (manifest.parent / relative_path(name)).unlink()
    report_path = manifest.parent / "skyrl-migration/snapshot-report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate / "report.md", report_path)


def transplant_pulls(snapshot_dir: Path, output: Path, baseline: str) -> list[PullOutcome]:
    """Apply each source PR on its staged parent and return publication outcomes."""
    record = read_json(snapshot_dir / "snapshot.json")
    chosen = read_json(snapshot_dir / "selection.json")
    manifest = tomllib.loads((snapshot_dir / "retention.toml").read_text())
    tree = output / "tree"
    objects = snapshot_dir / "source.git"
    git(tree, "fetch", "--quiet", "--no-tags", str(snapshot_dir / "source.git"), "+refs/snapshot/*:refs/source/*")
    parents = parent_pulls(record["pulls"], record["default_branch"])
    outcomes: dict[int, PullOutcome] = {}
    for pull in ordered_pulls(record["pulls"], parents):
        number, parent = pull["number"], parents[pull["number"]]
        branch = f"{PR_BRANCH_PREFIX}{number}"
        outcome = PullOutcome(number, pull["html_url"], parent, branch)
        outcomes[number] = outcome
        if parent is not None and outcomes[parent].status is not PullStatus.READY:
            outcome.status = PullStatus.BLOCKED_PARENT
            continue
        if parent is None and pull["base"]["ref"] != record["default_branch"]:
            outcome.status = PullStatus.UNMAPPED_BASE
            outcome.base = pull["base"]["ref"]
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
        outcome.selected = selected
        outcome.excluded = excluded
        if not selected:
            outcome.status = PullStatus.EXCLUDED
            continue
        base = outcomes[parent].commit if parent is not None else baseline
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
            outcome.status = PullStatus.CONFLICT
            outcome.diagnostic = applied.stderr.decode(errors="replace")
            git(tree, "reset", "--hard", "--quiet", base)
            git(tree, "clean", "-fd", "--quiet")
            continue
        if not git(tree, "diff", "--cached", "--name-only"):
            outcome.status = PullStatus.ALREADY_APPLIED
            continue
        outcome.status = PullStatus.NEEDS_DECISION if excluded else PullStatus.READY
        outcome.commit = commit(tree, f"Transplant {SOURCE}#{number}\n\nSource-head: {pull['head']['sha']}")
    git(tree, "checkout", "--quiet", "main")
    return list(outcomes.values())


def write_report(output: Path, report: StageReport) -> None:
    """Write the machine-readable outcomes and their review summary."""
    write_json(output / "report.json", asdict(report))
    rows = [
        "# SkyRL migration dry run",
        "",
        f"Source: `{report.source_sha}`. Retained {report.retained} files; discarded {report.discarded}.",
        "",
        "This is a source candidate. Runtime bootstrap and nightly acceptance remain required.",
        "",
        "| PR | parent | outcome | excluded paths |",
        "| --- | --- | --- | --- |",
    ]
    rows += [
        f"| [#{item.number}]({item.url}) | {item.parent or ''} | {item.status} | {len(item.excluded)} |"
        for item in report.pulls
    ]
    rows += [
        "",
        f"Selected issues: {len(report.issues)}. Complete source patches and review history remain in the snapshot.",
        "",
        "See docs/dev-guide/skyrl-migration.md for cutover and archive requirements.",
    ]
    (output / "report.md").write_text("\n".join(rows) + "\n")


def wheel_source_files(source: Path) -> dict[str, Path]:
    """Map the current package files and assets to their wheel member paths."""
    files = {}
    for path in source.rglob("*"):
        name = path.relative_to(source)
        if {".git", ".venv", "__pycache__"}.intersection(name.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        member = name.as_posix().removeprefix("skyrl-train/").removeprefix("skyrl-gym/")
        if member.split("/")[0] in WHEEL_RUNTIME_ROOTS and path.is_file():
            path.resolve().relative_to(source.resolve())
            files[member] = path
    return files


def audit_wheel(wheel: Path, source: Path) -> list[str]:
    """Check that the wheel contains the current checked-out source and assets."""
    with zipfile.ZipFile(wheel) as archive:
        members = sorted(archive.namelist())
        if PROVENANCE not in members or BUNDLE_MANIFEST not in members:
            raise ValueError("Wheel is missing runtime provenance or the launcher bundle manifest")
        expected = wheel_source_files(source)
        for packaged, path in expected.items():
            if packaged not in members or hashlib.sha256(archive.read(packaged)).hexdigest() != file_digest(path):
                raise ValueError(f"Wheel member missing or changed: {packaged}")
        unexpected = {
            name for name in members if name.split("/")[0] in WHEEL_RUNTIME_ROOTS and not name.endswith("/")
        } - expected.keys()
        if unexpected:
            raise ValueError(f"Wheel contains runtime files outside the checked-out source: {sorted(unexpected)}")
    return members


@dataclass(frozen=True)
class BuiltWheel:
    path: Path
    sha256: str
    members: list[str]


def build_audited_wheel(source: Path, output: Path) -> BuiltWheel:
    run("uv", "build", "--wheel", "--project", str(source), "--out-dir", str(output))
    [wheel] = output.glob("*.whl")
    return BuiltWheel(wheel, file_digest(wheel), audit_wheel(wheel, source))


def check_vendor(source: Path) -> None:
    """Check the independent lock, isolated Marin contract, and wheel contents."""
    run("uv", "lock", "--project", str(source), "--check")
    run(sys.executable, str(ROOT / "infra/ci/check_skyrl_marin.py"), "project", "--source", str(source))
    with tempfile.TemporaryDirectory(prefix="skyrl-wheel-") as temporary:
        wheel = build_audited_wheel(source, Path(temporary))
        print(json.dumps({"wheel_sha256": wheel.sha256, "members": len(wheel.members)}))


def build_wheel(candidate: Path) -> None:
    """Build and inventory a binary candidate without resolving or installing its dependencies."""
    source = candidate.resolve() / "tree" / VENDOR
    wheels = candidate.resolve() / "wheels"
    wheels.mkdir(exist_ok=False)
    wheel = build_audited_wheel(source, wheels)
    inputs = {
        path.relative_to(source).as_posix(): file_digest(path)
        for path in (source / "skyrl-train/tests").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    inputs.update({name: file_digest(source / name) for name in ("pyproject.toml", "uv.lock")})
    write_json(
        candidate / "wheel.json",
        {"path": str(wheel.path), "sha256": wheel.sha256, "members": wheel.members, "inputs": inputs},
    )
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
        + wheel.path.name
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
        if file_digest(candidate / "runtime-dependencies" / name) != wheel_record["inputs"][name]:
            raise ValueError(f"Runtime dependency manifest changed after staging: {name}")
    for name, digest in wheel_record["inputs"].items():
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


def validate_cutover(number: int, source_sha: str, patches: list[MigrationPatch]) -> dict:
    pull = gh(f"repos/{TARGET}/pulls/{number}")
    if not pull["merged"] or pull["base"]["repo"]["full_name"] != TARGET:
        raise ValueError("Publication requires a merged Marin cutover PR")
    provenance = gh(f"repos/{TARGET}/contents/{VENDOR}/{PROVENANCE}?ref={pull['merge_commit_sha']}")
    identity = json.loads(base64.b64decode(provenance["content"]))
    if identity["source"] != SOURCE or identity["source_sha"] != source_sha:
        raise ValueError("Merged cutover provenance does not match the reviewed source snapshot")
    if identity.get("patches", []) != [asdict(patch) for patch in patches]:
        raise ValueError("Merged cutover migration fixes do not match the reviewed candidate")
    return pull


def validate_fresh(record: dict) -> None:
    head = gh(f"repos/{SOURCE}/commits/{quote(record['default_branch'], safe='')}")["sha"]
    pulls = github_items(f"repos/{SOURCE}/pulls?state=open&per_page=100")
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
    patches = [MigrationPatch(**patch) for patch in read_json(snapshot_dir / "selection.json")["patches"]]
    validate_cutover(cutover, record["source_sha"], patches)
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
    report = read_stage_report(candidate / "report.json")
    snapshot_dir = Path(report.snapshot)
    record = read_json(snapshot_dir / "snapshot.json")
    cutover_pull = validate_cutover(cutover, record["source_sha"], report.patches)
    validate_fresh(record)
    incomplete = [item.number for item in report.pulls if item.status != PullStatus.READY]
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
    labels = github_items(f"repos/{TARGET}/labels?per_page=100")
    if not any(label["name"] == AGENT_LABEL for label in labels):
        gh(f"repos/{TARGET}/labels", method="POST", payload={"name": AGENT_LABEL, "color": "ededed"})
    state_path = candidate / "published-prs.json"
    state = read_json(state_path) if state_path.exists() else {}
    pulls = {pull["number"]: pull for pull in record["pulls"]}
    for outcome in report.pulls:
        if outcome.status != PullStatus.READY:
            continue
        number, parent, branch = outcome.number, outcome.parent, outcome.branch
        pull = pulls[number]
        marker = f"<!-- skyrl-migration: source={SOURCE}#{number} head={pull['head']['sha']} -->"
        existing = github_items(
            f"repos/{TARGET}/pulls?state=all&head=marin-community:{quote(branch, safe='')}&per_page=100"
        )
        base_branch = f"{PR_BRANCH_PREFIX}{parent}" if parent else default
        if existing:
            if len(existing) != 1 or marker not in existing[0]["body"] or existing[0]["base"]["ref"] != base_branch:
                raise ValueError(f"Destination PR identity mismatch: {branch}")
            destination = DESTINATION_PATTERN.search(existing[0]["body"])
            if destination is None or existing[0]["head"]["sha"] != destination[1]:
                raise ValueError(f"Destination PR head changed: {branch}")
            git(workspace, "fetch", "--quiet", "origin", existing[0]["head"]["sha"])
            git(workspace, "branch", "-f", branch, existing[0]["head"]["sha"])
            gh(
                f"repos/{TARGET}/issues/{existing[0]['number']}/labels",
                method="POST",
                payload={"labels": [AGENT_LABEL]},
            )
            state[str(number)] = existing[0]["html_url"]
            write_json(state_path, state)
            continue
        base = f"{PR_BRANCH_PREFIX}{parent}" if parent else target_base
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
            f"{marker}\n<!-- {DESTINATION_MARKER}: {intended} -->\n"
        )
        created = gh(
            f"repos/{TARGET}/pulls",
            method="POST",
            payload={"title": pull["title"], "body": body, "head": branch, "base": base_branch, "draft": True},
        )
        gh(f"repos/{TARGET}/issues/{created['number']}/labels", method="POST", payload={"labels": [AGENT_LABEL]})
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
    refresh = commands.add_parser("refresh", help="Refresh upstream and active work while preserving local SkyRL edits")
    refresh.add_argument("--previous", type=Path, required=True, help="Snapshot matching the current vendor origin")
    refresh.add_argument(
        "--output", type=Path, required=True, help="New snapshot, candidate, and conflict report directory"
    )
    refresh.add_argument("--source", type=Path, default=ROOT / VENDOR)
    refresh.add_argument("--ref", default="main")
    refresh.add_argument("--days", type=int, default=10)
    refresh.add_argument("--manifest", type=Path, default=MANIFEST)
    refresh.add_argument("--marin", type=Path, default=ROOT)
    refresh.add_argument(
        "--apply", action="store_true", help="Install a clean candidate and its normalized migration patch"
    )
    apply = commands.add_parser("apply-refresh", help="Install a previously previewed clean refresh")
    apply.add_argument("--report", type=Path, required=True)
    apply.add_argument("--source", type=Path, default=ROOT / VENDOR)
    apply.add_argument("--manifest", type=Path, default=MANIFEST)
    finish = commands.add_parser("finish-refresh", help="Finish a manually resolved candidate and rebuild its PR report")
    finish.add_argument("--report", type=Path, required=True)
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
    elif args.command == "refresh":
        args.output.mkdir(parents=True, exist_ok=False)
        updated = args.output / "snapshot"
        snapshot(updated, args.ref, args.days, datetime.now(UTC), objects_cache=args.previous / "source.git")
        plan(updated, args.manifest, args.marin)
        report = refresh_candidate(args.previous, args.source, updated, args.output / "candidate")
        if args.apply and report.status is RefreshStatus.READY:
            apply_refresh(report, args.source, args.manifest)
        print(
            json.dumps(
                {
                    "status": report.status,
                    "previous_sha": report.previous_sha,
                    "source_sha": report.source_sha,
                    "candidate": report.candidate,
                    "conflicts": report.conflicts,
                    "report": str(args.output / "refresh.json"),
                    "applied": args.apply and report.status is RefreshStatus.READY,
                }
            )
        )
        if report.status is not RefreshStatus.READY:
            raise SystemExit(1)
    elif args.command == "apply-refresh":
        apply_refresh(read_refresh_report(args.report), args.source, args.manifest)
    elif args.command == "finish-refresh":
        finish_refresh(read_refresh_report(args.report))
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
