# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject source-policy literals, unlanded checkouts and changed worker bundles."""

import ast
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.ingest_stages.ingest_config import GIT_REVISION, IngestConfig
from experiments.post_training.task_curation.ingest_stages.receipts import SHA256
from experiments.post_training.task_curation.release_stages.guard import (
    atlas_landed_tip,
    scan_release_literals,
    sha_in_landing_log,
    sha_is_ancestor,
)

STAGE_DIRECTORY = Path("experiments/post_training/task_curation/ingest_stages")
SOURCE_SUFFIXES = frozenset({".py", ".json", ".toml", ".yaml", ".yml", ".sh", ".lock", ".proto"})
PIN_LITERAL = re.compile(r"(?<![0-9a-f])(?:[0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{32})(?![0-9a-f])")
LOCATION_LITERAL = re.compile(r"(?:s3|gs|hf|file)://[^\s/]+|/(?:Users|home|workspace|tmp)/\S+")
MEASURED_FIELDS = frozenset(
    {
        "task_count",
        "expected_tasks",
        "measured_tasks",
        "retained_tasks",
        "population_count",
        "raw_count",
        "unique_count",
        "snapshot_id",
        "snapshot_name",
        "release_id",
        "population_path",
    }
)


@dataclass(frozen=True)
class SourceViolation:
    path: Path
    line: int
    reason: str


class SourceWitness(BaseModel):
    """Small source inventory carried with an explicitly pinned worker request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_revision: str = Field(pattern=GIT_REVISION)
    policy_sha256: str = Field(pattern=SHA256)
    files: dict[str, Annotated[str, Field(pattern=SHA256)]] = Field(min_length=1)

    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


def scan_stage_sources(paths: Iterable[Path]) -> list[SourceViolation]:
    """Report locations and reasons without echoing potentially sensitive values."""
    paths = tuple(paths)
    violations = [
        SourceViolation(row.path, row.line, "release or snapshot literal") for row in scan_release_literals(paths)
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if PIN_LITERAL.search(node.value) or LOCATION_LITERAL.search(node.value):
                    violations.append(SourceViolation(path, node.lineno, "pinned identity or artifact location literal"))
            fields: list[tuple[str, ast.expr]] = []
            if isinstance(node, ast.Assign):
                fields = [(target.id, node.value) for target in node.targets if isinstance(target, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                fields = [(node.target.id, node.value)]
            elif isinstance(node, ast.Call):
                fields = [(keyword.arg, keyword.value) for keyword in node.keywords if keyword.arg is not None]
            elif isinstance(node, ast.Dict):
                fields = [
                    (key.value, value)
                    for key, value in zip(node.keys, node.values, strict=True)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                ]
            for name, value in fields:
                if name.lower() in MEASURED_FIELDS and isinstance(value, ast.Constant) and value.value is not None:
                    violations.append(SourceViolation(path, value.lineno, "literal measured field"))
    return violations


def assert_stage_sources_clean(repo_root: Path) -> None:
    paths = tuple(sorted((repo_root / STAGE_DIRECTORY).rglob("*.py")))
    if not paths:
        raise ValueError("maintained ingest stage sources are missing")
    violations = scan_stage_sources(paths)
    if violations:
        detail = ", ".join(f"{row.path.relative_to(repo_root)}:{row.line} ({row.reason})" for row in violations)
        raise ValueError(f"ingest stage source guard refused: {detail}")


def converter_directory(repo_root: Path, policy: IngestConfig) -> Path:
    relative = Path(*policy.converter_module.split("."))
    if policy.converter_module.startswith("taskcompendium."):
        relative = Path("lib/taskcompendium/src") / relative
    module = repo_root / relative.with_suffix(".py")
    if not module.is_file():
        raise ValueError("the configured converter must resolve to a repository module file")
    return module.parent


def required_source_files(repo_root: Path, policy: IngestConfig) -> set[Path]:
    paths = set((repo_root / STAGE_DIRECTORY).rglob("*.py"))
    paths.update(
        path
        for path in converter_directory(repo_root, policy).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    return paths


def local_source_path(repo_root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source witness paths must stay inside the repository")
    path = repo_root / relative
    path.resolve().relative_to(repo_root.resolve())
    return path


def checkout_witness(repo_root: Path, policy: IngestConfig) -> SourceWitness:
    """Measure Git-visible source/config bytes, including current preflight edits."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    names = (
        subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repo_root)
        .decode()
        .split("\0")
    )
    required = required_source_files(repo_root, policy)
    paths = {repo_root / name for name in names if name and Path(name).suffix in SOURCE_SUFFIXES} | required
    files = {}
    for path in sorted(paths):
        name = path.relative_to(repo_root).as_posix()
        files[name] = file_sha256(StoragePath(str(local_source_path(repo_root, name))))
    return SourceWitness(source_revision=revision, policy_sha256=policy.digest(), files=files)


def verify_source_witness(repo_root: Path, policy: IngestConfig, witness: SourceWitness) -> None:
    if witness.policy_sha256 != policy.digest():
        raise ValueError("source witness belongs to another ingest policy")
    required = {path.relative_to(repo_root).as_posix() for path in required_source_files(repo_root, policy)}
    if required - witness.files.keys():
        raise ValueError("worker has unattested stage or converter files")
    for name, digest in witness.files.items():
        path = local_source_path(repo_root, name)
        if file_sha256(StoragePath(str(path))) != digest:
            raise ValueError(f"worker source bytes differ from the witness: {name}")
    assert_stage_sources_clean(repo_root)


def assert_checkout_allowed(repo_root: Path, policy: IngestConfig, witness: SourceWitness, landing_log: Path) -> None:
    """Require clean source at the attested landed revision before submission."""
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo_root)
    if dirty:
        raise ValueError("ingest submission requires a clean committed checkout")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    if head != witness.source_revision:
        raise ValueError("checkout revision differs from the worker source witness")
    verify_source_witness(repo_root, policy, witness)
    if sha_in_landing_log(head, landing_log):
        return
    if not sha_is_ancestor(head, atlas_landed_tip(repo_root), repo_root):
        raise ValueError("ingest source must land through atlas-push before submission")


def assert_bundle_allowed(repo_root: Path, policy: IngestConfig, witness: SourceWitness, landing_log: Path) -> None:
    """Verify a copied worker bundle using the explicit landing-log receipt."""
    if not sha_in_landing_log(witness.source_revision, landing_log):
        raise ValueError("worker source revision is absent from the bundled landing log")
    verify_source_witness(repo_root, policy, witness)
