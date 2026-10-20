# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content identities for converted models and their parity inputs."""

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path

MANIFEST_FILE = "mlx-build.json"


class Recipe(StrEnum):
    EIGHT_BIT = "8bit"
    MIXED_4BIT = "mixed4"


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_files(model: Path) -> dict[str, str]:
    return {
        str(path.relative_to(model)): file_digest(path)
        for path in sorted(model.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE and "__pycache__" not in path.parts
    }


def validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The source revision must be a full commit SHA")


def write_manifest(model: Path, *, repo: str, revision: str, recipe: Recipe) -> None:
    validate_revision(revision)
    manifest = {"source_repo": repo, "source_revision": revision, "recipe": recipe.value, "files": build_files(model)}
    (model / MANIFEST_FILE).write_text(json.dumps(manifest, sort_keys=True) + "\n")


def verified_build(model: Path) -> dict:
    manifest_path = model / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    if manifest["files"] != build_files(model):
        raise ValueError("Model files differ from the recorded MLX build")
    return {
        "build_sha256": file_digest(manifest_path),
        **{k: manifest[k] for k in ("source_repo", "source_revision", "recipe")},
    }


def validate_dump(dump: dict, *, model: Path, recipe: Recipe, golden: Path) -> dict:
    build = verified_build(model)
    if dump["build"] != build or recipe.value != build["recipe"]:
        raise ValueError("Parity dump belongs to a different model build or recipe")
    if dump["golden_sha256"] != file_digest(golden):
        raise ValueError("Parity dump belongs to a different golden reference")
    for case in json.loads(golden.read_text()):
        if case["source_revision"] != build["source_revision"] or case["source_repo"] != build["source_repo"]:
            raise ValueError("Golden reference belongs to a different source model")
    return build
