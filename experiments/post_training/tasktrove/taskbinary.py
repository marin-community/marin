# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor task binaries and their template fingerprints.

A TaskTrove row is a gzip tarball with ``instruction.md``, ``task.toml``, ``environment/``,
``tests/`` and optionally ``solution/``. Within a source, the verifier code and Dockerfile are
stamped from one template and only a few data files vary per task. The fingerprint hashes the
template code and ignores per-task data, so tasks that share a template share a fingerprint.
"""

import fnmatch
import gzip
import hashlib
import io
import re
import tarfile
from dataclasses import dataclass, field

INSTRUCTION = "instruction.md"
TASK_TOML = "task.toml"
DOCKERFILE = "environment/Dockerfile"
TEST_SH = "tests/test.sh"
SOLUTION_DIR = "solution/"

# Files under tests/ or environment/ that carry per-task data rather than template code.
DATA_FILE_PATTERNS = (
    "tests/verifier_data.json",
    "tests/gold.json",
    "tests/config.json",
    "tests/test_data.json",
    "tests/expected*.txt",
    "tests/solution.txt",
    "tests/reference.txt",
    "tests/judge.toml",
    "tests/conversation.txt",
    "tests/criterion_partition.json",
    "tests/inputs/*",
    "tests/outputs/*",
    "tests/*.patch",
    "tests/*.diff",
    "tests/trusted_test_paths.txt",
    "tests/trusted_patch_paths.txt",
    "tests/test_*.py",
    "tests/*_test.py",
    "tests/test_solution.*",
    "tests/solution_test.*",
    "tests/*Test.java",
    "tests/*_test.go",
    "tests/*.spec.*",
    "tests/*.test.*",
    "tests/*_spec.rb",
    "tests/test_impl.sh",
    "tests/test_state.py",
    "tests/checker.py",
    "tests/requirements.txt",
    "tests/pom.xml",
    "tests/r2e_tests/*",
    "tests/setup_files/*",
    "tests/*.json",
    "tests/*.txt",
    "tests/*.md",
    "tests/*.yml",
    "tests/*.yaml",
    "tests/*.java",
    "tests/*.php",
    "environment/write_solution.sh",
    "environment/setup_files/*",
    "environment/toolscale_runtime.py",
)

_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
_QUOTED_RE = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")
_DIGITS_RE = re.compile(r"\d+")
_PATH_TOKEN_RE = re.compile(r"\S*/\S*")


@dataclass
class TaskFiles:
    """Decoded contents of one task binary, keyed by path relative to the tarball root."""

    files: dict[str, bytes] = field(default_factory=dict)

    def text(self, path: str) -> str:
        return self.files[path].decode("utf-8", errors="replace")

    def get_text(self, path: str) -> str | None:
        blob = self.files.get(path)
        return None if blob is None else blob.decode("utf-8", errors="replace")

    @property
    def has_solution(self) -> bool:
        return any(p.startswith(SOLUTION_DIR) for p in self.files)

    def under(self, prefix: str) -> dict[str, bytes]:
        return {p: b for p, b in self.files.items() if p.startswith(prefix)}


@dataclass(frozen=True)
class TemplateFingerprint:
    """Identity of the template a task was stamped from."""

    template_id: str
    """Hash of test.sh plus every template-code file under tests/ and environment/."""
    dockerfile_id: str
    """Hash of the Dockerfile with commit hashes blanked, so per-repo SWE images still group."""
    test_sh_id: str
    code_files: tuple[str, ...]
    data_files: tuple[str, ...]


def read_task_binary(blob: bytes) -> TaskFiles:
    raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = _strip_root(member.name)
            handle = tf.extractfile(member)
            if handle is not None:
                files[name] = handle.read()
    return TaskFiles(files)


def write_task_binary(task: TaskFiles) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for path in sorted(task.files):
            data = task.files[path]
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mode = 0o755 if path.endswith(".sh") else 0o644
            tf.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue(), mtime=0)


def _strip_root(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def normalize_template_text(text: str, shell: bool = False) -> str:
    """Blank the per-task literals a template script embeds (test ids, commits, counts).

    Shell scripts reduce to their command skeleton: each line keeps only its first token, so a
    per-task ``pytest <paths>`` or ``pip install <pkgs>`` line does not fork the template.
    """
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = _QUOTED_RE.sub('""', text)
    text = _SHA_RE.sub("<sha>", text)
    text = _PATH_TOKEN_RE.sub("<path>", text)
    text = _DIGITS_RE.sub("#", text)
    if shell:
        heads = [line.split(None, 1)[0] for line in text.splitlines() if line and line != '""']
        text = "\n".join(heads)
    return text


def data_file_shape(path: str) -> str:
    """The template-level shape of a data file path: nested trees collapse to their top directory."""
    parts = path.split("/")
    if len(parts) > 2:
        return "/".join(parts[:2]) + "/*"
    suffix = parts[-1].rsplit(".", 1)[-1] if "." in parts[-1] else ""
    return f"{parts[0]}/*.{suffix}"


def is_data_file(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in DATA_FILE_PATTERNS)


def template_fingerprint(task: TaskFiles) -> TemplateFingerprint:
    code: list[str] = []
    data: list[str] = []
    for path in sorted(task.files):
        if not (path.startswith("tests/") or path.startswith("environment/")):
            continue
        if path == DOCKERFILE:
            continue
        (data if is_data_file(path) else code).append(path)
    digest = hashlib.sha256()
    for path in code:
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(normalize_template_text(task.text(path), shell=path.endswith(".sh")).encode())
        digest.update(b"\0")
    # Data file *names* are part of the template even though their contents are not.
    data_shape = sorted({data_file_shape(path) for path in data})
    digest.update("|".join(data_shape).encode())
    dockerfile = task.get_text(DOCKERFILE) or ""
    normalized_dockerfile = normalize_template_text(dockerfile, shell=True)
    return TemplateFingerprint(
        template_id=digest.hexdigest()[:12],
        dockerfile_id=hashlib.sha256(normalized_dockerfile.encode()).hexdigest()[:12],
        test_sh_id=hashlib.sha256(task.files.get(TEST_SH, b"")).hexdigest()[:12],
        code_files=tuple(code),
        data_files=tuple(data),
    )
