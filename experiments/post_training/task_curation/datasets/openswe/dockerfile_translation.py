# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OpenSWE per-task Dockerfile translation.

Upstream builds each task image with the row's own Dockerfile as the recipe and
a repository checkout as the build context (``scripts/build_images.py``: clone,
check out ``base_commit``, ``docker build`` with the repo as context). The
profile keeps the recipe bytes and replaces only what a declarative build
cannot carry:

- ``FROM openswe-python-<version>`` — the ten upstream base images are never
  published, so the recipe binds the profile's digest-pinned condensed base for
  that Python version instead; the fidelity gate proves outcome equivalence.
- other registry bases — passed through for the shared digest-pinning resolver.
- ``COPY repo <dest>`` — the build context was the repository checkout, which a
  declarative build context cannot carry, so the copy becomes a build-time
  clone+checkout of the pinned commit into the same destination. Every other
  byte passes through untouched.

Everything the translation cannot express faithfully is a typed rejection with
the offending bytes as evidence — never a silent rewrite.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

CONDA_BASE_PATTERN = re.compile(r"^openswe-python-(?P<version>[\d.]+)$")
DIGEST_PIN_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
REPO_COPY_PATTERN = re.compile(r"^COPY\b(?P<rest>.*)$", re.IGNORECASE)
REPO_SOURCE_PATTERN = re.compile(r"^repo(?:/\S+)*$", re.IGNORECASE)
STAGING_CLONE_ROOT = "/tmp/openswe-repo-staging"


class DockerfileRejected(ValueError):
    """A typed rejection carrying the offending Dockerfile evidence."""


def _heredoc_delimiter(argument: str) -> str | None:
    """The heredoc delimiter a shell command opens, honoring quoting.

    ``cat > f <<'EOF'`` opens a heredoc whose body is shell data; ``echo "a << b"``
    does not (the operator is inside quotes). Only unquoted ``<<`` counts.
    """
    quote: str | None = None
    index = 0
    while index < len(argument):
        char = argument[index]
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif argument.startswith("<<", index):
            index += 2
            if index < len(argument) and argument[index] == "-":
                index += 1
            while index < len(argument) and argument[index].isspace():
                index += 1
            if index < len(argument) and argument[index] in "'\"":
                quote_char = argument[index]
                index += 1
                end = argument.find(quote_char, index)
                if end == -1:
                    return None
                return argument[index:end]
            match = re.match(r"[\w.-]+", argument[index:])
            if match:
                return match.group(0)
            return None
        index += 1
    return None


@dataclass(frozen=True)
class Instruction:
    """One top-level Dockerfile instruction with its original line number."""

    name: str
    argument: str
    line_number: int


def dockerfile_instructions(text: str) -> list[Instruction]:
    """Scan top-level Dockerfile instructions; heredoc bodies and continuations excluded.

    OpenSWE's recipes are machine-authored and embed shell heredocs inside RUN
    steps whose bodies contain instruction-looking lines (``from typing import
    ...`` and even literal ``FROM`` text); those bytes are shell data, never
    instructions. A backslash-continued line joins before tokenizing, matching
    Docker's own parser.
    """
    instructions: list[Instruction] = []
    pending = ""
    pending_line = 0
    heredoc_delimiter: str | None = None
    for line_number, raw in enumerate([*text.splitlines(), ""], start=1):
        if heredoc_delimiter is not None:
            if raw.strip() == heredoc_delimiter:
                heredoc_delimiter = None
            continue
        if pending:
            raw = pending + raw.lstrip()
        else:
            pending_line = line_number
        if raw.rstrip().endswith("\\") and not raw.rstrip().endswith("\\\\"):
            pending = raw.rstrip()[:-1].rstrip() + " "
            continue
        pending = ""
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, _, argument = stripped.partition(" ")
        instructions.append(Instruction(name=name.upper(), argument=argument.strip(), line_number=pending_line))
        heredoc_delimiter = _heredoc_delimiter(argument)
    return instructions


@dataclass(frozen=True)
class Translation:
    """The translated Dockerfile plus the evidence the conversion records."""

    text: str
    base_group: str | None
    declared_bases: tuple[str, ...]
    clone_destination: str | None
    workdir: str


def _clone_block(repo: str, base_commit: str, destination: str) -> str:
    quoted_repo = repo.replace("'", "'\\''")
    return (
        f"RUN git clone https://github.com/{quoted_repo}.git '{destination}' \\\n"
        f" && git -C '{destination}' checkout --quiet '{base_commit}' \\\n"
        f" && git -C '{destination}' remote remove origin"
    )


def _rewrite_copy(
    lines: list[str],
    copies: list[tuple[int, list[str], str]],
    repo: str,
    base_commit: str,
) -> list[str]:
    """Replace the repo copies with build-time clone+copy steps.

    The first repo copy decides the clone destination. A full-repo copy (the
    single source ``repo``, ``repo/`` or ``repo/.``) clones straight into its
    destination; any other first copy (the layer-caching shape that copies
    ``repo/requirements.txt`` first) clones into a staging directory that every
    copy step reads from. Each copy becomes shell copy commands out of the
    cloned tree, preserving the recipe's order and destinations.
    """
    first_index, first_sources, first_destination = copies[0]
    full = len(first_sources) == 1 and first_sources[0] in ("repo", "repo/", "repo/.")
    clone_root = first_destination.rstrip("/") if full else STAGING_CLONE_ROOT
    rewritten = list(lines)
    for index, sources, destination in reversed(copies):
        target = destination.rstrip("/") or "/"
        if index == first_index and full:
            rewritten[index] = _clone_block(repo, base_commit, clone_root)
            continue
        commands = []
        for source in sources:
            relative = source[len("repo") :].lstrip("/")
            # Whole-tree copies (``repo/``, ``repo/.``, a trailing slash) need the
            # destination directory to exist for the ``/.`` content copy; Docker
            # creates the parent path for a plain file copy itself.
            tree_copy = relative in ("", ".") or relative.endswith("/")
            commands.append(f"mkdir -p '{target if tree_copy else target.rsplit('/', 1)[0] or '/'}'")
            commands.append(f"cp -a '{clone_root}/{relative or '.'}' '{target}'")
        rewritten[index] = "RUN " + " && ".join(commands)
    if not full:
        rewritten.insert(first_index, _clone_block(repo, base_commit, clone_root))
    return rewritten


def _repo_copies(instructions: list[Instruction]) -> list[tuple[int, list[str], str]]:
    """The repo-bearing COPY instructions: (line index, sources, destination).

    Docker COPY syntax is ``COPY [flags] <src>... <dst>``; flags on a repo copy
    have no faithful build-time clone form, so they are typed rejections.
    """
    copies: list[tuple[int, list[str], str]] = []
    for instruction in instructions:
        if instruction.name != "COPY":
            continue
        match = REPO_COPY_PATTERN.match("COPY " + instruction.argument)
        if match is None:
            continue
        tokens = match.group("rest").split()
        if not tokens:
            continue
        flags = [token for token in tokens if token.startswith("--")]
        if flags:
            raise DockerfileRejected(
                f"Line {instruction.line_number}: COPY flags {flags[0]!r} on a repo copy have no faithful "
                "build-time clone form"
            )
        sources, destination = tokens[:-1], tokens[-1]
        if not sources or not all(REPO_SOURCE_PATTERN.fullmatch(source) for source in sources):
            continue
        copies.append((instruction.line_number - 1, sources, destination))
    return copies


def _resolve_from(
    instruction: Instruction,
    lines: list[str],
    condensed_bases: dict[str, str],
    resolve_base: Callable[[str], str] | None,
) -> str | None:
    """Digest-pin one root FROM base in place; a stage re-reference returns None.

    The condensed base binding covers ``openswe-python-<version>``; every other
    literal registry base goes through the shared resolver. A variable reference
    or an unresolved base is a typed rejection.
    """
    fields = [token for token in instruction.argument.split() if not token.startswith("--")]
    if not fields:
        raise DockerfileRejected(f"Line {instruction.line_number}: an empty FROM instruction")
    base = fields[0]
    if base == "scratch" or DIGEST_PIN_PATTERN.fullmatch(base):
        return None
    if "$" in base:
        raise DockerfileRejected(f"Line {instruction.line_number}: FROM {base!r} is not a literal reference")
    match = CONDA_BASE_PATTERN.fullmatch(base)
    if match:
        pinned = condensed_bases.get(match.group("version"))
        if not pinned or not DIGEST_PIN_PATTERN.fullmatch(pinned):
            raise DockerfileRejected(
                f"Line {instruction.line_number}: no digest-pinned condensed base covers Python {match.group('version')}"
            )
    elif resolve_base is None:
        raise DockerfileRejected(
            f"Line {instruction.line_number}: base {base!r} is not digest-pinned and no resolver is staged"
        )
    else:
        try:
            pinned = resolve_base(base)
        except (ValueError, OSError) as error:
            raise DockerfileRejected(
                f"Line {instruction.line_number}: base {base!r} does not resolve: {error}"
            ) from error
        if not DIGEST_PIN_PATTERN.fullmatch(pinned):
            raise DockerfileRejected(
                f"Line {instruction.line_number}: base {base!r} resolved to {pinned!r}, not a digest pin"
            )
    lines[instruction.line_number - 1] = lines[instruction.line_number - 1].replace(base, pinned, 1)
    return base


def translate_dockerfile(
    text: str,
    *,
    repo: str,
    base_commit: str,
    condensed_bases: dict[str, str],
    resolve_base: Callable[[str], str] | None = None,
) -> Translation:
    """Translate one row Dockerfile into the profile's declarative build recipe.

    ``condensed_bases`` maps an upstream ``openswe-python-<version>`` base to its
    digest-pinned condensed reference; a missing version is a typed rejection
    (the base build failed or was not staged). ``resolve_base`` digest-pins any
    other registry base through the shared resolver.
    """
    instructions = dockerfile_instructions(text)
    lines = text.splitlines()
    if not any(instruction.name == "FROM" for instruction in instructions):
        raise DockerfileRejected("The Dockerfile declares no FROM instruction")
    stages: set[str] = set()
    declared_bases: list[str] = []
    base_group = None
    for instruction in instructions:
        if instruction.name != "FROM":
            continue
        fields = [token for token in instruction.argument.split() if not token.startswith("--")]
        stage = None
        if len(fields) >= 3 and fields[1].upper() == "AS":
            stage = fields[2]
        base = fields[0] if fields else ""
        if base.lower() in stages or base == "scratch":
            if stage:
                stages.add(stage.lower())
            continue
        declared = _resolve_from(instruction, lines, condensed_bases, resolve_base)
        if declared is not None:
            declared_bases.append(declared)
            match = CONDA_BASE_PATTERN.fullmatch(declared)
            if match:
                base_group = match.group("version")
        if stage:
            stages.add(stage.lower())

    copies = _repo_copies(instructions)
    if copies:
        lines = _rewrite_copy(lines, copies, repo, base_commit)

    workdirs = [instruction.argument.strip() for instruction in instructions if instruction.name == "WORKDIR"]
    workdir = workdirs[-1] if workdirs else "/testbed"
    translated = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    return Translation(
        text=translated,
        base_group=base_group,
        declared_bases=tuple(declared_bases),
        clone_destination=copies[0][2] if copies else None,
        workdir=workdir,
    )
