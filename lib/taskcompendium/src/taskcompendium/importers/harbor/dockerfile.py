# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the external images a Dockerfile builds from and pin them to registry digests.

Builds receive no build arguments, so a ``FROM`` that names a global ``ARG`` resolves to the
argument's default. References to earlier stages and ``scratch`` are not external images.
"""

import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass

from taskcompendium.importers.harbor.config import UnsupportedHarborFeature
from taskcompendium.models import DOCKER_IMAGE_PATTERN

PINNED_IMAGE = re.compile(DOCKER_IMAGE_PATTERN)
_VARIABLE = re.compile(r"\$(?:\{(\w+)(?::-([^}]*))?\}|(\w+))")
_HEREDOC = re.compile(r"(?<!<)<<(?!<)(-?)([\"']?)([A-Za-z_]\w*)\2")
_ESCAPE_DIRECTIVE = re.compile(r"#\s*escape\s*=\s*(\S)", re.IGNORECASE)
_HEREDOC_INSTRUCTIONS = frozenset({"RUN", "COPY", "ADD"})


@dataclass(frozen=True)
class _Instruction:
    keyword: str
    arguments: str
    lines: range
    """Physical lines holding the instruction, excluding heredoc bodies."""


@dataclass(frozen=True)
class _ImageUse:
    line: int
    reference: str
    from_flag: str | None
    """For ``COPY --from``, the flag's text; ``None`` for a ``FROM`` instruction."""


@dataclass(frozen=True)
class _Scan:
    lines: list[str]
    uses: tuple[_ImageUse, ...]
    workdir: str | None


def _instructions(lines: list[str]) -> list[_Instruction]:
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#"):
            break
        escape = _ESCAPE_DIRECTIVE.fullmatch(line.strip())
        if escape is not None and escape[1] != "\\":
            raise UnsupportedHarborFeature("Dockerfiles with a non-default escape character")
    instructions = []
    index = 0
    while index < len(lines):
        if not lines[index].strip() or lines[index].lstrip().startswith("#"):
            index += 1
            continue
        start = index
        parts = []
        while index < len(lines):
            text = lines[index].rstrip()
            index += 1
            if not text.endswith("\\"):
                parts.append(text)
                break
            parts.append(text[:-1])
            # Docker drops comment lines inside a continued instruction.
            while index < len(lines) and lines[index].lstrip().startswith("#"):
                index += 1
        keyword, _, arguments = " ".join(parts).strip().partition(" ")
        keyword = keyword.upper()
        instructions.append(_Instruction(keyword, arguments.strip(), range(start, index)))
        if keyword in _HEREDOC_INSTRUCTIONS:
            for strip_tabs, _, terminator in _HEREDOC.findall(arguments):
                while True:
                    if index == len(lines):
                        raise UnsupportedHarborFeature(f"Unterminated Dockerfile heredoc {terminator}")
                    body = lines[index].rstrip("\r\n")
                    index += 1
                    if (body.lstrip("\t") if strip_tabs else body) == terminator:
                        break
    return instructions


def _substitute(text: str, arguments: dict[str, str | None]) -> str:
    def value(match: re.Match[str]) -> str:
        name = match[1] or match[3]
        current = arguments.get(name)
        if current:
            return current
        if match[2] is not None:
            return match[2]
        if current is not None:
            return current
        raise UnsupportedHarborFeature(f"Dockerfile image {text!r} depends on build argument {name} without a default")

    resolved = _VARIABLE.sub(value, text)
    if "$" in resolved:
        raise UnsupportedHarborFeature(f"Dockerfile image {text!r} uses an unsupported substitution")
    return resolved


def _global_arguments(arguments: str, known: dict[str, str | None]) -> None:
    for declaration in arguments.split():
        name, separator, default = declaration.partition("=")
        known[name] = _substitute(default.strip("\"'"), known) if separator else None


def _scan(text: str) -> _Scan:
    lines = text.splitlines(keepends=True)
    arguments: dict[str, str | None] = {}
    workdirs: list[str | None] = []
    aliases: dict[str, int] = {}
    uses: list[_ImageUse] = []

    def is_stage(name: str) -> bool:
        return name.lower() in aliases or (name.isdecimal() and int(name) < len(workdirs))

    for instruction in _instructions(lines):
        if instruction.keyword == "ARG" and not workdirs:
            _global_arguments(instruction.arguments, arguments)
        elif instruction.keyword == "FROM":
            if len(instruction.lines) != 1:
                raise UnsupportedHarborFeature("FROM instructions spanning several lines")
            tokens = [token for token in instruction.arguments.split() if not token.startswith("--")]
            if len(tokens) not in (1, 3) or (len(tokens) == 3 and tokens[1].upper() != "AS"):
                raise UnsupportedHarborFeature(f"Unrecognized FROM instruction: {instruction.arguments!r}")
            image = _substitute(tokens[0], arguments)
            if image.lower() in aliases:
                workdirs.append(workdirs[aliases[image.lower()]])
            elif image == "scratch":
                workdirs.append("/")
            else:
                workdirs.append(None)
                uses.append(_ImageUse(instruction.lines.start, image, None))
            if len(tokens) == 3:
                aliases[tokens[2].lower()] = len(workdirs) - 1
        elif instruction.keyword == "WORKDIR" and workdirs:
            target = instruction.arguments.strip("\"'")
            current = workdirs[-1]
            if "$" in target:
                workdirs[-1] = None
            elif target.startswith("/"):
                workdirs[-1] = posixpath.normpath(target)
            elif current is not None:
                workdirs[-1] = posixpath.normpath(posixpath.join(current, target))
        elif instruction.keyword == "COPY":
            for flag in instruction.arguments.split():
                source = flag.removeprefix("--from=")
                if source == flag or is_stage(source):
                    continue
                if "$" in source:
                    raise UnsupportedHarborFeature(f"COPY {flag} depends on a build argument")
                line = next(index for index in instruction.lines if flag in lines[index])
                uses.append(_ImageUse(line, source, flag))
        elif instruction.keyword == "RUN":
            for flag in instruction.arguments.split():
                if not flag.startswith("--mount="):
                    continue
                for option in flag.removeprefix("--mount=").split(","):
                    key, _, value = option.partition("=")
                    if key == "from" and not is_stage(value):
                        raise UnsupportedHarborFeature("RUN --mount from an external image")
    if not workdirs:
        raise UnsupportedHarborFeature("Dockerfile has no FROM instruction")
    return _Scan(lines, tuple(uses), workdirs[-1])


def dockerfile_images(text: str) -> tuple[str, ...]:
    """External images the build pulls, in order and without repeats."""
    return tuple(dict.fromkeys(use.reference for use in _scan(text).uses))


def dockerfile_workdir(text: str) -> str | None:
    """The final stage's ``WORKDIR``, or ``None`` when it is the external base image's own."""
    return _scan(text).workdir


def pin_dockerfile(text: str, pin: Callable[[str], str]) -> str:
    """Rewrite every external image reference to ``pin(reference)``, keeping every other line unchanged.

    A ``FROM`` line keeps its indentation and line ending, with its tokens separated by single spaces.
    """
    scan = _scan(text)
    lines = list(scan.lines)
    for use in scan.uses:
        pinned = pin(use.reference)
        line = lines[use.line]
        if use.from_flag is not None:
            lines[use.line] = line.replace(use.from_flag, f"--from={pinned}", 1)
            continue
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        indent = body[: len(body) - len(body.lstrip())]
        tokens = body.split()
        image = next(index for index, token in enumerate(tokens) if index > 0 and not token.startswith("--"))
        tokens[image] = pinned
        lines[use.line] = indent + " ".join(tokens) + ending
    return "".join(lines)
