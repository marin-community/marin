# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep a task's specified environment as part of its data.

A task must run in the environment its archive specifies. Two mechanical checks
enforce that, and neither is a judgement call:

- ``provably_no_op_against`` proves that a Dockerfile changes nothing relative to
  the agent image, so the task may declare that image directly. Anything it cannot
  prove — an extra package, a version mismatch, a copied fixture — means the task
  needs its own environment built.
- ``environment_unused`` flags a task whose archive defines an environment its
  spec does not carry, which happens when a conversion replaces it.

The rule applies to agentic tasks: a conversation-only task needs no machine.
"""

import base64
import re
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

FROM = re.compile(r"^\s*FROM\s+(\S+)", re.IGNORECASE | re.MULTILINE)
STEP = re.compile(r"^\s*(RUN|COPY|ADD|ENV|ARG|USER|WORKDIR|VOLUME|ENTRYPOINT|CMD)\b", re.IGNORECASE | re.MULTILINE)
ENVIRONMENT_DOCKERFILE = "environment/Dockerfile"
SETUP_PREFIX = "setup_files/"


def dockerfile_of(files: Mapping[str, bytes | str]) -> str | None:
    """The archive's environment Dockerfile, when it defines one.

    ``unpack_task_binary`` exposes archived files as base64 text, so both forms are
    accepted; anything that is not text is rejected rather than guessed at.
    """
    blob = files.get(ENVIRONMENT_DOCKERFILE)
    if blob is None:
        return None
    if isinstance(blob, str):
        blob = base64.b64decode(blob, validate=True)
    return blob.decode(errors="replace")


def provably_no_op_against(dockerfile: str, files: Iterable[str], *, agent_image: str) -> tuple[bool, str]:
    """Prove that ``dockerfile`` changes nothing relative to ``agent_image``.

    ``agent_image`` is a digest-pinned reference. A no-op must start from exactly
    that image and carry no step that could install, copy or configure anything.
    A copied fixture is never a no-op, because the agent image cannot contain it.
    """
    froms = FROM.findall(dockerfile)
    if len(froms) != 1:
        return False, f"expected one FROM, found {len(froms)}"
    if froms[0] != agent_image:
        return False, f"FROM {froms[0]} is not the agent image"
    steps = [match.group(1).upper() for match in STEP.finditer(dockerfile)]
    blocking = [step for step in steps if step not in {"WORKDIR"}]
    if blocking:
        return False, f"steps change the image: {', '.join(sorted(set(blocking)))}"
    fixtures = [name for name in files if name.startswith(SETUP_PREFIX)]
    if fixtures:
        return False, f"copies fixtures: {len(fixtures)} file(s)"
    return True, "no changes relative to the agent image"


def environment_unused(
    spec: Mapping, archive_files: Mapping[str, bytes | str], *, bases: Mapping[str, str] | None = None
) -> str | None:
    """Why a converted task ignores its archive's environment, if it does.

    ``spec`` is a TaskCompendium task dict. A task whose archive defines
    ``environment/Dockerfile`` must declare that build context; declaring an image
    instead means the conversion replaced the task's own environment.
    """
    dockerfile = dockerfile_of(archive_files)
    if dockerfile is None:
        return None
    requirements = spec.get("environment_requirements") or {}
    build = requirements.get("docker_build")
    if build is None:
        # A spec may use the archive's environment by replaying its steps on a pinned
        # base instead of declaring a build context. Anything else replaced it.
        try:
            lowered = lower_environment(dockerfile, bases or PINNED_BASES)
        except UnlowerableRecipe as error:
            return f"archive environment is not lowered: {error}"
        if requirements.get("docker_image") != lowered.base_ref:
            return "spec does not use the archive's base image"
        if tuple(requirements.get("setup_commands") or ()) != lowered.setup_commands:
            return "spec does not replay the archive's steps"
        return None
    paths = {resource["path"] for resource in build.get("files") or ()}
    if "Dockerfile" not in paths:
        return "build context does not carry the archive's Dockerfile"
    return None


def missing_environment(spec: Mapping) -> str | None:
    """Why an agentic task cannot be admitted, if it declares no environment.

    A task that declares nothing resolves to the host image at run time, which is
    not the environment the task defines. It is rejected rather than run.
    """
    requirements = spec.get("environment_requirements") or {}
    declared = (
        requirements.get("docker_image"),
        requirements.get("docker_build"),
        requirements.get("packages_lock"),
        requirements.get("package_spec"),
    )
    if any(item is not None for item in declared):
        return None
    if not any(requirements.get("capabilities") or ()):
        return None
    return "agentic task declares no environment"


@dataclass(frozen=True)
class LoweredEnvironment:
    """A recipe expressed as a pinned base image plus the commands that configure it."""

    base_ref: str
    working_directory: str
    setup_commands: tuple[str, ...]


class UnlowerableRecipe(ValueError):
    """A recipe step that cannot be expressed as setup commands without changing the task."""


UNSUPPORTED_DIRECTIVES = frozenset({"COPY", "ADD", "USER", "VOLUME", "ENTRYPOINT", "CMD", "ARG", "SHELL", "HEALTHCHECK"})


def _logical_lines(dockerfile: str) -> list[str]:
    """Recipe lines with backslash continuations joined, comments and blanks dropped."""
    lines: list[str] = []
    buffer: str | None = None
    for raw in dockerfile.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if buffer is None:
            buffer = ""
        if line.endswith("\\"):
            buffer += line[:-1].rstrip() + " "
            continue
        buffer += line
        lines.append(buffer.strip())
        buffer = None
    if buffer:
        lines.append(buffer.strip())
    return lines


PINNED_BASES = {
    # Digest-pinned references for the bases a conversion may lower onto. A tag would let
    # the environment drift under a published cohort; changing one is a deliberate edit.
    "ubuntu:24.04": "ubuntu:24.04@sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d",
    "python:3.13-slim-bookworm": (
        "python:3.13-slim-bookworm@sha256:f3dac913ab53b78c64f26d5518fcd7043d5a1a46522d6a493b598c3ed12f62a6"
    ),
    "python:3.12-slim": "python:3.12-slim@sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177",
    "python:3.12-slim-bookworm": (
        "python:3.12-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4"
    ),
    "python:3.11-slim-bookworm": (
        "python:3.11-slim-bookworm@sha256:e8c69d86735f6df3efa06b324694fe51f0422785f83a3bd463052636c2421944"
    ),
    "python:3.10-slim-bookworm": (
        "python:3.10-slim-bookworm@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8"
    ),
    "debian:12-slim": "debian:12-slim@sha256:c67e92c8099ba22853ab4cfda7cfb494d253d6012ca82dee98ba8976772af018",
    "debian:13.0-slim": "debian:13.0-slim@sha256:f2306da7a8fb6440535937baed0ce0018736742d6dc5beec9d6a31355b259726",
}
"""The digest-pinned bases a conversion may lower onto."""


def lower_environment(dockerfile: str, bases: Mapping[str, str]) -> LoweredEnvironment:
    """Express a recipe as a pinned base plus setup commands, or refuse it.

    The base must be one of ``bases``, whose values are digest-pinned references; every
    ``RUN`` becomes a setup command run from the active ``WORKDIR`` with the recipe's
    ``ENV`` applied. Anything that installs files from the build context, changes the
    user or declares an entrypoint is refused: expressing it as a command would change
    what the task's environment is.
    """
    froms = FROM.findall(dockerfile)
    if len(froms) != 1:
        raise UnlowerableRecipe(f"expected one FROM, found {len(froms)}")
    base = froms[0]
    if base not in bases:
        raise UnlowerableRecipe(f"unmapped base: {base}")
    base_ref = bases[base]
    if "@sha256:" not in base_ref:
        raise UnlowerableRecipe(f"base is not digest-pinned: {base_ref}")

    workdir = "/"
    environment: list[str] = []
    commands: list[str] = []
    for line in _logical_lines(dockerfile):
        directive, _, body = line.partition(" ")
        directive = directive.upper()
        if directive == "FROM":
            continue
        if directive in UNSUPPORTED_DIRECTIVES:
            raise UnlowerableRecipe(f"unsupported step: {directive}")
        if directive == "WORKDIR":
            workdir = body.strip()
            commands.append(f"mkdir -p {shlex.quote(workdir)}")
            continue
        if directive == "ENV":
            assignments = body.strip()
            if "=" not in assignments or " " in assignments.split("=", 1)[1].strip():
                raise UnlowerableRecipe(f"unsupported ENV form: {body!r}")
            environment.append(f"{assignments.split('=', 1)[0]}={shlex.quote(assignments.split('=', 1)[1].strip())}")
            continue
        if directive == "RUN":
            prefix = f"{' '.join(environment)} " if environment else ""
            commands.append(f"(cd {shlex.quote(workdir)} && {prefix}{body.strip()})")
            continue
        raise UnlowerableRecipe(f"unknown step: {directive}")
    return LoweredEnvironment(base_ref=base_ref, working_directory=workdir, setup_commands=tuple(commands))
