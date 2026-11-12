# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep a task's specified environment as part of its data.

A task must run in the environment its archive specifies.
``environment_unused`` flags a task whose archive defines an environment its
spec does not carry, which happens when a conversion replaces it.

The rule applies to agentic tasks: a conversation-only task needs no machine.
"""

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass

from taskcompendium.models import CommandSemantics, EnvironmentRequirements

from experiments.post_training.task_curation.environment import dockerfile_lines

FROM = re.compile(r"^\s*FROM\s+(\S+)", re.IGNORECASE | re.MULTILINE)
ENVIRONMENT_PREFIX = "environment/"
"""The tar prefix under which a native task archive carries its environment."""

ENVIRONMENT_DOCKERFILE = f"{ENVIRONMENT_PREFIX}Dockerfile"

DEFAULT_WORKDIR = "/app"
"""The workdir a task gets when neither the recipe nor the spec declares one."""


def _utf8_dockerfile(dockerfile: bytes) -> str:
    try:
        return dockerfile.decode()
    except UnicodeDecodeError as error:
        raise UnlowerableRecipe("archive environment/Dockerfile is not UTF-8") from error


def environment_unused(spec: Mapping, dockerfile: bytes | None) -> str | None:
    """Why a converted task ignores its archive's environment, if it does.

    ``spec`` is a TaskCompendium task dict. An archive that defines an environment
    Dockerfile must be declared as a build context or replayed as lowered steps,
    working directory and variables on the matching digest-pinned base.
    """
    if dockerfile is None:
        return None
    requirements = spec.get("environment_requirements") or {}
    build = requirements.get("docker_build")
    if build is None:
        # A spec may use the archive's environment by replaying its steps on a pinned
        # base instead of declaring a build context. Anything else replaced it.
        try:
            lowered = lower_environment(_utf8_dockerfile(dockerfile), PINNED_BASES)
        except UnlowerableRecipe as error:
            return f"archive environment is not lowered: {error}"
        if requirements.get("docker_image") != lowered.base_ref:
            return "spec does not use the archive's base image"
        if tuple(requirements.get("setup_commands") or ()) != lowered.setup_commands:
            return "spec does not replay the archive's steps"
        if requirements.get("working_directory") != lowered.working_directory:
            return "spec does not use the archive's working directory"
        if (requirements.get("environment_variables") or {}) != lowered.environment_variables:
            return "spec does not retain the archive's environment variables"
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
    environment_variables: dict[str, str]


class UnlowerableRecipe(ValueError):
    """A recipe step that cannot be expressed as setup commands without changing the task."""


UNSUPPORTED_DIRECTIVES = frozenset({"COPY", "ADD", "USER", "VOLUME", "ENTRYPOINT", "CMD", "ARG", "SHELL", "HEALTHCHECK"})


PINNED_BASES = {
    "aweaiteam/calibforge:tb2-verifier-base-v1": (
        "aweaiteam/calibforge:tb2-verifier-base-v1@sha256:"
        "0f61b2776c40231d5d868278ed6042295ddbc9d24b036cd72f0b88ebe9bd621a"
    ),
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
    "python:3.10-slim": "python:3.10-slim@sha256:7b6bd969dda3ecfca0655524e338b57a8b8cd692a9362e3d55b5a3ef49b07256",
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
    variables: dict[str, str] = {}
    commands: list[str] = []
    for line in dockerfile_lines(dockerfile):
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
            # One ENV line may set several variables: ENV A=1 B=2.
            for assignment in shlex.split(body):
                key, separator, value = assignment.partition("=")
                if not separator or not key or "$" in value or "`" in value:
                    raise UnlowerableRecipe(f"unsupported ENV form: {body!r}")
                environment.append(f"{key}={shlex.quote(value)}")
                variables[key] = value
            continue
        if directive == "RUN":
            prefix = f"export {' '.join(environment)}; " if environment else ""
            commands.append(f"(cd {shlex.quote(workdir)} && {prefix}{body.strip()})")
            continue
        raise UnlowerableRecipe(f"unknown step: {directive}")
    return LoweredEnvironment(
        base_ref=base_ref,
        working_directory=workdir,
        setup_commands=tuple(commands),
        environment_variables=variables,
    )


def archive_environment(dockerfile: bytes | None):
    """A TaskTrove archive's own environment, lowered onto a pinned base.

    The archive defines how the task's machine is built; using anything else would run the
    task somewhere it never specified. A recipe the lowering cannot express raises, so the
    row is refused rather than quietly given a different environment.
    """
    if dockerfile is None:
        raise ValueError("archive defines no environment/Dockerfile")
    lowered = lower_environment(_utf8_dockerfile(dockerfile), PINNED_BASES)
    return EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=lowered.base_ref,
        working_directory=lowered.working_directory,
        setup_commands=lowered.setup_commands,
        environment_variables=lowered.environment_variables,
    )
