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

import re
from collections.abc import Iterable, Mapping

FROM = re.compile(r"^\s*FROM\s+(\S+)", re.IGNORECASE | re.MULTILINE)
STEP = re.compile(r"^\s*(RUN|COPY|ADD|ENV|ARG|USER|WORKDIR|VOLUME|ENTRYPOINT|CMD)\b", re.IGNORECASE | re.MULTILINE)
ENVIRONMENT_DOCKERFILE = "environment/Dockerfile"
SETUP_PREFIX = "setup_files/"


def dockerfile_of(files: Mapping[str, bytes]) -> str | None:
    """The archive's environment Dockerfile, when it defines one."""
    blob = files.get(ENVIRONMENT_DOCKERFILE)
    return None if blob is None else blob.decode(errors="replace")


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


def environment_unused(spec: Mapping, archive_files: Mapping[str, bytes]) -> str | None:
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
        return "archive defines environment/Dockerfile but the spec declares no build context"
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
