# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading the candidate text for output-file modes."""

from pathlib import Path

from tasktrove_verify.spec import DEFAULT_WORKSPACE, Spec


def local_output_path(output: str, workspace: Path) -> Path:
    """``output`` re-rooted onto ``workspace`` when it sits under the container's default workspace."""
    path = Path(output)
    prefix = Path(DEFAULT_WORKSPACE).parts
    if path.is_absolute() and path.parts[: len(prefix)] == prefix:
        return workspace.joinpath(*path.parts[len(prefix) :])
    return path


def read_output(spec: Spec, workspace: Path) -> str | None:
    """The candidate text, or ``None`` when the agent wrote nothing.

    ``spec.output`` is absolute inside the task container. Under a local gate the workspace is a
    temporary directory, so an absolute path under ``/app`` is re-rooted onto ``workspace``.
    """
    output = local_output_path(spec.output, workspace)  # type: ignore[union-attr] - only output-file specs reach here
    if not output.is_file():
        return None
    text = output.read_text(errors="replace")
    return text if text.strip() else None
