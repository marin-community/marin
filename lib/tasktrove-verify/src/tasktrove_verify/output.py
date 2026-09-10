# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading the candidate text for output-file modes."""

from pathlib import Path

from tasktrove_verify.spec import Spec


def read_output(spec: Spec, workspace: Path) -> str | None:
    """The candidate text, or ``None`` when the agent wrote nothing.

    ``spec.output`` is absolute inside the task container. Under a local gate the workspace is a
    temporary directory, so an absolute path under ``/app`` is re-rooted onto ``workspace``.
    """
    output = Path(spec.output)  # type: ignore[union-attr] - only output-file specs reach here
    if output.is_absolute() and output.parts[:2] == ("/", "app"):
        output = workspace.joinpath(*output.parts[2:])
    if not output.is_file():
        return None
    text = output.read_text(errors="replace")
    return text if text.strip() else None
