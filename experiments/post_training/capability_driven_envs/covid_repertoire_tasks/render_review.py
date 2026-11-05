# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render generated tasks and hidden answers for private human design review."""

import argparse
import json
from pathlib import Path

from taskcompendium.models import TaskSpec, TextMessage, VerifyitGrader, verifyit_spec
from verifyit.spec import StructuredExactSpec


def render(tasks_path: Path, *, limit: int | None = None) -> str:
    """Return a private Markdown review sheet for generated tasks."""
    tasks = [TaskSpec.model_validate_json(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    if limit is not None:
        tasks = tasks[:limit]
    sections = [
        "# Private task-design review",
        "",
        "> Contains hidden verifier answers. Do not expose this file to solvers.",
        "",
    ]
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task.grader, VerifyitGrader):
            raise ValueError(f"{task.id} does not use a VerifyIt grader")
        spec = verifyit_spec(task.grader)
        if not isinstance(spec, StructuredExactSpec):
            raise ValueError(f"{task.id} does not use StructuredExactSpec")
        instruction = task.context.events[0]
        if not isinstance(instruction, TextMessage):
            raise ValueError(f"{task.id} has no text instruction")
        sections.extend(
            (
                f"## {index}. `{task.id}`",
                "",
                f"**Family:** `{task.tags[-1]}`",
                "",
                "### Solver instruction",
                "",
                str(instruction.content),
                "",
                "### Hidden expected answer",
                "",
                "```json",
                json.dumps(spec.expected, indent=2, sort_keys=True),
                "```",
                "",
                "### Reviewer notes",
                "",
                "- Requires more than transcription: TODO",
                "- Uses biological knowledge or reference material: TODO",
                "- Has one defensible answer: TODO",
                "- Distractors/failure modes are meaningful: TODO",
                "- Keep, revise, or reject: TODO",
                "",
            )
        )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.output.write_text(render(args.tasks, limit=args.limit))
    print(f"wrote private review sheet to {args.output}")


if __name__ == "__main__":
    main()
