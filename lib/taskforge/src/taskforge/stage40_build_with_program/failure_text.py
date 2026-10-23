# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The failure text a program's author reads: where in the program the build failed, then the error.

``failure_head`` finds the program's frames in an exception's tracebacks by the files of its step
functions (``stage30_author.loader.program_file``, the name the loader registers in ``linecache``)
and reads, for an error raised in the program, in library code called from it, or with no program
frame at all::

    program.py:12, in step numbers: return scale * 2
    NameError: name 'scale' is not defined

    program.py:12, in step numbers: return TaskResource(
    raised in library code at pydantic/main.py:253 (__init__)
    ValidationError: 1 validation error for TaskResource

    step numbers, outside the program, at stage40_build_with_program/run.py:124 (run)
    ValueError: ...

A frame in a function nested in a step names it after the step: ``program.py:40, in step grader
(lex): ...``. ``failure_text`` puts the head above the bounded tail of the traceback.

A build in a child process (``build_process``) computes the head in the child, where the exception's
frames are, and the parent raises ``BuildCrash`` with that head and the child's traceback, so both
texts read the same for a build in either process.
"""

import traceback
from collections.abc import Collection
from pathlib import Path

from taskforge.stage30_author.author import PROGRAM_FILE
from taskforge.stage40_build_with_program.context import BuildFailure
from taskforge.stage40_build_with_program.infrastructure import exception_chain


class BuildCrash(Exception):
    """A program's own exception, other than a build outcome, from a build run in a child process.

    ``str()`` is ``head``, the child's ``failure_head`` of the exception; ``traceback`` is the
    child's formatted traceback.
    """

    def __init__(self, head: str, traceback: str):
        super().__init__(head)
        self.head = head
        self.traceback = traceback


def error_line(error: BaseException) -> str:
    """The error's type and the first line of its message, or the type alone for an empty message."""
    message = error.message if isinstance(error, BuildFailure) else str(error)
    first = message.partition("\n")[0]
    return f"{type(error).__name__}: {first}" if first else type(error).__name__


def failure_head(
    error: BaseException, files: Collection[str], steps: Collection[str], *, shown_as: str = PROGRAM_FILE
) -> str:
    """Locate ``error`` in the program whose step functions are in ``files``, then name it.

    The head's first line is the deepest program frame from the step's own frame on: ``shown_as``,
    its line number, the step and the source line. It picks a traceback from the error's chain: the
    innermost exception whose traceback passes through the program, since the runner re-raises a
    step's ``BuildFailure`` from the original, otherwise the innermost one with a traceback. The step
    is ``BuildFailure.step``, or else the outermost program frame running one of ``steps``. A
    ``BuildCrash`` carries its head.
    """
    if isinstance(error, BuildCrash):
        return error.head
    named = error.step if isinstance(error, BuildFailure) else None
    tracebacks = [traceback.extract_tb(link.__traceback__) for link in reversed(exception_chain(error))]
    passing = [frames for frames in tracebacks if any(frame.filename in files for frame in frames)]
    frames = passing[0] if passing else next((frames for frames in tracebacks if frames), [])
    headline = error_line(error)
    if not frames:
        return headline if named is None else f"step {named}\n{headline}"
    last = frames[-1]
    library = f"{Path(*Path(last.filename).parts[-2:])}:{last.lineno} ({last.name})"
    inside = [index for index, frame in enumerate(frames) if frame.filename in files]
    wanted = {named} if named is not None else set(steps)
    entries = [index for index in inside if frames[index].name in wanted]
    step = named or (frames[entries[0]].name if entries else None)
    deepest = [index for index in inside if not entries or index >= entries[0]]
    if not deepest:
        return f"{f'step {step}, ' if step else ''}outside the program, at {library}\n{headline}"
    frame = frames[deepest[-1]]
    function = "" if frame.name == step else f" ({frame.name})"
    where = f"in step {step}{function}" if step else f"in {frame.name}"
    location = f"{shown_as}:{frame.lineno}, {where}: {(frame.line or '').strip()}"
    if deepest[-1] == len(frames) - 1:
        return f"{location}\n{headline}"
    return f"{location}\nraised in library code at {library}\n{headline}"


def failure_text(head: str, error: BaseException, limit: int) -> str:
    """``head``, a blank line, then the last ``limit`` characters of ``error``'s traceback.

    A ``BuildCrash``'s traceback is the child's, where the program ran.
    """
    trace = error.traceback if isinstance(error, BuildCrash) else "".join(traceback.format_exception(error))
    return f"{head}\n\n{trace[-limit:]}"
