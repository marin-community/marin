# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The graceful-resolution lane: traces that end well instead of looping.

For a task the generator cannot solve within its band budget, the lane continues from the
best failed attempt under a form contract: bounded re-derivation, then exactly one of two
closings -- commit to a best-effort answer, or state the uncertainty explicitly. The
structural checks below enforce the contract on every graceful trace; the store carries
only traces that pass them.
"""

from dataclasses import dataclass

from experiments.post_training.task_curation.balanced_reasoning.style import StyleReport, style_report

BEST_EFFORT_MARKER = "Final answer (best effort):"
UNCERTAIN_MARKER = "Final answer (uncertain):"
CLOSING_MARKERS: tuple[str, ...] = (BEST_EFFORT_MARKER, UNCERTAIN_MARKER)
"""The two contract closings. The graceful prompt requires one of these lines verbatim, so
the check is a literal prefix match and never a fuzzy classifier."""

RESOLUTION_BEST_EFFORT = "graceful_best_effort"
RESOLUTION_UNCERTAIN = "graceful_uncertain"

RESOLUTION_BY_MARKER = {BEST_EFFORT_MARKER: RESOLUTION_BEST_EFFORT, UNCERTAIN_MARKER: RESOLUTION_UNCERTAIN}

CONVERGENCE_PROMPT = """\
Your previous attempt did not reach a verified answer. Converge now:

1. Spend a short, bounded effort re-deriving only the parts you are least sure of. Do not \
restart the problem from scratch more than once.
2. The FINAL line of your reply must be exactly one of these two lines, with the prefix \
copied verbatim:

{best_effort_line}
{uncertain_line}

The first commits to your single best candidate answer plus one sentence on why you believe \
it despite not verifying it. The second states that the answer is uncertain, gives your best \
candidate, and names the concrete blocker. Never leave the reply without this final line; the \
line must start with exactly "{best_effort_prefix}" or "{uncertain_prefix}".
"""

BEST_EFFORT_LINE = f"{BEST_EFFORT_MARKER} <answer> -- <one sentence on why this is your best candidate>"
UNCERTAIN_LINE = f"{UNCERTAIN_MARKER} <answer> -- <the concrete blocker>"

CORRECTION_PROMPT = """\
Your reply did not end with the required closing line. Reply once more with the same bounded \
effort, and make the LAST line exactly one of:

{best_effort_line}
{uncertain_line}

Copy the prefix verbatim; nothing may follow that line.
"""


@dataclass(frozen=True, slots=True)
class GracefulCheck:
    """The structural form verdict on one graceful trace."""

    resolution: str | None
    """``graceful_best_effort`` or ``graceful_uncertain``; ``None`` when the contract fails."""
    reason: str | None
    style: StyleReport

    @property
    def passed(self) -> bool:
        return self.resolution is not None


def graceful_convergence_prompt() -> str:
    """The user turn that converts a failed attempt into a bounded, committed reply."""
    return CONVERGENCE_PROMPT.format(
        best_effort_line=BEST_EFFORT_LINE,
        uncertain_line=UNCERTAIN_LINE,
        best_effort_prefix=BEST_EFFORT_MARKER,
        uncertain_prefix=UNCERTAIN_MARKER,
    )


def graceful_correction_prompt() -> str:
    """The user turn that retries a reply missing its contract closing."""
    return CORRECTION_PROMPT.format(best_effort_line=BEST_EFFORT_LINE, uncertain_line=UNCERTAIN_LINE)


def _closing_resolution(content: str) -> tuple[str | None, str | None]:
    for marker in CLOSING_MARKERS:
        if marker in content:
            return RESOLUTION_BY_MARKER[marker], None
    return None, f"graceful reply lacks both contract closings ({'; '.join(CLOSING_MARKERS)})"


def check_graceful_trace(reasoning: str, content: str, finish_reason: str) -> GracefulCheck:
    """Enforce the graceful form contract on one generated reply."""
    style = style_report(f"{reasoning}\n{content}")
    if finish_reason != "stop":
        return GracefulCheck(None, f"graceful reply did not commit (finish_reason={finish_reason})", style)
    if not content.strip():
        return GracefulCheck(None, "graceful reply has no final content", style)
    if not reasoning.strip():
        return GracefulCheck(None, "graceful reply has no reasoning", style)
    resolution, reason = _closing_resolution(content)
    if reason is not None:
        return GracefulCheck(None, reason, style)
    if style.degenerate:
        return GracefulCheck(None, f"graceful reply carries a rejecting style signature: {style.reason()}", style)
    return GracefulCheck(resolution, None, style)
