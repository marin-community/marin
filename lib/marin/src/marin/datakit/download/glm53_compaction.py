# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 compaction summaries paired with their original AgentTrove histories."""

import hashlib
import json

from marin.datakit.download.referenced_completion import ReferencedCompletion, chat_normalize_steps, completion_document
from marin.execution.step_spec import StepSpec

AGENTTROVE_NAME = "agenttrove-glm53-compactions"
AGENTTROVE_REPO = "open-athena/" + AGENTTROVE_NAME
AGENTTROVE_REVISION = "6daf898de0743872e972c2bfb7c399bd052bd31f"
OPENCODE_REVISION = "0033bb35599a359def31b53d73e885eb4c44d815"
# Verbatim SUMMARY_TEMPLATE from packages/core/src/session/compaction.ts at OPENCODE_REVISION.
COMPACTION_TEMPLATE = (
    "Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do "
    "not include the <template> tags in your response.\n"
    "<template>\n"
    "## Objective\n"
    "- [one or two brief sentences describing what the user is trying to accomplish]\n"
    "\n"
    "## Important Details\n"
    "- [constraints/preferences, decisions and why, important facts/assumptions, exact context needed to "
    'continue, or "(none)"]\n'
    "\n"
    "## Work State\n"
    "### Completed\n"
    '- [finished work, verified facts, or changes made; otherwise "(none)"]\n'
    "\n"
    "### Active\n"
    '- [current work, partial changes, or investigation state; otherwise "(none)"]\n'
    "\n"
    "### Blocked\n"
    '- [blockers, failing commands, or unknowns; otherwise "(none)"]\n'
    "\n"
    "## Next Move\n"
    '1. [immediate concrete action, or "(none)"]\n'
    '2. [next action if known, or "(none)"]\n'
    "\n"
    "## Relevant Files\n"
    '- [file or directory path: why it matters, or "(none)"]\n'
    "</template>\n"
    "\n"
    "Rules:\n"
    "- Keep every section, even when empty.\n"
    "- Use terse bullets, not prose paragraphs.\n"
    "- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers when known.\n"
    "- Do not mention the summary process or that context was compacted."
)


def compaction_document(row: dict, source: dict) -> dict:
    """Recover the exact compaction request, keeping the old trajectory inside its user turn."""
    history = source["conversations"]
    if hashlib.sha256(json.dumps(history, ensure_ascii=False).encode("utf-8")).hexdigest() != row["trace_id"]:
        raise ValueError("AgentTrove source history hash mismatch")
    if row["opencode_revision"] != OPENCODE_REVISION:
        raise ValueError("Unexpected OpenCode compaction prompt revision")
    context = "\n\n".join(f"[{m['role'].capitalize()}]: {m['content']}" for m in history)
    prompt = "\n\n".join(
        [
            f"Here is the conversation so far:\n\n<conversation>\n{context}\n</conversation>",
            "Create a new anchored summary from the conversation history in the <conversation> tags above "
            "so another coding agent can continue the work.",
            COMPACTION_TEMPLATE,
        ]
    )
    return completion_document(prompt, row["compaction"], AGENTTROVE_REPO, row["trace_id"])


COMPACTIONS = ReferencedCompletion(
    AGENTTROVE_NAME,
    AGENTTROVE_REVISION,
    AGENTTROVE_REPO,
    "open-thoughts/AgentTrove",
    "b395a4307a2bc9950a90dc899438f149e115fc60",
    ("conversations",),
    compaction_document,
)


def glm53_compaction_chat_normalize_steps() -> tuple[StepSpec, ...]:
    return chat_normalize_steps(COMPACTIONS, "train", prompt_hash_attrs={"opencode_revision": OPENCODE_REVISION})
