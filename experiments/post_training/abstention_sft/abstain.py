# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the "I don't know" target for a question the student cannot answer.

A writer model drafts a short reply per question, so the wording varies and the student learns a
judgement rather than one fixed sentence. A reply that contains the answer key is written again: it
would teach the student to state the fact it does not know.
"""

from openai import OpenAI

from experiments.post_training.abstention_sft.problems import normalized_question

ABSTAIN_WRITER_PROMPT = (
    "Write the reply of an assistant that does not know the answer to the question below.\n"
    "Rules:\n"
    "- One or two sentences.\n"
    "- Say plainly that you do not know or cannot verify the answer.\n"
    "- Name the specific information or source that would answer it.\n"
    "- Do not guess, hint at an answer, or state any fact about the subject.\n"
    "- Output only the reply.\n"
    "\n"
    "Question: {question}"
)
# Sample the writer so replies to similar questions do not share one wording.
WRITER_TEMPERATURE = 1.0


def leaks_answer(reply: str, answer: str) -> bool:
    key = normalized_question(answer)
    assert key, "an empty answer key cannot be checked"
    return f" {key} " in f" {normalized_question(reply)} "


def abstain_reply(question: str, answer: str, client: OpenAI, model: str, attempts: int = 3) -> str:
    """Ask the writer for an abstain reply until one does not contain the answer key."""
    for _ in range(attempts):
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ABSTAIN_WRITER_PROMPT.format(question=question)}],
            temperature=WRITER_TEMPERATURE,
        )
        reply = (completion.choices[0].message.content or "").strip()
        if reply and not leaks_answer(reply, answer):
            return reply
    raise ValueError(f"no abstain reply without the answer after {attempts} attempts: {question!r}")
