# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EconEvals synthetic labor-market benchmarks, judged pairwise against a released baseline answer.

EconEvals (https://econevals.com/, https://arxiv.org/abs/2607.19375) asks chat questions that
O*NET workers might bring to a chatbot and scores each response with an Arena-Hard judge against
a baseline model's released answer; the paper's baseline is o3-mini. Both orders are judged; a tie
counts half and a decisive verdict counts as three games, so 0.5 means parity with the baseline.

The judge prompts are EconEvals' ``arena-leaderboards`` "ei" system prompt and multi-turn
template, adapted from Arena-Hard-Auto (Apache-2.0). Synthetic queries are single user turns, so
the template's previous-turn block is empty, as upstream renders it.
"""

import json
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from verifyit.spec import RUBRIC_PAIRWISE, EmptyOutputPolicy, JudgeSpec

from taskcompendium.convert.answers import answer_task, source_defect, source_evidence
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow

BASELINE_INPUT = "baseline"
CATEGORIES_INPUT = "categories"
BASELINE_FILENAME = "baseline.jsonl"
CATEGORIES_FILENAME = "categories.json"

JUDGE_SYSTEM_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. Any preceding turns in the conversation will be displayed directly before the user prompt.\n\nYou will be given assistant A's answer and assistant B's answer to the user prompt. Your job is to evaluate which assistant's answer is better. You must identify and correct any mistakes or inaccurate information.\n\nConsider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."  # noqa: E501 - exact source prompt

JUDGE_TEMPLATE = "<|The Start of the Previous Turns|>\n\n<|The End of the Previous Turns|>\n\n<|The Start of the User Prompt|>\n{question}\n<|The End of the User Prompt|>\n\n<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"  # noqa: E501 - exact source template

VERDICT_PATTERNS = (r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]")
# Assistant A's score for each verdict, and how many games it counts as (Arena-Hard's weight 3).
VERDICT_SCORES = {
    "A>B": 1.0,
    "A>>B": 1.0,
    "A=B": 0.5,
    "A<<B": 0.0,
    "A<B": 0.0,
    "B>A": 0.0,
    "B>>A": 0.0,
    "B=A": 0.5,
    "B<<A": 1.0,
    "B<A": 1.0,
}
VERDICT_WEIGHTS = {"A>>B": 3.0, "A<<B": 3.0, "B>>A": 3.0, "B<<A": 3.0}
JUDGE_MAX_TOKENS = 16000


def econevals_spec(question: str, baseline: str) -> JudgeSpec:
    """Compare a candidate with the baseline answer using the judge chosen by the launcher."""
    return JudgeSpec(
        references=(baseline,),
        question=question,
        rubric=RUBRIC_PAIRWISE,
        # The launcher's judge binding selects the model; the paper used gpt-4.1-mini.
        model="",
        exact_gate=False,
        empty_output=EmptyOutputPolicy.GRADE,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        prompt_template=JUDGE_TEMPLATE,
        label_scores=VERDICT_SCORES,
        label_weights=VERDICT_WEIGHTS,
        verdict_patterns=VERDICT_PATTERNS,
        label_case="upper",
        # Reasoning judges may echo candidate verdict labels before their final one.
        strip_reasoning_blocks=True,
        strip_candidate_reasoning_blocks=True,
        max_completion_tokens=JUDGE_MAX_TOKENS,
        request_timeout=300,
        transport_attempts=3,
        transport_budget=900,
    )


def read_econevals(
    file: StoragePath, context: ConversionContext, *, baseline_model: str, per_category: int | None
) -> Iterator[dict[str, Any]]:
    """Join Arena-Hard questions with the baseline model's answers and category names, in question order.

    With ``per_category``, keep a fixed random sample of that many questions from each category.
    """
    with (context.inputs[CATEGORIES_INPUT] / CATEGORIES_FILENAME).open("r") as stream:
        table = json.load(stream)
    categories = {int(index): table["subcategory_name"][key] for key, index in table["subcategory_idx"].items()}
    baselines = {}
    with (context.inputs[BASELINE_INPUT] / BASELINE_FILENAME).open("r") as stream:
        for line in stream:
            if line.strip():
                answer = json.loads(line)
                if answer["model"] != baseline_model or answer["uid"] in baselines:
                    raise ValueError(f"Unexpected baseline answer {answer['ans_id']}")
                baselines[answer["uid"]] = answer["messages"]
    rows = []
    with file.open("r") as stream:
        for line in stream:
            if not line.strip():
                continue
            question = json.loads(line)
            messages = baselines.pop(question["uid"])
            if messages[:-1] != question["convo"] or messages[-1]["role"] != "assistant":
                raise ValueError(f"Baseline answer {question['uid']} does not continue its question")
            rows.append(
                {
                    "uid": question["uid"],
                    "category": categories[int(question["subcategory"])],
                    "conversation": question["convo"],
                    "baseline": messages[-1]["content"],
                }
            )
    if baselines:
        raise ValueError(f"{len(baselines)} baseline answers have no question")
    if per_category is None:
        yield from rows
        return
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["uid"])
    rng = random.Random(0)
    kept = {uid for category in sorted(by_category) for uid in rng.sample(by_category[category], per_category)}
    yield from (row for row in rows if row["uid"] in kept)


def convert_econevals(row: RawRow, _context: ConversionContext, *, benchmark: str) -> TaskSpec | ImportRejection:
    conversation = row.data["conversation"]
    if len(conversation) != 1 or conversation[0]["role"] != "user" or not conversation[0]["content"].strip():
        return source_defect("not_single_user_turn", "Synthetic EconEvals queries are one nonempty user turn")
    prompt = conversation[0]["content"]
    task = answer_task(
        row,
        prompt=prompt,
        spec=econevals_spec(prompt, row.data["baseline"]),
        evidence={"uid": row.data["uid"], "category": row.data["category"]},
    )
    return task.model_copy(
        update={
            "tags": ("eval", "source:econevals", "benchmark:" + benchmark),
            "labels": {"category": row.data["category"]},
        }
    )


def econevals_item(task: TaskSpec) -> EvaluationItem:
    """Judge final answers only, grouped by the DWA or occupation EconEvals reports."""
    item = chat_item(task, response_selection="final_or_complete_reasoning")
    return item.model_copy(update={"groups": {"category": source_evidence(task)["category"]}})
