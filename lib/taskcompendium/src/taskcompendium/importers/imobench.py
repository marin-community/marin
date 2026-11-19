# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Google DeepMind's IMO-AnswerBench, graded by the paper's AnswerAutoGrader prompt.

Data and prompt are from "Towards Robust Mathematical Reasoning" (Luong et al., EMNLP 2025,
arXiv:2511.01846), CC BY 4.0. The prompt is the appendix A.5 text, transcribed from the paper's
LaTeX source as Markdown, plus one rule that marks hedged answers incorrect. The solver sees only
the problem statement.
"""

from typing import Any

from verifyit.spec import JudgeCompletionPolicy, JudgeSpec

from taskcompendium.convert.answers import answer_task, nonempty_text, source_defect, source_evidence
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow

ANSWER_GRADER_TEMPLATE = """**# System Role: Deterministic Mathematical Autograder**

You are a precise, automated grading system. Your sole function is to determine if the final answer provided in the `Model Solution` is mathematically equivalent to the `Golden Answer`. You must NOT grade the reasoning or steps, only the final result.

**# 1. Grading Guidelines (Equivalence Rules)**

Equivalence is mandatory for a correct grade. You must rigorously verify if the answers represent the exact same mathematical value or expression, even if the format differs.
* **Algebraic Equivalence:** e.g., `n(n+1)/2` is equivalent to `n^2/2 + n/2`. You must verify the algebra.
* **Numerical Equivalence:** e.g., `1/2` is equivalent to `0.5`; `sqrt(2)/2` is equivalent to `1/sqrt(2)`.
* **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order of elements does not matter (e.g., {{1, 2}} is equivalent to {{2, 1}}).
* **Partial Credit:** No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.
* **No Answers:** If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.
* **Hedged Answers:** If the solution offers more than one candidate final answer without clearly committing to one of them, it is incorrect.

**# 3. Output Protocol (Strict Compliance Required)**

You must execute the task using a two-part structure. Failure to follow this structure will result in task failure.

**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags. Make your thinking concise. This section details your reasoning process and must follow these steps sequentially:
1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer based on the Extraction Protocol. If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. Detail the steps taken to verify mathematical equivalence (e.g., simplification, algebraic manipulation). You must actively try to prove they are the same before concluding they are different.
4. **Conclusion:** State the final determination ("Correct" or "Incorrect").

**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output **ONLY** the final grade.
* If Correct: \\boxed{{Correct}}
* If Incorrect: \\boxed{{Incorrect}}

**CRITICAL CONSTRAINT: Do not add any text, explanations, or formatting outside the <thinking> tags or the final \\boxed{{}} output.**

---
Output exmaple:

<thinking>
1. **Golden Answer:** $(-\\infty, -4) \\cup (-4, \\infty)$
2. **Extracted Model Answer:** $\\emptyset$ (the empty set)
3. **Equivalence Analysis:**
   The Golden Answer is a non-empty set of real numbers.
   The Model Answer is the empty set.
   These two sets are not equivalent. The empty set contains no elements, while the Golden Answer contains an infinite number of elements.
4. **Conclusion:** Incorrect
</thinking>
\\boxed{{Incorrect}}

---
**# 4. Input Data**
Here is the problem, model solution, and golden answer to grade:
Problem: {question}
Model Solution: {candidate}
Golden Answer: {reference}
"""  # noqa: E501 - source prompt plus the Hedged Answers rule

ANSWERBENCH_CATEGORIES = frozenset({"Algebra", "Combinatorics", "Geometry", "Number theory"})


def answer_judge_spec(problem: str, answer: str) -> JudgeSpec:
    """AnswerAutoGrader: the verdict is the box after the judge's <thinking> block."""
    return JudgeSpec(
        question=problem,
        references=(answer,),
        # Use the judge deployment selected by the evaluation launcher.
        model="",
        rubric="labels",
        prompt_template=ANSWER_GRADER_TEMPLATE,
        exact_gate=False,
        label_scores={r"\boxed{Correct}": 1.0, r"\boxed{Incorrect}": 0.0},
        label_scan="whole",
        label_case="upper",
        strip_reasoning_blocks=True,
        max_completion_tokens=8192,
        completion_policy=JudgeCompletionPolicy.CONTENT,
        request_timeout=300,
        transport_attempts=3,
        transport_budget=900,
    )


def answerbench_row_selected(data: dict[str, Any], _context: ConversionContext) -> bool:
    """Drop rows outside the four categories: answerbench_v2's algebra-036 has shifted columns."""
    return data.get("Category") in ANSWERBENCH_CATEGORIES


def convert_answerbench(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    problem, answer = nonempty_text(row.data.get("Problem")), nonempty_text(row.data.get("Short Answer"))
    if problem is None or answer is None:
        return source_defect("missing_field", "IMO-AnswerBench rows require Problem and Short Answer")
    task = answer_task(row, prompt=problem, spec=answer_judge_spec(problem, answer), evidence=dict(row.data))
    return task.model_copy(update={"tags": ("eval", "source:imobench", "benchmark:IMO-AnswerBench")})


def answerbench_item(task: TaskSpec) -> EvaluationItem:
    """Judge the final answer and report per-category accuracy."""
    evidence = source_evidence(task)
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"problem_id": evidence["Problem ID"], "category": evidence["Category"]}}
    )
