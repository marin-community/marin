# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Google DeepMind's IMO-AnswerBench and IMO-ProofBench, graded by the paper's autograder prompts.

Data and prompts are from "Towards Robust Mathematical Reasoning" (Luong et al., EMNLP 2025,
arXiv:2511.01846), CC BY 4.0. The prompts are the appendix A.5 AnswerAutoGrader and B.5
ProofAutoGrader texts, transcribed from the paper's LaTeX source as Markdown; the answer grader
adds one rule that marks hedged answers incorrect. Each solver sees only the problem statement.
"""

from typing import Any

from verifyit.spec import JudgeCompletionPolicy, JudgeSamplingParams, JudgeSpec

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

PROOF_GRADER_TEMPLATE = """You are an expert grader for the International Mathematics Olympiad (IMO). Your task is to evaluate a proposed solution strictly and rigorously. Keep in mind the standards at the IMO are extremely high: only arguments that are logically sound, complete, and precise should be rewarded.

### General Scoring Rubric
Scores are assigned on a 0-7 scale. The general guidelines are:
* **7 Points (Correct):** The solution is complete, correct, and fully rigorous. If the submission contains incorrect attempts or lines of reasoning but ultimately presents a complete and correct solution, it should still be awarded full points; the presence of earlier, discarded work does not detract from the final correct proof.
* **6 Points (Almost Correct):** The solution is almost correct with a sound core argument, but contains minor errors in calculation or small gaps in logic. Missing proofs for major components, unjustified claims, or sketchy arguments are **not** eligible for 6 points.
* **1 Point (Partial Progress):** The solution demonstrates substantial progress explicitly mentioned in the grading guidelines. Initial observations, reformulating the problem without making substantive headway, or proving partial results not mentioned in the grading guidelines are generally **not** eligible for this score.
* **0 Points (Incorrect):** The solution doesn't make substantial progress that is a key step in the full solution or is fundamentally flawed. All partial progress without key results or lacking rigor also fall in this category.

### Input Data and Interpretation
You are provided with the following:
1. **Problem Statement:** The IMO problem.
2. **Ground Truth Solution:** A reference solution. Assume this solution is correct. It demonstrates one valid approach.
3. **Specific Grading Guidelines:** Criteria for awarding credit for this specific problem. These guidelines take precedence over the General Scoring Rubric, especially for partial credit.
4. **Proposed Solution:** The student submission.

### Evaluation Process
You must follow this structured process:
1. **Analyze References:** Meticulously read and understand the problem and Ground Truth Solution check the Specific Grading Guidelines. Identify the key steps for a complete solution and the criteria for partial credit.
2. **Step-by-Step Verification:** Verify the logical validity and rigor of every step. Identify all flaws, gaps, assumptions, and errors. **Make sure you fully understand every piece of logic behind each step of the proposed solution, you must be careful for solutions that 'pretend' to be correct.**
3. **Assess Progress:** Determine the extent of non-trivial progress made.
4. **Score Determination:** Compare the findings against the Specific Grading Guidelines and the General Rubric to determine the final score.

### Output Requirements
You must provide your final score in the format <points>N out of 7</points>. Ensure the `<points>` block is used **only once**, as your answer will be parsed based on the first <points> </points> block that appears in your whole response.

---
**PROBLEM STATEMENT**
{question}

**GROUND-TRUTH SOLUTION**
{reference}

**SPECIFIC GRADING GUIDELINES**
{guidelines}

**PROPOSED SOLUTION**
{candidate}
---

Present your detailed thought process and formal justification based on the scoring rubric and grading guidelines, and finally present your final score in the format below.

[Select one of the following options]
<points>7 out of 7</points>
<points>6 out of 7</points>
<points>1 out of 7</points>
<points>0 out of 7</points>
"""  # noqa: E501 - exact source prompt

PROOF_POINTS = (7, 6, 1, 0)
ANSWERBENCH_CATEGORIES = frozenset({"Algebra", "Combinatorics", "Geometry", "Number theory"})


def answer_judge_spec(problem: str, answer: str) -> JudgeSpec:
    """Score the boxed verdict, ignoring the judge's thinking block."""
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


def proof_judge_spec(problem: str, solution: str, guidelines: str) -> JudgeSpec:
    """Score a <points> line as 7, 6, 1 or 0 points out of 7."""
    # Guidelines are literal LaTeX inside a format template; braces must survive formatting.
    template = PROOF_GRADER_TEMPLATE.replace("{guidelines}", guidelines.replace("{", "{{").replace("}", "}}"))
    return JudgeSpec(
        question=problem,
        references=(solution,),
        # Use the judge deployment selected by the evaluation launcher.
        model="",
        rubric="labels",
        prompt_template=template,
        exact_gate=False,
        label_scores={f"<points>{points} out of 7</points>": points / 7 for points in PROOF_POINTS},
        label_scan="lines",
        # GLM-5.3-Flash reasons for up to ~85K tokens on IMO-ProofBench proofs, some past 450 s, and
        # can reason without end at temperature 0. Together returns 503s under load.
        sampling=JudgeSamplingParams(),
        max_completion_tokens=131072,
        completion_policy=JudgeCompletionPolicy.CONTENT,
        request_timeout=900,
        transport_attempts=6,
        transport_budget=1200,
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


def convert_proofbench(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    problem, solution, guidelines = (
        nonempty_text(row.data.get(column)) for column in ("Problem", "Solution", "Grading guidelines")
    )
    if problem is None or solution is None or guidelines is None:
        return source_defect("missing_field", "IMO-ProofBench rows require Problem, Solution and Grading guidelines")
    task = answer_task(
        row, prompt=problem, spec=proof_judge_spec(problem, solution, guidelines), evidence=dict(row.data)
    )
    return task.model_copy(update={"tags": ("eval", "source:imobench", "benchmark:IMO-ProofBench")})


def answerbench_item(task: TaskSpec) -> EvaluationItem:
    """Judge the final answer and report per-category accuracy."""
    evidence = source_evidence(task)
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"problem_id": evidence["Problem ID"], "category": evidence["Category"]}}
    )


def proofbench_item(task: TaskSpec) -> EvaluationItem:
    """Judge the final proof; group by subset (Basic, Advanced), level, category and source."""
    evidence = source_evidence(task)
    return chat_item(task, response_selection="final_only").model_copy(
        update={
            "groups": {
                "problem_id": evidence["Problem ID"],
                "subset": evidence["Problem ID"].split("-")[1],
                "level": evidence["Level"],
                "category": evidence["Category"],
                "source": evidence["Source"],
            }
        }
    )
