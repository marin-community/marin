# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prompts for the Self-Instill synthetic data generation pipeline.

This module contains all prompt templates used for:
- Generation: Instructions for generating reasoning responses
- Summarization: Condensing long reasoning into clean explanations
- Validation: LLM-as-judge prompts for quality verification

All prompts are designed to work with base language models and produce
structured outputs that can be parsed programmatically.
"""

# =============================================================================
# GENERATION PROMPTS
# =============================================================================

REASONING_LONG_INSTRUCTION = (
    "Solve the problem by thinking out loud in plain text.\n"
    "Write it like a working scratchpad: short paragraphs, informal reasoning, and intermediate steps.\n"
    "Avoid structured writeups: do NOT use Markdown headings (###), numbered lists, or section titles like 'Plan'/'Step-by-step'.\n"
    "As you go, question your own steps and check for mistakes; if you notice an issue or a better idea, correct course and continue.\n"
    "Before finishing, do a quick sanity check (edge cases / constraints / re-check a key step).\n"
    "Put your final answer within \\boxed{{}}."
)
"""
Long-form reasoning instruction that encourages:
- Informal, scratchpad-style thinking
- Self-correction and mistake checking
- Sanity checks before finalizing
- Final answer in \\boxed{{}} format
"""

REASONING_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{{}}."
"""
Short reasoning instruction for the final formatted output.
Used when constructing the conversation format after summarization.
"""


# =============================================================================
# SUMMARIZATION PROMPTS
# =============================================================================

SUMMARIZATION_PROMPT_TEMPLATE = """Summarize the solution as a clear explanation (like the final write-up), not a one-liner.
Include the key reasoning steps that justify the result (setup -> method -> crucial computations -> conclusion).
Paraphrase, don't copy sentences from the original.
End with the final answer within \\boxed{{...}}

Original solution:
{text}

Explanation:
"""
"""
Template for summarizing long reasoning into a clean explanation.

The summarizer takes verbose reasoning output and produces:
- A clear, concise explanation of the solution approach
- Key reasoning steps that justify the result
- The final answer in \\boxed{{}} format

Args:
    text: The original long-form reasoning to summarize
"""


# =============================================================================
# VALIDATION PROMPTS - Cycle Consistency
# =============================================================================

CYCLE_QUESTION_GENERATION_PROMPT = """Write ONE short question that this answer would correctly respond to.

Answer:
{answer}

Output the question only.
Question:"""
"""
Prompt for generating an inferred question from an answer.
Used in cycle consistency validation to verify the answer addresses the original question.
"""

CYCLE_COMPARISON_PROMPT = """You are a grader. Decide if the inferred question matches the original question (same core ask).

Original question:
{original_question}

Inferred question (from the answer):
{inferred_question}

Rules:
- [[Y]] only if both questions ask for the same core thing.
- [[N]] if they differ in topic, goal, or key constraints.
- If unsure, output [[N]].

Output EXACTLY one token: [[Y]] or [[N]].
Decision:"""
"""
Prompt for comparing the inferred question with the original.
Returns [[Y]] if they match, [[N]] otherwise.
"""


# =============================================================================
# VALIDATION PROMPTS - Factual Error Check
# =============================================================================

FACTUAL_ERROR_PROMPT = """You are a strict checker. Decide if the answer contains ANY clear error.

Question:
{question}

Answer:
{answer}

What counts as an error:
- wrong math/arithmetic
- contradicts the question constraints
- incorrect factual claim (clearly false)
- invalid logic that changes the conclusion

Rules:
- [[Y]] only if you see NO clear errors.
- [[N]] if you see at least one clear error.
- If unsure, output [[N]].

Output EXACTLY one token: [[Y]] or [[N]].
Decision:"""
"""
Prompt for checking factual/logical errors in an answer.
Returns [[Y]] if no errors found, [[N]] if any error is detected.
"""


# =============================================================================
# VALIDATION PROMPTS - Total Correctness
# =============================================================================

TOTAL_CORRECTNESS_PROMPT = """You are a grader. Decide if the answer is a COMPLETE and CORRECT solution.

Question:
{question}

Answer:
{answer}

Accept [[Y]] ONLY if ALL are true:
1) Final result is correct.
2) Reasoning does not contain a clear mistake that would mislead.
3) It addresses all key parts of the question (not partial).

If any condition fails, output [[N]].
If unsure, output [[N]].

Output EXACTLY one token: [[Y]] or [[N]].
Decision:"""
"""
Prompt for verifying complete correctness of an answer.
This is the strictest validation - requires correct result, sound reasoning,
and complete coverage of the question.
"""


# =============================================================================
# VALIDATION PROMPTS - Relevance Check
# =============================================================================

RELEVANCE_PROMPT = """You are a grader. Decide if the ANSWER addresses what the QUESTION asks.

QUESTION:
{question}

ANSWER:
{answer}

Rules:
- [[Y]] only if the answer directly addresses the main request.
- [[N]] if it is off-topic, dodges the question, or misses key requirements.
- If unsure, output [[N]].

Output EXACTLY one token: [[Y]] or [[N]].
Decision:"""
"""
Prompt for checking if an answer is relevant to the question.
Used as a basic relevance filter before more detailed validation.
"""


# =============================================================================
# VALIDATION PROMPTS FOR INSTRUCTION-TUNED MODELS (THINKING MODELS)
# =============================================================================
# These prompts are longer and allow the model to think before giving a verdict.
# The model will output reasoning and then [[Y]] or [[N]] at the end.
# =============================================================================

CYCLE_QUESTION_GENERATION_PROMPT_INSTRUCT = """Given an answer, please generate the most likely question that would have prompted this answer. Focus on inferring the core question that this answer is addressing. Output only the inferred question, without any additional explanation.

Answer:
{answer}

Inferred Question:"""
"""
Instruction-tuned version of cycle question generation.
Used for models that can reason before responding.
"""

CYCLE_COMPARISON_PROMPT_INSTRUCT = """You are evaluating whether an answer is relevant to the original question and touches the core of the question by comparing the original question with an inferred question derived only from the answer.

Original Question: {original_question}
Inferred Question: {inferred_question}

Compare the two questions and determine:
1. If the original question and inferred question are asking about the same core topic
2. If the original question and inferred question share the same key elements and requirements
3. If answering one question would effectively address the other question

After your analysis, provide your decision: [[Y]] if the questions are semantically equivalent and address the same core problem, or [[N]] if they are asking about different things."""
"""
Instruction-tuned version of cycle comparison.
Allows the model to reason before giving verdict.
"""

FACTUAL_ERROR_PROMPT_INSTRUCT = """Please act as an impartial judge and carefully analyze the following answer for any factual errors, logical flaws, or misleading information.

Question: {question}
Answer: {answer}

Consider the credibility of the claims made in the answer and determine if they align with established knowledge. Evaluate:
1. Are there any incorrect facts, dates, numbers, formulas, or claims?
2. Is there any faulty logic, reasoning, or problem-solving approach?
3. Are there any misleading, incomplete, or ambiguous explanations?
4. Does the answer introduce any misconceptions or propagate common errors?

Minor typos or grammatical errors are acceptable. But be strict about any factual error, calculation error, or logical flaw. When unsure, lean toward accepting statements unless they contain clear errors.

After a thorough analysis, provide your decision: [[Y]] if the answer has no factual errors or major flaws, or [[N]] if it contains important factual errors or logical flaws that would mislead the user."""
"""
Instruction-tuned version of factual error check.
Allows the model to reason before giving verdict.
"""

TOTAL_CORRECTNESS_PROMPT_INSTRUCT = """Please act as an impartial judge and evaluate whether the response is completely correct in both process and conclusion.

Question: {question}
Answer: {answer}

Consider correctness, usefulness, completeness and depth in your assessment. Consider whether this answer completely solves the question.

You should rely on your own reasoning to form a reference solution and compare the answer to your reasoning.

Begin your evaluation by giving a brief summary of your thoughts on the response. Focus on whether it is accurate, addresses the question well, and is reasonably detailed. Be precise about any errors or gaps you notice.

Notes:
1. If the answer is partial, high-level, or just states that this is an open problem, you should not accept it.
2. If the answer lacks details or is not comprehensive, you should not accept it.
3. If the answer contains any errors, you should not accept it.
4. You should only accept the answer if it is at least 95% correct and solves the question.

After providing your explanation, decide whether this answer is correct. Think twice about whether this answer solves the question.
Format: Accepted: [[Y]] if you accept the answer or Accepted: [[N]] if you do not accept."""
"""
Instruction-tuned version of total correctness check.
Allows the model to reason before giving verdict.
"""


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

FINAL_OUTPUT_TEMPLATE = """<think>
{reasoning}
</think>

{summary}"""
"""
Template for formatting the final output combining reasoning and summary.

Args:
    reasoning: The long-form thinking/reasoning process
    summary: The condensed explanation with final answer
"""


def format_generation_prompt(original_prompt: str, use_long_instruction: bool = True) -> str:
    """
    Format a prompt with reasoning instructions for generation.

    Args:
        original_prompt: The original question/problem to solve
        use_long_instruction: If True, use detailed reasoning instructions.
                             If False, use short instruction.

    Returns:
        The formatted prompt with reasoning instructions appended.
    """
    instruction = REASONING_LONG_INSTRUCTION if use_long_instruction else REASONING_INSTRUCTION
    return f"{original_prompt}\n{instruction}\n"


def format_summarization_prompt(text: str) -> str:
    """
    Format a summarization prompt for condensing reasoning.

    Args:
        text: The long-form reasoning text to summarize

    Returns:
        The formatted summarization prompt
    """
    return SUMMARIZATION_PROMPT_TEMPLATE.format(text=text)


def format_final_output(reasoning: str, summary: str) -> str:
    """
    Format the final output combining reasoning and summary.

    Args:
        reasoning: The long-form thinking/reasoning process
        summary: The condensed explanation with final answer

    Returns:
        The formatted final output with <think> tags
    """
    return FINAL_OUTPUT_TEMPLATE.format(reasoning=reasoning, summary=summary)
