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
    "Put your final answer within \\boxed{}."
)
"""
Long-form reasoning instruction that encourages:
- Informal, scratchpad-style thinking
- Self-correction and mistake checking
- Sanity checks before finalizing
- Final answer in \\boxed{} format
"""

REASONING_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
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
- The final answer in \\boxed{} format

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
