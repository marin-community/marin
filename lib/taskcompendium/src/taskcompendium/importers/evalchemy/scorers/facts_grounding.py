# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Grounding v2 grading: baseline answer, eligibility rating, then sentence-level grounding.

Prompts and parsers are copied from the FACTS Grounding V2 starter notebook
(kaggle.com/code/prathameshbang/facts-grounding-v2-benchmark-starter, version 4; Apache 2.0, Copyright 2024
DeepMind Technologies Limited). The notebook asks Gemini 2.5 Flash and then GPT-5; this scorer asks one
judge for every step, so a response is eligible when that judge reports no major instruction-following
issue, and its score is that judge's grounding verdict.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from verifyit.grade import Reward, scored
from verifyit.modes.grade_judge import JudgeReply

ELIGIBILITY_SYSTEM_PROMPT = "You are a helpful assistant. Your task is to evaluate the quality of a Response."

ELIGIBILITY_PROMPT = """Your mission is to judge the response from an AI model, the *test* response, calibrating your judgement using a *baseline* response.
Please use the following rubric criteria to judge the responses:

<START OF RUBRICS>
Your task is to analyze the test response based on the criterion of "Instruction Following". Start your analysis with "Analysis".

**Instruction Following**
Please first list the instructions in the user query.
In general, an instruction is VERY important if it is specifically asked for in the prompt and deviates from the norm. Please highlight such specific keywords.
You should also derive the task type from the user query and include the task-specific implied instructions.
Sometimes, no instruction is available in the user query.
It is your job to infer if the instruction is to autocomplete the user query or is asking the LLM for follow-ups.
After listing the instructions, you should rank them in order of importance.
After that, INDEPENDENTLY check if the test response and the baseline response meet each of the instructions.
You should itemize, for each instruction, whether the response meets, partially meets, or does not meet the requirement, using reasoning.
You should start reasoning first before reaching a conclusion about whether the response satisfies the requirement.
Citing examples while reasoning is preferred.

Important: Only select `Major Issue(s)` for issues that are extremely egregious and critical to the response, such that they give little to no value to the user.

Note: Both the `baseline` and `test` responses were made based on a given context document **which was redacted** in your analysis. You do not have access to the context document which is likely referenced in the user request and model responses (both baseline and test) - assume that it contains the information referenced adequately.

Reflect on your answer and consider the possibility that you are wrong.
If you are wrong, explain clearly what needs to be clarified, improved, or changed in the rubric criteria and guidelines.

In the end, express your final verdict as one of the following three json objects:

```json
{{
    "Instruction Following": "No Issues"
}}
```

```json
{{
    "Instruction Following": "Minor Issue(s)"
}}
```

```json
{{
    "Instruction Following": "Major Issue(s)"
}}
```

<END OF RUBRICS>

# Your task
## User query
<|begin_of_query|>
{user_request}
<|end_of_query|>

**NOTE: CONTEXT DOCUMENT REDACTED FROM USER QUERY. ASSUME THAT IT CONTAINS THE NECESSARY INFORMATION WHICH IS REFERENCED IN THE RESPONSES BELOW.**

## Test Response:
<|begin_of_test_response|>
{response_a}
<|end_of_test_response|>

## Baseline Response:
<|begin_of_baseline_response|>
{response_b}
<|end_of_baseline_response|>

Please write your analysis and final verdict for the test response."""  # noqa: E501 - exact source prompt

GROUNDING_SYSTEM_PROMPT = "You are a helpful assistant. You will be provided with a textual context and a model-generated response. Your task is to analyze the response sentence by sentence and classify each sentence according to its relationship with the provided context."  # noqa: E501 - exact source prompt

GROUNDING_PROMPT = """## Task: Evaluate Factual Accuracy of Model Response Against Provided Context

Your goal is to evaluate whether each sentence in the model's response is factually accurate **solely based on the provided context document**. You must act as a strict fact-checker. Do not use any external knowledge or make assumptions beyond what is stated in the text.

**Output Format:**
For each sentence in the response, output a JSON object with the following fields, representing an evaluation of a single sentence from the model's response:

* `sentence`: (string) The exact sentence from the model's response being evaluated.
* `label`: (string) Your verdict for the sentence. Must be one of:
    * `supported`: The sentence is fully and accurately supported by the provided context document.
    * `not_supported`: The sentence is NOT accurately supported by the provided context document. This includes sentences that are partially correct but contain inaccuracies, unsupported inferences, misattributions, or ignore crucial context.
    * `no_rad`: The sentence is introductory, concluding, conversational filler, or otherwise does not make a factual claim that can be verified against the context (e.g., "Here is the summary:", "Key aspects include:").
* `rationale`: (string) A detailed justification for your `label`.
    * If `supported`, explain *how* the provided excerpt(s) directly and accurately support the *entire* claim made in the sentence. Point out specific phrases or data points.
    * If `not_supported`, clearly explain *why* the sentence is inaccurate. Specify the inaccuracy (e.g., contradicts the text, makes an unsupported inference, misrepresents nuance, misattributes information, ignores contradictory context, uses incorrect terminology). Reference the relevant part(s) of the text or the lack thereof.
    * If `no_rad`, briefly explain why the sentence doesn't require factual verification against the context.
* `excerpt`: (string or null) One or more direct quotes (excerpts) from the context document that are **most relevant** to supporting or refuting the sentence. Concatenate multiple relevant excerpts with a semicolon and space (`; `). If the label is `not_supported` due to a complete lack of information, or if the label is `no_rad`, this field should be `null`.

**Evaluation Criteria - Strict Adherence Required:**

1.  **Explicit Support or Trivial/Strongly-Implied Inferences Only:** A sentence is only `supported` if the information is **explicitly stated** in the context or are logically inferred statements that are strongly implied by the context or trivially implied. Do NOT mark sentences as `supported` if they require inference, assumption, synthesis beyond direct statements, or understanding of implied meaning beyond a very strong or trivial implication. A non-trivial inference that goes well beyond what is strongly or trivially implied by the context is NOT supported.
2.  **Precision and Nuance:** Pay close attention to specific terminology, qualifiers (e.g., "significantly", "most", "some"), quantities, and causal relationships mentioned in the text. Simplifications in the model response are only acceptable if they maintain **full factual accuracy** and do not misrepresent the nuance of the source.
3.  **Completeness and Context:** Evaluate the sentence in the context of the *entire* provided document. A sentence is `not_supported` if it cherry-picks information while ignoring contradictory or significantly qualifying information present elsewhere *in the same document*. The response should not be misleading even if technically based on a fragment of the text.
4.  **Accurate Attribution:** Ensure that claims, descriptions, or characteristics are attributed to the correct subject, object, or scope as defined in the text. Misattributions make a sentence `not_supported`.
5.  **No Hallucinations:** The model response must not introduce information, terms, or concepts not present in the context. Such additions make a sentence `not_supported`.
6.  **Sentence-Level Evaluation:** Evaluate each sentence independently based on these criteria.

Output each JSON object on a new line.

**Example Structure:**

```json
{{"sentence": "The first sentence of the model response.", "label": "supported/not_supported/no_rad", "rationale": "Detailed explanation referencing the criteria and text.", "excerpt": "Relevant quote(s) from context; null if none."}}
{{"sentence": "The second sentence of the model response.", "label": "supported/not_supported/no_rad", "rationale": "Detailed explanation referencing the criteria and text.", "excerpt": "Relevant quote(s) from context; null if none."}}
// ... more sentence evaluations
```

**DO NOT** use RAW quotation marks (") in the values of the sentence, rationale, or excerpt fields in the response. Replace any quotations (") with tick (`)
Example: "The response states, `This is the way it goes`". DO NOT attempt any escaping of the quotations marks - just replace them with ticks.

**Now, please analyze the following context and response:**

**User Query:**
{user_query}

**Context:**
{context}

**Response:**
{response}"""  # noqa: E501 - exact source prompt

MAJOR_ISSUES = "Major Issue(s)"

Ask = Callable[[list[ChatCompletionMessageParam]], JudgeReply]


def extract_instruction_json(ans: str) -> dict:
    pattern = re.compile(
        r"{\s*" r'"Instruction Following"\s*:\s*' r'"(No Issues|Minor Issue\(s\)|Major Issue\(s\))"\s*' r"}"
    )
    match = pattern.search(ans)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {"Instruction Following": "Invalid"}
    return {"Instruction Following": "Invalid"}


def clean_json_content_8(text: str, start_field: str, end_field: str | None = None) -> str:
    """Strip quotes and escapes from one field's value in a JSON-like sentence verdict."""
    if end_field:
        trailer_pattern = rf'(,\s*"{re.escape(end_field)}":)'
    else:
        trailer_pattern = r"(\s*})"

    pattern = re.compile(rf'"{re.escape(start_field)}":(.+?){trailer_pattern}', re.DOTALL)

    def _clean_match(match: re.Match[str]) -> str:
        content = match.group(1)
        trailer = match.group(2)
        content = content.replace('"', "")
        content = content.replace("\\", "")
        content = content.replace("\x02", "")
        content = content.strip()
        cleaned_content = f'"{content}"' if content != "null" else "null"
        return f'"{start_field}":{cleaned_content}{trailer}'

    cleaned_text, num_subs = pattern.subn(_clean_match, text, count=1)
    return cleaned_text if num_subs > 0 else text


def parse_ufg_rev21_verdict(answer: str) -> tuple[bool, list[dict[str, Any]]]:
    """Return the notebook's grounding decision and parsed sentence verdicts."""
    # The notebook splits verdict lines on "}\n" only; a judge that writes CRLF would merge them all.
    answer = answer.replace("\r\n", "\n")
    if "```json" in answer:
        jsonl_chunks = []
        for a in answer.split("```json")[1:]:
            jsonl_chunks.append(a.split("```")[0])
        answer = "\n".join(jsonl_chunks)
    answer = answer.strip()
    answer = answer.replace("}\n", "}\n@\n@\n")
    answer = answer.replace("} \n", "}\n@\n@\n")
    answer = answer.replace("}  \n", "}\n@\n@\n")
    answer = answer.replace("<ctrl75>", "")
    parsed_answers = []
    for line in answer.split("\n@\n@\n"):
        raw_line = line
        try:
            line = line.replace("\n", " ")
            line = line.replace("\\'", "'")
            line = line.replace("<ctrl75>", "")
            line = line.lstrip(",")
            line = line.replace("\n", " ")
            line = clean_json_content_8(line, "sentence", "label")
            line = clean_json_content_8(line, "label", "rationale")
            line = clean_json_content_8(line, "rationale", "excerpt")
            line = clean_json_content_8(line, "excerpt")
            parsed = json.loads(line)
            if "label" not in parsed:
                parsed["label"] = "unknown"
            parsed_answers.append(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            if '"not_supported"' in raw_line:
                parsed_answers.append({"sentence": "", "rationale": "", "excerpt": "", "label": "not_supported"})
    if parsed_answers:
        grounded = all(parsed["label"] in ("supported", "no_rad", "unknown") for parsed in parsed_answers)
    else:
        grounded = False
    return grounded, parsed_answers


def grade_facts_grounding(full_prompt: str, user_request: str, context_document: str, response: str, ask: Ask) -> Reward:
    """Score with the notebook's eligibility and sentence-label rules."""
    baseline = ask([{"role": "user", "content": full_prompt}])
    eligibility_prompt = ELIGIBILITY_PROMPT.format(
        user_request=user_request, response_a=response, response_b=baseline.text
    )
    eligibility = ask(
        [
            {"role": "system", "content": ELIGIBILITY_SYSTEM_PROMPT},
            {"role": "user", "content": eligibility_prompt},
        ]
    )
    rating = extract_instruction_json(eligibility.text)["Instruction Following"]
    # An unparseable rating counts as eligible, as in the reference notebook.
    eligible = MAJOR_ISSUES not in rating
    detail: dict[str, Any] = {
        "baseline": baseline.text,
        "baseline_judge": baseline.detail,
        "eligibility_rating": rating,
        "eligibility_text": eligibility.text,
        "eligibility_judge": eligibility.detail,
    }
    grounded = False
    if eligible:
        grounding_prompt = GROUNDING_PROMPT.format(user_query=user_request, context=context_document, response=response)
        grounding = ask(
            [
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": grounding_prompt},
            ]
        )
        grounded, sentences = parse_ufg_rev21_verdict(grounding.text)
        detail.update(
            grounding_text=grounding.text,
            grounding_judge=grounding.detail,
            parsed_sentences=len(sentences),
            unsupported_sentences=[sentence for sentence in sentences if sentence["label"] == "not_supported"],
        )
    factuality = float(eligible and grounded)
    return scored(factuality, **detail, metrics={"factuality": factuality, "eligible": float(eligible)})
