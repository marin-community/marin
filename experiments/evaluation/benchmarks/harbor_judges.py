# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grading rules for Harbor datasets whose package tests call an LLM judge.

A judged package's ``tests/test.sh`` runs in the agent's container with the host's judge
credentials, which a TaskSpec never gives the agent machine. Each rule here captures the package's
answer file and grades it on the host with a verifyit judge that sends the request the package's
script sends, for the judge the dataset's Harbor policy named. The protocol declaring a rule must
require a judge; the run's judge identity must map the rule's model to its revision.
"""

import json
from pathlib import Path

from taskcompendium.grader import verifyit_package
from taskcompendium.importers.harbor.package import AnswerFile
from verifyit.spec import EmptyOutputPolicy, ExtractGroup, JudgeSamplingParams, JudgeSpec, StripOuterWhitespace

JUDGE_REQUEST_TIMEOUT = 600
"""Each package's verifier timeout, which bounds its script's judge call and SDK retries."""
JUDGE_TRANSPORT_ATTEMPTS = 3
"""The OpenAI SDK's default of one request and two retries of transient failures."""

SIMPLEQA_ANSWER = "/workspace/answer.txt"
SIMPLEQA_JUDGE_MODEL = "openai/gpt-oss-120b"
"""The judge that Harbor's ``simpleqa-recovery`` policy serves from Together."""
# From tests/llm_judge.py of Harbor's simpleqa@1.1 packages, after its ``.strip()``, typos included.
SIMPLEQA_GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 5: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 6: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 7: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it."""  # noqa: E501 - exact source prompt

FINANCEAGENT_ANSWER = "/app/answer.txt"
FINANCEAGENT_JUDGE_MODEL = "gpt-5-2025-08-07"
"""Every package's ``[verifier.env] MODEL_NAME``, without LiteLLM's ``openai/`` provider prefix."""
FINANCEAGENT_MAX_OUTPUT_TOKENS = 128_000
"""GPT-5's output limit; the package's script sets none."""
# From tests/run_test.py of DCAgent/financeagent_terminal@86f788bc, a str.format template.
FINANCEAGENT_JUDGE_PROMPT = """You are a fair judge evaluating financial question answers.

## Question
{question}

## Reference Answer (Ground Truth)
{reference_answer}

## Model's Answer
{model_answer}

## Your Task
Determine if the model's answer is correct. Consider:
1. The core factual content must match (numbers, dates, names)
2. Minor phrasing differences are acceptable
3. Additional context is fine as long as the key answer is correct
4. For numerical answers, small rounding differences are acceptable

Respond with ONLY valid JSON in this exact format:
{{
    "judgment": "CORRECT" or "INCORRECT",
    "explanation": "Brief explanation of your judgment"
}}"""

BIXBENCH_ANSWER = "/workspace/answer.txt"
BIXBENCH_JUDGE_MODEL = "MiniMaxAI/MiniMax-M3-MXFP8"
"""The judge the campaign co-hosted for ``bixbench-pi``, which Harbor's runner gave the package as ``MODEL_NAME``."""
# From tests/llm_judge.py of penfever/harbor-datasets@454d04b5's bixbench-cli-marin-corrections-1,
# a str.format template, double space included.
BIXBENCH_EVAL_PROMPT = """
Here is a question, the correct answer to the question, and a proposed answer.
Question: {question}
Correct answer: {target}
Proposed answer: {predicted}
You must respond with a binary score for whether the proposed answer is equivalent  to the correct answer.
\nNothing else is permitted.
"""
BIXBENCH_RESPONSE_SCHEMA = {
    "properties": {"binary_score": {"title": "Binary Score", "type": "boolean"}},
    "required": ["binary_score"],
    "title": "JudgeResponse",
    "type": "object",
}
"""The package script's ``JudgeResponse.model_json_schema()``."""
BIXBENCH_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "JudgeResponse",
        "schema": {**BIXBENCH_RESPONSE_SCHEMA, "additionalProperties": False},
        "strict": True,
    },
}
"""The ``response_format`` the OpenAI SDK's ``chat.completions.parse`` sends for ``JudgeResponse``."""


def simpleqa_answer(directory: Path) -> AnswerFile:
    """SimpleQA's ``tests/llm_judge.py``: grade A (correct) scores 1; B (incorrect) and C (not attempted) score 0.

    The script sends the stripped answer through the Responses API with a 1,024-token output limit
    and low reasoning effort, and accepts a reply that is exactly one grade letter in any case.
    """
    truth = json.loads((directory / "tests" / "ground_truth.json").read_text())
    judge = JudgeSpec(
        question=truth["question"],
        references=(truth["expected_answer"],),
        model=SIMPLEQA_JUDGE_MODEL,
        rubric="labels",
        prompt_template=SIMPLEQA_GRADER_TEMPLATE.replace("{target}", "{reference}").replace(
            "{predicted_answer}", "{candidate}"
        ),
        label_scores={"A": 1.0, "B": 0.0, "C": 0.0},
        label_scan="whole",
        label_case="upper",
        candidate_steps=(StripOuterWhitespace(),),
        api="responses",
        sampling=JudgeSamplingParams(),
        parse_attempts=1,
        max_completion_tokens=1024,
        reasoning_effort="low",
        request_timeout=JUDGE_REQUEST_TIMEOUT,
        transport_attempts=JUDGE_TRANSPORT_ATTEMPTS,
        transport_budget=JUDGE_REQUEST_TIMEOUT,
    )
    return AnswerFile(SIMPLEQA_ANSWER, verifyit_package(judge).grader)


def financeagent_answer(directory: Path) -> AnswerFile:
    """FinanceAgent's ``tests/test.sh``: the reference answer scores 1 without a judge call.

    The gate compares texts without outer whitespace, where the script removes every newline, so an
    answer that differs from the reference only in its line breaks goes to the judge here. Other
    answers go to GPT-5, whose JSON ``judgment``, optionally in a Markdown code fence, scores 1 for
    CORRECT and 0 for INCORRECT in any case. The judge calls Chat Completions without sampling
    options, as the script does.
    """
    config = json.loads((directory / "tests" / "config.json").read_text())
    judge = JudgeSpec(
        question=config["question"],
        references=(config["expected_answer"],),
        model=FINANCEAGENT_JUDGE_MODEL,
        rubric="labels",
        prompt_template=FINANCEAGENT_JUDGE_PROMPT.replace("{reference_answer}", "{reference}").replace(
            "{model_answer}", "{candidate}"
        ),
        label_scores={"CORRECT": 1.0, "INCORRECT": 0.0},
        exact_gate_answers=(config["expected_answer"].strip(),),
        exact_gate_label="CORRECT",
        label_scan="json",
        label_field="judgment",
        label_case="upper",
        candidate_steps=(StripOuterWhitespace(),),
        sampling=JudgeSamplingParams(),
        parse_attempts=1,
        max_completion_tokens=FINANCEAGENT_MAX_OUTPUT_TOKENS,
        request_timeout=JUDGE_REQUEST_TIMEOUT,
        transport_attempts=JUDGE_TRANSPORT_ATTEMPTS,
        transport_budget=JUDGE_REQUEST_TIMEOUT,
    )
    return AnswerFile(FINANCEAGENT_ANSWER, verifyit_package(judge).grader)


def bixbench_answer(directory: Path) -> AnswerFile:
    """BixBench's ``tests/llm_judge.py``: a ``binary_score`` of true scores 1 and false scores 0.

    The script judges the first ``<answer>...</answer>`` span within one line, or the whole file
    when there is none, without stripping it; an empty answer still goes to the judge, and a missing
    file scores 0. It states the JSON schema in a developer message and requests it as a strict
    structured output, with a 1,024-token limit and no sampling options, so the endpoint's defaults apply.
    """
    truth = json.loads((directory / "tests" / "ground_truth.json").read_text())
    schema = json.dumps(BIXBENCH_RESPONSE_SCHEMA).replace("{", "{{").replace("}", "}}")
    judge = JudgeSpec(
        question=truth["question"],
        references=(truth["ideal_answer"],),
        model=BIXBENCH_JUDGE_MODEL,
        rubric="labels",
        system_prompt=f"Return only a JSON object matching this schema: {schema}",
        prompt_template=BIXBENCH_EVAL_PROMPT.replace("{target}", "{reference}").replace("{predicted}", "{candidate}"),
        label_scores={"true": 1.0, "false": 0.0},
        label_scan="json",
        label_field="binary_score",
        parse_attempts=1,
        max_completion_tokens=1024,
        request_timeout=JUDGE_REQUEST_TIMEOUT,
        transport_attempts=JUDGE_TRANSPORT_ATTEMPTS,
        transport_budget=JUDGE_REQUEST_TIMEOUT,
        empty_output=EmptyOutputPolicy.GRADE,
        candidate_steps=(ExtractGroup(r"<answer>(.*)</answer>"),),
        system_role="developer",
        response_format=BIXBENCH_RESPONSE_FORMAT,
        sampling=JudgeSamplingParams(),
    )
    return AnswerFile(BIXBENCH_ANSWER, verifyit_package(judge).grader)
