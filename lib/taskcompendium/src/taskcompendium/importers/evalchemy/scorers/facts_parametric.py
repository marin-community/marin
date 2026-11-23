# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Parametric grading: one judge verdict per answer, read from the reply's last ``Output:`` match.

``GRADER_TEMPLATE`` is the starter notebook's prompt byte for byte
(kaggle.com/code/yulongt/facts-parametric-benchmark-starter-code, version 10; Apache 2.0), as vendored in
NVIDIA-NeMo/Gym#3588. The notebook samples three grades from Gemini 2.5 Pro and maps each reply to a label
by substring search; this scorer asks the run's configured judge once and reads the last ``Output:``
match, because the judge's explanation often names other labels.
"""

import re
from collections.abc import Callable

from openai.types.chat import ChatCompletionMessageParam
from verifyit.grade import GradingInfraError, Reward, scored
from verifyit.modes.grade_judge import JudgeReply

GRADER_TEMPLATE = """Your task is to grade the correctness of a predicted answer given a question and its gold answer.

Assign a grade as follows:

A: CORRECT - Matches gold answer without contradiction. Minor granularity differences permitted if the question allows. Hedging ("I think," "possibly," etc.) is acceptable if correct and not contradicted. If the predicted answer provides more information than the gold answer, the predicted answer must contain the gold answer and the additional information may not contradict the gold answer.
B: MISTAKE - Directly contradicts the gold answer or provided context, even with hedging. Numeric answers significantly differ from gold target. In cases of lists, if the lists are not equal, the grade is incorrect.
C: UNKNOWN - Contains relevant information neither confirmed nor contradicted by gold answer.
D: NOT_ATTEMPTED - Does not provide the required information; answer is missing key details or explicitly refuses to answer.

Examples:
```
Question: What are the names of Barack Obama's children?
Gold answer: Malia and Sasha
```

The following are examples of CORRECT predicted answers.
```
Predicted answer 1: sasha and malia obama
Predicted answer 2: Malia Ann and Natasha.
Predicted answer 4: Barack Obama was the 41th president of the United States. He has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
  - They fully contain the important information in the gold target.
  - They do not contain any information that contradicts the gold target.
  - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
  - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.
  - In ``Predicted answer 2'', I know that Sasha is used in many cases as a nickname for Natasha. It is very likely the case here as well.

The following are examples of MISTAKE predicted answers.
```
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary.
```
These predicted answers are all MISTAKE because:
  - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it's possible that", "although i'm not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Predicted answer 1: I cannot answer any political related questions.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
  - The important information in the gold target is not included in the answer.
  - No statements in the answer contradict the gold target.


```
Question: When was Shimon Peres born?
Gold answer: August 2, 1923
```

The following are examples of CORRECT predicted answers.
```
Predicted answer 1: 1923
Predicted answer 2: August 1923
Predicted answer 3: 2.8.1923, it was 20th of Av, 5683 according to the hebrew calender
```

The following are examples of MISTAKE predicted answers.
```
Predicted answer 1:: August 3, 1923
```

The following are examples of UNKNOWN predicted answers.
```
Predicted: 20th of Av, 5683
```
This predicted answer is UNKNOWN because:
- It refers to the Hebrew calender. It clearly answers the question, and might be true, but is not supported by the context or the gold answer.

```
Question: Who owns Google
Gold answer: Alphabet
```
The following is an example of a CORRECT predicted answer.
```
Predicted answer 1: Alphabet Inc.
```
This predicted answer is CORRECT because:
- It is aligned with the gold answer. It contains an additional information, that Google is an incorporated business. This extra information is not confirmed by the gold answer, but it does not contradict it as well.is not

### Special Guidelines:

- **Numeric answers:** Must match to the last significant figure ("120k" and "124k" correct for "120k"; "100k" incorrect; "around 100k" is NOT_ATTEMPTED).
- **Granularity:** Predicted answer must adhere to the granularity specified in the question. If unspecified, broader granularity is permitted if supported by the gold answer or the context.
- **Inference/Omissions:** Do not penalize for omitting details clearly inferred from the question (e.g., omitting state when city is specified).
- **Typos:** Minor typos in names or answers do not affect correctness if clearly identifiable.

### Step-by-Step Grading Process which you need to follow:

1. Identify the direct predicted answer, ignoring background, hedging, or additional information. If the model only hedges, output NOT_ATTEMPTED and finish.
2. Check match with the gold answer:
   - If matching clearly → output CORRECT and finish.
   - If clearly contradicting → output MISTAKE and finish.
   - If unclear and you cannot decide → output UNKNOWN and finsh.


### Response format:

Briefly explain your decision-making steps clearly.
End your response explicitly with:
```
Output: [CORRECT/MISTAKE/UNKNOWN/NOT_ATTEMPTED]
```

Now grade this new example:

```
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {prediction}
```"""  # noqa: E501 - exact source prompt

# The notebook's label for each verdict; MISTAKE and INCORRECT are the same verdict.
VERDICTS = {
    "CORRECT": "correct",
    "MISTAKE": "mistake",
    "INCORRECT": "mistake",
    "UNKNOWN": "unknown",
    "NOT_ATTEMPTED": "not_attempted",
}
# Matches "Output: CORRECT", "Output: [NOT_ATTEMPTED]" and markdown-bold variants, but not the
# prompt's own "Output: [CORRECT/MISTAKE/UNKNOWN/NOT_ATTEMPTED]" if the judge echoes it.
OUTPUT_LINE = re.compile(
    r"Output:?\**\s*\[?\s*\**\s*(CORRECT|MISTAKE|INCORRECT|UNKNOWN|NOT[_ ]ATTEMPTED)\s*\**\s*\]?(?![\w/])",
    re.IGNORECASE,
)

Ask = Callable[[list[ChatCompletionMessageParam]], JudgeReply]


def grader_prompt(question: str, gold_answer: str, prediction: str) -> str:
    return GRADER_TEMPLATE.format(question=question, gold_answer=gold_answer, prediction=prediction)


def output_verdict(judgment: str) -> str | None:
    """Return the verdict from the reply's last ``Output:`` match, or None if absent."""
    matches = OUTPUT_LINE.findall(judgment)
    if not matches:
        return None
    return VERDICTS[matches[-1].upper().replace(" ", "_")]


def grade_facts_parametric(question: str, gold_answer: str, response: str, ask: Ask) -> Reward:
    """Score 1 for a CORRECT verdict; UNKNOWN counts as an attempted, incorrect answer, as in the paper's metrics."""
    reply = ask([{"role": "user", "content": grader_prompt(question, gold_answer, response)}])
    verdict = output_verdict(reply.text)
    if verdict is None:
        raise GradingInfraError("judge reply has no Output verdict line", judgment=reply.text, **reply.detail)
    correct = float(verdict == "correct")
    metrics = {
        "accuracy": correct,
        "incorrect": float(verdict in ("mistake", "unknown")),
        "not_attempted": float(verdict == "not_attempted"),
        "unknown": float(verdict == "unknown"),
        "judge_failed": 0.0,
    }
    return scored(
        correct,
        verdict=verdict,
        judgment=reply.text,
        judge=reply.detail,
        metrics=metrics,
    )
