#  Open-Ended Question Answering Template
OEQA = """
## Question: 

{question}


## Instruction 

Please answer this question by first reasoning and then providing your answer.
Present your reasoning and solution in the following json format. 
Please show your final answer in the `answer` field, e.g.,`"answer": "42"`.

```json
{
    "reasoning": "___",
    "answer": "___"
}
```
"""

OEQA_DIRECT = """

## Question: 

{question}


## Instruction 

Please solve this question directly by providing your answer.
Please show your final answer in the `answer` field. without explanation in the following json format. 

```json
{
    "answer": "___"
}
```
"""
