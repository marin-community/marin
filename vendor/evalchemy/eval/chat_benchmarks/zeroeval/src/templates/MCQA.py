# Multiple Choice Question Answer Template
MCQA = """
## Question: 

{question}

## Choices:

{choices}

## Instruction 

Please answer this question by first reasoning and then selecting the correct choice. 
Present your reasoning and solution in the following json format. 
Please show your choice in the `answer` field with only the choice letter, e.g.,`"answer": "C"`.

```json
{
    "reasoning": "___",
    "answer": "___"
}
```
"""
