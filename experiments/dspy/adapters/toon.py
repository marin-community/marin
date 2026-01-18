import dspy

from typing import Any

class ToonAdapter(dspy.Adapter):
    def call(self, example: dspy.Example) -> dspy.Example:
        return example

    def parse(self, signature: type[dspy.Signature], completion: str) -> dict[str, Any]:
        return {
            "result": completion
        }

    def format(self, signature: type[dspy.Signature], inputs: dict[str, Any]) -> str:
        return f"### User: {inputs['user']}\n### Assistant: {inputs['assistant']}"
