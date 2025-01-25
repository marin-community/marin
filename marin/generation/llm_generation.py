"""
Wrapper around vLLM generation.

User -> Preset Prompt, Engine Kwargs, File -> Generate Text
"""

from typing import Any, ClassVar

from vllm import LLM, SamplingParams


class LLMProvider:
    DEFAULT_ENGINE_KWARGS: ClassVar[dict[str, Any]] = {
        "tensor_parallel_size": 1,
        "enforce_eager": True,
    }

    DEFAULT_GENERATION_KWARGS: ClassVar[dict[str, Any]] = {
        "temperature": 0.1,
        "top_p": 0.3,
        "max_tokens": 128,
    }

    def __init__(self, model_name: str, generation_kwargs: dict[str, Any], engine_kwargs: dict[str, Any]):
        self.model_name = model_name
        self.engine_kwargs = {**LLMProvider.DEFAULT_ENGINE_KWARGS, **engine_kwargs}
        self.generation_kwargs = {**LLMProvider.DEFAULT_GENERATION_KWARGS, **generation_kwargs}

        self.llm = LLM(model=self.model_name, **self.engine_kwargs)
        self.sampling_params = SamplingParams(**self.generation_kwargs)

    def generate(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_text: list[str] = []
        for output in outputs:
            generated_text.append(" ".join([o.text for o in output.outputs]))
        return generated_text
