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
Wrapper around vLLM generation.

User -> Preset Prompt, Engine Kwargs, File -> Generate Text
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar
import os

# Disable multiprocessing since running on Ray, easier to clean
# resources if not using a multiprocess.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
except ImportError:
    logger.warning("vLLM is not installed, so we will not be able to generate text.")
    TokensPrompt = Any


class BaseLLMProvider(ABC):
    @abstractmethod
    def __init__(self, model_name: str):
        pass

    @abstractmethod
    def generate(self, prompts: list[str]) -> list[str]:
        """The input is a list of prompts and the output is a list of generated text."""
        raise NotImplementedError


class vLLMProvider(BaseLLMProvider):
    DEFAULT_ENGINE_KWARGS: ClassVar[dict[str, Any]] = {
        "tensor_parallel_size": 1,
        "enforce_eager": False,
    }

    DEFAULT_GENERATION_KWARGS: ClassVar[dict[str, Any]] = {
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model_name)

        self.model_name = model_name
        self.engine_kwargs = {**vLLMProvider.DEFAULT_ENGINE_KWARGS, **engine_kwargs}
        self.generation_kwargs = {**vLLMProvider.DEFAULT_GENERATION_KWARGS, **generation_kwargs}

        self.llm = LLM(model=self.model_name, **self.engine_kwargs)
        self.sampling_params = SamplingParams(**self.generation_kwargs)

    def generate(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_text: list[str] = []
        for output in outputs:
            generated_text.append(" ".join([o.text for o in output.outputs]))
        return generated_text

    def generate_multi_sample(self, prompts: list[str]) -> list[list[str]]:
        """Generate multiple samples per prompt, returning them separately.

        Unlike generate() which joins samples with spaces, this returns
        each sample individually for downstream selection/filtering.

        Args:
            prompts: List of prompts to generate from.

        Returns:
            List of lists, where each inner list contains all samples for one prompt.
        """
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_samples: list[list[str]] = []
        for output in outputs:
            samples = [o.text for o in output.outputs]
            generated_samples.append(samples)
        return generated_samples
