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

import logging
from typing import Any

from marin.generation.llm_generation import BaseLLMProvider, vLLMProvider
from marin.generation.templates import STEP_BY_STEP_TEMPLATE

logger = logging.getLogger(__name__)

try:
    from vllm.inputs.data import TokensPrompt
except ImportError:
    logger.warning("vLLM is not installed, so we will not be able to generate text.")
    TokensPrompt = Any


class TextGeneration:
    def __init__(
        self,
        llm: BaseLLMProvider,
        template: list[str] | str | None = None,
        num_generations: int = 1,
        prompt_column: str = "text",
        save_templated_prompt: bool = False,
        generated_text_column_name: str = "generated_text",
    ):
        self.llm = llm

        # Template is a string that contains a placeholder for "example"
        # which will be replaced with the actual example
        self.template = template or STEP_BY_STEP_TEMPLATE
        self.num_generations = num_generations
        self.prompt_column = prompt_column
        self.save_templated_prompt = save_templated_prompt
        self.generated_text_column_name = generated_text_column_name

    def _update_batch(self, batch: dict[str, Any], generated_text: list[str], prompts: list[str]) -> dict[str, Any]:
        batch.update({self.generated_text_column_name: generated_text})

        if self.save_templated_prompt:
            batch.update({"prompt": prompts})

        return batch

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Generate a batch of text using an LLM where the example text is in dolma format in the "text" column."""

        if isinstance(self.template, list):
            assert len(self.template) == len(
                batch[self.prompt_column]
            ), "The number of templates must match the number of examples."
            prompts = [
                template.format(example=example)
                for template, example in zip(self.template, batch[self.prompt_column], strict=False)
            ]
        else:
            prompts = [self.template.format(example=example) for example in batch[self.prompt_column]]

        generated_text = self.llm.generate(prompts)

        return self._update_batch(batch, generated_text, prompts)


class vLLMTextGeneration(TextGeneration):
    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        template: list[str] | str | None = None,
        num_generations: int = 1,
        num_instances: tuple[int, int] = (1, 4),
        prompt_column: str = "text",
        apply_chat_template: bool = True,
        save_templated_prompt: bool = False,
        max_doc_tokens: int = 7000,
        generated_text_column_name: str = "generated_text",
    ):
        # Initialize the LLM Provider here for the pipeline since we need the model
        # to be placed in the same placement group as the pipeline
        llm = vLLMProvider(model_name, engine_kwargs, generation_kwargs)

        super().__init__(
            llm, template, num_generations, prompt_column, save_templated_prompt, generated_text_column_name
        )
        self.apply_chat_template = apply_chat_template
        self.max_doc_tokens = max_doc_tokens
        self.tokenizer = self.llm.llm.get_tokenizer()

    def _truncate_example(self, example: str) -> str:
        example_tokens = self.tokenizer.encode(example)
        if len(example_tokens) > self.max_doc_tokens:
            example_tokens = example_tokens[: self.max_doc_tokens]
            example = self.tokenizer.decode(example_tokens)

        return example

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        prompts = []

        if isinstance(self.template, list):
            assert len(self.template) == len(
                batch[self.prompt_column]
            ), "The number of templates must match the number of examples."

            for template, example in zip(self.template, batch[self.prompt_column], strict=False):
                example = self._truncate_example(example)

                if self.apply_chat_template:
                    try:
                        chat_example = [{"role": "user", "content": template.format(example=example)}]
                    except Exception as e:
                        print(f"Error formatting template: {e}")
                        print(f"Template: {template}")
                        print(f"Example: {example}")
                        raise e
                    prompts.append(
                        self.tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    prompts.append(example)
        else:
            for example in batch[self.prompt_column]:
                example = self._truncate_example(example)
                if self.apply_chat_template:
                    chat_example = [{"role": "user", "content": self.template.format(example=example)}]
                    prompts.append(
                        self.tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    prompts.append(example)

        generated_text = self.llm.generate(prompts)
        return self._update_batch(batch, generated_text, prompts)
