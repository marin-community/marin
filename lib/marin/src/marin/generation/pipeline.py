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
        """Initializes a text generation pipeline that takes an input batch and generates text for each example.

        Inputs:
            llm: The LLM provider to use for the pipeline.
            template: The template to use for the pipeline. This can be a string or a list of strings.
                If it is a string, it will be used for all examples.
                If it is a list of strings, we will zip the template with each example
                (i.e. iterate through template, example in zip(templates, examples))
            num_generations: The number of generations to generate for each example.
            prompt_column: The column name of the prompt in the input batch.
            save_templated_prompt: Whether to save the templated prompt.
            generated_text_column_name: The column name of the generated text in the output batch.
        """
        self.llm = llm

        # Normalize template to a list for consistent handling at call time
        # Accept either a single string template or a list of per-example templates.
        base_template = template or STEP_BY_STEP_TEMPLATE
        self.templates: list[str] = base_template if isinstance(base_template, list) else [base_template]
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

        examples = batch[self.prompt_column]

        if len(self.templates) == 1:
            # Broadcast single template to all examples
            prompts = [self.templates[0].format(example=example) for example in examples]
        else:
            assert len(self.templates) == len(examples), "The number of templates must match the number of examples."
            prompts = [
                template.format(example=example) for template, example in zip(self.templates, examples, strict=False)
            ]

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

        examples = batch[self.prompt_column]

        if len(self.templates) == 1:
            single_template = self.templates[0]
            for example in examples:
                example = self._truncate_example(example)
                if self.apply_chat_template:
                    try:
                        chat_example = [{"role": "user", "content": single_template.format(example=example)}]
                    except Exception as e:
                        logger.error(f"Error formatting template: {e}")
                        logger.error(f"Template: {single_template}")
                        logger.error(f"Example: {example}")
                        raise e
                    prompts.append(
                        self.tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    prompts.append(example)
        else:
            assert len(self.templates) == len(examples), "The number of templates must match the number of examples."
            for template, example in zip(self.templates, examples, strict=False):
                example = self._truncate_example(example)
                if self.apply_chat_template:
                    try:
                        chat_example = [{"role": "user", "content": template.format(example=example)}]
                    except Exception as e:
                        logger.error(f"Error formatting template: {e}")
                        logger.error(f"Template: {template}")
                        logger.error(f"Example: {example}")
                        raise e
                    prompts.append(
                        self.tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    prompts.append(example)

        generated_text = self.llm.generate(prompts)
        return self._update_batch(batch, generated_text, prompts)
