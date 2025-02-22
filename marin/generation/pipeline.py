from typing import Any

from marin.generation.llm_generation import BaseLLMProvider, vLLMProvider
from marin.generation.templates import STEP_BY_STEP_TEMPLATE


class TextGeneration:
    def __init__(
        self,
        llm: BaseLLMProvider,
        template: str | None = None,
        num_generations: int = 1,
        prompt_column: str = "text",
        save_templated_prompt: bool = False,
    ):
        self.llm = llm

        # Template is a string that contains a placeholder for "example"
        # which will be replaced with the actual example
        self.template = template or STEP_BY_STEP_TEMPLATE
        self.num_generations = num_generations
        self.prompt_column = prompt_column
        self.save_templated_prompt = save_templated_prompt

    def _update_batch(self, batch: dict[str, Any], generated_text: list[str], prompts: list[str]) -> dict[str, Any]:
        batch.update({"generated_text": generated_text})

        if self.save_templated_prompt:
            batch.update({"prompt": prompts})

        return batch

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Generate a batch of text using an LLM where the example text is in dolma format in the "text" column."""

        prompts = [self.template.format(example=example) for example in batch[self.prompt_column]]
        generated_text = self.llm.generate(prompts)

        return self._update_batch(batch, generated_text, prompts)


class vLLMTextGeneration(TextGeneration):
    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        template: str | None = None,
        num_generations: int = 1,
        num_instances: tuple[int, int] = (1, 4),
        prompt_column: str = "text",
        apply_chat_template: bool = True,
        save_templated_prompt: bool = False,
    ):
        # Initialize the LLM Provider here for the pipeline since we need the model
        # to be placed in the same placement group as the pipeline
        llm = vLLMProvider(model_name, engine_kwargs, generation_kwargs)

        super().__init__(llm, template, num_generations, prompt_column, save_templated_prompt)
        self.apply_chat_template = apply_chat_template

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.apply_chat_template:
            prompts = []
            tokenizer = self.llm.llm.get_tokenizer()
            for example in batch[self.prompt_column]:
                chat_example = [{"role": "user", "content": self.template.format(example=example)}]
                prompts.append(tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True))
        else:
            prompts = [self.template.format(example=example) for example in batch[self.prompt_column]]

        generated_text = self.llm.generate(prompts)
        return self._update_batch(batch, generated_text, prompts)
