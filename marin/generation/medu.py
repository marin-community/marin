import logging
from dataclasses import dataclass
from typing import Any

import ray
from transformers import AutoTokenizer

from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from marin.generation.llm_generation import vLLMProvider
from marin.generation.ray_utils import get_scheduling_strategy_fn
from marin.generation.templates import (
    MEDU_BENCHMARK_DESCRIPTION_MERGING_TEMPLATE,
    MEDU_BENCHMARK_DESCRIPTION_TEMPLATE,
    MEDU_DOCUMENT_LABELING_PROMPT,
)

logger = logging.getLogger("ray")


@dataclass
class MEDUPipelineConfig:
    model_name: str
    dev_sets: list[str]
    input_path: str
    output_path: str
    prompt_column: str = "text"
    filetype: str = "jsonl.gz"
    tensor_parallel_size: int = 1
    engine_kwargs: dict[str, Any] | None = None
    generation_kwargs: dict[str, Any] | None = None
    num_instances: tuple[int, int] = (1, 4)
    save_templated_prompt: bool = False


@ray.remote
class MEDUPipeline:
    def __init__(
        self,
        model_name: str,
        dev_sets: list[str],
        input_path: str,
        output_path: str,
        prompt_column: str = "text",
        filetype: str = "jsonl.gz",
        tensor_parallel_size: int = 1,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        num_instances: tuple[int, int] = (1, 4),
        save_templated_prompt: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = vLLMProvider(model_name, engine_kwargs, generation_kwargs)
        self.generated_benchmark_descriptions = []
        self.final_benchmark_description_prompt = ""
        self.dev_sets = dev_sets
        self.input_path = input_path
        self.output_path = output_path
        self.tensor_parallel_size = tensor_parallel_size
        self.num_instances = num_instances
        self.save_templated_prompt = save_templated_prompt

    def _get_final_medu_prompt(self, benchmark_description: str) -> str:
        return MEDU_DOCUMENT_LABELING_PROMPT.format(test_description=benchmark_description, example="{example}")

    # Stage 1: Get benchmark description prompt
    def get_benchmark_description_prompt(self) -> str:
        logger.info(f"Starting benchmark description prompt generation for {len(self.dev_sets)} dev sets")
        prompts = []
        for dev_set in self.dev_sets:
            corpus = "\n\n".join(dev_set)
            prompt = MEDU_BENCHMARK_DESCRIPTION_TEMPLATE.format(corpus=corpus)
            prompts.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        self.generated_benchmark_descriptions = self.llm.generate(prompts)
        logger.info(f"Generated {len(self.generated_benchmark_descriptions)} benchmark descriptions")
        return self.generated_benchmark_descriptions

    # Stage 2: Merge benchmark description prompts
    def merge_benchmark_description_prompts(self, benchmark_description_prompts: list[str]) -> str:
        logger.info(f"Starting benchmark description merging for {len(benchmark_description_prompts)} prompts")
        while len(self.generated_benchmark_descriptions) > 1:
            description_merging_prompt = MEDU_BENCHMARK_DESCRIPTION_MERGING_TEMPLATE.format(
                description_a=self.generated_benchmark_descriptions[0],
                description_b=self.generated_benchmark_descriptions[1],
            )
            chat_prompt = [{"role": "user", "content": description_merging_prompt}]
            description_merging_prompt = self.tokenizer.apply_chat_template(
                chat_prompt, tokenize=False, add_generation_prompt=True
            )
            new_benchmark_descriptions = self.llm.generate([description_merging_prompt])
            self.generated_benchmark_descriptions.extend(new_benchmark_descriptions)

            # Pop the first two descriptions
            self.generated_benchmark_descriptions.pop(0)
            self.generated_benchmark_descriptions.pop(0)

        # We only have one description left
        self.final_benchmark_description_prompt = self._get_final_medu_prompt(self.generated_benchmark_descriptions[0])
        logger.info(f"Final benchmark description prompt: {self.final_benchmark_description_prompt}")

    # Stage 3: Label documents
    def label_documents(self) -> list[str]:
        text_generation_config = TextGenerationInferenceConfig(
            input_path=self.input_path,
            output_path=self.output_path,
            model_name=self.model_name,
            engine_kwargs=self.engine_kwargs,
            generation_kwargs=self.generation_kwargs,
            template=self.final_benchmark_description_prompt,
            num_instances=self.num_instances,
            tensor_parallel_size=self.tensor_parallel_size,
            save_templated_prompt=self.save_templated_prompt,
            prompt_column=self.prompt_column,
            filetype=self.filetype,
        )
        inference_future = run_inference.remote(text_generation_config)
        return inference_future


def run_medu_pipeline(config: MEDUPipelineConfig):
    scheduling_strategy = get_scheduling_strategy_fn(config.tensor_parallel_size)

    pipeline = MEDUPipeline.options(scheduling_strategy=scheduling_strategy).remote(**config)
    pipeline.get_benchmark_description_prompt()
    pipeline.merge_benchmark_description_prompts()
    pipeline.label_documents()
    return pipeline.output_path
