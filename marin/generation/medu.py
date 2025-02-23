import logging
import os
from dataclasses import dataclass
from typing import Any

import ray
from transformers import AutoTokenizer

from marin.generation.dataset import DatasetOutputProcessorConfig, DatasetSampler, MeduDatasetOutputProcessor
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
    dev_sets: list[list[str]]
    input_path: str
    output_path: str
    prompt_column: str = "text"
    filetype: str = "jsonl.gz"
    tensor_parallel_size: int = 1
    engine_kwargs: dict[str, Any] | None = None
    generation_kwargs: dict[str, Any] | None = None
    num_instances: tuple[int, int] = (1, 4)
    save_templated_prompt: bool = False
    output_filetype_override: str = "jsonl.gz"


@ray.remote
class MEDUPipeline:
    def __init__(
        self,
        model_name: str,
        dev_sets: list[list[str]],
        tensor_parallel_size: int = 1,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = vLLMProvider(model_name, engine_kwargs, generation_kwargs)
        self.generated_benchmark_descriptions = []
        self.final_benchmark_description_prompt = ""
        self.dev_sets = dev_sets
        self.tensor_parallel_size = tensor_parallel_size

    def _get_final_medu_prompt(self, benchmark_description: str) -> str:
        return MEDU_DOCUMENT_LABELING_PROMPT.format(test_description=benchmark_description, example="{example}")

    def _get_final_benchmark_description_prompt(self) -> str:
        return self.final_benchmark_description_prompt

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
    def merge_benchmark_description_prompts(self) -> str:
        logger.info(f"Starting benchmark description merging for {len(self.generated_benchmark_descriptions)} prompts")
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
def label_documents(config: MEDUPipelineConfig, final_benchmark_description_prompt: str) -> list[str]:
    text_generation_config = TextGenerationInferenceConfig(
        input_path=config.input_path,
        output_path=config.output_path,
        model_name=config.model_name,
        engine_kwargs=config.engine_kwargs,
        generation_kwargs=config.generation_kwargs,
        template=final_benchmark_description_prompt,
        num_instances=config.num_instances,
        tensor_parallel_size=config.tensor_parallel_size,
        save_templated_prompt=config.save_templated_prompt,
        prompt_column=config.prompt_column,
        filetype=config.filetype,
        output_filetype_override=config.output_filetype_override,
    )
    inference_future = run_inference.remote(text_generation_config)
    return inference_future


def run_medu_labeling_pipeline(config: MEDUPipelineConfig):
    ray_remote_args = get_scheduling_strategy_fn(config.tensor_parallel_size)()

    pipeline = MEDUPipeline.options(**ray_remote_args).remote(
        config.model_name, config.dev_sets, config.tensor_parallel_size, config.engine_kwargs, config.generation_kwargs
    )

    futures = []
    futures.append(pipeline.get_benchmark_description_prompt.remote())
    futures.append(pipeline.merge_benchmark_description_prompts.remote())

    ray.get(futures)

    logger.info(f"Starting document labeling pipeline for {config.input_path}")
    final_benchmark_description_prompt = ray.get(pipeline._get_final_benchmark_description_prompt.remote())
    label_documents_future = label_documents(config, final_benchmark_description_prompt)
    ray.get(label_documents_future)
    logger.info(f"Finished document labeling pipeline for {config.output_path}")


def run_medu_dataset_sampling_pipeline(config: DatasetOutputProcessorConfig):
    logger.info(f"Starting MEDU dataset sampling pipeline for {config.input_path}")

    convert_output_path = os.path.join(config.output_path, "converted")
    processor = MeduDatasetOutputProcessor(config.input_path, convert_output_path)
    dataset_score_distribution = processor.convert_dataset()
    logger.info(f"Dataset score distribution: {dataset_score_distribution}")
    # Keep all labels equally weighted
    label_weights = {score: 1 for score in dataset_score_distribution.keys()}
    sampler_output_path = os.path.join(config.output_path, "sampled")
    sampler = DatasetSampler(convert_output_path, sampler_output_path, label_weights)
    sampler.sample_dataset()
    logger.info(f"Finished MEDU dataset sampling pipeline for {sampler_output_path}")
