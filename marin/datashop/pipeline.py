import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Literal

import fsspec
import ray
from transformers import AutoTokenizer

from experiments.evals.resource_configs import TPU_V6E_8_STRICT_PACK, ResourceConfig
from marin.datashop.dataset_processor import MeduDatasetOutputProcessor
from marin.datashop.templates import (
    MEDU_BENCHMARK_DESCRIPTION_MERGING_TEMPLATE,
    MEDU_BENCHMARK_DESCRIPTION_TEMPLATE,
    MEDU_DOCUMENT_LABELING_PROMPT,
)
from marin.generation.dataset import DatasetOutputProcessorConfig, DatasetSampler
from marin.generation.llm_generation import vLLMProvider
from marin.generation.ray_utils import scheduling_strategy_fn

logger = logging.getLogger("ray")

MEDU_BENCHMARK_DESCRIPTION_PROMPT_FILENAME = "final_benchmark_description_prompt.txt"


@dataclass
class CorpusContent:
    # The content of the corpus.
    content: list[str] | str

    # Type of corpus content:
    #  1. str_list: The content is a list of strings. Usually a direct example of the corpus content.
    #  2. filepath: The content is a filepath to a file that contains the corpus.
    content_type: Literal["str_list", "filepath"]

    # The column name of the prompt in the corpus. This can be "text" for Dolma datasets for example.
    prompt_column: str = "text"


@dataclass
class MEDUPipelineConfig:
    """Configuration for the MEDU pipeline.

    Inputs:
        model_name: The path to the model to use for the pipeline. It should be a path to a directory in the
                    GCSFuse mount path. Check experiments/models.py for example models and how to download them.
        corpus_contents: The list of corpus content to use for the pipeline.
        input_path: The path to the input data.
        output_path: The path to write the output data.
        prompt_column: The column name of the prompt in the input data. In dolma, it is "text".
        filetype: The filetype of the input data.
        engine_kwargs: The kwargs to pass to the vLLM engine.
        generation_kwargs: The kwargs to pass for vLLM sampling parameters.
        num_instances: The number of instances to autoscale. It is a tuple of (min_workers, max_workers).
        save_templated_prompt: Whether to save the templated prompt. Use to debug the prompt passed into the model.
        output_filetype_override: The filetype to write the output data. We default to jsonl.gz to match dolma format.
        resource_config: The type of TPU hardware to use for the pipeline.
        medu_benchmark_description_template: The template that prompts an LLM to generate a description of the skills
                                             that the data should have.
    """

    model_name: str
    corpus_contents: list[CorpusContent]
    input_path: str
    output_path: str
    prompt_column: str = "text"
    filetype: str = "jsonl.gz"
    engine_kwargs: dict[str, Any] | None = None
    generation_kwargs: dict[str, Any] | None = None
    num_instances: tuple[int, int] = (1, 4)
    save_templated_prompt: bool = False
    output_filetype_override: str = "jsonl.gz"
    resource_config: ResourceConfig = field(default_factory=lambda: TPU_V6E_8_STRICT_PACK)
    medu_benchmark_description_template: str = MEDU_BENCHMARK_DESCRIPTION_TEMPLATE


@ray.remote(max_restarts=-1)  # NOTE(chris): We use Spot TPUs, so we need to be able to restart the pipeline if it fails.
class MEDUPipeline:
    """The pipeline that generates a benchmark description prompt given a list of corpus content.

    Inputs:
        model_name: The name of the model to use for the pipeline.
        corpus_contents: The list of corpus content to use for the pipeline.
        tensor_parallel_size: The number of TPUs to use for the pipeline.
        engine_kwargs: The kwargs to pass to the vLLM engine.
        generation_kwargs: The kwargs to pass for vLLM sampling parameters.
        final_benchmark_description_prompt: The final benchmark description prompt. This allows a user to pass in a
                                            prompt directly to the pipeline instead of getting the LLM to generate one.
        medu_benchmark_description_template: The template that prompts an LLM to generate a description of the skills
                                             that the data should have.
    """

    def __init__(
        self,
        model_name: str,
        corpus_contents: list[CorpusContent],
        tensor_parallel_size: int = 1,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        medu_benchmark_description_template: str = MEDU_BENCHMARK_DESCRIPTION_TEMPLATE,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = vLLMProvider(model_name, engine_kwargs, generation_kwargs)
        self.generated_benchmark_descriptions = []
        self.final_benchmark_description_prompt = ""
        self.medu_benchmark_description_template = medu_benchmark_description_template
        self.corpus_contents = corpus_contents
        self.tensor_parallel_size = tensor_parallel_size

    def _get_final_medu_prompt(self, benchmark_description: str) -> str:
        return MEDU_DOCUMENT_LABELING_PROMPT.format(test_description=benchmark_description, example="{example}")

    def _get_final_benchmark_description_prompt(self) -> str:
        return self.final_benchmark_description_prompt

    def _get_chat_templated_prompt(self, prompt: str) -> str:
        chat_prompt = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)

    # Stage 1: Get benchmark description prompt
    def get_benchmark_description_prompt(self) -> str:
        """Given the corpus content, generate a description of the corpus content and the type of data and skills
        that the data should have.
        """
        logger.info(f"Starting benchmark description prompt generation for {len(self.corpus_contents)} dev sets")
        prompts = []
        corpus = ""
        for dev_set in self.corpus_contents:
            if dev_set.content_type == "str_list":
                corpus = "\n\n".join(dev_set.content)
                corpus += "\n\n"
            elif dev_set.content_type == "filepath":
                with fsspec.open(dev_set.content, "r", compression="infer") as f:
                    for line in f:
                        row = json.loads(line)
                        if dev_set.prompt_column not in row:
                            raise ValueError(
                                f"The file {dev_set.content} does not contain a '{dev_set.prompt_column}' key, "
                                "please include it in the JSONL file."
                            )
                        corpus += row[dev_set.prompt_column] + "\n\n"

            prompt = self.medu_benchmark_description_template.format(corpus=corpus)
            prompts.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        self.generated_benchmark_descriptions = self.llm.generate(prompts)
        # Set random seed for reproducibility before shuffling
        random.seed(42)

        # Shuffle the benchmark descriptions to ensure diversity in merging
        random.shuffle(self.generated_benchmark_descriptions)
        logger.info(f"Generated {len(self.generated_benchmark_descriptions)} benchmark descriptions")
        return self.generated_benchmark_descriptions

    # Stage 2: Merge benchmark description prompts
    def merge_benchmark_description_prompts(self) -> str:
        """Hierarchically merge the corpus content description prompts into a single prompt.

        The idea is there are many descriptions of the skills that the data should have based on the
        corpus contents provided. We take each consecutive pair of descriptions and merge them into a
        single description. We repeat this process until we have a single description.
        Then, this final description is what we use to score other documents.
        """
        logger.info(f"Starting benchmark description merging for {len(self.generated_benchmark_descriptions)} prompts")
        while len(self.generated_benchmark_descriptions) > 1:
            # Odd number of prompts, do a single merge to make the number even.
            if len(self.generated_benchmark_descriptions) % 2 != 0 and len(self.generated_benchmark_descriptions) > 1:
                description_merging_prompts = [
                    MEDU_BENCHMARK_DESCRIPTION_MERGING_TEMPLATE.format(
                        description_a=self.generated_benchmark_descriptions[0],
                        description_b=self.generated_benchmark_descriptions[1],
                    )
                ]
                description_merging_prompts_templated = [
                    self._get_chat_templated_prompt(prompt) for prompt in description_merging_prompts
                ]
                new_benchmark_descriptions = self.llm.generate(description_merging_prompts_templated)
                self.generated_benchmark_descriptions.extend(new_benchmark_descriptions)

                # Pop the first two descriptions
                self.generated_benchmark_descriptions.pop(0)
                self.generated_benchmark_descriptions.pop(0)

            pairs = zip(
                self.generated_benchmark_descriptions[0::2], self.generated_benchmark_descriptions[1::2], strict=False
            )
            description_merging_prompts = [
                MEDU_BENCHMARK_DESCRIPTION_MERGING_TEMPLATE.format(
                    description_a=description_a,
                    description_b=description_b,
                )
                for description_a, description_b in pairs
            ]
            description_merging_prompts_templated = [
                self._get_chat_templated_prompt(prompt) for prompt in description_merging_prompts
            ]

            new_benchmark_descriptions = self.llm.generate(description_merging_prompts_templated)
            self.generated_benchmark_descriptions = new_benchmark_descriptions

        # We only have one description left
        self.final_benchmark_description_prompt = self._get_final_medu_prompt(self.generated_benchmark_descriptions[0])
        logger.info(f"Final benchmark description prompt: {self.final_benchmark_description_prompt}")


def _get_final_benchmark_description_prompt_output_path(output_path: str):
    return os.path.join(output_path, MEDU_BENCHMARK_DESCRIPTION_PROMPT_FILENAME)


def _write_final_benchmark_description_prompt(final_benchmark_description_prompt: str, output_path: str):
    with fsspec.open(_get_final_benchmark_description_prompt_output_path(output_path), "w", compression="infer") as f:
        f.write(final_benchmark_description_prompt)


def _run_benchmark_prompt_generation_pipeline(config: MEDUPipelineConfig):
    scheduling_strategy = scheduling_strategy_fn(config.resource_config.num_tpu, config.resource_config.strategy)
    pipeline = MEDUPipeline.options(scheduling_strategy=scheduling_strategy).remote(
        config.model_name,
        config.corpus_contents,
        config.resource_config.num_tpu,
        config.engine_kwargs,
        config.generation_kwargs,
        config.medu_benchmark_description_template,
    )

    futures = []
    futures.append(pipeline.get_benchmark_description_prompt.remote())
    futures.append(pipeline.merge_benchmark_description_prompts.remote())
    ray.get(futures)

    logger.info(f"Starting document labeling pipeline for {config.input_path}")
    final_benchmark_description_prompt = ray.get(pipeline._get_final_benchmark_description_prompt.remote())
    return final_benchmark_description_prompt


def run_data_filter_prompt_generation_pipeline(config: MEDUPipelineConfig):
    """Runs the pipeline that generates a data filter prompt given some targeted corpus content.

    The user can either pass in a final benchmark description prompt or let the pipeline generate one
    given some targeted corpus content.
    """
    final_benchmark_description_prompt = _run_benchmark_prompt_generation_pipeline(config)
    _write_final_benchmark_description_prompt(final_benchmark_description_prompt, config.output_path)


def run_medu_dataset_sampling_pipeline(config: DatasetOutputProcessorConfig):
    """Runs the pipeline that converts the labeled documents into a dataset of sampled documents.

    This pipeline is split into two stages:
    1. Convert the labeled documents into a dataset of parsed scores.
    2. Sample the dataset to get a final dataset of documents with key "text" and "label" corresponding
    to the text and its respective score graded by the model.
    """
    logger.info(f"Starting MEDU dataset sampling pipeline for {config.input_path}")

    convert_output_path = os.path.join(config.output_path, "converted")
    processor = MeduDatasetOutputProcessor(config.input_path, convert_output_path)
    dataset_score_distribution = processor.convert_dataset()
    logger.info(f"Dataset score distribution: {dataset_score_distribution}")
    if -1 in dataset_score_distribution:
        logger.warning(f"Found {dataset_score_distribution[-1]} examples with score -1. These will not be sampled.")
    # Keep all labels equally weighted
    label_weights = {score: 1 for score in dataset_score_distribution.keys() if score != -1}
    sampler_output_path = os.path.join(config.output_path, "sampled")
    sampler = DatasetSampler(convert_output_path, sampler_output_path, label_weights)
    sampler.sample_dataset()
    logger.info(f"Finished MEDU dataset sampling pipeline for {sampler_output_path}")
