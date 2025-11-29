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

import os
from dataclasses import replace

from experiments.anneal_config import AnnealConfig
from experiments.datashop.default_configs import (
    default_consolidate_config_kwargs,
    default_consolidate_filter_config_kwargs,
    default_dataset_output_processor_config_kwargs,
    default_engine_kwargs,
    default_generation_kwargs,
    default_inference_config_kwargs,
    default_medu_config_kwargs,
    default_quality_filter_train_config_kwargs,
    default_text_generation_config_kwargs,
)
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.defaults import default_anneal, default_tokenize
from experiments.evals.resource_configs import ResourceConfig
from experiments.llama import llama3_tokenizer
from marin.classifiers.hf.launch_ray_training import LaunchConfig, launch_training_with_ray
from marin.datashop.dataset_processor import DatasetOutputProcessorConfig
from marin.datashop.pipeline import (
    MEDU_BENCHMARK_DESCRIPTION_PROMPT_FILENAME,
    CorpusContent,
    MEDUPipelineConfig,
    run_data_filter_prompt_generation_pipeline,
    run_medu_dataset_sampling_pipeline,
)
from marin.download.filesystem.transfer import TransferConfig, transfer_files
from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path
from marin.generation.inference import TextGenerationInferenceConfig
from marin.generation.inference import run_inference as run_generation_inference
from marin.processing.classification.config.inference_config import InferenceConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import run_inference
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config
from marin.resources import TpuPodConfig


def default_label(
    documents_to_be_labeled: str | ExecutorStep,
    targeted_documents: list[CorpusContent],
    experiment_name: str,
    resource_config: ResourceConfig,
    annotator_model_name_or_path: str = "meta-llama/Llama-3.3-70B-Instruct",
    data_filter_prompt: str | None = None,
    medu_pipeline_config_kwargs: dict | None = None,
    text_generation_inference_config_kwargs: dict | None = None,
):
    """Label a set of documents with an LLM given some targeted documents.

    Inputs:
        documents_to_be_labeled: Input path to documents to be labeled.
        targeted_documents: A list of CorpusContent objects that define the corpus content to use for
            generating the data filter prompt. It can either point to a filepath or a list of strings that
            represent the text corpus.
        experiment_name: The name of the experiment.
        data_filter_prompt: The user's prompt for the annotator model.
        medu_pipeline_config_kwargs: Keyword arguments for the MEDU pipeline which is used to generate
            a data filter prompt given some existing data.
        text_generation_inference_config: Keyword arguments for the text generation inference which is
            used to label the documents based on their quality.

    Outputs:
        An ExecutorStep that represents the labeled documents. Each document is .jsonl.gz file with
        each row containing an additional key "generated_text" that contains the LLM's response.
    """
    if isinstance(documents_to_be_labeled, ExecutorStep):
        documents_to_be_labeled = output_path_of(documents_to_be_labeled)

    assert (data_filter_prompt is not None and targeted_documents == []) or (
        data_filter_prompt is None and targeted_documents != []
    ), "Must provide either a data filter prompt or targeted documents, but not both"

    # Quality teacher model that is used to label the initial data pool that will then be used
    # to train the quality filter model.
    # TODO(chris): Make this a parameter and support other models. As of now, we make Llama-70B-Instruct the
    # default quality teacher model.
    # If the user does not provide a data filter prompt, we generate one using the MEDU pipeline.
    if data_filter_prompt is None:
        if medu_pipeline_config_kwargs is None:
            medu_pipeline_config_kwargs = default_medu_config_kwargs
        else:
            # Merge default kwargs with passed-in kwargs, with passed-in values taking precedence
            medu_pipeline_config_kwargs = {**default_medu_config_kwargs, **medu_pipeline_config_kwargs}

        medu_pipeline_config = MEDUPipelineConfig(
            corpus_contents=targeted_documents,
            input_path=documents_to_be_labeled,
            output_path=this_output_path(),
            resource_config=resource_config,
            model_name=annotator_model_name_or_path,
            **medu_pipeline_config_kwargs,
        )

        data_filter_generated_prompt = ExecutorStep(
            name=f"documents/datashop-prompts/{experiment_name}",
            fn=run_data_filter_prompt_generation_pipeline,
            config=medu_pipeline_config,
        )
        data_filter_prompt_path = output_path_of(
            data_filter_generated_prompt, MEDU_BENCHMARK_DESCRIPTION_PROMPT_FILENAME
        )
    else:
        data_filter_prompt_path = None

    # NOTE(chris): Assuming we are filtering from a jsonl.zst file such as DCLM.
    if text_generation_inference_config_kwargs is None:
        text_generation_inference_config_kwargs = default_text_generation_config_kwargs
    else:
        text_generation_inference_config_kwargs = {
            **default_text_generation_config_kwargs,
            **text_generation_inference_config_kwargs,
        }

    text_generation_inference_config = TextGenerationInferenceConfig(
        input_path=documents_to_be_labeled,
        output_path=this_output_path(),
        template=data_filter_prompt,
        template_path=data_filter_prompt_path,
        tensor_parallel_size=resource_config.num_tpu,
        resource_config=resource_config,
        model_name=annotator_model_name_or_path,
        **text_generation_inference_config_kwargs,
    )

    return ExecutorStep(
        name=f"documents/datashop-labels/{experiment_name}",
        fn=run_generation_inference,
        config=text_generation_inference_config,
        override_output_path=f"documents/datashop-labels/{experiment_name}",
    )


def default_train_quality_model(
    labeled_documents: ExecutorStep,
    experiment_name: str,
    resource_config: ResourceConfig,
    dataset_processor_config_kwargs: dict | None = None,
    quality_train_config_kwargs: dict | None = None,
):
    """Train a quality filter model based on the set of labeled documents.

    Inputs:
        labeled_documents: An ExecutorStep that represents the labeled documents.
        experiment_name: The name of the experiment.
        resource_config: The resource config to use for training the quality filter model.
        dataset_processor_config_kwargs: Keyword arguments for the dataset processor which is used to
            process the labeled documents into a dataset of "text" and "label" columns.
        quality_train_config_kwargs: Keyword arguments for the quality filter training config - matches
            the huggingface trainer config.

    Outputs:
        An ExecutorStep that represents the quality filter model.
    """
    if dataset_processor_config_kwargs is None:
        dataset_processor_config_kwargs = default_dataset_output_processor_config_kwargs
    else:
        dataset_processor_config_kwargs = {
            **default_dataset_output_processor_config_kwargs,
            **dataset_processor_config_kwargs,
        }

    dataset_output_processor_config = DatasetOutputProcessorConfig(
        input_path=output_path_of(labeled_documents), output_path=this_output_path(), **dataset_processor_config_kwargs
    )

    dataset = ExecutorStep(
        name=f"documents/datashop-datasets/{experiment_name}",
        fn=run_medu_dataset_sampling_pipeline,
        config=dataset_output_processor_config,
    ).cd("sampled")

    if quality_train_config_kwargs is None:
        quality_train_config_kwargs = default_quality_filter_train_config_kwargs
    else:
        quality_train_config_kwargs = {**default_quality_filter_train_config_kwargs, **quality_train_config_kwargs}

    training_config = quality_train_config_kwargs["training_config"]
    training_config = replace(
        training_config,
        train_dataset=dataset,
        tpu_num_cores=resource_config.num_tpu,
        run_name=f"datashop-classifier-{experiment_name}",
    )

    quality_train_config = LaunchConfig(training_config=training_config, resource_config=resource_config)

    datashop_classifier_remote = ExecutorStep(
        name=f"classifiers/datashop-bert/{experiment_name}", fn=launch_training_with_ray, config=quality_train_config
    )
    # Download the model locally to GCSFuse mount path for inference
    datashop_classifier = ExecutorStep(
        name=f"gcsfuse_mount/datashop-models/{experiment_name}-classifier",
        fn=transfer_files,
        config=TransferConfig(
            input_path=output_path_of(datashop_classifier_remote),
            output_path=this_output_path(),
        ),
        override_output_path=f"gcsfuse_mount/datashop-models/{experiment_name}-classifier",
        pip_dependency_groups=[
            # NOTE(Chris): USE MAIN for now since newest accelerate 1.6.0 still uses
            # xrt_world_size which has since been deprecated after Pytorch XLA 2.7.0
            "https://github.com/huggingface/accelerate/archive/refs/heads/main.zip"
        ],
    )

    return datashop_classifier


def default_quality_filter(
    encoder_model: ExecutorStep | str,
    input_data_path: str | ExecutorStep,
    input_data_name: str,
    experiment_name: str,
    inference_config_kwargs: dict | None = None,
):
    """Runs quality filtering and consolidation on an input dataset given the quality filter model.

    Inputs:
        encoder_model: The model to use for quality filtering.
        input_data_path: The path to the input data, usually a large pretraining corpus to filter.
        input_data_name: The name of the input data used for storage purposes.
        experiment_name: The name of the experiment.
        inference_config_kwargs: Keyword arguments for the inference config which uses the encoder
            model to score documents based on their quality.
        filter_config_kwargs: Keyword arguments for the filter config which is used to filter the
            documents based on their quality.
        consolidate_config_kwargs: Keyword arguments for the consolidate config which is used to
            consolidate the filtered documents.

    Outputs:
        An ExecutorStep that represents the filtered documents.
    """

    if isinstance(encoder_model, str):
        model_path = encoder_model
    elif isinstance(encoder_model, ExecutorStep):
        # Model path is from the previous step in GCSFuse Mount.
        model_path = os.path.join("/opt", encoder_model.name)
    else:
        raise ValueError(f"Invalid encoder_model type: {type(encoder_model)}")

    if isinstance(input_data_path, ExecutorStep):
        input_data_path = output_path_of(input_data_path)

    if inference_config_kwargs is None:
        inference_config_kwargs = default_inference_config_kwargs
    else:
        inference_config_kwargs = {**default_inference_config_kwargs, **inference_config_kwargs}

    inference_config = InferenceConfig(
        input_path=input_data_path,
        output_path=this_output_path(),
        model_name=model_path,
        attribute_name=f"datashop-{experiment_name}",
        **inference_config_kwargs,
    )

    attributes = ExecutorStep(
        name=f"attributes/quality_filtering/datashop/{input_data_name}-{experiment_name}",
        fn=run_inference,
        config=inference_config,
        pip_dependency_groups=[
            "--find-links https://storage.googleapis.com/libtpu-releases/index.html",
            "--find-links https://storage.googleapis.com/libtpu-wheels/index.html",
            "fasttext",
            "datasets",
            "filelock",
            "torch~=2.7.0",
            "torch_xla[tpu]~=2.7.0",
        ],
    )

    return attributes


def default_consolidate(
    attributes: ExecutorStep,
    input_data_path: str | ExecutorStep,
    input_data_name: str,
    experiment_name: str,
    filter_config_kwargs: dict | None = None,
    consolidate_config_kwargs: dict | None = None,
):
    """Runs quality filtering and consolidation on an input dataset given the quality filter model.

    Inputs:
        encoder_model: The model to use for quality filtering.
        input_data_path: The path to the input data, usually a large pretraining corpus to filter.
        input_data_name: The name of the input data used for storage purposes.
        experiment_name: The name of the experiment.
        inference_config_kwargs: Keyword arguments for the inference config which uses the encoder
            model to score documents based on their quality.
        filter_config_kwargs: Keyword arguments for the filter config which is used to filter the
            documents based on their quality.
        consolidate_config_kwargs: Keyword arguments for the consolidate config which is used to
            consolidate the filtered documents.

    Outputs:
        An ExecutorStep that represents the filtered documents.
    """
    if filter_config_kwargs is None:
        filter_config_kwargs = default_consolidate_filter_config_kwargs
    else:
        filter_config_kwargs = {**default_consolidate_filter_config_kwargs, **filter_config_kwargs}

    if consolidate_config_kwargs is None:
        consolidate_config_kwargs = default_consolidate_config_kwargs
    else:
        consolidate_config_kwargs = {**default_consolidate_config_kwargs, **consolidate_config_kwargs}

    filtered_documents = ExecutorStep(
        name=f"documents/quality_filtering/datashop/{input_data_name}-{experiment_name}",
        fn=consolidate,
        config=ConsolidateConfig(
            input_path=input_data_path,
            output_path=this_output_path(),
            filters=[
                FilterConfig(
                    attribute_path=output_path_of(attributes),
                    name=f"datashop-{experiment_name}",
                    **filter_config_kwargs,
                ),
            ],
            **consolidate_config_kwargs,
        ),
        pip_dependency_groups=["ddsketch"],
    )

    return filtered_documents


def _get_anneal_config(candidate_tokenized: TokenizerStep | None, tpu_type: str, experiment_name: str):
    if candidate_tokenized is None:
        return AnnealConfig(
            dataset_config=lm_mixture_data_config(
                components={"dclm": dclm_components_llama3["dclm_baseline"]},
                weights={"dclm": 1.0},
                permutation_type="linear",
            ),
            resources=TpuPodConfig(tpu_type=tpu_type, slice_count=2),
            use_default_validation=True,
        )
    else:
        return AnnealConfig(
            dataset_config=lm_mixture_data_config(
                components={"dclm": dclm_components_llama3["dclm_baseline"], "candidate": candidate_tokenized},
                weights={"dclm": 0.70, "candidate": 0.30},
                permutation_type="linear",
            ),
            resources=TpuPodConfig(tpu_type=tpu_type, slice_count=2),
            use_default_validation=True,
        )


def default_candidate_anneal(documents: ExecutorStep | None, tpu_type: str, experiment_name: str):
    """Evaluates the quality of a set of candidate documents

    To analyze the quality of the filtered dataset, we need to train two models:
    1. A control model that is trained on 100% DCLM-baseline.
    2. A candidate model that is trained on the filtered dataset.

    We then compare the performance of the candidate model to the control model to
    see if there is an improvement. We specify whether we are training the control
    model or the candidate model by passing in `documents` as None.

    Inputs:
        documents: An ExecutorStep that represents the documents to be annealed. If None, we
                will simply train the control model (100% DCLM-baseline).
        tpu_type: The type of TPU to use for training.
        experiment_name: The name of the experiment.

    Outputs:
        An ExecutorStep that represents the final model after annealing.
    """
    if documents is not None:
        candidate_tokenized = default_tokenize(
            name=f"datashop-candidate-{experiment_name}",
            dataset=output_path_of(documents),
            tokenizer=llama3_tokenizer,
        )
        model_name = f"datashop-candidate-{experiment_name}"
    else:
        candidate_tokenized = None
        model_name = "datashop-control"

    anneal_config = _get_anneal_config(candidate_tokenized, tpu_type, experiment_name)

    return default_anneal(
        name=model_name,
        anneal_config=anneal_config,
    )


def default_synthetic_data_generation(
    input_path: ExecutorStep | InputName,
    output_path: str,
    model_name_or_path: str,
    data_generation_template: str,
    input_filetype: str,
    prompt_column: str,
    resource_config: ResourceConfig,
    generated_text_column_name: str = "generated_text",
    engine_kwargs: dict = default_engine_kwargs,
    generation_kwargs: dict = default_generation_kwargs,
) -> ExecutorStep:
    """
    Generates synthetic data using a specified model and prompt template.

    Args:
        input_path (ExecutorStep | InputName): The input data to generate from. This is usually
            either a dataset from Huggingface or a filtered dataset from the datashop.
        model_name_or_path (str): The name or path of the model to use for generation
            (e.g. "meta-llama/Llama-3.1-8B-Instruct")
        data_generation_template (str): The template string used to generate prompts for the model.
            It must include a placeholder for the input data (e.g. "{example}").
        input_filetype (str): The file type of the input data (e.g., "jsonl.zst").
            This is used to read the input data correctly.
        prompt_column (str): The column in the input data to use as the prompt for generation.
        resource_config (ResourceConfig): The resource configuration specifying hardware requirements.
        engine_kwargs (dict, optional): Keyword arguments for the model engine.
            Defaults to default_engine_kwargs.
        generation_kwargs (dict, optional): Keyword arguments for text generation.
            Defaults to default_generation_kwargs.
        output_path (str | None, optional): The output path for the generated data.
            If None, a default path is used.

    Returns:
        (ExecutorStep): An ExecutorStep of the synthetic dataset.
    """
    return ExecutorStep(
        name=output_path,
        fn=run_generation_inference,
        config=TextGenerationInferenceConfig(
            input_path=input_path,
            output_path=this_output_path(),
            model_name=model_name_or_path,
            engine_kwargs=engine_kwargs,
            generation_kwargs=generation_kwargs,
            template=data_generation_template,
            tensor_parallel_size=resource_config.num_tpu,
            prompt_column=prompt_column,
            filetype=input_filetype,
            output_filetype_override="jsonl.gz",
            one_to_one_input_output_mapping=False,
            generated_text_column_name=generated_text_column_name,
            resource_config=resource_config,
            batch_size=512,
        ),
        pip_dependency_groups=["vllm"],
    )
