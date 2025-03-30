import os

from transformers import AutoTokenizer

from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal, default_tokenize
from experiments.evals.resource_configs import ResourceConfig
from experiments.llama import llama3_tokenizer
from marin.classifiers.hf.launch_ray_training import LaunchConfig, launch_training_with_ray
from marin.classifiers.hf.train_classifier import HFTrainingConfig
from marin.core.runtime import TaskConfig
from marin.datashop.pipeline import (
    MEDU_BENCHMARK_DESCRIPTION_PROMPT_FILENAME,
    MEDUPipelineConfig,
    run_data_filter_prompt_generation_pipeline,
    run_medu_dataset_sampling_pipeline,
)
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path
from marin.generation.dataset import DatasetOutputProcessorConfig
from marin.generation.inference import TextGenerationInferenceConfig
from marin.generation.inference import run_inference as run_generation_inference
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import run_inference
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config
from operations.download.filesystem.transfer import TransferConfig, transfer_files

# Quality teacher model that is used to label the initial data pool that will then be used
# to train the quality filter model.
# TODO(chris): Make this a parameter and support other models. As of now, we make Llama-70B-Instruct the
# default quality teacher model.
quality_teacher_model = "/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(quality_teacher_model)


def default_label(
    documents_to_be_labeled: str | ExecutorStep,
    targeted_documents: list[list[str] | str],
    experiment_name: str,
    resource_config: ResourceConfig,
    data_filter_prompt: str | None = None,
):
    """Label a set of documents with an LLM given some targeted documents.

    Inputs:
        documents_to_be_labeled: Input path to documents to be labeled.
        targeted_documents: A list of strings or filepaths of documents that is being targeted for labeling.
        experiment_name: The name of the experiment.
        data_filter_prompt: The user's prompt for the annotator model.

    Outputs:
        An ExecutorStep that represents the labeled documents. Each document is .jsonl.gz file with
        each row containing an additional key "generated_text" that contains the LLM's response.
    """
    if isinstance(documents_to_be_labeled, ExecutorStep):
        documents_to_be_labeled = output_path_of(documents_to_be_labeled)

    assert (data_filter_prompt is not None and targeted_documents == []) or (
        data_filter_prompt is None and targeted_documents != []
    ), "Must provide either a data filter prompt or targeted documents, but not both"

    default_engine_kwargs = {
        "tensor_parallel_size": resource_config.num_tpu,
        "enforce_eager": False,
        "max_model_len": 8192,
    }

    default_generation_kwargs = {
        "temperature": 0.1,
        "max_tokens": 1024,
        "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    }

    # If the user does not provide a data filter prompt, we generate one using the MEDU pipeline.
    if data_filter_prompt is None:
        data_filter_prompt = ExecutorStep(
            name=f"documents/datashop-prompts/{experiment_name}",
            fn=run_data_filter_prompt_generation_pipeline,
            config=MEDUPipelineConfig(
                model_name=quality_teacher_model,
                corpus_contents=targeted_documents,
                input_path=documents_to_be_labeled,
                output_path=this_output_path(),
                engine_kwargs=default_engine_kwargs,
                generation_kwargs=default_generation_kwargs,
                filetype="jsonl.zst",
                output_filetype_override="jsonl.gz",
                resource_config=resource_config,
            ),
        )
        data_filter_prompt = output_path_of(data_filter_prompt, MEDU_BENCHMARK_DESCRIPTION_PROMPT_FILENAME)
        template_type = "file"
    else:
        template_type = "string"

    # NOTE(chris): Assuming we are filtering from a jsonl.zst file such as DCLM.
    return ExecutorStep(
        name=f"documents/datashop-labels/{experiment_name}",
        fn=run_generation_inference,
        config=TextGenerationInferenceConfig(
            model_name=quality_teacher_model,
            input_path=documents_to_be_labeled,
            output_path=this_output_path(),
            engine_kwargs=default_engine_kwargs,
            generation_kwargs=default_generation_kwargs,
            template=data_filter_prompt,
            template_type=template_type,
            num_instances=(1, 128),
            tensor_parallel_size=resource_config.num_tpu,
            save_templated_prompt=False,
            prompt_column="text",
            filetype="jsonl.zst",
            output_filetype_override="jsonl.gz",
            resource_config=resource_config,
        ),
        override_output_path=f"documents/datashop-labels/{experiment_name}",
    )


def default_train_quality_model(labeled_documents: ExecutorStep, experiment_name: str, resource_config: ResourceConfig):
    """Train a quality filter model based on the set of labeled documents.

    Inputs:
        labeled_documents: An ExecutorStep that represents the labeled documents.
        experiment_name: The name of the experiment.
        resource_config: The resource config to use for training the quality filter model.

    Outputs:
        An ExecutorStep that represents the quality filter model.
    """
    dataset = ExecutorStep(
        name=f"documents/datashop-datasets/{experiment_name}",
        fn=run_medu_dataset_sampling_pipeline,
        config=DatasetOutputProcessorConfig(
            input_path=output_path_of(labeled_documents),
            output_path=this_output_path(),
        ),
    ).cd("sampled")

    max_length = 512
    datashop_classifier_remote = ExecutorStep(
        name=f"classifiers/datashop-bert/{experiment_name}",
        fn=launch_training_with_ray,
        config=LaunchConfig(
            training_config=HFTrainingConfig(
                train_dataset=dataset,
                output_dir=this_output_path(),
                num_labels=1,
                target_column="label",
                max_length=max_length,
                train_size=0.9,
                eval_steps=100,
                save_steps=100,
                logging_steps=10,
                run_name=f"datashop-classifier-{experiment_name}-max-length-{max_length}",
                tpu_num_cores=resource_config.num_tpu,
            ),
            resource_config=resource_config,
        ),
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
    )

    return datashop_classifier


def default_quality_filter_and_consolidate(
    encoder_model: ExecutorStep | str, input_data_path: str | ExecutorStep, input_data_name: str, experiment_name: str
):
    """Runs quality filtering and consolidation on an input dataset given the quality filter model.

    Inputs:
        encoder_model: The model to use for quality filtering.
        input_data_path: The path to the input data, usually a large pretraining corpus to filter.
        input_data_name: The name of the input data used for storage purposes.
        experiment_name: The name of the experiment.

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

    attributes = ExecutorStep(
        name=f"attributes/quality_filtering/datashop/{input_data_name}-{experiment_name}",
        fn=run_inference,
        config=InferenceConfig(
            input_path=input_data_path,
            output_path=this_output_path(),
            model_name=model_path,
            model_type="gte",
            attribute_name=f"datashop-{experiment_name}",
            runtime=RuntimeConfig(
                memory_limit_gb=12,
                resources={"TPU": 1},
            ),
            task=TaskConfig(max_in_flight=500),
            filetype="jsonl.zst",
            classifier_kwargs={"max_length": 512},
        ),
        pip_dependency_groups=[
            "--find-links https://storage.googleapis.com/libtpu-releases/index.html",
            "--find-links https://storage.googleapis.com/libtpu-wheels/index.html",
            "fasttext",
            "datasets",
            "filelock",
            "torch~=2.6.0",
            "torch_xla[tpu]~=2.6.0",
        ],
    )

    filtered_documents = ExecutorStep(
        name=f"documents/quality_filtering/datashop/{input_data_name}-{experiment_name}",
        fn=consolidate,
        config=ConsolidateConfig(
            input_path=input_data_path,
            output_path=this_output_path(),
            filters=[
                FilterConfig(
                    type="classify",
                    attribute_path=output_path_of(attributes),
                    name=f"datashop-{experiment_name}",
                    label="score",
                    keep_fraction=0.10,
                ),
            ],
            ray_memory_limit_gb=12,
            filetype="jsonl.zst",
        ),
        pip_dependency_groups=["ddsketch"],
    )

    return filtered_documents


def _get_anneal_config(candidate_tokenized: TokenizerStep | None, tpu_type: str, experiment_name: str):
    if candidate_tokenized is None:
        return AnnealConfig(
            dataset_config=lm_mixture_data_config(
                components={"dclm": dclm_components_llama3["dclm_baseline"]}, weights={"dclm": 1.0}
            ),
            tpu_type=tpu_type,
        )
    else:
        return AnnealConfig(
            dataset_config=lm_mixture_data_config(
                components={"dclm": dclm_components_llama3["dclm_baseline"], "candidate": candidate_tokenized},
                weights={"dclm": 0.70, "candidate": 0.30},
            ),
            tpu_type=tpu_type,
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
