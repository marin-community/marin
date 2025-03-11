import os

from transformers import AutoTokenizer

from experiments.evals.resource_configs import ResourceConfig
from marin.classifiers.hf.launch_ray_training import LaunchConfig, launch_training_with_ray
from marin.classifiers.hf.train_classifier import HFTrainingConfig
from marin.core.runtime import TaskConfig
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path
from marin.generation.dataset import DatasetOutputProcessorConfig
from marin.generation.medu import MEDUPipelineConfig, run_medu_dataset_sampling_pipeline, run_medu_labeling_pipeline
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import run_inference
from operations.download.gcs.model import DownloadFromGCSConfig, download_model_from_gcs

model_name = "/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tensor_parallel_size = 8


def default_label(
    documents_to_be_labeled: str | ExecutorStep, targeted_documents: list[list[str] | str], experiment_name: str
):
    if isinstance(documents_to_be_labeled, ExecutorStep):
        documents_to_be_labeled = output_path_of(documents_to_be_labeled)

    # NOTE(chris): Assuming we are filtering from a jsonl.zst file such as DCLM.
    return ExecutorStep(
        name=f"documents/medu-labels/{experiment_name}",
        fn=run_medu_labeling_pipeline,
        config=MEDUPipelineConfig(
            model_name=model_name,
            dev_sets=targeted_documents,
            input_path=documents_to_be_labeled,
            tensor_parallel_size=tensor_parallel_size,
            output_path=this_output_path(),
            engine_kwargs={"tensor_parallel_size": tensor_parallel_size, "enforce_eager": False, "max_model_len": 8192},
            generation_kwargs={
                "temperature": 0.1,
                "max_tokens": 1024,
                "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            },
            filetype="jsonl.zst",
            output_filetype_override="jsonl.gz",
        ),
        override_output_path=f"documents/medu-labels/{experiment_name}",
    )


def default_quality_filter_model(labeled_documents: ExecutorStep, experiment_name: str, tpu_type: str = "TPU-v4-8"):
    dataset = ExecutorStep(
        name=f"documents/medu-datasets/{experiment_name}",
        fn=run_medu_dataset_sampling_pipeline,
        config=DatasetOutputProcessorConfig(
            input_path=output_path_of(labeled_documents),
            output_path=this_output_path(),
        ),
    ).cd("sampled")

    # TODO(chris): Change to v4-8 when we aren't running inference on v4-16 anymore?
    tpu_type_to_resource_config = {
        "TPU-v4-16": ResourceConfig(num_tpu=4, tpu_type="TPU-v4-16"),
        "TPU-v6e-8": ResourceConfig(num_tpu=8, tpu_type="TPU-v6e-8"),
    }
    resource_config = tpu_type_to_resource_config[tpu_type]

    max_length = 512
    medu_classifier_remote = ExecutorStep(
        name=f"classifiers/medu-bert/{experiment_name}",
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
                run_name=f"medu-classifier-{experiment_name}-max-length-{max_length}",
                tpu_num_cores=resource_config.num_tpu,
            ),
            resource_config=resource_config,
        ),
    )
    # Download the model locally to GCSFuse mount path for inference
    medu_classifier = ExecutorStep(
        name=f"gcsfuse_mount/medu-models/{experiment_name}-classifier",
        fn=download_model_from_gcs,
        config=DownloadFromGCSConfig(
            gcs_path=output_path_of(medu_classifier_remote),
            destination_path=this_output_path(),
        ),
        override_output_path=f"gcsfuse_mount/medu-models/{experiment_name}-classifier",
    )

    return medu_classifier


def default_quality_filter_and_consolidate(
    encoder_model: ExecutorStep | str, input_data_path: str | ExecutorStep, input_data_name: str, experiment_name: str
):
    if isinstance(encoder_model, str):
        model_path = encoder_model
    else:
        model_path = os.path.join("/opt", encoder_model.name)

    if isinstance(input_data_path, ExecutorStep):
        input_data_path = output_path_of(input_data_path)

    attributes = ExecutorStep(
        name=f"attributes/quality_filtering/medu/{input_data_name}-{experiment_name}",
        fn=run_inference,
        config=InferenceConfig(
            input_path=input_data_path,
            output_path=this_output_path(),
            model_name=model_path,
            model_type="gte",
            attribute_name=f"medu-{experiment_name}",
            runtime=RuntimeConfig(
                memory_limit_gb=12,
                resources={"TPU": 1},
            ),
            task=TaskConfig(max_in_flight=500),
            filetype="jsonl.zst",
            classifier_kwargs={"max_length": 512},
        ),
        pip_dependency_groups=["fasttext", "datasets", "filelock"],
    )

    filtered_documents = ExecutorStep(
        name=f"documents/quality_filtering/medu/{input_data_name}-{experiment_name}",
        fn=consolidate,
        config=ConsolidateConfig(
            input_path=input_data_path,
            output_path=this_output_path(),
            filters=[
                FilterConfig(
                    type="classify",
                    attribute_path=output_path_of(attributes),
                    name=f"medu-{experiment_name}",
                    label="int_score",
                    threshold=3,
                ),
            ],
            ray_memory_limit_gb=12,
            filetype="jsonl.zst",
        ),
        pip_dependency_groups=["ddsketch"],
    )

    return filtered_documents
