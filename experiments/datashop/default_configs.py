from transformers import AutoTokenizer

from marin.classifiers.hf.launch_ray_training import LaunchConfig
from marin.classifiers.hf.train_classifier import HFTrainingConfig
from marin.datashop.pipeline import MEDUPipelineConfig
from marin.execution.executor import this_output_path
from marin.generation.dataset import DatasetOutputProcessorConfig
from marin.generation.inference import TextGenerationInferenceConfig
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, TaskConfig

quality_teacher_model = "/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(quality_teacher_model)

default_engine_kwargs = {
    "tensor_parallel_size": 8,
    "enforce_eager": False,
    "max_model_len": 8192,
}

default_generation_kwargs = {
    "temperature": 0.1,
    "max_tokens": 1024,
    "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
}

default_medu_config = (
    MEDUPipelineConfig(
        model_name=quality_teacher_model,
        # corpus_contents=targeted_documents,
        # input_path=documents_to_be_labeled,
        output_path=this_output_path(),
        engine_kwargs=default_engine_kwargs,
        generation_kwargs=default_generation_kwargs,
        filetype="jsonl.zst",
        output_filetype_override="jsonl.gz",
        # resource_config=resource_config,
    ),
)

default_text_generation_config = TextGenerationInferenceConfig(
    model_name=quality_teacher_model,
    # input_path=documents_to_be_labeled,
    output_path=this_output_path(),
    engine_kwargs=default_engine_kwargs,
    generation_kwargs=default_generation_kwargs,
    # template=data_filter_prompt,
    # template_path=data_filter_prompt_path,
    num_instances=(1, 128),
    # tensor_parallel_size=resource_config.num_tpu,
    save_templated_prompt=False,
    prompt_column="text",
    filetype="jsonl.zst",
    output_filetype_override="jsonl.gz",
    # resource_config=resource_config,
)

default_dataset_output_processor_config = DatasetOutputProcessorConfig(
    # input_path=output_path_of(labeled_documents),
    output_path=this_output_path(),
    processor_type="medu",
)

default_quality_filter_train_config = LaunchConfig(
    training_config=HFTrainingConfig(
        output_dir=this_output_path(),
        num_labels=1,
        target_column="label",
        max_length=512,
        train_size=0.9,
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
    ),
)

default_inference_config = InferenceConfig(
    # input_path=input_data_path,
    output_path=this_output_path(),
    # model_name=model_path,
    model_type="gte",
    # attribute_name=f"datashop-{experiment_name}",
    runtime=RuntimeConfig(
        memory_limit_gb=12,
        resources={"TPU": 1},
    ),
    task=TaskConfig(max_in_flight=500),
    filetype="jsonl.zst",
    classifier_kwargs={"max_length": 512},
)
