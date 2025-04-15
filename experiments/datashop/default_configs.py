from transformers import AutoTokenizer

from marin.classifiers.hf.train_classifier import HFTrainingConfig
from marin.execution.executor import this_output_path
from marin.processing.classification.config.inference_config import RuntimeConfig, TaskConfig

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

default_medu_config_kwargs = {
    "model_name": quality_teacher_model,
    "engine_kwargs": default_engine_kwargs,
    "generation_kwargs": default_generation_kwargs,
    "filetype": "jsonl.zst",
    "output_filetype_override": "jsonl.gz",
}

default_text_generation_config_kwargs = {
    "model_name": quality_teacher_model,
    "engine_kwargs": default_engine_kwargs,
    "generation_kwargs": default_generation_kwargs,
    "num_instances": (1, 128),
    "save_templated_prompt": False,
    "prompt_column": "text",
    "filetype": "jsonl.zst",
    "output_filetype_override": "jsonl.gz",
}

default_dataset_output_processor_config_kwargs = {
    "processor_type": "medu",
}

default_quality_filter_train_config_kwargs = {
    "training_config": HFTrainingConfig(
        output_dir=this_output_path(),
        num_labels=1,
        target_column="label",
        max_length=512,
        train_size=0.9,
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
    )
}

default_inference_config_kwargs = {
    "model_type": "gte",
    "runtime": RuntimeConfig(
        memory_limit_gb=12,
        resources={"TPU": 1},
    ),
    "task": TaskConfig(max_in_flight=500),
    "filetype": "jsonl.zst",
    "classifier_kwargs": {"max_length": 512},
}

default_consolidate_filter_config_kwargs = {
    "type": "classify",
    "label": "score",
    "keep_fraction": 0.10,
}

default_consolidate_config_kwargs = {
    "filetype": "jsonl.zst",
    "ray_memory_limit_gb": 12,
}
