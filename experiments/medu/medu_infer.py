from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, TaskConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import run_inference
from marin.processing.tokenize import TokenizeConfig, tokenize

input_data_path = "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_01_of_10"
medu_inference = ExecutorStep(
    name="attributes/quality_filtering/economic-bert/dclm-global-shard-01-of-10",
    fn=run_inference,
    config=InferenceConfig(
        input_path=input_data_path,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/economic-bert-large-8/checkpoint-651",
        model_type="gte",
        attribute_name="economic-bert",
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

medu_consolidate = ExecutorStep(
    name="documents/quality_filtering/dclm-global-shard-01-of-10-medu-economics-3plus",
    fn=consolidate,
    config=ConsolidateConfig(
        input_path=input_data_path,
        output_path=this_output_path(),
        filters=[
            FilterConfig(
                type="classify",
                attribute_path=output_path_of(medu_inference),
                name="economic-bert",
                label="int_score",
                threshold=3,
            ),
        ],
        ray_memory_limit_gb=12,
        filetype="jsonl.zst",
    ),
    pip_dependency_groups=["ddsketch"],
)

BASE_MEDU_PATH = "gs://marin-us-east5/documents/quality_filtering/dclm-global-shard-01-of-10-medu-economics-3plus-bbb96b"
FILE_PATTERN = "**/*.jsonl.zst"
medu_economics_tokenized = ExecutorStep(
    name="tokenized/medu-economics-3plus",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[f"{BASE_MEDU_PATH}/{FILE_PATTERN}"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=llama3_tokenizer,
    ),
    pip_dependency_groups=["tokenize_train"],
)

if __name__ == "__main__":
    executor_main([medu_inference, medu_consolidate, medu_economics_tokenized])
