import os

from experiments.medu.medu_train import medu_econ_classifier
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, TaskConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import run_inference

# NOTE(chris): We copy the direct path to the DCLM dataset because it is a large file that we do not want to redownload
# especially since we only want a single shard of the data. Ideally, this should use the output_path_of function
# to get the path, but we do not have DCLM in the us-east5 region yet.
input_data_path = "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_01_of_10"
model_path = os.path.join("/opt", medu_econ_classifier.name)
print(model_path)
medu_inference = ExecutorStep(
    name="attributes/quality_filtering/economic-bert/dclm-global-shard-01-of-10",
    fn=run_inference,
    config=InferenceConfig(
        input_path=input_data_path,
        output_path=this_output_path(),
        model_name=model_path,
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

if __name__ == "__main__":
    executor_main([medu_inference, medu_consolidate])
