import ray

from experiments.medu_inference import medu_inference
from marin.classifiers.bert.train_classifier import (
    ScriptArguments,
    train_classifier_distributed,
)
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path
from operations.transform.medu.convert_classifier_dataset import (
    ConvertClassifierDatasetConfig,
    convert_classifier_dataset_func,
)

convert_to_classifier_dataset_format = ExecutorStep(
    name="documents/fineweb-economics-llama-70b-annotations",
    fn=convert_classifier_dataset_func,
    config=ConvertClassifierDatasetConfig(
        input_path=output_path_of(medu_inference),
        output_path=this_output_path(),
    ),
)


# NOTE(chris): BTW NEED PIP_DEPS accelerate>=0.26.0 set in the commandline
@ray.remote(resources={"TPU": 8, "TPU-v6e-8-head": 1})
def train_classifier_tpu():
    train_classifier_distributed(
        ScriptArguments(
            train_dataset="gs://marin-us-east5/documents/fineweb-economics-llama-70b-annotations-497aa8/56_000000_000000.jsonl.gz",
            num_labels=5,
            target_column="label",
            # TODO(chris): change later
            output_dir="~/.cache",
            remove_unused_columns=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            report_to="wandb",
            logging_steps=10,
            tpu_num_cores=1,
        )
    )


if __name__ == "__main__":
    # executor_main(steps=[convert_to_classifier_dataset_format])
    x = ray.get(train_classifier_tpu.remote())
