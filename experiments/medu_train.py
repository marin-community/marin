import ray

from experiments.medu_inference import medu_inference
from marin.classifiers.bert.train_classifier import (
    ScriptArguments,
    # train_classifier_distributed,
    train_classifier,
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


NUM_TPU_DEVICES = 1


# NOTE(chris): BTW NEED PIP_DEPS accelerate>=0.26.0 set in the commandline
@ray.remote(resources={"TPU": NUM_TPU_DEVICES, "TPU-v6e-8-head": 1}, runtime_env={"pip": ["accelerate>=0.26.0"]})
def train_classifier_tpu():
    import os

    os.mkdir("/opt/gcsfuse_mount/economic-bert")

    train_classifier(
        rank=0,
        args=ScriptArguments(
            train_dataset="gs://marin-us-east5/documents/fineweb-economics-llama-70b-annotations-497aa8/56_000000_000000.jsonl.gz",
            num_labels=5,
            target_column="label",
            # TODO(chris): change later
            output_dir="/opt/gcsfuse_mount/economic-bert",
            remove_unused_columns=False,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=16,
            report_to="wandb",
            logging_steps=10,
            max_length=512,  # TODO(CHRIS): Can change later
            # tpu_num_cores=NUM_TPU_DEVICES,
        ),
    )


if __name__ == "__main__":
    # executor_main(steps=[convert_to_classifier_dataset_format])
    x = ray.get(train_classifier_tpu.remote())
