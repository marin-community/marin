import os

import ray

from experiments.medu.medu_label import DATASET_SAMPLING_STEP_OUTPUT_PATH
from marin.classifiers.hf.train_classifier import HFTrainingConfig, train_classifier_distributed

dataset_path = os.path.join(os.getenv("MARIN_PREFIX"), DATASET_SAMPLING_STEP_OUTPUT_PATH, "sampled")
classifier_output_path = os.path.join(os.getenv("MARIN_PREFIX"), "classifiers", "test-medu-dclm-classifier-training")


# NOTE(chris): We cannot combine the executor steps here because there is an issue with the TPU topology when running
# with the Executor.
@ray.remote(resources={"TPU": 8, "TPU-v6e-8-head": 1})
def run_train():
    train_classifier_distributed(
        HFTrainingConfig(
            train_dataset=dataset_path,
            output_dir=classifier_output_path,
            num_labels=1,
            target_column="label",
            tpu_num_cores=8,
            max_length=512,
            train_size=0.9,
            eval_steps=100,
            save_steps=100,
            logging_steps=10,
        )
    )


if __name__ == "__main__":
    future = ray.get(run_train.remote())
