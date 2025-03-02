import os

from experiments.medu.medu_label import DATASET_SAMPLING_STEP_OUTPUT_PATH
from marin.classifiers.hf.train_classifier import HFTrainingConfig, train_classifier_distributed
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from operations.download.gcs.model import DownloadFromGCSConfig, download_model_from_gcs

dataset_path = os.path.join(os.getenv("MARIN_PREFIX"), DATASET_SAMPLING_STEP_OUTPUT_PATH, "sampled")
classifier_output_path = os.path.join(os.getenv("MARIN_PREFIX"), "classifiers", "test-medu-dclm-classifier-training")

medu_econ_classifier_remote = ExecutorStep(
    name="classifiers/medu-bert-test",
    fn=train_classifier_distributed,
    config=HFTrainingConfig(
        train_dataset=dataset_path,
        output_dir=this_output_path(),
        num_labels=1,
        target_column="label",
        tpu_num_cores=8,
        max_length=512,
        train_size=0.9,
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
    ),
)

# Download the model locally to GCSFuse mount path for inference
medu_econ_classifier = ExecutorStep(
    name="gcsfuse_mount/models/medu-econ-classifier",
    fn=download_model_from_gcs,
    config=DownloadFromGCSConfig(
        gcs_path=output_path_of(medu_econ_classifier_remote),
        destination_path=this_output_path(),
    ),
    override_output_path="gcsfuse_mount/models/medu-econ-classifier",
)


if __name__ == "__main__":
    executor_main([medu_econ_classifier, medu_econ_classifier_remote])
