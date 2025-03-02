from experiments.medu.medu_label import dataset_sampling_step
from marin.classifiers.hf.train_classifier import HFTrainingConfig, train_classifier_distributed
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from operations.download.gcs.model import DownloadFromGCSConfig, download_model_from_gcs

medu_econ_classifier_remote = ExecutorStep(
    name="classifiers/medu-bert-test",
    fn=train_classifier_distributed,
    config=HFTrainingConfig(
        train_dataset=output_path_of(dataset_sampling_step),
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
