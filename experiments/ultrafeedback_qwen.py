# download openbmb/UltraFeedback

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig, download

ultrafeedback_download = ExecutorStep(
    name="HuggingFaceH4/ultrafeedback_binarized",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceH4/ultrafeedback_binarized",
        revision=versioned("3949bf5"),
        gcs_output_path=this_output_path(),
        wait_for_completion=False,
    ),
    override_output_path="raw/huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/3949bf5",
)


if __name__ == "__main__":
    executor_main(steps=[])
