from experiments.models import get_model_local_path
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf

huggingface_dataset_id = "HuggingFaceH4/MATH-500"

dataset_name = huggingface_dataset_id.replace("/", "--").replace(".", "-").replace("#", "-")
download_dataset_step = ExecutorStep(
    name=f"raw/{dataset_name}-retry-2",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=huggingface_dataset_id,
        revision=versioned("ff5b202"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

inference_step = ExecutorStep(
    name="documents/synthetic_data_llama_8b",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=output_path_of(download_dataset_step),
        output_path=this_output_path(),
        model_name=get_model_local_path("meta-llama/Llama-3.1-8B-Instruct"),
        engine_kwargs={
            "max_model_len": 8192,
            "enforce_eager": True,
            "tensor_parallel_size": 8,
        },
        generation_kwargs={
            "temperature": 0.8,
            "max_tokens": 512,
        },
        template="You will be given a problem. Please reason step by step, \
            and put your final answer within \boxed{{}}:\n{example}",
        tensor_parallel_size=8,
        prompt_column="problem",
        filetype="jsonl",
    ),
)

steps = [download_dataset_step, inference_step]

if __name__ == "__main__":
    executor_main(steps)
