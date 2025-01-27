import os

import ray

from experiments.models import GCS_FUSE_MOUNT_PATH, LOCAL_PREFIX
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.classification.classifier import AutoClassifier
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, TaskConfig
from marin.processing.classification.inference import run_inference

PRESET_PROMPT = """
Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren’t too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.
The extract: {example}
After examining the extract: - Briefly justify your total score, up to 100 words. - Conclude with the score using the format: “Score: <total points>”
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria.
"""  # noqa: E501, RUF001

# ALSO BROKEN if you import!
# from experiments.instruction_datasets import get_directory_friendly_dataset_name


def get_directory_friendly_dataset_name(model_name: str) -> str:
    return model_name.replace("/", "--").replace(".", "-").replace("#", "-")


def get_model_local_path(model_name: str) -> str:
    model_name = get_directory_friendly_dataset_name(model_name)
    return os.path.join(LOCAL_PREFIX, GCS_FUSE_MOUNT_PATH, model_name)


@ray.remote(resources={"TPU-v6e-8-head": 1})
def test():
    prompts = ["What is 2 + 2?", "What is the capital of France?"]

    sampling_params = {
        "temperature": 0.1,
        "n": 1,
        "max_tokens": 256,
    }

    engine_kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
    }

    # llm_provider = LLMProvider(
    #     model_name=get_model_local_path("meta-llama/Llama-3.1-8B-Instruct"),
    #     generation_kwargs=sampling_params,
    #     engine_kwargs=engine_kwargs,
    # )

    classifier = AutoClassifier.from_model_path(
        model_name_or_path=get_model_local_path("meta-llama/Llama-3.1-8B-Instruct"),
        generation_kwargs=sampling_params,
        engine_kwargs=engine_kwargs,
        attribute_name="test",
        preset_prompt=PRESET_PROMPT,
        model_type="vllm",
    )

    batch = {"text": prompts}
    batch = classifier(batch)
    print(batch)


def test_vllm_inference():
    # input_data_path = "gs://marin-us-east5/documents/chris-test"
    input_data_path = "gs://marin-us-central2/documents/fineweb-small-resiliparse-preserve-formatting-e8c6ec/md/CC-MAIN-2024-18/000_00000"

    input_basename = os.path.basename(os.path.normpath(input_data_path))
    inference_step = ExecutorStep(
        name="attributes/chris-vllm-fineweb-shard-8b-multi",
        fn=run_inference,
        config=InferenceConfig(
            input_path=input_data_path,
            output_path=this_output_path(input_basename),
            model_name=get_model_local_path("meta-llama/Llama-3.1-8B-Instruct"),
            model_type="vllm",
            attribute_name=versioned("chris-vllm-quality"),
            runtime=RuntimeConfig(
                memory_limit_gb=120,
                resources={"TPU-v6e-8-head": 1},
            ),
            # NOTE(CHRIS): change back to higher number, just for debug
            task=TaskConfig(max_in_flight=1),
            kwargs={
                "preset_prompt": PRESET_PROMPT,
                "generation_kwargs": {
                    "temperature": 0.1,
                    "n": 1,
                    "max_tokens": 512,
                },
                "engine_kwargs": {
                    "enforce_eager": True,
                    "tensor_parallel_size": 8,
                    "max_model_len": 8192,
                },
            },
        ),
        pip_dependency_groups=["fasttext", "datasets", "filelock"],
    )

    return [inference_step]


@ray.remote
def test_llm_call():
    import os
    import shutil

    import vllm.envs as envs
    from vllm import LLM

    # Remove VLLM XLA CACHE PATH
    # os.remove(envs.VLLM_XLA_CACHE_PATH)
    if os.path.exists(envs.VLLM_XLA_CACHE_PATH):
        print(f"Removing {envs.VLLM_XLA_CACHE_PATH}")
        shutil.rmtree(envs.VLLM_XLA_CACHE_PATH)

        import glob

        for file in glob.glob(os.path.join(envs.VLLM_XLA_CACHE_PATH, "*")):
            print(file)
    else:
        print(f"No {envs.VLLM_XLA_CACHE_PATH} found")

    llm = LLM(model=get_model_local_path("meta-llama/Llama-3.1-8B-Instruct"), enforce_eager=True, max_model_len=8192)

    output = llm.generate(["What is 2 + 2?"])

    return output


@ray.remote
def coordinator():
    first_call = ray.get(test_llm_call.options(resources={"TPU-v6e-8-head": 1}).remote())
    print(f"First call: {first_call}")

    second_call = ray.get(test_llm_call.options(resources={"TPU-v6e-8-head": 1}).remote())
    print(f"Second call: {second_call}")


@ray.remote
def collect_env():
    import subprocess

    # Run the wget command to download collect_env.py
    wget_cmd = "wget https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py"
    subprocess.run(wget_cmd, shell=True, check=True)

    # Run collect_env.py and capture output
    collect_env_cmd = "python collect_env.py"
    result = subprocess.run(collect_env_cmd, shell=True, capture_output=True, text=True)

    # Print the output
    print("collect_env.py output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


if __name__ == "__main__":
    bar = ray.get(coordinator.remote())
    # bar = ray.get(collect_env.remote())
