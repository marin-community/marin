"""Synthetic data generation for the GSM8K dataset in the style of MIND.

Inspiration from the Olmo-2 paper where they utilize the MIND rewrite technique to generate
synthetic math datasets from existing datasets.
"""

from transformers import AutoTokenizer

from experiments.models import get_model_local_path, llama_3_1_8b_instruct, llama_3_3_70b_instruct
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from marin.utils import get_directory_friendly_name
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf

huggingface_dataset_id = "openai/gsm8k"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tensor_parallel_size = 1

dataset_name = get_directory_friendly_name(huggingface_dataset_id)
gsm8k = ExecutorStep(
    name=f"raw/{dataset_name}",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=huggingface_dataset_id,
        revision=versioned("e53f048"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
).cd("main/train-00000-of-00001.parquet")

tokenizer = AutoTokenizer.from_pretrained(model_name)
mind_rewrite_student_teacher = ExecutorStep(
    name="documents/gsm8k-llama8b-mind/student_teacher",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=gsm8k,
        output_path=this_output_path(),
        model_name=get_model_local_path(llama_3_1_8b_instruct),
        engine_kwargs={
            "max_model_len": 8192,
            "enforce_eager": False,
            "tensor_parallel_size": tensor_parallel_size,
        },
        generation_kwargs={
            "temperature": 0.8,
            "max_tokens": 1024,
            "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        },
        template=(
            "{example}\n\nConvert the context above as a multi-turn discussions between a teacher and a student. "
            "The student has questions about the context and the teacher solves each of them step-by-step. "
            "Make sure that their discussions strictly adhere to the context above and remains faithful to "
            "information in the context. Please DO NOT add any new information/reference other than the context."
        ),
        tensor_parallel_size=tensor_parallel_size,
        prompt_column="question",
        filetype="parquet",
        one_to_one_input_output_mapping=False,
    ),
)

mind_rewrite_problem_solving = ExecutorStep(
    name="documents/gsm8k-llama8b-mind/problem_solving",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=gsm8k,
        output_path=this_output_path(),
        model_name=get_model_local_path(llama_3_1_8b_instruct),
        engine_kwargs={
            "max_model_len": 8192,
            "enforce_eager": False,
            "tensor_parallel_size": tensor_parallel_size,
        },
        generation_kwargs={
            "temperature": 0.8,
            "max_tokens": 1024,
            "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        },
        template=(
            "{example}\n\nConvert the context above as a multi-turn problem-solving conversation where participants "
            "analyze challenges or scenarios presented in the content and brainstorm solutions within the context "
            "of the provided material, avoiding speculation or unrelated discussions. Make sure that their conversation "
            "strictly adhere to the context above and remains faithful to information in the context. "
            "Please DO NOT add any new information/reference other than the context."
        ),
        tensor_parallel_size=tensor_parallel_size,
        prompt_column="question",
        filetype="parquet",
        one_to_one_input_output_mapping=False,
    ),
)

tensor_parallel_size_70b = 8
mind_rewrite_student_teacher_70b = ExecutorStep(
    name="documents/gsm8k-llama70b-mind/student_teacher",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=gsm8k,
        output_path=this_output_path(),
        model_name=get_model_local_path(llama_3_3_70b_instruct),
        engine_kwargs={
            "max_model_len": 8192,
            "enforce_eager": False,
            "tensor_parallel_size": tensor_parallel_size_70b,
        },
        generation_kwargs={
            "temperature": 0.8,
            "max_tokens": 1024,
            "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        },
        template=(
            "{example}\n\nConvert the context above as a multi-turn discussions between a teacher and a student. "
            "The student has questions about the context and the teacher solves each of them step-by-step. "
            "Make sure that their discussions strictly adhere to the context above and remains faithful to "
            "information in the context. Please DO NOT add any new information/reference other than the context."
        ),
        tensor_parallel_size=tensor_parallel_size_70b,
        prompt_column="question",
        filetype="parquet",
        one_to_one_input_output_mapping=False,
    ),
)

mind_rewrite_problem_solving_70b = ExecutorStep(
    name="documents/gsm8k-llama70b-mind/problem_solving",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=gsm8k,
        output_path=this_output_path(),
        model_name=get_model_local_path(llama_3_3_70b_instruct),
        engine_kwargs={
            "max_model_len": 8192,
            "enforce_eager": False,
            "tensor_parallel_size": tensor_parallel_size_70b,
        },
        generation_kwargs={
            "temperature": 0.8,
            "max_tokens": 1024,
            "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        },
        template=(
            "{example}\n\nConvert the context above as a multi-turn problem-solving conversation where participants "
            "analyze challenges or scenarios presented in the content and brainstorm solutions within the context "
            "of the provided material, avoiding speculation or unrelated discussions. Make sure that their conversation "
            "strictly adhere to the context above and remains faithful to information in the context. "
            "Please DO NOT add any new information/reference other than the context."
        ),
        tensor_parallel_size=tensor_parallel_size_70b,
        prompt_column="question",
        filetype="parquet",
        one_to_one_input_output_mapping=False,
    ),
)

steps = [
    mind_rewrite_student_teacher,
    mind_rewrite_problem_solving,
    mind_rewrite_student_teacher_70b,
    mind_rewrite_problem_solving_70b,
]

if __name__ == "__main__":
    executor_main(steps)
