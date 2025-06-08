import copy

from experiments.datashop.default_configs import default_engine_kwargs, default_generation_kwargs
from experiments.datashop.defaults import default_synthetic_data_generation
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from experiments.exp1361_datashop_medical import datashop_runner
from marin.execution.executor import executor_main

qa_rephrase_prompt = """
A chat between a curious patient and an educated doctor.
The doctor gives helpful, detailed, and polite answers to the questions.
Convert the following paragraph into a conversational format with
multiple tags of "Question:" followed by "Answer:".

{example}
"""

engine_kwargs = copy.deepcopy(default_engine_kwargs)
engine_kwargs["tensor_parallel_size"] = 1
generation_kwargs = copy.deepcopy(default_generation_kwargs)
generation_kwargs["max_tokens"] = 1024

synthetic_medical_data = default_synthetic_data_generation(
    datashop_runner.filtered_documents,
    "documents/datashop-medical-qa",
    "meta-llama/Llama-3.1-8B-Instruct",
    qa_rephrase_prompt,
    "jsonl.zst",
    "text",
    checkpoint_id_column={"metadata": "WARC-Record-ID"},
    engine_kwargs=engine_kwargs,
    generation_kwargs=generation_kwargs,
    resource_config=SINGLE_TPU_V6E_8,
)

if __name__ == "__main__":
    executor_main([synthetic_medical_data])
