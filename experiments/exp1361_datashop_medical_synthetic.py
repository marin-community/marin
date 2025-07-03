import copy

from experiments.datashop.default_configs import default_engine_kwargs, default_generation_kwargs
from experiments.datashop.defaults import default_synthetic_data_generation
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import AnnealConfig, default_anneal, default_tokenize
from experiments.evals.resource_configs import TPU_V6E_8_STRICT_PACK
from experiments.exp1361_datashop_medical import datashop_runner
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig

qa_rephrase_prompt = """
A chat between a curious patient and an educated doctor.
The doctor gives helpful, detailed, and polite answers to the questions.
Convert the following paragraph into a conversational format with
multiple tags of "Question:" followed by "Answer:".

{example}
"""

engine_kwargs = copy.deepcopy(default_engine_kwargs)
engine_kwargs["tensor_parallel_size"] = 8
generation_kwargs = copy.deepcopy(default_generation_kwargs)
generation_kwargs["max_tokens"] = 1024

synthetic_medical_data = default_synthetic_data_generation(
    # datashop_runner.filtered_documents,
    output_path_of(datashop_runner.filtered_documents),
    "documents/datashop-medical-qa-whole",
    "meta-llama/Llama-3.1-8B-Instruct",
    qa_rephrase_prompt,
    "jsonl.zst",
    "text",
    checkpoint_id_column={"metadata": "WARC-Record-ID"},
    engine_kwargs=engine_kwargs,
    generation_kwargs=generation_kwargs,
    resource_config=TPU_V6E_8_STRICT_PACK,
)

synthetic_medical_data_tokenized = default_tokenize(
    "datashop-medical-qa-whole",
    synthetic_medical_data,
    llama3_tokenizer,
)

# total tokens was around 3B, let's epoch around 10B tokens
# so 10B * 1/ 0.3 = ~33B tokens
anneal_model = default_anneal(
    "datshop-medical-qa",
    AnnealConfig(
        dataset_config=lm_mixture_data_config(
            {
                "synthetic_medical_data": synthetic_medical_data_tokenized,
                "dclm": dclm_components_llama3["dclm_baseline"],
            },
            {
                "synthetic_medical_data": 0.3,
                "dclm": 0.7,
            },
        ),
        num_anneal_training_tokens=33_000_000_000,
        resources=TpuPodConfig(tpu_type="v6e-128", slice_count=2),
    ),
)

if __name__ == "__main__":
    executor_main([anneal_model])
