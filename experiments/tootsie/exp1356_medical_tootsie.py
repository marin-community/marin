from experiments.anneal_config import AnnealConfig
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal, default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.midtraining_datasets import lavita_medical_qa_datasets
from experiments.tootsie.exp600_tootsie import phoenix_phase4_checkpoint_for_phase5
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig
from marin.transform.medical.lavita_to_dolma import LavitaToDolmaConfig, convert_lavita_split_to_dolma

lavita_pubmed = ExecutorStep(
    name="documents/lavita_pubmed",
    fn=convert_lavita_split_to_dolma,
    config=LavitaToDolmaConfig(input_path=lavita_medical_qa_datasets, output_path=this_output_path(), split="pubmed-qa"),
)

lavita_medmcqa = ExecutorStep(
    name="documents/lavita_medmcqa",
    fn=convert_lavita_split_to_dolma,
    config=LavitaToDolmaConfig(input_path=lavita_medical_qa_datasets, output_path=this_output_path(), split="medmcqa"),
)

lavita_allprocessed = ExecutorStep(
    name="documents/lavita_allprocessed",
    fn=convert_lavita_split_to_dolma,
    config=LavitaToDolmaConfig(
        input_path=lavita_medical_qa_datasets,
        output_path=this_output_path(),
        split="all-processed",
    ),
)

lavita_allprocessed_tokenized = default_tokenize(
    "tokenized/lavita_allprocessed",
    lavita_allprocessed,
    tokenizer=llama3_tokenizer,
)

lavita_medmcqa_tokenized = default_tokenize(
    "tokenized/lavita_medmcqa",
    lavita_medmcqa,
    tokenizer=llama3_tokenizer,
)

lavita_pubmedqa_tokenized = default_tokenize(
    "tokenized/lavita_pubmedqa",
    lavita_pubmed,
    tokenizer=llama3_tokenizer,
)

pubmed_qa_tokens = 78_993_593
allprocessed_tokens = 58_717_739
medmcqa_tokens = 30_779_801
all_medical_tokens = pubmed_qa_tokens + allprocessed_tokens + medmcqa_tokens
medical_token_proportion = 0.3
dclm_token_proportion = 1 - medical_token_proportion
num_anneal_tokens = int(all_medical_tokens * 4 / medical_token_proportion)
anneal_config = AnnealConfig(
    initialize_from_checkpoint_path=phoenix_phase4_checkpoint_for_phase5,
    dataset_config=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "lavita_pubmedqa": lavita_pubmedqa_tokenized,
            "lavita_allprocessed": lavita_allprocessed_tokenized,
            "lavita_medmcqa": lavita_medmcqa_tokenized,
        },
        weights={
            "dclm": dclm_token_proportion,
            "lavita_pubmedqa": medical_token_proportion * pubmed_qa_tokens / all_medical_tokens,
            "lavita_allprocessed": medical_token_proportion * allprocessed_tokens / all_medical_tokens,
            "lavita_medmcqa": medical_token_proportion * medmcqa_tokens / all_medical_tokens,
        },
    ),
    resources=TpuPodConfig(tpu_type="v6e-128", slice_count=2),
)
medical_tootsie_anneal = default_anneal(
    name="checkpoints/medical_tootsie",
    anneal_config=anneal_config,
)


if __name__ == "__main__":
    executor_main(
        [
            medical_tootsie_anneal,
        ]
    )
