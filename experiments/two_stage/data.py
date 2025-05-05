from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.midtraining_datasets import finemath_3_plus_tokenized

dolma_components = tokenize_dolma_steps()

c4_tokenized = dolma_components["dolma/c4"]
# dclm_tokenized = # TODO
# slimpajama_tokenized = # TODO

# python_tokenized = dolma_components["python"]
# cpp_tokenized = # TODO
starcoder_tokenized = dolma_components["dolma/starcoder"]
wiki_tokenized = dolma_components["dolma/wiki"]
flan_tokenized = dolma_components["dolma/flan"]

data_dict = {
    "c4": c4_tokenized,
    # "dclm": dclm_tokenized,
    # "spj": slimpajama_tokenized,
    # "python": python_tokenized,
    # "cpp": cpp_tokenized,
    "starcoder": starcoder_tokenized,
    "wiki": wiki_tokenized,
    "flan": flan_tokenized,
    "finemath": finemath_3_plus_tokenized,
}

