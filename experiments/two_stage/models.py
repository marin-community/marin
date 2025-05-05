import dataclasses
from levanter.models.llama import LlamaConfig
from experiments.llama import llama_150m

llama_150m_4096_config = dataclasses.replace(
    llama_150m,
    seq_len=4096,
)

model_dict = {
    "150m4k": llama_150m_4096_config,
}