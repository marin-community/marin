from experiments.defaults import default_download, default_tokenize
from experiments.llama import llama3_tokenizer

finemath_commit_hash = "8f233cf"
finemath = default_download(
    name="raw/finemath",
    hf_dataset_id="HuggingFaceTB/finemath",
    revision=finemath_commit_hash,
)

finemath_3_plus = finemath.cd("finemath-3plus")
finemath_3_plus_tokenized = default_tokenize(
    name="finemath_3_plus",
    dataset=finemath_3_plus,
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/finemath_3_plus-a26b0f/")
