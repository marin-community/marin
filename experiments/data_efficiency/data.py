import dataclasses

from levanter.data.text import TextLmDatasetFormat

from experiments.defaults import default_download, default_tokenize
from experiments.pretraining_datasets import dclm_baseline, starcoderdata
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

dclm_tokenized = dataclasses.replace(
    default_tokenize(
        name="dclm_baseline",
        dataset=dclm_baseline,
        tokenizer=llama3_tokenizer,
    ).with_output_path("tokenized/dclm_baseline-0206f1/"),
)

starcoderdata_tokenized = default_tokenize(
    name="starcoderdata",
    dataset=starcoderdata,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="content"),
).with_output_path("tokenized/starcoderdata-12f018/")

# self-distill with 300M model trained on 200M tokens (using optimal hparams for an infinite ensemble)
sd0715 = default_download(
    name="raw/data_efficiency/sd0715",
    hf_dataset_id="konwoo/300Me1-200Mdata-4.45Mgen",
    revision="b14955b",
    override_output_path="raw/data_efficiency/sd0715",
)

sd0715_tokenized = default_tokenize(
    name="data_efficiency/sd0715",
    dataset=sd0715,
    tokenizer=llama3_tokenizer,
)

sd0805 = default_download(
    name="raw/data_efficiency/sd0805",
    hf_dataset_id="konwoo/300Me1-200Mdata-16.1Mgen",
    revision="b5d2a92",
    override_output_path="raw/data_efficiency/sd0805",
)

sd0805_tokenized = default_tokenize(
    name="data_efficiency/sd0805",
    dataset=sd0805,
    tokenizer=llama3_tokenizer,
)

# [BUGGED] distilling a 2 ensemble with 300M model trained on 200M tokens (using optimal hparams for an infinite ensemble)
ens2d0715 = default_download(
    name="raw/data_efficiency/ens2d0715",
    hf_dataset_id="konwoo/300Me2-200Mdata-4.1Mgen",
    revision="6f268b6",
    override_output_path="raw/data_efficiency/ens2d0715",
)

ens2d0715_tokenized = default_tokenize(
    name="data_efficiency/ens2d0715",
    dataset=ens2d0715,
    tokenizer=llama3_tokenizer,
)

# distilling a 2 ensemble with 300M model trained on 200M tokens
ens2d0717 = default_download(
    name="raw/data_efficiency/ens2d0717",
    hf_dataset_id="konwoo/300Me2-200Mdata-4.5Mgen",
    revision="79b78ea",
    override_output_path="raw/data_efficiency/ens2d0717",
)

ens2d0717_tokenized = default_tokenize(
    name="data_efficiency/ens2d0717",
    dataset=ens2d0717,
    tokenizer=llama3_tokenizer,
)

ens4d0721 = default_download(
    name="raw/data_efficiency/ens4d0721",
    hf_dataset_id="konwoo/300Me4-200Mdata-4.8Mgen",
    revision="e0342a3",
    override_output_path="raw/data_efficiency/ens4d0721",
)

ens4d0721_tokenized = default_tokenize(
    name="data_efficiency/ens4d0721",
    dataset=ens4d0721,
    tokenizer=llama3_tokenizer,
)

ens4x0728 = default_download(
    name="raw/data_efficiency/ens4x0728",
    hf_dataset_id="konwoo/300Me1-200Mdata-mix4seeds",
    revision="e7c195b",
    override_output_path="raw/data_efficiency/ens4x0728",
)

ens4x0728_tokenized = default_tokenize(
    name="data_efficiency/ens4x0728",
    dataset=ens4x0728,
    tokenizer=llama3_tokenizer,
)

ens8x0730 = default_download(
    name="raw/data_efficiency/ens8x0730",
    hf_dataset_id="konwoo/300Me1-200Mdata-mix8seeds",
    revision="d16b474",
    override_output_path="raw/data_efficiency/ens8x0730",
)

ens8x0730_tokenized = default_tokenize(
    name="data_efficiency/ens8x0730",
    dataset=ens8x0730,
    tokenizer=llama3_tokenizer,
)

octothinker_megamath = default_download(
    name="raw/octothinker-megamath",
    hf_dataset_id="OctoThinker/MegaMath-Web-Pro-Max",
    revision="b5129b6",
    override_output_path="raw/octothinker-megamath",
)

octothinker_megamath_tokenized = default_tokenize(
    name="octothinker-megamath",
    dataset=octothinker_megamath,
    tokenizer=llama3_tokenizer,
)

data_dict = {
    "dclm": dclm_tokenized,
    "code": starcoderdata_tokenized,
    "sd0715": sd0715_tokenized,
    "sd0805": sd0805_tokenized,
    "ens2d0715": ens2d0715_tokenized,
    "ens2d0717": ens2d0717_tokenized,
    "ens4d0721": ens4d0721_tokenized,
    "ens4x0728": ens4x0728_tokenized,
    "ens8x0730": ens8x0730_tokenized,
    "octo": octothinker_megamath_tokenized,
}

