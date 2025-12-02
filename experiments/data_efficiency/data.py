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

dclm_200m_train = default_download(
    name="raw/data_efficiency/dclm_200m_train",
    hf_dataset_id="konwoo/dclm-164k-docs-train",
    revision="c4f5716",
    override_output_path="raw/data_efficiency/dclm_200m_train",
)

dclm_200m_tokenized_train = default_tokenize(
    name="data_efficiency/dclm_200m_train",
    dataset=dclm_200m_train,
    tokenizer=llama3_tokenizer,
)

dclm_200m_val = default_download(
    name="raw/data_efficiency/dclm_200m_val",
    hf_dataset_id="konwoo/dclm-164k-docs-val",
    revision="fb8ee7b",
    override_output_path="raw/data_efficiency/dclm_200m_val",
)

dclm_200m_tokenized_val = default_tokenize(
    name="data_efficiency/dclm_200m_val",
    dataset=dclm_200m_val,
    tokenizer=llama3_tokenizer,
)

hq_cpr16 = default_download(
    name="raw/data_efficiency/hq_cpr16",
    hf_dataset_id="konwoo/dclm-164k-instruct-hq-cpr16-ml512-train",
    revision="981b0e3",
    override_output_path="raw/data_efficiency/hq_cpr16",
)

hq_cpr16_tokenized = default_tokenize(
    name="data_efficiency/hq_cpr16",
    dataset=hq_cpr16,
    tokenizer=llama3_tokenizer,
)

sd_cpr16 = default_download(
    name="raw/data_efficiency/sd_cpr16",
    hf_dataset_id="konwoo/dclm-164k-300m-raw-cpr16-ml1024",
    revision="a938788",
    override_output_path="raw/data_efficiency/sd_cpr16",
)

sd_cpr16_tokenized = default_tokenize(
    name="data_efficiency/sd_cpr16",
    dataset=sd_cpr16,
    tokenizer=llama3_tokenizer,
)


sd_cpr200 = default_download(
    name="raw/data_efficiency/sd_cpr200",
    hf_dataset_id="konwoo/dclm-164k-300m-cpr200-ml1024",
    revision="818adca",
    override_output_path="raw/data_efficiency/sd_cpr200",
)

sd_cpr200_tokenized = default_tokenize(
    name="data_efficiency/sd_cpr200",
    dataset=sd_cpr200,
    tokenizer=llama3_tokenizer,
)

synth_mixed_cpr16 = default_download(
    name="raw/data_efficiency/synth_mixed_cpr16",
    hf_dataset_id="konwoo/dclm-164k-old-cpr16-combined",
    revision="e3ae5bc",
    override_output_path="raw/data_efficiency/synth_mixed_cpr16",
)

synth_mixed_cpr16_tokenized = default_tokenize(
    name="data_efficiency/synth_mixed_cpr16",
    dataset=synth_mixed_cpr16,
    tokenizer=llama3_tokenizer,
)

sd_cpr200_new = default_download(
    name="raw/data_efficiency/sd_cpr200_new",
    hf_dataset_id="konwoo/dclm-164k-new-300m-raw-cpr200-ml1024",
    revision="fa7bffb",
    override_output_path="raw/data_efficiency/sd_cpr200_new",
)

sd_cpr200_new_tokenized = default_tokenize(
    name="data_efficiency/sd_cpr200_new",
    dataset=sd_cpr200_new,
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
    "dclm_200m": dclm_200m_tokenized_train,
    "dclm_200m_val": dclm_200m_tokenized_val,
    "hq_cpr16": hq_cpr16_tokenized,
    "sd_cpr16": sd_cpr16_tokenized,
    "sd_cpr200": sd_cpr200_tokenized,
    "symx_c16": synth_mixed_cpr16_tokenized,
    "sdn_c200": sd_cpr200_new_tokenized,
}

