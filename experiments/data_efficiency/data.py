import dataclasses

from levanter.data.text import TextLmDatasetFormat

from experiments.defaults import default_download, default_tokenize
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

dclm_tokenized = dclm_components_llama3["dclm_baseline"]

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

sbp_cpr16 = default_download(
    name="raw/data_efficiency/sbp_cpr16",
    hf_dataset_id="konwoo/dclm-164k-sbp-pf-cpr16-ml1024",
    revision="1d64554",
    override_output_path="raw/data_efficiency/sbp_cpr16",
)

sbp_cpr16_tokenized = default_tokenize(
    name="data_efficiency/sbp_cpr16",
    dataset=sbp_cpr16,
    tokenizer=llama3_tokenizer,
)

## Multi News

multi_news_long_context_train = default_download(
    name="raw/data_efficiency/multi_news_long_context_train",
    hf_dataset_id="kothasuhas/multi_news_long_context_train",
    revision="226e5a2",
    override_output_path="raw/data_efficiency/multi_news_long_context_train",
)

multi_news_long_context_train_tokenized = default_tokenize(
    name="data_efficiency/multi_news_long_context_train",
    dataset=multi_news_long_context_train,
    tokenizer=llama3_tokenizer,
)

multi_news_long_context_val = default_download(
    name="raw/data_efficiency/multi_news_long_context_val",
    hf_dataset_id="kothasuhas/multi_news_long_context_validation",
    revision="0d9afb4",
    override_output_path="raw/data_efficiency/multi_news_long_context_val",
)

multi_news_long_context_val_tokenized = default_tokenize(
    name="data_efficiency/multi_news_long_context_is_val",
    dataset=multi_news_long_context_val,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

multi_news_split_train = default_download(
    name="raw/data_efficiency/multi_news_split_train",
    hf_dataset_id="kothasuhas/multi_news_split_shuffled_train",
    revision="68f4515",
    override_output_path="raw/data_efficiency/multi_news_split_train",
)

multi_news_split_train_tokenized = default_tokenize(
    name="data_efficiency/multi_news_split_train",
    dataset=multi_news_split_train,
    tokenizer=llama3_tokenizer,
)

multi_news_split_val = default_download(
    name="raw/data_efficiency/multi_news_split_val",
    hf_dataset_id="kothasuhas/multi_news_split_shuffled_validation",
    revision="f1c1dc0",
    override_output_path="raw/data_efficiency/multi_news_split_val",
)

multi_news_split_val_tokenized = default_tokenize(
    name="data_efficiency/multi_news_split_is_val",
    dataset=multi_news_split_val,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

## DCLM with explicit shuffling or sorting

dclm_200m_shuffled_train = default_download(
    name="raw/data_efficiency/dclm_200m_shuffled_train",
    hf_dataset_id="konwoo/dclm-train-164k-shuffled",
    revision="ed2ba77",
    override_output_path="raw/data_efficiency/dclm_200m_shuffled_train",
)

dclm_200m_shuffled_tokenized_train = default_tokenize(
    name="data_efficiency/dclm_200m_shuffled_train",
    dataset=dclm_200m_shuffled_train,
    tokenizer=llama3_tokenizer,
)

dclm_200m_shuffled_validation = default_download(
    name="raw/data_efficiency/dclm_200m_shuffled_val",
    hf_dataset_id="konwoo/dclm-val-1k-shuffled",
    revision="3af1ab8",
    override_output_path="raw/data_efficiency/dclm_200m_shuffled_val",
)

dclm_200m_shuffled_tokenized_validation = default_tokenize(
    name="data_efficiency/dclm_200m_shuffled_is_val",
    dataset=dclm_200m_shuffled_validation,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

dclm_200m_sorted_train = default_download(
    name="raw/data_efficiency/dclm_200m_sorted_train",
    hf_dataset_id="konwoo/dclm-train-164k-sorted",
    revision="ccf737b",
    override_output_path="raw/data_efficiency/dclm_200m_sorted_train",
)

dclm_200m_sorted_tokenized_train = default_tokenize(
    name="data_efficiency/dclm_200m_sorted_train",
    dataset=dclm_200m_sorted_train,
    tokenizer=llama3_tokenizer,
)

dclm_200m_sorted_validation = default_download(
    name="raw/data_efficiency/dclm_200m_sorted_val",
    hf_dataset_id="konwoo/dclm-val-1k-sorted",
    revision="56e44c5",
    override_output_path="raw/data_efficiency/dclm_200m_sorted_val",
)

dclm_200m_sorted_tokenized_validation = default_tokenize(
    name="data_efficiency/dclm_200m_sorted_is_val",
    dataset=dclm_200m_sorted_validation,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

dclm_200m_tsp_train = default_download(
    name="raw/data_efficiency/dclm_200m_tsp_train",
    hf_dataset_id="konwoo/dclm-train-164k-tsp",
    revision="8d03be0",
    override_output_path="raw/data_efficiency/dclm_200m_tsp_train",
)

dclm_200m_tsp_tokenized_train = default_tokenize(
    name="data_efficiency/dclm_200m_tsp_train",
    dataset=dclm_200m_tsp_train,
    tokenizer=llama3_tokenizer,
)

dclm_200m_tsp_validation = default_download(
    name="raw/data_efficiency/dclm_200m_tsp_val",
    hf_dataset_id="konwoo/dclm-val-1k-tsp-s50-d20",
    revision="99657fd",
    override_output_path="raw/data_efficiency/dclm_200m_tsp_val",
)

dclm_200m_tsp_tokenized_validation = default_tokenize(
    name="data_efficiency/dclm_200m_tsp_val",
    dataset=dclm_200m_tsp_validation,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

dclm_200m_tsp_validation_shuffled = default_download(
    name="raw/data_efficiency/dclm_200m_tsp_val_shuffled",
    hf_dataset_id="konwoo/dclm-val-1k-tsp-s50-d20-shuffled",
    revision="c93c775",
    override_output_path="raw/data_efficiency/dclm_200m_tsp_val_shuffled",
)

dclm_200m_tsp_tokenized_validation_shuffled = default_tokenize(
    name="data_efficiency/dclm_200m_tsp_val_shuffled",
    dataset=dclm_200m_tsp_validation_shuffled,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

## 10x dense TSP data

dclm_train_164k_t10x_tsp_d20 = default_download(
    name="raw/data_efficiency/dclm_train_164k_t10x_tsp_d20",
    hf_dataset_id="konwoo/dclm-train-164k-t10x-tsp-d20",
    revision="a60ffef",
    override_output_path="raw/data_efficiency/dclm_train_164k_t10x_tsp_d20",
)

dclm_train_164k_t10x_tsp_d20_tokenized = default_tokenize(
    name="data_efficiency/dclm_train_164k_t10x_tsp_d20",
    dataset=dclm_train_164k_t10x_tsp_d20,
    tokenizer=llama3_tokenizer,
)

dclm_train_164k_t10x_tsp_d20_shuffled = default_download(
    name="raw/data_efficiency/dclm_train_164k_t10x_tsp_d20_shuffled",
    hf_dataset_id="konwoo/dclm-train-164k-t10x-tsp-d20-shuffled",
    revision="1adda8f",
    override_output_path="raw/data_efficiency/dclm_train_164k_t10x_tsp_d20_shuffled",
)

dclm_train_164k_t10x_tsp_d20_shuffled_tokenized = default_tokenize(
    name="data_efficiency/dclm_train_164k_t10x_tsp_d20_shuffled",
    dataset=dclm_train_164k_t10x_tsp_d20_shuffled,
    tokenizer=llama3_tokenizer,
)

dclm_val_1k_t10x_tsp_d20 = default_download(
    name="raw/data_efficiency/dclm_val_1k_t10x_tsp_d20",
    hf_dataset_id="konwoo/dclm-val-1k-t10x-tsp-d20",
    revision="2d0cdf2",
    override_output_path="raw/data_efficiency/dclm_val_1k_t10x_tsp_d20",
)

dclm_val_1k_t10x_tsp_d20_tokenized = default_tokenize(
    name="data_efficiency/dclm_val_1k_t10x_tsp_d20",
    dataset=dclm_val_1k_t10x_tsp_d20,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

dclm_val_1k_t10x_tsp_d20_shuffled = default_download(
    name="raw/data_efficiency/dclm_val_1k_t10x_tsp_d20_shuffled",
    hf_dataset_id="konwoo/dclm-val-1k-t10x-tsp-d20-shuffled",
    revision="5113f9e",
    override_output_path="raw/data_efficiency/dclm_val_1k_t10x_tsp_d20_shuffled",
)

dclm_val_1k_t10x_tsp_d20_shuffled_tokenized = default_tokenize(
    name="data_efficiency/dclm_val_1k_t10x_tsp_d20_shuffled",
    dataset=dclm_val_1k_t10x_tsp_d20_shuffled,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

dclm_train_164k_t10x_shuffled = default_download(
    name="raw/data_efficiency/dclm_train_164k_t10x_shuffled",
    hf_dataset_id="konwoo/dclm-train-164k-t10x-shuffled",
    revision="e8b08b4",
    override_output_path="raw/data_efficiency/dclm_train_164k_t10x_shuffled",
)

dclm_train_164k_t10x_shuffled_tokenized = default_tokenize(
    name="data_efficiency/dclm_train_164k_t10x_shuffled",
    dataset=dclm_train_164k_t10x_shuffled,
    tokenizer=llama3_tokenizer,
)

## 1.64M doc TSP 

dclm_train_1m640k_tsp = default_download(
    name="raw/data_efficiency/dclm_train_1m640k_tsp",
    hf_dataset_id="konwoo/dclm-train-1.64m-tsp",
    revision="86ebf05",
    override_output_path="raw/data_efficiency/dclm_train_1m640k_tsp",
)

dclm_train_1m640k_tsp_tokenized = default_tokenize(
    name="data_efficiency/dclm_train_1m640k_tsp",
    dataset=dclm_train_1m640k_tsp,
    tokenizer=llama3_tokenizer,
)

dclm_train_1m640k_tsp_shuffled = default_download(
    name="raw/data_efficiency/dclm_train_1m640k_tsp_shuffled",
    hf_dataset_id="konwoo/dclm-train-1.64m-shuffled",
    revision="5877fda",
    override_output_path="raw/data_efficiency/dclm_train_1m640k_tsp_shuffled",
)

dclm_train_1m640k_tsp_shuffled_tokenized = default_tokenize(
    name="data_efficiency/dclm_train_1m640k_tsp_shuffled",
    dataset=dclm_train_1m640k_tsp_shuffled,
    tokenizer=llama3_tokenizer,
)

dclm_val_1k_normal = default_download(
    name="raw/data_efficiency/dclm_val_1k_normal",
    hf_dataset_id="konwoo/dclm-val-1k-normal-shuffled",
    revision="7f9cdc0",
    override_output_path="raw/data_efficiency/dclm_val_1k_normal",
)

dclm_val_1k_normal_tokenized = default_tokenize(
    name="data_efficiency/dclm_val_1k_normal",
    dataset=dclm_val_1k_normal,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

## 1.3M doc TSP 

dclm_train_1m300k_tsp = default_download(
    name="raw/data_efficiency/dclm_train_1m300k_tsp",
    hf_dataset_id="konwoo/dclm-train-1.3m-tsp",
    revision="857fb3f",
    override_output_path="raw/data_efficiency/dclm_train_1m300k_tsp",
)

dclm_train_1m300k_tsp_tokenized = default_tokenize(
    name="data_efficiency/dclm_train_1m300k_tsp",
    dataset=dclm_train_1m300k_tsp,
    tokenizer=llama3_tokenizer,
)

dclm_train_1m300k_tsp_shuffled = default_download(
    name="raw/data_efficiency/dclm_train_1m300k_tsp_shuffled",
    hf_dataset_id="konwoo/dclm-train-1.3m-shuffle",
    revision="785b137",
    override_output_path="raw/data_efficiency/dclm_train_1m300k_tsp_shuffled",
)

dclm_train_1m300k_tsp_shuffled_tokenized = default_tokenize(
    name="data_efficiency/dclm_train_1m300k_tsp_shuffled",
    dataset=dclm_train_1m300k_tsp_shuffled,
    tokenizer=llama3_tokenizer,
)

data_dict = {
    "dclm": dclm_tokenized,
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
    "sbp_cpr16": sbp_cpr16_tokenized,
    # multi news
    "mn_lc": multi_news_long_context_train_tokenized,
    "mn_lc_val": multi_news_long_context_val_tokenized,
    "mn_split": multi_news_split_train_tokenized,
    "mn_split_val": multi_news_split_val_tokenized,
    # dclm with explicit shuffling or sorting
    "dclm_200m_shuffled": dclm_200m_shuffled_tokenized_train,
    "dclm_200m_shuffled_val": dclm_200m_shuffled_tokenized_validation,
    "dclm_200m_sorted": dclm_200m_sorted_tokenized_train,
    "dclm_200m_sorted_val": dclm_200m_sorted_tokenized_validation,
    # tsp based dclm data
    "dclm_tsp": dclm_200m_tsp_tokenized_train,
    "dclm_tsp_val": dclm_200m_tsp_tokenized_validation,
    "dclm_tsp_val_shuffled": dclm_200m_tsp_tokenized_validation_shuffled,
    "dclm_shuffled": dclm_200m_shuffled_tokenized_train,
    # 10x dense TSP data
    "dc_t10x": dclm_train_164k_t10x_tsp_d20_tokenized,
    "dc_t10x_shuffled": dclm_train_164k_t10x_tsp_d20_shuffled_tokenized,
    "dc_t10x_val": dclm_val_1k_t10x_tsp_d20_tokenized,
    "dc_t10x_val_shuffled": dclm_val_1k_t10x_tsp_d20_shuffled_tokenized,
    "dc_shuffled": dclm_train_164k_t10x_shuffled_tokenized,
    # 1.64M doc TSP data
    "dc_1m": dclm_train_1m640k_tsp_tokenized,
    "dc_1m_mix": dclm_train_1m640k_tsp_shuffled_tokenized,
    "dc_1k_val_normal": dclm_val_1k_normal_tokenized,
    # 1.3M doc TSP data
    "dc_1_3m": dclm_train_1m300k_tsp_tokenized,
    "dc_1_3m_mix": dclm_train_1m300k_tsp_shuffled_tokenized,
}
