from experiments.defaults import default_download, default_tokenize
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.llama import llama3_tokenizer

dclm_tokenized = dclm_components_llama3["dclm_baseline"]

def tokenize_data_efficiency_dataset(name, hf_dataset_id, revision, *, is_validation=False, tokenized_name=None):
    downloaded = default_download(
        name=f"raw/data_efficiency/{name}",
        hf_dataset_id=hf_dataset_id,
        revision=revision,
        override_output_path=f"raw/data_efficiency/{name}"
    )

    tokenized = default_tokenize(
        name=f"data_efficiency/{tokenized_name}" if tokenized_name else f"data_efficiency/{name}",
        dataset=downloaded,
        tokenizer=llama3_tokenizer,
        is_validation=is_validation,
    )

    return tokenized

# self-distill with 300M model trained on 200M tokens (using optimal hparams for an infinite ensemble)
sd0715_tokenized = tokenize_data_efficiency_dataset(
    "sd0715",
    "konwoo/300Me1-200Mdata-4.45Mgen",
    "b14955b",
)

sd0805_tokenized = tokenize_data_efficiency_dataset(
    "sd0805",
    "konwoo/300Me1-200Mdata-16.1Mgen",
    "b5d2a92",
)

# [BUGGED] distilling a 2 ensemble with 300M model trained on 200M tokens (using optimal hparams for an infinite ensemble)
ens2d0715_tokenized = tokenize_data_efficiency_dataset(
    "ens2d0715",
    "konwoo/300Me2-200Mdata-4.1Mgen",
    "6f268b6",
)

# distilling a 2 ensemble with 300M model trained on 200M tokens
ens2d0717_tokenized = tokenize_data_efficiency_dataset(
    "ens2d0717",
    "konwoo/300Me2-200Mdata-4.5Mgen",
    "79b78ea",
)

ens4d0721_tokenized = tokenize_data_efficiency_dataset(
    "ens4d0721",
    "konwoo/300Me4-200Mdata-4.8Mgen",
    "e0342a3",
)

ens4x0728_tokenized = tokenize_data_efficiency_dataset(
    "ens4x0728",
    "konwoo/300Me1-200Mdata-mix4seeds",
    "e7c195b",
)

ens8x0730_tokenized = tokenize_data_efficiency_dataset(
    "ens8x0730",
    "konwoo/300Me1-200Mdata-mix8seeds",
    "d16b474",
)

octothinker_megamath_tokenized = tokenize_data_efficiency_dataset(
    "octothinker-megamath",
    "OctoThinker/MegaMath-Web-Pro-Max",
    "b5129b6",
)

dclm_200m_tokenized_train = tokenize_data_efficiency_dataset(
    "dclm_200m_train",
    "konwoo/dclm-164k-docs-train",
    "c4f5716",
)

dclm_200m_tokenized_val = tokenize_data_efficiency_dataset(
    "dclm_200m_val",
    "konwoo/dclm-164k-docs-val",
    "fb8ee7b",
)

hq_cpr16_tokenized = tokenize_data_efficiency_dataset(
    "hq_cpr16",
    "konwoo/dclm-164k-instruct-hq-cpr16-ml512-train",
    "981b0e3",
)

sd_cpr16_tokenized = tokenize_data_efficiency_dataset(
    "sd_cpr16",
    "konwoo/dclm-164k-300m-raw-cpr16-ml1024",
    "a938788",
)

sd_cpr200_tokenized = tokenize_data_efficiency_dataset(
    "sd_cpr200",
    "konwoo/dclm-164k-300m-cpr200-ml1024",
    "818adca",
)

synth_mixed_cpr16_tokenized = tokenize_data_efficiency_dataset(
    "synth_mixed_cpr16",
    "konwoo/dclm-164k-old-cpr16-combined",
    "e3ae5bc",
)

sd_cpr200_new_tokenized = tokenize_data_efficiency_dataset(
    "sd_cpr200_new",
    "konwoo/dclm-164k-new-300m-raw-cpr200-ml1024",
    "fa7bffb",
)

sbp_cpr16_tokenized = tokenize_data_efficiency_dataset(
    "sbp_cpr16",
    "konwoo/dclm-164k-sbp-pf-cpr16-ml1024",
    "1d64554",
)

## Multi News

multi_news_long_context_train_tokenized = tokenize_data_efficiency_dataset(
    "multi_news_long_context_train",
    "kothasuhas/multi_news_long_context_train",
    "226e5a2",
)

multi_news_long_context_val_tokenized = tokenize_data_efficiency_dataset(
    "multi_news_long_context_val",
    "kothasuhas/multi_news_long_context_validation",
    "0d9afb4",
    tokenized_name="multi_news_long_context_is_val",
    is_validation=True,
)

multi_news_split_train_tokenized = tokenize_data_efficiency_dataset(
    "multi_news_split_train",
    "kothasuhas/multi_news_split_shuffled_train",
    "68f4515",
)

multi_news_split_val_tokenized = tokenize_data_efficiency_dataset(
    "multi_news_split_val",
    "kothasuhas/multi_news_split_shuffled_validation",
    "f1c1dc0",
    tokenized_name="multi_news_split_is_val",
    is_validation=True,
)

## DCLM with explicit shuffling or sorting

dclm_200m_shuffled_tokenized_train = tokenize_data_efficiency_dataset(
    "dclm_200m_shuffled_train",
    "konwoo/dclm-train-164k-shuffled",
    "ed2ba77",
)

dclm_200m_shuffled_tokenized_validation = tokenize_data_efficiency_dataset(
    "dclm_200m_shuffled_val",
    "konwoo/dclm-val-1k-shuffled",
    "3af1ab8",
    tokenized_name="dclm_200m_shuffled_is_val",
    is_validation=True,
)

dclm_200m_sorted_tokenized_train = tokenize_data_efficiency_dataset(
    "dclm_200m_sorted_train",
    "konwoo/dclm-train-164k-sorted",
    "ccf737b",
)

dclm_200m_sorted_tokenized_validation = tokenize_data_efficiency_dataset(
    "dclm_200m_sorted_val",
    "konwoo/dclm-val-1k-sorted",
    "56e44c5",
    tokenized_name="dclm_200m_sorted_is_val",
    is_validation=True,
)

dclm_200m_tsp_tokenized_train = tokenize_data_efficiency_dataset(
    "dclm_200m_tsp_train",
    "konwoo/dclm-train-164k-tsp",
    "8d03be0",
)

dclm_200m_tsp_tokenized_validation = tokenize_data_efficiency_dataset(
    "dclm_200m_tsp_val",
    "konwoo/dclm-val-1k-tsp-s50-d20",
    "99657fd",
    is_validation=True,
)

dclm_200m_tsp_tokenized_validation_shuffled = tokenize_data_efficiency_dataset(
    "dclm_200m_tsp_val_shuffled",
    "konwoo/dclm-val-1k-tsp-s50-d20-shuffled",
    "c93c775",
    is_validation=True,
)

## 10x dense TSP data

dclm_train_164k_t10x_tsp_d20_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t10x_tsp_d20",
    "konwoo/dclm-train-164k-t10x-tsp-d20",
    "a60ffef",
)

dclm_train_164k_t10x_tsp_d20_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t10x_tsp_d20_shuffled",
    "konwoo/dclm-train-164k-t10x-tsp-d20-shuffled",
    "1adda8f",
)

dclm_val_1k_t10x_tsp_d20_tokenized = tokenize_data_efficiency_dataset(
    "dclm_val_1k_t10x_tsp_d20",
    "konwoo/dclm-val-1k-t10x-tsp-d20",
    "2d0cdf2",
    is_validation=True,
)

dclm_val_1k_t10x_tsp_d20_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_val_1k_t10x_tsp_d20_shuffled",
    "konwoo/dclm-val-1k-t10x-tsp-d20-shuffled",
    "5113f9e",
    is_validation=True,
)

dclm_train_164k_t10x_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t10x_shuffled",
    "konwoo/dclm-train-164k-t10x-shuffled",
    "e8b08b4",
)

# 40x dense TSP data

dclm_train_164k_t40x_tsp_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t40x_tsp",
    "konwoo/dclm-40x-tsp-train",
    "3df9749",
)

dclm_train_164k_t40x_tsp_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t40x_tsp_shuffled",
    "konwoo/dclm-40x-tsp-train-shuffled",
    "eff3412",
)

dclm_val_1k_t40x_tsp_tokenized = tokenize_data_efficiency_dataset(
    "dclm_val_1k_t40x_tsp",
    "konwoo/dclm-40x-tsp-val",
    "71666ae",
    is_validation=True,
)

dclm_val_1k_t40x_tsp_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_val_1k_t40x_tsp_shuffled",
    "konwoo/dclm-40x-tsp-val-shuffled",
    "b1ab43d",
    is_validation=True,
)

## 100x dense TSP data

dclm_train_164k_t100x_tsp_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t100x_tsp",
    "konwoo/dclm-100x-tsp-train",
    "7ee6e7c",
)

dclm_train_164k_t100x_tsp_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t100x_tsp_shuffled",
    "konwoo/dclm-100x-tsp-train-shuffled",
    "6e8055f",
)

dclm_val_1k_t100x_tsp_tokenized = tokenize_data_efficiency_dataset(
    "dclm_val_1k_t100x_tsp",
    "konwoo/dclm-100x-tsp-val",
    "2f609cb",
    is_validation=True,
)

dclm_val_1k_t100x_tsp_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_val_1k_t100x_tsp_shuffled",
    "konwoo/dclm-100x-tsp-val-shuffled",
    "728dd4f",
    is_validation=True,
)

dclm_train_164k_t100x_tsp_tokenized_v2 = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t100x_tsp_v2",
    "konwoo/dclm-100x-tsp-v2",
    "06a49d1",
)

dclm_train_164k_t100x_tsp_shuffled_tokenized_v2 = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t100x_tsp_shuffled_v2",
    "konwoo/dclm-100x-tsp-v2-shuffled",
    "ceb72a7",
)

dclm_val_1k_t100x_tsp_tokenized_v2 = tokenize_data_efficiency_dataset(
    "dclm_val_1k_t100x_tsp_v2",
    "konwoo/dclm-100x-tsp-v2-val",
    "0006aff",
    is_validation=True,
)

dclm_val_1k_t100x_tsp_shuffled_tokenized_v2 = tokenize_data_efficiency_dataset(
    "dclm_val_1k_t100x_tsp_shuffled_v2",
    "konwoo/dclm-100x-tsp-v2-val-shuffled",
    "c40959c",
    is_validation=True,
)

dclm_train_164k_t100x_tsp_mixed_tokenized_v2 = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t100x_tsp_mixed_v2",
    "kothasuhas/dclm-100x-tsp-v2-mixed",
    "b5e8d49",
)

dclm_train_164k_t100x_tsp_submixed_tokenized_v2 = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t100x_tsp_submixed_v2",
    "kothasuhas/dclm-100x-tsp-v2-submixed",
    "24a85ff",
)

dclm_train_164k_t100x_tsp_doubled_tokenized_v2 = tokenize_data_efficiency_dataset(
    "dclm_train_164k_t100x_tsp_doubled_v2",
    "kothasuhas/dclm-100x-tsp-v2-double",
    "62cb243",
)

# generic val set to use

dclm_val_1k_1_5_tokenized = tokenize_data_efficiency_dataset(
    "dclm_val_1k_1_5",
    "konwoo/dclm-01-05-26-val-1000",
    "147eb15",
    is_validation=True,
)
## 1.64M doc TSP 

dclm_train_1m640k_tsp_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_1m640k_tsp",
    "konwoo/dclm-train-1.64m-tsp",
    "86ebf05",
)

dclm_train_1m640k_tsp_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_1m640k_tsp_shuffled",
    "konwoo/dclm-train-1.64m-shuffled",
    "5877fda",
)

dclm_val_1k_normal_tokenized = tokenize_data_efficiency_dataset(
    "dclm_val_1k_normal",
    "konwoo/dclm-val-1k-normal-shuffled",
    "7f9cdc0",
    is_validation=True,
)

## 1.3M doc TSP

dclm_train_1m300k_tsp_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_1m300k_tsp",
    "konwoo/dclm-train-1.3m-tsp",
    "857fb3f",
)

dclm_train_1m300k_tsp_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "dclm_train_1m300k_tsp_shuffled",
    "konwoo/dclm-train-1.3m-shuffle",
    "785b137",
)

## wrap icpt

hq_cpr4_tokenized = tokenize_data_efficiency_dataset(
    "hq_cpr4",
    "konwoo/dclm-164k-8b-instruct-hq-cpr4-ml1024",
    "c2f99ab",
)

wrap_icpt_cpr1_fixed_tokenized = tokenize_data_efficiency_dataset(
    "wrap_icpt_cpr1_fixed",
    "konwoo/wrap_icpt_cpr1_fixed",
    "f688684",
)

train_as_val_tokenized = tokenize_data_efficiency_dataset(
    "train_as_val",
    "konwoo/dclm-1k-train-as-val",
    "69dc3a0",
    is_validation=True,
)

## wrap icpt fixed

hq_cpr4_real_tokenized = tokenize_data_efficiency_dataset(
    "hq_cpr4_real",
    "konwoo/dclm-164k-real-train-8b-instruct-hq-cpr4-ml1024",
    "9b7cdca",
)

hq_cpr4_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "hq_cpr4_shuffled",
    "konwoo/dclm-hq-cpr4-shuffled",
    "c680220",
)

wrap_icpt_cpr1_real_tokenized = tokenize_data_efficiency_dataset(
    "wrap_icpt_cpr1_real",
    "konwoo/wic_real_train_cpr1",
    "851ce78",
)

wrap_icpt_cpr4_real_tokenized = tokenize_data_efficiency_dataset(
    "wrap_icpt_cpr4_real",
    "konwoo/wic_real_train_cpr4",
    "d66b6b0",
)

wrap_icpt_cpr4_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "wrap_icpt_cpr4_shuffled",
    "konwoo/dclm-hq-cpr4anddoc-shuffled",
    "3f27b12",
)

dclm_real_train_tokenized = tokenize_data_efficiency_dataset(
    "dclm_real_train",
    "konwoo/dclm-164k-real-train",
    "a982851",
)

# new wrap baselines (cpr2)


wrap_icpt_back_cpr2_tokenized = tokenize_data_efficiency_dataset(
    "wrap_icpt_back_cpr2",
    "konwoo/wicpt_back_cpr2",
    "13255ab",
)

wrap_icpt_back_cpr2_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "wrap_icpt_back_cpr2_shuffled",
    "konwoo/wicpt_back_cpr2-shuffled",
    "f60f0bc",
)

wrap_icpt_front_cpr2_tokenized = tokenize_data_efficiency_dataset(
    "wrap_icpt_front_cpr2",
    "konwoo/wicpt_front_cpr2",
    "27b0c4e",
)

# paired wrap baselines 


paired_wrap_back_cpr4_tokenized = tokenize_data_efficiency_dataset(
    "fixed_paired_wrap_back_cpr4",
    "konwoo/wicpt_paired_cpr4",
    "224f4a6",
)

paired_wrap_back_cpr4_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "fixed_paired_wrap_back_cpr4_shuffled",
    "konwoo/wicpt_paired_cpr4-shuffled",
    "bf3d61a",
)

# wrap synthetic data scaling 

wrap_back_cpr16_tokenized = tokenize_data_efficiency_dataset(
    "wrap_back_cpr16",
    "konwoo/wicpt_back_cpr16",
    "ee2b717",
)

wrap_back_cpr16_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "wrap_back_cpr16_shuffled",
    "konwoo/wicpt_back_cpr16-shuffled",
    "d2f2836",
)

wrap_back_cpr8_tokenized = tokenize_data_efficiency_dataset(
    "wrap_back_cpr8",
    "konwoo/wicpt_back_cpr8",
    "832b978",
)

wrap_back_cpr8_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "wrap_back_cpr8_shuffled",
    "konwoo/wicpt_back_cpr8-shuffled",
    "dbff3f8",
)

wrap_back_cpr4_tokenized = tokenize_data_efficiency_dataset(
    "wrap_back_cpr4",
    "konwoo/wicpt_back_cpr4",
    "a5e6a79",
)

wrap_back_cpr4_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "wrap_back_cpr4_shuffled",
    "konwoo/wicpt_back_cpr4-shuffled",
    "07ac50c",
)

wrap_back_cpr2_tokenized = tokenize_data_efficiency_dataset(
    "wrap_back_cpr2_new",
    "konwoo/wicpt_back_cpr2_new",
    "17b865a",
)

wrap_back_cpr2_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "wrap_back_cpr2_new_shuffled",
    "konwoo/wicpt_back_cpr2_new-shuffled",
    "e2a0dc5",
)

### no doc

wrap_nodoc_cpr8_tokenized = tokenize_data_efficiency_dataset(
    "wrap_nodoc_cpr8",
    "konwoo/wicpt_cpr8_nodoc",
    "bc6c2b3",
)

wrap_nodoc_cpr8_shuffled_tokenized = tokenize_data_efficiency_dataset(
    "wrap_nodoc_cpr8_shuffled",
    "konwoo/wicpt_cpr8_nodoc-shuffled",
    "f73340b",
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
    # 40x dense TSP data
    "dc_40x": dclm_train_164k_t40x_tsp_tokenized,
    "dc_40x_shuffled": dclm_train_164k_t40x_tsp_shuffled_tokenized,
    "dc_40x_val": dclm_val_1k_t40x_tsp_tokenized,
    "dc_40x_val_shuffled": dclm_val_1k_t40x_tsp_shuffled_tokenized,
    # 100x dense TSP data
    "dc_100x": dclm_train_164k_t100x_tsp_tokenized,
    "dc_100x_shuffled": dclm_train_164k_t100x_tsp_shuffled_tokenized,
    "dc_100x_val": dclm_val_1k_t100x_tsp_tokenized,
    "dc_100x_val_shuffled": dclm_val_1k_t100x_tsp_shuffled_tokenized,
    "dc_100x_v2": dclm_train_164k_t100x_tsp_tokenized_v2,
    "dc_100x_v2_shuffled": dclm_train_164k_t100x_tsp_shuffled_tokenized_v2,
    "dc_100x_v2_mixed": dclm_train_164k_t100x_tsp_mixed_tokenized_v2,
    "dc_100x_v2_submixed": dclm_train_164k_t100x_tsp_submixed_tokenized_v2,
    "dc_100x_v2_doubled": dclm_train_164k_t100x_tsp_doubled_tokenized_v2,
    "dc_100x_v2_val": dclm_val_1k_t100x_tsp_tokenized_v2,
    "dc_100x_v2_val_shuffled": dclm_val_1k_t100x_tsp_shuffled_tokenized_v2,
    # generic val set to use
    "dc_1k_val_1_5": dclm_val_1k_1_5_tokenized,
    # 1.64M doc TSP data
    "dc_1m": dclm_train_1m640k_tsp_tokenized,
    "dc_1m_mix": dclm_train_1m640k_tsp_shuffled_tokenized,
    "dc_1k_val_normal": dclm_val_1k_normal_tokenized,
    # 1.3M doc TSP data
    "dc_1_3m": dclm_train_1m300k_tsp_tokenized,
    "dc_1_3m_mix": dclm_train_1m300k_tsp_shuffled_tokenized,
    # wrap icpt
    "hq4": hq_cpr4_tokenized,
    "wic1": wrap_icpt_cpr1_fixed_tokenized,
    "wrap_ic1": wrap_icpt_cpr1_fixed_tokenized,
    "tav": train_as_val_tokenized,
    # fixed wrap icpt
    "hqr": hq_cpr4_real_tokenized,
    "wir": wrap_icpt_cpr1_real_tokenized,
    "dcr": dclm_real_train_tokenized,
    "hqs": hq_cpr4_shuffled_tokenized,
    "w4": wrap_icpt_cpr4_real_tokenized,
    "w4s": wrap_icpt_cpr4_shuffled_tokenized,
    # new wrap baselines (cpr2)
    "w2": wrap_icpt_back_cpr2_tokenized, 
    "w2s": wrap_icpt_back_cpr2_shuffled_tokenized,
    "w2f": wrap_icpt_front_cpr2_tokenized,
    # paired wrap baselines (cpr4)
    "p4b": paired_wrap_back_cpr4_tokenized,
    "p4s": paired_wrap_back_cpr4_shuffled_tokenized,
    # cpr 16 (and synth data scaling)
    "b16": wrap_back_cpr16_tokenized,
    "s16": wrap_back_cpr16_shuffled_tokenized,
    "b8": wrap_back_cpr8_tokenized,
    "s8": wrap_back_cpr8_shuffled_tokenized,
    "b4": wrap_back_cpr4_tokenized,
    "s4": wrap_back_cpr4_shuffled_tokenized,
    "b2": wrap_back_cpr2_tokenized,
    "s2": wrap_back_cpr2_shuffled_tokenized,
    # nodoc baselines
    "n8": wrap_nodoc_cpr8_tokenized,
    "n8s": wrap_nodoc_cpr8_shuffled_tokenized,
}
