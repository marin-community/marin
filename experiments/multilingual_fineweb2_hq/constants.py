# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FINEWEB2_DATASETS = {
    "arb_Arab": ["arb_Arab/*.parquet"],
    "rus_Cyrl": ["rus_Cyrl/*.parquet"],
    "cmn_Hani": ["cmn_Hani/*.parquet"],
    "deu_Latn": ["deu_Latn/*.parquet"],
    "spa_Latn": ["spa_Latn/*.parquet"],
    "jpn_Jpan": ["jpn_Jpan/*.parquet"],
    "fra_Latn": ["fra_Latn/*.parquet"],
    "ita_Latn": ["ita_Latn/*.parquet"],
    "por_Latn": ["por_Latn/*.parquet"],
    "pol_Latn": ["pol_Latn/*.parquet"],
    "nld_Latn": ["nld_Latn/*.parquet"],
    "ind_Latn": ["ind_Latn/*.parquet"],
    "tur_Latn": ["tur_Latn/*.parquet"],
    "ces_Latn": ["ces_Latn/*.parquet"],
    "fas_Arab": ["fas_Arab/*.parquet"],
    "hun_Latn": ["hun_Latn/*.parquet"],
    "swe_Latn": ["swe_Latn/*.parquet"],
    "ell_Grek": ["ell_Grek/*.parquet"],
    "dan_Latn": ["dan_Latn/*.parquet"],
    "vie_Latn": ["vie_Latn/*.parquet"],
}


FINEWEB2_HQ_MIXTURE_BYTES = {  # From https://huggingface.co/datasets/epfml/FineWeb2-HQ
    "arb_Arab": 94 / 1024,
    "rus_Cyrl": 1.2,  # TiB
    "cmn_Hani": 784 / 1024,  # in GiB
    "deu_Latn": 618 / 1024,
    "spa_Latn": 515 / 1024,
    "jpn_Jpan": 393 / 1024,
    "fra_Latn": 483 / 1024,
    "ita_Latn": 269 / 1024,
    "por_Latn": 222 / 1024,
    "pol_Latn": 168 / 1024,
    "nld_Latn": 160 / 1024,
    "ind_Latn": 125 / 1024,
    "tur_Latn": 100 / 1024,
    "ces_Latn": 104 / 1024,
    "fas_Arab": 69 / 1024,
    "hun_Latn": 79 / 1024,
    "swe_Latn": 61 / 1024,
    "ell_Grek": 84 / 1024,
    "dan_Latn": 61 / 1024,
    "vie_Latn": 59 / 1024,
}

"""
To calculate tokens per byte, we sample 10_000 documents per language and tokenize.
Please take into account the below factors that affect tokens per byte for each language:
-Character to byte ratio per language in UTF-8. Some languages take up more bytes per character than others.
-Tokens per (set of UTF-8 characters in the sampled documents)
See: https://gist.github.com/pruksmhc/6f70c6f41b93fe2fdd16344181e062ec
"""

LLAMA3_TOKENS_PER_BYTE = {
    "rus_Cyrl": 0.17677821397793153,
    "cmn_Hani": 0.2959771019692867,
    "deu_Latn": 0.28022908170020344,
    "jpn_Jpan": 0.25653320627781934,
    "spa_Latn": 0.26257939432458355,
    "fra_Latn": 0.2680086987523188,
    "ita_Latn": 0.28830668455618513,
    "por_Latn": 0.27618470534916084,
    "pol_Latn": 0.34932138076807834,
    "nld_Latn": 0.29540389884839874,
    "ind_Latn": 0.29528557247940446,
    "tur_Latn": 0.26928567422981753,
    "ces_Latn": 0.29385531687368166,
    "vie_Latn": 0.21473460462772323,
    "swe_Latn": 0.3080962816189239,
    "fas_Arab": 0.18324142427489984,
    "arb_Arab": 0.22182239425150832,
    "ell_Grek": 0.21557501301254528,
    "dan_Latn": 0.31682970618405476,
    "hun_Latn": 0.37565856717870233,
}
