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

from experiments.audio.tokenize_yodas import tokenize_yodas_steps, tokenize_yodas_asr_en_steps
from experiments.audio.tokenize_emilia import (
    tokenize_emilia_steps,
    tokenize_emilia_conversational_steps,
    tokenize_emilia_fix_steps,
)
from experiments.audio.tokenize_mls_en import tokenize_mls_en_steps, tokenize_mls_en_tts0_steps
from experiments.audio.tokenize_nemotron import tokenize_nemotron_hq_actual_step
from experiments.audio.tokenize_finetune import (
    tokenize_peoples_speech_steps,
    tokenize_common_voice_17_steps,
    tokenize_librispeech_steps,
    tokenize_libritts_steps,
)
from marin.processing.tokenize.data_configs import lm_mixture_data_config, LMMixtureDatasetConfig


def mix3_v1_english_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized mix3-v1 English only."""

    # Mix3-v1: MLS-EN (35B) + Emilia-En (37B) + Emilia-Yodas2-En +Yodas2-En (73B + 131B)
    # We want to under sample Emilia-Yodas2-En +Yodas2-En (as we just treat them as one source)

    mls_en_tokenized = tokenize_mls_en_steps()["mls-en"]
    emilia_en_tokenized = tokenize_emilia_steps()["Emilia/EN"]
    emilia_yodas_en_tokenized = tokenize_emilia_steps()["Emilia-YODAS/EN"]
    yodas2_tokenized = tokenize_yodas_steps()["yodas2/en"]

    tokenized = {
        "mls-en": mls_en_tokenized,
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_tokenized,
    }

    weights = {
        "mls-en": 0.33333,
        "emilia-en": 0.33333,
        "emilia-yodas-en": 0.33333 * (73 / (73 + 131)),
        "yodas2-en": 0.33333 * (131 / (73 + 131)),
    }

    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def mix2_v1_english_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized mix2-v1 English only."""

    # Mix2-v1: Emilia-En (37B) + Emilia-Yodas2-En +Yodas2-En (73B + 131B)
    # We want to under sample Yodas2-En (as we just treat them as one source)

    emilia_en_tokenized = tokenize_emilia_steps()["Emilia/EN"]
    emilia_yodas_en_tokenized = tokenize_emilia_steps()["Emilia-YODAS/EN"]
    yodas2_tokenized = tokenize_yodas_steps()["yodas2/en"]
    tokenized = {
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_tokenized,
    }
    weights = {
        "emilia-en": 0.5,
        "emilia-yodas-en": 0.5 * (73 / (73 + 131)),
        "yodas2-en": 0.5 * (131 / (73 + 131)),
    }

    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def mix3_v2_english_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized mix3-v2 English only."""

    # Mix3-v2: MLS-EN (35B) + Emilia-En (37B) + Emilia-Yodas2-En +Yodas2-En (73B + 131B)
    # v2: use Yodas only for ASR and MLS-en & Emilia for TTS

    mls_en_tokenized = tokenize_mls_en_tts0_steps()["mls-en-tts0"]
    emilia_yodas_en_tokenized = tokenize_emilia_conversational_steps()["Emilia-YODAS/EN"]
    yodas2_tokenized = tokenize_yodas_asr_en_steps()["yodas2/en"]

    tokenized = {
        "mls-en": mls_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_tokenized,
    }

    weights = {
        "mls-en": 0.50 * (51 / (51 + 114)),  # ratio in GB
        "emilia-yodas-en": 0.50 * (114 / (51 + 114)),  # ratio in GB
        "yodas2-en": 0.50,
    }

    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def mix3_v3_english_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized mix3-v3 English only."""

    # Mix3-v3: MLS-EN (35B) + Emilia-En (37B) + Emilia-Yodas2-En +Yodas2-En (73B + 131B)
    # similar to v1, but use Emilia-Pretrain-Fix instead of Emilia-Pretrain

    mls_en_tokenized = tokenize_mls_en_steps()["mls-en"]
    emilia_en_tokenized = tokenize_emilia_fix_steps()["Emilia/EN"]
    emilia_yodas_en_tokenized = tokenize_emilia_fix_steps()["Emilia-YODAS/EN"]
    yodas2_tokenized = tokenize_yodas_steps()["yodas2/en"]

    tokenized = {
        "mls-en": mls_en_tokenized,
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_tokenized,
    }

    weights = {
        "mls-en": 0.33333,
        "emilia-en": 0.33333,
        "emilia-yodas-en": 0.33333 * (73 / (73 + 131)),
        "yodas2-en": 0.33333 * (131 / (73 + 131)),
    }

    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def mix3_v4_english_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized mix3-v4 English only."""

    # Mix3-v4: MLS-EN (35B) + Emilia-En (37B) + Emilia-Yodas2-En (73B) + "Yodas2-En-ASR" (131/2 B)
    # similar to Mix3-v3, but use Yodas2-En-ASR instead of Yodas2-En (ASR only for Yodas2)

    mls_en_tokenized = tokenize_mls_en_steps()["mls-en"]
    emilia_en_tokenized = tokenize_emilia_fix_steps()["Emilia/EN"]
    emilia_yodas_en_tokenized = tokenize_emilia_fix_steps()["Emilia-YODAS/EN"]
    yodas2_tokenized = tokenize_yodas_asr_en_steps()["yodas2/en"]

    tokenized = {
        "mls-en": mls_en_tokenized,
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_tokenized,
    }

    weights = {
        "mls-en": 0.33333,
        "emilia-en": 0.33333,
        "emilia-yodas-en": 0.33333 * (73 / (73 + 65)),
        "yodas2-en": 0.33333 * (65 / (73 + 65)),
    }

    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def mix2_v2_english_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized mix2-v2 English only."""

    # Mix2-v2: MLS-EN (35B) + Emilia-En (37B) + Emilia-Yodas2-En (73B)
    # similar to Mix3-v4, but remove Yodas2 from the mix!

    mls_en_tokenized = tokenize_mls_en_steps()["mls-en"]
    emilia_en_tokenized = tokenize_emilia_fix_steps()["Emilia/EN"]
    emilia_yodas_en_tokenized = tokenize_emilia_fix_steps()["Emilia-YODAS/EN"]

    tokenized = {
        "mls-en": mls_en_tokenized,
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
    }

    weights = {
        "mls-en": 0.33333,
        "emilia-en": 0.33333,
        "emilia-yodas-en": 0.33333,
    }

    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def mix2_v3_english_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized mix2-v3 English only."""

    # Mix2-v3: MLS-EN (35B) + Emilia-En (37B) + "Emilia-Yodas2-En-ASR" (73B/2)
    # similar to Mix3-v4, but remove MLS-EN from the mix!

    emilia_en_tokenized = tokenize_emilia_fix_steps()["Emilia/EN"]
    emilia_yodas_en_tokenized = tokenize_emilia_fix_steps()["Emilia-YODAS/EN"]
    yodas2_tokenized = tokenize_yodas_asr_en_steps()["yodas2/en"]

    tokenized = {
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_tokenized,
    }

    weights = {
        "emilia-en": 0.33333,
        "emilia-yodas-en": 0.33333 * (73 / (73 + 65)),
        "yodas2-en": 0.33333 * (65 / (73 + 65)),
    }

    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


def mix3_v5_english_yodas_sweep_mixture_config(yodas_weight: float = 1.0) -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized mix3-v3 English only."""

    # it's like Mix3-v3, but we sweep the yodas ratio

    mls_en_tokenized = tokenize_mls_en_steps()["mls-en"]
    emilia_en_tokenized = tokenize_emilia_fix_steps()["Emilia/EN"]
    emilia_yodas_en_tokenized = tokenize_emilia_fix_steps()["Emilia-YODAS/EN"]
    yodas2_tokenized = tokenize_yodas_steps()["yodas2/en"]

    tokenized = {
        "mls-en": mls_en_tokenized,
        "emilia-en": emilia_en_tokenized,
        "emilia-yodas-en": emilia_yodas_en_tokenized,
        "yodas2-en": yodas2_tokenized,
    }

    weights = {
        "mls-en": 0.333,
        "emilia-en": 0.333,
        "emilia-yodas-en": 0.119,
        "yodas2-en": 0.214 * yodas_weight,
    }

    return lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="feistel",
    )


### Cooldown Data Mixes ###


def cooldown_mix_exp1(n_weight: float = 1.0) -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized cooldown mix exp1."""

    yodas2_en_tokenized = tokenize_yodas_steps()["yodas2/en"]
    emilia_yodas_en_tokenized = tokenize_emilia_fix_steps()["Emilia-YODAS/EN"]
    emilia_en_tokenized = tokenize_emilia_fix_steps()["Emilia/EN"]
    mls_en_tokenized = tokenize_mls_en_steps()["mls-en"]
    nemotron_tokenized = tokenize_nemotron_hq_actual_step()

    # cooldown mix
    peoples_speech_tokenized = tokenize_peoples_speech_steps()["peoples-speech-clean"]
    common_voice_17_tokenized = tokenize_common_voice_17_steps()["commonvoice17-en"]
    librispeech_tokenized = tokenize_librispeech_steps()["librispeech-train"]
    libritts_tokenized = tokenize_libritts_steps()["libritts-train"]

    return lm_mixture_data_config(
        components={
            "mls-en": mls_en_tokenized,
            "emilia-en": emilia_en_tokenized,
            "emilia-yodas-en": emilia_yodas_en_tokenized,
            "yodas2-en": yodas2_en_tokenized,
            "nemotron_cc/hq_actual": nemotron_tokenized,
            "peoples-speech-clean": peoples_speech_tokenized,
            "commonvoice17-en": common_voice_17_tokenized,
            "librispeech-train": librispeech_tokenized,
            "libritts-train": libritts_tokenized,
        },
        weights={
            # yodas and nemotron components always at 5%
            "yodas2-en": 0.05,
            "nemotron_cc/hq_actual": 0.05,
            # core components
            "mls-en": 0.205,
            "emilia-en": 0.217 - ((n_weight - 1.0) * 0.053 * 0.336),
            "emilia-yodas-en": 0.428 - ((n_weight - 1.0) * 0.053 * 0.664),
            # ASR / TTS data
            "peoples-speech-clean": 0.035 * n_weight,
            "commonvoice17-en": 0.01 * n_weight,
            "librispeech-train": 0.005 * n_weight,
            "libritts-train": 0.003 * n_weight,
        },
        permutation_type="feistel",
    )
