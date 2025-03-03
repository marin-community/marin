from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from marin.execution.executor import executor_main

dolmino_tinygsm = get_dolmino_step("math/tinyGSM-MIND")
dolmino_gsm8k = get_dolmino_step("math/gsm8k")
dolmino_mathcoder2 = get_dolmino_step("math/mathcoder2-synthmath")

# TinyGSM is 6.48B so normal 50B tokens is roughly a bit more than
# 1 repetition
tinygsm_model = default_quality_ablation(candidate_tokenized=dolmino_tinygsm, config=QualityAblationConfig())

# GSM8K is 2.74M tokens, this is assuming 0.15 weight for candidate
# and 4 repetition
gsm8k_model = default_quality_ablation(
    candidate_tokenized=dolmino_gsm8k,
    config=QualityAblationConfig(
        num_anneal_tokens=73_064_000,
    ),
)

# MathCoder2 is 3.87B tokens, which should assume 1 / 0.15 * 3.87B tokens = 25.8B
# so this is about 2 repetition
mathcoder2_model = default_quality_ablation(
    candidate_tokenized=dolmino_mathcoder2,
    config=QualityAblationConfig(
        num_anneal_tokens=25_800_000_000,
    ),
)

# 28.7M tokens, 4 repetitions
dolmino_synthmath = get_dolmino_step("math/dolmino_math_synth")
synthmath_model = default_quality_ablation(
    candidate_tokenized=dolmino_synthmath,
    config=QualityAblationConfig(
        num_anneal_tokens=765_333_000,
    ),
)

# 230M tokens, 4 repetitions
dolmino_tulu_math = get_dolmino_step("math/tulu_math")
tulu_math_model = default_quality_ablation(
    candidate_tokenized=dolmino_tulu_math,
    config=QualityAblationConfig(
        num_anneal_tokens=6_133_000_000,
    ),
)

# 84M tokens, 4 repetitions
metamath_owmfilter = get_dolmino_step("math/metamath-owmfilter")
metamath_owmfilter_model = default_quality_ablation(
    candidate_tokenized=metamath_owmfilter,
    config=QualityAblationConfig(
        num_anneal_tokens=2_245_000_000,
    ),
)

if __name__ == "__main__":
    executor_main(
        [tinygsm_model, gsm8k_model, mathcoder2_model, synthmath_model, tulu_math_model, metamath_owmfilter_model]
    )
