# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Registry of midtraining math datasets, grouped by provenance bucket.

Buckets:
    BUCKET_1 — no model-rewritten text (filter / score / extract only)
    BUCKET_2 — non-Qwen model rewriting allowed
    BUCKET_3 — Qwen / QwQ model rewriting allowed

Only datasets already tokenized in Marin are exposed here. Datasets not yet
ingested (OpenWebMath standalone, MathPile, InfiMM-WebMath, MathCode-Pile,
Nemotron-MIND standalone, UltraData-Math) are tracked separately.
"""

from experiments.midtraining_datasets import (
    finemath_3_plus_tokenized,
    megamath_tokenized,
)
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets.nemotron_v2 import tokenize_nemotron_v2_family

# ----------------------------------------------------------------------------
# Nemotron v2 families (resolved once at import; all subsets per family).
# ----------------------------------------------------------------------------

nemotron_cc_math_v1 = tokenize_nemotron_v2_family("nemotron_cc_math_v1")
nemotron_cc_code_v1 = tokenize_nemotron_v2_family("nemotron_cc_code_v1")
nemotron_cc_v2 = tokenize_nemotron_v2_family("nemotron_cc_v2")
nemotron_cc_v2_1 = tokenize_nemotron_v2_family("nemotron_cc_v2_1")
nemotron_pretraining_code_v1 = tokenize_nemotron_v2_family("nemotron_pretraining_code_v1")
nemotron_pretraining_code_v2 = tokenize_nemotron_v2_family("nemotron_pretraining_code_v2")
nemotron_pretraining_specialized_v1 = tokenize_nemotron_v2_family("nemotron_pretraining_specialized_v1")
# Nemotron-Pretraining-SFT-v1 deliberately excluded — it's SFT/instruction data, not midtraining.


# ============================================================================
# BUCKET 1 — no model rewriting (filter / score / extract only)
# ============================================================================

BUCKET_1 = {
    "proofpile_2": dclm_components_llama3["proofpile_2"],
    "finemath-3-plus": finemath_3_plus_tokenized,
    "megamath/web": megamath_tokenized["megamath/web"],
}


# ============================================================================
# BUCKET 2 — non-Qwen model rewriting allowed
# ============================================================================

BUCKET_2 = {
    # Nemotron-CC-Math v1 — Phi-4-cleaned CC math
    "nemotron_cc_math_v1/4plus": nemotron_cc_math_v1["nemotron_cc_math_v1/4plus"],
    "nemotron_cc_math_v1/3": nemotron_cc_math_v1["nemotron_cc_math_v1/3"],
    # Phi-4-regenerated synthetic dialogues on top of 4plus
    "nemotron_cc_math_v1/4plus_mind": nemotron_cc_math_v1["nemotron_cc_math_v1/4plus_mind"],
    # MegaMath-Web-Pro — Llama-3.3-70B refinement of MegaMath-Web
    "megamath/web_pro": megamath_tokenized["megamath/web_pro"],
}


# ============================================================================
# BUCKET 3 — Qwen / QwQ model rewriting allowed
# ============================================================================

BUCKET_3 = {
    # MegaMath synthetic splits (Qwen2.5-72B / Qwen2.5-Coder-32B among generators)
    "megamath/qa": megamath_tokenized["megamath/qa"],
    "megamath/translated_code": megamath_tokenized["megamath/translated_code"],
    "megamath/text_code_block": megamath_tokenized["megamath/text_code_block"],
    # Nemotron-CC-v2 / v2.1 — Qwen3-30B-A3B rephrases of CC
    **nemotron_cc_v2,
    **nemotron_cc_v2_1,
    # Nemotron-Pretraining-Code — includes Qwen-synth in v2
    **nemotron_cc_code_v1,
    **nemotron_pretraining_code_v1,
    **nemotron_pretraining_code_v2,
    # Nemotron-Pretraining-Specialized-v1 — Qwen3 / QwQ / DeepSeek-R1 generated.
    # Exclude `stem_sft` because it's SFT-shaped despite the "pretraining" family label.
    **{k: v for k, v in nemotron_pretraining_specialized_v1.items() if not k.endswith("/stem_sft")},
}


# ============================================================================
# Convenience unions
# ============================================================================

BUCKET_1_PLUS_2 = {**BUCKET_1, **BUCKET_2}
ALL_BUCKETS = {**BUCKET_1, **BUCKET_2, **BUCKET_3}
