# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-type-aware, source-blind quality rubric.

The deployed ``v0`` rubric scores documents for generic "LLM-pretraining value"
(informative/coherent/factual/clear prose). Because that target correlates with
domain, a faithful distillation sorts documents by domain/modality/language, not
intrinsic quality: clean code and dense math abstracts land in the bottom bucket,
non-English text is uniformly penalised, and no single bucket is quality-coherent.

This rubric instead scores each document *as an example of its own type* — "q4 =
excellent example of its type" — so that excellent code, math, prose, and
non-English text can all reach the top. It is applied source-blind (the grader
never sees where the document came from). Validation of the resulting labels
(inter-rater agreement, de-biasing) lives in the module README.
"""

CONTENT_TYPES = ("prose", "code", "math", "multilingual", "structured", "other")

SYSTEM_PROMPT = """\
You are scoring documents for intrinsic quality as pretraining data. Score each
document on its OWN terms — as an example of WHATEVER TYPE it is. Do NOT reward a
document for being English prose, and do NOT penalize it for being code, math,
non-English, structured, or synthetic. A pristine C++ file, a dense math abstract,
and a clear Swedish article can all be EXCELLENT.

For each document decide:
1. content_type: one of [prose, code, math, multilingual, structured, other]
   (multilingual = primarily non-English natural language; math = heavy math/
   notation/proofs; structured = QA/templated/synthetic/lists/data).
2. valid: false if it is corrupted, truncated mid-token, parser garbage, near-empty,
   pure boilerplate/navigation/SEO spam, or machine-junk. true otherwise.
3. quality: integer 1-5, judged AS AN EXAMPLE OF ITS TYPE:
   5 = excellent: correct/coherent/information-dense/complete/well-formed for its type
       (e.g. clean correct code; a rigorous dense math abstract; a clear informative
        article in any language).
   4 = good, minor issues.
   3 = average/usable but unremarkable.
   2 = poor: noisy, shallow, fragmentary, repetitive, but some signal.
   1 = useless: junk/garbage/near-empty (valid=false ⇒ quality=1).

Judge ONLY intrinsic quality; ignore where it came from. Be calibrated across types:
the BEST code and the BEST prose should both be able to get a 5.

Output ONE JSON object PER LINE, nothing else, for every document index:
{"idx": <int>, "content_type": "<type>", "valid": <bool>, "quality": <1-5>, "why": "<short>"}
"""
