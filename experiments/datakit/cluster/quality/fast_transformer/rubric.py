# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-type-aware, source-blind quality rubric.

The original ``v0`` rubric scored documents for generic "LLM-pretraining value"
(informative/coherent/factual/clear prose). Because that target correlates with
domain, a faithful distillation sorts documents by domain/modality/language, not
intrinsic quality: clean code and dense math abstracts land in the bottom bucket,
non-English text is uniformly penalised, and no single bucket is quality-coherent.
The type-aware framing below — score each document *as an example of its own type*,
source-blind — was the fix for that.

Measurements on the 292-source 50M sample show the framing did not survive into
the labels, so ``v2`` targets the two specific ways it failed:

* **The scale collapsed to its middle.** The deployed scorer puts 2.4% of documents
  in the top bucket and 2.3% in the bottom, with 82% in the two middle buckets. The
  labels it learned from are the cause: 75 of 5,578 (1.3%) are quality-5. A grader
  that almost never spends a 5 teaches a model that cannot rank the top of the
  distribution, which is the end of the scale data selection actually uses. ``v2``
  therefore states the intended spread explicitly rather than leaving "excellent"
  to the grader's caution.
* **Type parity is still not real.** Math lands 88.5% of its documents in the top
  two buckets; multilingual lands 23.5%. Every one of the 18 ``finepdfs`` language
  splits scores at or below English (Arabic 0.400, Thai 0.473, English 0.529).
  Saying "do not penalize non-English" was not enough, so ``v2`` makes per-type
  parity a checkable instruction: each type's own best work must reach 5.

``agentic`` is new. Roughly half the registry's sources are now tool-use and
multi-turn trajectories (``penfever-traces/*``, ``swe-*``, ``agenttrove``,
``nemotron-gym``), which v0 had to file under ``structured`` or ``other`` — a
category too broad to score against a shared standard.

Labeling is offline; :mod:`experiments.datakit.cluster.quality.glm52_vllm` serves
the grader. Validation of the resulting labels lives in the module README.
"""

CONTENT_TYPES = ("prose", "code", "math", "multilingual", "structured", "agentic", "other")

SYSTEM_PROMPT = """\
You are scoring documents for intrinsic quality as pretraining data. Score each
document on its OWN terms — as an example of WHATEVER TYPE it is. Do NOT reward a
document for being English prose, and do NOT penalize it for being code, math,
non-English, structured, synthetic, or a machine-generated trajectory. A pristine
C++ file, a dense math abstract, a clear Swedish article, and a clean tool-use
trace can all be EXCELLENT.

For each document decide:
1. content_type: one of [prose, code, math, multilingual, structured, agentic, other]
   (multilingual = primarily non-English natural language; math = heavy math/
   notation/proofs; structured = QA/templated/synthetic/lists/data; agentic =
   tool-call or multi-turn agent/assistant trajectories, terminal or IDE sessions).
2. valid: false if it is corrupted, truncated mid-token, parser garbage, near-empty,
   pure boilerplate/navigation/SEO spam, or machine-junk. true otherwise.
   A long document may arrive as an excerpt, ending with an explicit
   "[Excerpt ends here ...]" marker. That is this harness shortening the document,
   NOT damage in the source: judge the text shown on its own merits and never mark a
   document invalid, or lower its quality, for ending at that marker.
   A document may also arrive as a WINDOW from the middle or end of a longer
   document, announced by a bracketed "[This is a window from the ...]" notice
   above it; such a window may begin or end mid-sentence. That too is the harness
   slicing the document, NOT damage in the source: judge the text shown on its own
   merits and never mark a window invalid, or lower its quality, merely for
   starting or ending abruptly.
3. quality: integer 1-5, judged AS AN EXAMPLE OF ITS TYPE:
   5 = excellent: the best work of its kind. Concretely, per type:
       prose — a clear, informative, well-structured article that teaches something;
       code — correct, idiomatic, readable, with real logic (not a config stub);
       math — a rigorous derivation or dense, correct, self-contained statement;
       multilingual — the same bar as prose, applied in that language, judged by a
         fluent reader of it — NOT by how well it survives translation to English;
       structured — clean schema, accurate content, genuinely useful QA/data;
       agentic — a coherent goal pursued with sensible tool calls and a real
         resolution, not a truncated or flailing session.
   4 = good, minor issues.
   3 = average/usable but unremarkable.
   2 = poor: noisy, shallow, fragmentary, repetitive, but some signal.
   1 = useless: junk/garbage/near-empty (valid=false ⇒ quality=1).

5 IS NOT PERFECTION. It is the level a competent example of that type reaches when
nothing is wrong with it. Ask "would a practitioner of this type be glad to have
written this?" — if yes, it is a 5, even if you can imagine something better. Do not
hold 5 back for an ideal document; an unreachable top makes the scale useless. A
clean, correct, complete function is a 5. A clear, accurate, self-contained article
is a 5. Reserve 4 for work with a specific flaw you can name.

PARITY ACROSS TYPES IS THE POINT. Each type's own best work reaches 5 at a similar
rate. If code or non-English documents are landing consistently lower than English
prose, you are grading the type rather than the document. Judge a non-English
document exactly as a fluent reader of that language would judge it.

Judge ONLY intrinsic quality; ignore where it came from.

Output ONE JSON object PER LINE, nothing else, for every document index:
{"idx": <int>, "content_type": "<type>", "valid": <bool>, "quality": <1-5>, "why": "<short>"}
"""
