# Frozen Hero sanity panel

These are the exact 100 prompts, seeds, order and review references used for the
October 22, 2026 sanity check: 40 GSM8K questions, 40 authored Python tasks and
20 authored fixed histories with one fresh assistant response each.

GSM8K questions and solutions come from `openai/gsm8k`, revision
`e53f048856ff4f594e959d75785d2c2d37b678ee`, `main/test`, rows 500–539
(zero based), under the upstream MIT license. Their source links appear in the
panel. Python tasks and fixed histories are Marin-authored. References support
human review; they are not an executable grader or a quality acceptance gate.

The frozen panel records the original implementation revision. Preparation
rebinds only that field to the caller's source revision. Questions, seeds,
temperature, tokenizer, template and token limits remain fixed. Preparation
checks exact normalized overlap against both supplied SFT splits; this does not
establish absence from pretraining or detect paraphrases.

Reference caveats in `references.json` remain unchanged. Review also found
GSM8K row 515 ambiguous about packs versus individual bags; a coherent answer
under a stated interpretation should not be marked wrong solely by its numeric
reference. The drawing-supply follow-up can count a pad and pencils as either
one kit or two physical items.
