# SFT new mix

Tracking: [#10788](https://github.com/marin-community/marin/issues/10788).

## Candidate v1 and control

`sources.candidate_sft_sources_v1()` defines the candidate input map for future
materialization. It starts with `all_sft_sources()` and replaces only the four
`nemotron_sft_v3/science_v2/{rqa,so,syn_mcq,vendor}` components. The shared registry
remains the unchanged control. Other source objects, keys, and rough token
weights are preserved. Historical caches and existing training launches are
unchanged.

The v1 science subset is frozen at Hugging Face revision
`1583049c1df02b418e07d14047eb37db9df46eba` in
[open-athena/Nemotron-SFT-Science-v2-GLM-Accepted-2026.11.05](https://huggingface.co/datasets/open-athena/Nemotron-SFT-Science-v2-GLM-Accepted-2026.11.05/tree/1583049c1df02b418e07d14047eb37db9df46eba).
Keep this pin and v1 selection unchanged; future candidate changes get a new
versioned factory. The base registry follows the checkout, so comparisons must
record and use the same code revision for candidate and control.

The dataset is private. Materialization requires Hugging Face credentials with
access to it. Exact serialized text and duplicate rows are retained without
Harmony rendering. The source map normalizes text; it does not consume the
archived token IDs or loss masks directly.

## Selection and provenance

Of 268,164 science rows, 99,453 were accepted: 70,387 unchanged passes and 29,066
approved repairs. The 168,711 excluded rows include holds, failures, reserved
rows, and unprocessed rows; exclusion is not a scientific-error verdict.

The accepted set includes 31,984 model-only audits and 67,469 optional-web audits.
Acceptance does not guarantee factual correctness. The dataset contains the
original records, review evidence, request receipts, and manifest hashes.
It retains NVIDIA attribution and the source's
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) terms;
see the dataset card for attribution and provenance details.

## Later training comparison

Matched-token control and candidate training recipes in the `snowball_h100`
style are later work. This directory currently defines sources only; it does
not launch training or claim a completed matched-token comparison.
