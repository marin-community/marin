# Debugging log for MRCR full-render offsets

Determine why the live paired MRCR transform rejects the full-render offset tokenization, then preserve evidence-distance offsets and canonical cache tokenization without independently tokenizing substrings.

## Initial status

The live transform raises `ValueError: Offset tokenization differs from the canonical full-render MRCR tokenization` in `experiments/datasets/mrcr.py`. The failing comparison checks the chat preprocessor's IDs against IDs returned by a fast tokenizer call over `prompt + target + eos`.

## Hypothesis 1

The chat preprocessor splits the rendered template at `{% generation %}` boundaries before tokenization so it can construct the assistant mask. A BPE tokenizer can merge bytes across the prompt/target boundary when the complete rendered string is tokenized once. Therefore the two ID arrays can differ even though both paths render the same characters. EOS or wrapper-added special tokens are alternative possible first divergences.

## Changes to make

- Reproduce both tokenization paths with the real Marin tokenizer when available locally or through the configured tokenizer loader.
- Add a focused BPE regression tokenizer whose vocabulary merges across the generation boundary.
- Change only the evidence-distance implementation and its tests, retaining full-render offsets, target-only scoring, canonical bins, and source identities.

## Results

The real `marin-community/marin-tokenizer` reproduced a generation-boundary mismatch:

- Chat preprocessing of `Assistant: pre` followed by scored body `fix` produced IDs ending in `[72803, 25, 864, 5862, 128001]`.
- One offset-bearing tokenization of the complete rendered text produced `[72803, 25, 9436, 128001]`; token `9436` spans `prefix` across the prompt/body boundary.
- When no boundary merge was available, the preprocessor and complete-render IDs agreed, including the `<|end_of_text|>` EOS token.

This isolates the mismatch to generation-boundary segmentation. The HF offset wrapper and EOS behavior are not the cause. `ChatProcessor` deliberately tokenizes text on each side of `{% generation %}` separately to construct the assistant mask. The evidence-distance contract deliberately requires offsets from one complete rendered tokenization. Requiring those two ID arrays to be equal incorrectly rejects valid BPE merges.

The fix removes only that equality requirement. Canonical binning and scored target IDs still come from the shared `ChatLmDatasetFormat` preprocessor. Evidence distance still tokenizes the complete two-shot render once and derives both the selected-response endpoint and scored-body endpoint from that call's offsets. A BPE regression test now runs the public transform with `pre` at the end of the masked prompt and `fix` as the response body; the complete render merges them to `prefix`, while the canonical chat path does not. The transform completes, preserves the paired scored body, and records the distance from the complete-render offsets.

Focused results:

- The regression failed with the original equality guard and passed after its removal.
- `tests/test_mrcr_dataset.py`: 7 passed.
- Ruff check and format validation passed for the implementation and focused tests.
- Pyrefly reported zero errors for the implementation and focused tests.

## Future work

- No follow-up work identified for this failure.
