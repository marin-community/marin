# Shared100 chat-template ablation

This diagnostic compares three overlays on the same pinned Shared100 weights. It does not change the published checkpoint or claim eval-policy conformance. `variants.yaml` pins both source checkpoint revisions and file hashes. `launch.py` verifies those files, writes each effective model config and its SHA-256 provenance to the experiment artifacts directory, and submits MATH500, one-repeat AIME24, and BFCL Parity with the campaign's unchanged benchmark configs. The one-repeat AIME task uses the same prompt, seed, and grader as AIME24; its separate Evalchemy revision is pinned by this Marin branch.

From the Marin checkout root:

```bash
uv run python experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/shared100_template/validate.py
uv run python experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/shared100_template/launch.py reasoning_only
uv run python experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/shared100_template/launch.py reasoning_only --submit
```

Replace `reasoning_only` with `template_only` or `template_and_tokenizer` for the other arms. Omit `--submit` for a launch-plan dry run. The launcher does not set a policy version. Jobs and records capture the effective model config and benchmark pins. `validate.py` writes exact rendered prompt evidence under `artifacts/shared100-chat-template-ablation-20261013/`.

The Antidoom tokenizer arm points vLLM's tokenizer to the pinned Antidoom repository. Both repositories have byte-identical `tokenizer_config.json`; the verified `tokenizer.json` differs. The model location and weights remain Shared100 in every arm. The reasoning-only arm inserts historical `reasoning_content` into think blocks in Shared100's two assistant branches. Without historical reasoning, its rendered prompt is byte-identical to the original Shared100 template for the checked tool-call conversation.
