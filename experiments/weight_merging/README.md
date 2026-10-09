# Snowball checkpoint merging

This experiment implements aligned checkpoint merging for the Snowball/Grug 67B MoE family. It includes weighted averaging, task arithmetic, TIES, DARE-linear, DARE-TIES, RAM/RAM+, tensor and expert-row overrides, serial construction, learned coefficients, and calibrated curvature grafting. The learned-coefficient and curvature paths are adaptations of the cited methods, with native MoE component groups and tensor-local graft selection.

The [public campaign archive](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09) contains the historical recipes, calibration settings, selected expert IDs, model revisions, scores, and method citations. Generated coefficients, model configurations for completed evaluations, and historical result files are kept there instead of in the source tree.

## Static merge

Download each parent checkpoint at the revision recorded in the recipe, preserving its sharded safetensors index. Replace the archived object-storage paths with your local paths and choose an empty output directory. `revision` records provenance; the reader does not resolve or verify a remote revision. Verify the downloaded revision before starting.

```sh
uv run python -m experiments.weight_merging.merge \
  --recipe /path/to/recipe.json --threads 16 --code-revision YOUR_GIT_SHA
```

A recipe has `anchor` and `donors` entries with `path` and `revision`, an `output` path, and `parameters` with `method`, `coefficients`, `density`, `scale`, and `seed`. It also provides `preserve_rows` and `tensor_coefficients` mappings; `row_overrides` is optional. In an average, donor coefficients leave `1 - sum(coefficients)` on the anchor. In task arithmetic, they scale donor-minus-anchor updates. TIES pruning is per tensor. DARE randomness depends on the seed, tensor name, and donor position.

The merger reads one shard per source at a time and writes one output shard per tensor. It copies tokenizer and configuration files from the anchor and writes a completion manifest with object hashes last. Failed outputs remain incomplete; do not load or serve them. Existing nonempty output directories are rejected. Serial merging uses a completed output as a source for the next recipe.

## Calibrated methods

`calibration_data.py` generates deterministic synthetic numeric, tool-call, and retention examples. `calibrate.py` either learns component/chunk coefficients against specialist logits and hidden states or collects squared per-sequence response-loss gradients. The latter are calibration estimates, not saved optimizer moments. `merge_calibrated.py` exports either result through the static checkpoint writer.

Calibration requires CUDA GPUs, sufficient CPU RAM for frozen parent checkpoints, and the archived MarinSkyRL revision `f1ad008a3b260bcb4065a67ec60222e92a5f7e4e`. Run `calibration_runtime.py` in a disposable worker environment: it installs the campaign's pinned Transformers and Loguru versions and fetches the explicit `--skyrl-revision`. Provide the archived calibration recipe through `--recipe` and a source revision through `--code-revision`. CPU merging does not require MarinSkyRL.

`select_experts.py` converts saved benchmark activation comparisons into expert-row recipes with matched random controls. `diagnose.py` measures parent/update Gram matrices. `publish_checkpoint.py` verifies a completion manifest and publishes a checkpoint with an explicitly supplied model card and license.

The archived campaign optimized NUPA200 and BFCL-Parity and did not demonstrate broad capability preservation. These historical tools do not enforce the later campaign rule that at least 75% of Base weights remain unchanged; a new experiment must measure that constraint separately. Raw benchmark transcripts are excluded from the public archive, so exact activation-capture replay is not possible from the archive alone.

Run the CPU behavior tests with:

```sh
uv run pytest tests/test_weight_merging.py experiments/weight_merging/test_select_experts.py
```
