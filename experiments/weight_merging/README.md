# Snowball checkpoint merging

This experiment implements aligned checkpoint merging for the Snowball/Grug 67B MoE family. It includes weighted averaging, task arithmetic, TIES, DARE-linear, DARE-TIES, RAM/RAM+, tensor and expert-row overrides, serial construction, learned coefficients, and calibrated curvature grafting. The learned-coefficient and curvature paths are adaptations of the cited methods, with native MoE component groups and tensor-local graft selection.

The [public campaign archive](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09) contains the historical recipes, calibration settings, selected expert IDs, model revisions, scores, and method citations. Generated coefficients, model configurations for completed evaluations, and historical result files are kept there instead of in the source tree.

Campaign 2 adds exports that preserve at least 75% of the original Base scalar weights and every non-weight file. Its [54 recipes](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/tree/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes) and [result evidence](https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/blob/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/results-evidence.json) cover 48 evaluated candidates; [issue #9975](https://github.com/marin-community/marin/issues/9975) records the completed study. The historical workflow below remains available for the first campaign.

Constrained exports keep a temporary `.merge-incomplete` marker until their change cap passes and the external audit is written and read back. Marin's checkpoint reader rejects marked directories. Successful finalization removes the marker, leaving exactly Base's original file inventory. Failed exports retain the marker and must not be served.

## Static merge

Download each parent checkpoint at the revision recorded in the recipe, preserving its sharded safetensors index. Replace the archived object-storage paths with your local paths and choose an empty output directory. `revision` records provenance; the reader does not resolve or verify a remote revision. Verify the downloaded revision before starting.

```sh
uv run --package marin-core --extra cpu python -m experiments.weight_merging.merge \
  --recipe /path/to/recipe.json --threads 16 --code-revision YOUR_GIT_SHA
```

A recipe has `anchor` and `donors` entries with `path` and `revision`, an `output` path, and `parameters` with `method`, `coefficients`, `density`, `scale`, and `seed`. It also provides `preserve_rows` and `tensor_coefficients` mappings; `row_overrides` is optional. In an average, donor coefficients leave `1 - sum(coefficients)` on the anchor. In task arithmetic, they scale donor-minus-anchor updates. TIES pruning is per tensor. DARE randomness depends on the seed, tensor name, and donor position.

The merger reads one shard per source at a time and writes one output shard per tensor. It copies tokenizer and configuration files from the anchor and writes a completion manifest with object hashes last. Failed outputs remain incomplete; do not load or serve them. Existing nonempty output directories are rejected. Serial merging uses a completed output as a source for the next recipe.

## Calibrated methods

`calibration_data.py` generates deterministic synthetic numeric, tool-call, and retention examples. `calibrate.py` either learns component/chunk coefficients against specialist logits and hidden states or collects squared per-sequence response-loss gradients. The latter are calibration estimates, not saved optimizer moments. `merge_calibrated.py` exports either result through the static checkpoint writer.

Calibration requires CUDA GPUs and sufficient CPU RAM for frozen parent checkpoints. Run `calibration_runtime.py` in a disposable worker environment with `--module experiments.weight_merging.calibrate` for the historical calibration or `--module experiments.weight_merging.campaign_calibrate` for Campaign 2. The launcher installs the pinned Transformers and Loguru versions, fetches the explicit `--skyrl-revision`, and applies the recipe's `skyrl_patch` after checking `skyrl_patch_sha256` when present. Use the archived revision `f1ad008a3b260bcb4065a67ec60222e92a5f7e4e` for historical recipes; Campaign 2 records its revision in `skyrl_revision`. Provide the recipe through `--recipe` and a source revision through `--code-revision`. The temporary SkyRL checkout is removed when calibration finishes or fails. CPU merging does not require MarinSkyRL.

`select_experts.py` converts saved benchmark activation comparisons into expert-row recipes with matched random controls. `diagnose.py` measures parent/update Gram matrices. `publish_checkpoint.py` verifies a completion manifest and publishes a checkpoint with an explicitly supplied model card and license.

The archived campaign optimized NUPA200 and BFCL-Parity and did not demonstrate broad capability preservation. These historical tools do not enforce the later campaign rule that at least 75% of Base weights remain unchanged; a new experiment must measure that constraint separately. Raw benchmark transcripts are excluded from the public archive, so exact activation-capture replay is not possible from the archive alone.

Run the CPU behavior tests with:

```sh
uv run --package marin-core --extra cpu --group test pytest tests/test_weight_merging.py experiments/weight_merging/test_select_experts.py
```
