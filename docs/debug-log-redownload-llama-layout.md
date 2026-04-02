# Debugging log for redownloading Llama layouts with current Marin HF download

Confirm whether the claim in
`docs/debug-log-vllm-mistral-mismatch-review.md` is still true in current code:

- old Marin downloads could flatten `original/*` files into the artifact root
- current Marin downloads should preserve `original/*`
- determine whether re-downloading Llama 3 checkpoints via `experiments/models.py`
  would avoid the `params.json` -> Mistral parsing issue

## Initial status

- `experiments/models.py` is the model-download entrypoint for cached HF model
  artifacts.
- It delegates directly to `marin.datakit.download.huggingface.download_hf`.
- The earlier investigation tied the bad 8B artifact layout to older path logic
  in `download_hf.py`.

## Hypothesis 1

Current `experiments/models.py` still uses the old flattening logic, so
re-downloading would reproduce the same bad root layout.

## Changes to make

None. Inspect only.

## Results

Hypothesis was false.

Current `experiments/models.py` is only a thin wrapper that constructs a
`DownloadConfig` and calls `download_hf`; it does not manipulate file paths.

Current `download_hf.py` uses `_relative_path_in_source()` to compute the
relative path under the source repo, then writes each file to:

`os.path.join(output_path, relative_file_path)`

For a Meta Llama source path like:

- `meta-llama/Llama-3.1-8B-Instruct@0e9e39f/config.json`
- `meta-llama/Llama-3.1-8B-Instruct@0e9e39f/original/params.json`

the current helper resolves to:

- `config.json`
- `original/params.json`

So current code preserves `original/` and does not flatten `params.json` to the
artifact root.

## Hypothesis 2

The earlier markdown's claim about an old broken layout is consistent with git
history.

## Results

Confirmed.

Commit `3fdd982b2` replaced the old path construction:

`os.path.join(output_path, file.split("/", 3)[-1])`

with the current `_relative_path_in_source()` logic.

That old expression would flatten:

`meta-llama/Llama-3.1-8B-Instruct@0e9e39f/original/params.json`

to:

`params.json`

which exactly matches the bad artifact shape from the earlier investigation.

## Hypothesis 3

Re-running `experiments/models.py` today on the same model/revision path is
enough to fix an existing bad artifact.

## Results

Not necessarily.

`experiments/models.py` uses a stable overridden output path of the form:

`models/<repo>--<revision>`

for the same model and revision. Current downloads would write corrected nested
files like `original/params.json`, but they would not automatically delete
stale bad root files such as:

- `params.json`
- `consolidated*.pth`
- `tokenizer.model`

if those already exist from an older flattened artifact.

Therefore:

- re-download to a **fresh output path** should avoid this issue
- re-download to the **same old output path** may leave the problem intact

## Hypothesis 4

The `marin-us-east5` 8B-Instruct cache either already has the repaired layout
or disproves the earlier region-specific diagnosis.

## Results

Confirmed that the `marin-us-east5` 8B-Instruct cache still has the exact bad
layout signature discussed in the earlier investigation.

Direct GCS listing of:

`gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`

shows:

- root `config.json`
- root `params.json`
- root `consolidated.00.pth`
- no objects under `original/**`

That is the broken hybrid artifact shape that makes vLLM treat the cache like a
Mistral-format checkpoint instead of a pure HF Llama checkpoint.

For contrast, the `marin-us-east5` 70B-Instruct cache at:

`gs://marin-us-east5/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/`

looks repaired:

- no root `params.json`
- no root `consolidated.*.pth`
- `original/params.json` exists
- `original/consolidated.00.pth` through `original/consolidated.07.pth` exist

So the layout problem is not merely a `us-central1` artifact. At minimum, the
`marin-us-east5` 8B-Instruct prefix is also still broken, while the `us-east5`
70B-Instruct prefix appears clean.

## Hypothesis 5

Deleting the broken `marin-us-east5` 8B-Instruct cache prefix in place is a
safe targeted cleanup that leaves adjacent model caches untouched.

## Results

Confirmed on April 1, 2026.

Ran a recursive delete for:

`gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`

After deletion:

- `gcloud storage ls 'gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/**'`
  returns no objects
- the top-level `models/` listing no longer shows the 8B-Instruct prefix
- the adjacent 70B-Instruct prefix
  `gs://marin-us-east5/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/`
  is still present

So the broken `us-east5` 8B cache has been removed cleanly without touching the
70B cache.

## Future Work

- [ ] If we want the strongest artifact-level guarantee, expose
      `hf_urls_glob` in `experiments/models.py` and whitelist only HF-native
      files for Meta Llama caches, omitting `original/**` entirely.
- [ ] Add a layout/version suffix for model downloads so corrected re-downloads
      land in a fresh directory instead of reusing legacy artifact paths.
- [ ] If legacy paths must be reused, add an explicit cleanup step for stale
      root files before or after download.
