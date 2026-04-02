# Debugging log for vLLM Mistral mismatch review

This document records a first-principles root-cause trace of why Llama model
artifacts can show up inside vLLM/TPU logs as `MistralForCausalLM`, why RL
hot-reload then crashes, which layer is responsible for each step, and what
the viable fixes are.

Date of investigation: 2026-03-24

## Initial status

Observed symptom set:

- vLLM logs `Resolved architecture: MistralForCausalLM` for cached Llama GCS
  artifacts.
- `tpu-inference` logs:
  `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`
- RL hot-reload then crashes later on the first weight update with:
  `AttributeError: 'dict' object has no attribute 'flat_state'`

Primary questions:

1. What exact code and package versions are running on Marin `main`?
2. Is the bug in Marin, `vllm-tpu`, or `tpu-inference`?
3. Why can HF repo IDs resolve as Llama while GCS artifacts resolve as
   Mistral?
4. What fixes already exist in local forks and branches?
5. What is the minimally correct fix, and what is the principled long-term fix?

## Investigation method

No code changes were made during the investigation. The following were
inspected directly:

- Marin `main` dependency pins in `lib/marin/pyproject.toml` and `uv.lock`
- exact downloaded wheels for:
  - `vllm-tpu==0.13.2.post6`
  - `tpu-inference==0.13.2.post6`
- exact downloaded public wheels for:
  - `vllm-tpu==0.13.3`
  - `tpu-inference==0.13.3`
- local forks under:
  - `~/code/tpu-inference`
  - `~/code/vllm_tpu_multi/vllm`
- Marin local branches/worktrees, especially:
  - `origin/main`
  - `origin/iris_rl`
  - `origin/alignment_function`
- live GCS artifacts:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`
  - `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
- upstream HF repo layouts via `huggingface_hub.HfFileSystem`

Representative commands used:

```bash
python3 -m pip download --no-deps vllm-tpu==0.13.2.post6 tpu-inference==0.13.2.post6
python3 -m pip download --no-deps vllm-tpu==0.13.3 tpu-inference==0.13.3
gcloud storage ls gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/
gcloud storage cat gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/config.json
gcloud storage cat gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/provenance.json
uv run python - <<'PY'
from huggingface_hub import HfFileSystem
...
PY
git -C ~/code/tpu-inference log -S 'MistralForCausalLM' --oneline -- tpu_inference/models/common/model_loader.py
```

## Ground truth: what Marin `main` actually pins

Marin `main` explicitly pins:

- `vllm-tpu==0.13.2.post6`

Source:
- `lib/marin/pyproject.toml`, lines 171-173

The lockfile resolves both:

- `vllm-tpu==0.13.2.post6`
- `tpu-inference==0.13.2.post6`

Sources:
- `uv.lock`, `[[package]] name = "vllm-tpu"`
- `uv.lock`, `[[package]] name = "tpu-inference"`

Critical packaging fact:

- The downloaded `vllm_tpu-0.13.2.post6` wheel metadata contains:
  `Requires-Dist: tpu-inference==0.13.2.post6`

This means we cannot cleanly move only `tpu-inference` forward under normal
dependency resolution. Any dependency-based fix has to treat the pair as a
coupled versioned stack, or use local/forked overrides.

## Ground truth: mapping the pinned wheels to local fork history

The exact pinned wheels were downloaded and unpacked. Their source files were
then matched against local fork history by hashing file contents.

### Pinned `tpu-inference==0.13.2.post6`

The wheel's `tpu_inference/models/common/model_loader.py` matches local fork
commit:

- `1ea43b1ddab5eaa8c1b3286b1001f90e4f2cc910`

`git describe` for that commit in `~/code/tpu-inference` is:

- `v0.11.1-392-g1ea43b1d`

Important nuance:

- This is a file-content match against local history.
- It does **not** mean the whole PyPI wheel version is `0.11.1`.
- It does mean the relevant file predates later Marin fork fixes by a large
  margin.

### Pinned `vllm-tpu==0.13.2.post6`

The wheel's `vllm/transformers_utils/config.py` matches local fork commit:

- `19c583398aec0ef2b9fe42ba020bc3c39e7e001f`

`git describe` for that commit in `~/code/vllm_tpu_multi/vllm` is:

- `v0.13.0rc1-290-g19c583398`

Again, this is a file-content ancestry observation, not a claim that the whole
wheel is literally that git tag.

## Ground truth: the fixes already exist in the local `tpu-inference` fork

In `~/code/tpu-inference`, the first relevant fixes are:

- `b2c90c99861d234595d53f9163974d3444dde46a`
  `fix: add model_type fallback for vLLM architecture remapping`
- `a74f614221995e150be6b6631510b59119650557`
  `fix: register MistralForCausalLM as alias for LlamaForCausalLM`

Current local fork head:

- `4abb68f4`

The fork now contains:

- `_ABSTRACT_BOOTSTRAP_ARCHITECTURES` includes `MistralForCausalLM`
- `_MODEL_TYPE_TO_REGISTRY_KEY` fallback
- `_MODEL_REGISTRY["MistralForCausalLM"] = LlamaForCausalLM`
- tests for both alias and fallback

Relevant code in the fork:

- `tpu_inference/models/common/model_loader.py`
- `tests/models/common/test_model_loader.py`

The local fork fix is materially better than a one-line local shim because it
encodes the compatibility in the dependency layer where the incompatibility
actually occurs.

## Ground truth: public PyPI `0.13.3` still does not fix this

Public wheels were also downloaded and inspected:

- `vllm-tpu==0.13.3`
- `tpu-inference==0.13.3`

Findings:

- `vllm_tpu-0.13.3` still hard-pins `tpu-inference==0.13.3`
- public `tpu-inference==0.13.3` still does **not** contain:
  - `MistralForCausalLM` alias
  - `_MODEL_TYPE_TO_REGISTRY_KEY`

So:

- bumping Marin from public `0.13.2.post6` to public `0.13.3` is **not**
  sufficient
- the fix exists in the local fork, but not in the public release line we
  inspected

## Ground truth: HF repo layout vs cached GCS artifact layout

This is the most important part of the investigation because it explains why a
direct HF repo path can behave differently from a cached GCS artifact.

### HF repo layout: `meta-llama/Llama-3.1-8B-Instruct@0e9e39f`

Checked via `huggingface_hub.HfFileSystem`:

- `config.json`: present
- `model.safetensors.index.json`: present
- `params.json`: **not** present at repo root
- `consolidated.00.pth`: **not** present at repo root
- `original/params.json`: present
- `original/consolidated.00.pth`: present

Equivalent result for `meta-llama/Llama-3.3-70B-Instruct@6f6073b`:

- root `params.json`: absent
- root `consolidated.00.pth`: absent
- `original/params.json`: present
- `original/consolidated.00.pth`: present

This matters because pinned `vllm-tpu` auto-detects Mistral format only when
`params.json` exists at the model root.

### Cached GCS artifact layout

For the cached 8B artifact:

- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/config.json`
- `.../params.json`
- `.../consolidated.00.pth`
- `.../model.safetensors.index.json`

For the cached 70B artifact:

- `.../config.json`
- `.../params.json`
- `.../consolidated.00.pth` ... `consolidated.07.pth`
- `.../model.safetensors.index.json`

So the GCS artifacts are **hybridized**:

- HF files are present at root
- Meta original-format files are also present at root

This is the layout that triggers pinned `vllm-tpu` to parse the artifact as
Mistral format.

### Provenance: the source files were under `original/`

For the 8B artifact, `provenance.json` records source links including:

- `.../original/consolidated.00.pth`
- `.../original/params.json`
- `.../original/tokenizer.model`

For the 70B artifact, `provenance.json` similarly records:

- `.../original/consolidated.00.pth` ... `original/consolidated.07.pth`
- `.../original/params.json`
- `.../original/tokenizer.model`

But the actual GCS object layout has:

- `consolidated.00.pth` at the root
- `params.json` at the root
- `tokenizer.model` at the root

This proves there was a path-flattening step somewhere between source and final
artifact.

## Historical Marin bug: path flattening explains the 8B artifact exactly

The 8B artifact's provenance access time is:

- `2026-01-23T08:18:04.981719+00:00`

That predates commit:

- `3fdd982b2`
  `Support hf://buckets paths in default_download and default_tokenize (#3793)`
  on 2026-03-19

Current `download_hf.py` preserves relative paths via
`_relative_path_in_source()`.

Before `3fdd982b2`, the path-mapping code was:

```python
fsspec_file_path = os.path.join(output_path, file.split("/", 3)[-1])
```

For a source file like:

```text
meta-llama/Llama-3.1-8B-Instruct@0e9e39f/original/params.json
```

that old code produces:

```text
params.json
```

Likewise:

```text
meta-llama/Llama-3.1-8B-Instruct@0e9e39f/original/consolidated.00.pth
-> consolidated.00.pth
```

Therefore:

- the 8B cached artifact's root-level `params.json` and `consolidated.00.pth`
  are fully explained by a historical Marin bug in `download_hf.py`
- this is not speculation; it follows directly from:
  - source repo layout
  - cached GCS artifact layout
  - artifact creation date
  - exact old code

## 70B artifact flattening: same shape, but provenance is not fully closed

The 70B artifact shows the same hybridized root layout:

- root `params.json`
- root `consolidated.*.pth`

and its `provenance.json` also points back to source files under `original/`.

However, unlike the 8B artifact, its provenance access time is 2026-03-24,
which is after `3fdd982b2` landed.

What is proven:

- the 70B source repo has `original/params.json`, not root `params.json`
- the 70B GCS artifact has root `params.json`
- the current `download_hf.py` path logic would preserve `original/`, not drop it

What is **not** yet proven:

- whether the 70B artifact was produced by:
  - older code running on a stale branch or environment
  - another path-flattening helper outside current `download_hf.py`
  - branch-local code not visible in this worktree snapshot

So for the 70B artifact:

- flattening is definitely real
- its exact producing code path remains partially unresolved

This should be documented explicitly to avoid overstating certainty.

## Exact runtime cause in the pinned stack

Once a cached artifact has root-level `params.json`, the runtime chain is:

### Step 1: pinned `vllm-tpu` chooses Mistral config parsing

In the pinned wheel:

- `vllm/transformers_utils/config.py`

`config_format="auto"` checks:

1. `params.json`
2. only then `config.json`

So if root `params.json` exists, the model is parsed as Mistral-format.

This is a behavior of pinned `vllm-tpu`, not Marin.

### Step 2: pinned Mistral parser rewrites architectures

In the pinned wheel:

- `vllm/transformers_utils/configs/mistral.py`

for the dense non-MoE case:

```python
config_dict["architectures"] = ["MistralForCausalLM"]
```

This means the effective architecture seen downstream is no longer
`LlamaForCausalLM`, even if the original HF `config.json` said Llama.

### Step 3: pinned vLLM registry treats that remap as acceptable

In the pinned wheel:

- `vllm/model_executor/models/registry.py`

there is a direct mapping:

```python
"MistralForCausalLM": ("llama", "LlamaForCausalLM")
```

From vLLM's perspective, Llama and Mistral can share the same backend model
class, so this is not inherently an error.

Then:

- `vllm/config/model.py`

logs:

```text
Resolved architecture: MistralForCausalLM
```

### Step 4: pinned `tpu-inference` cannot translate the remap back to JAX

In the pinned `tpu-inference==0.13.2.post6` wheel:

- `tpu_inference/models/common/model_loader.py`

`_get_model_architecture()` only checks literal `hf_config.architectures`
against `_MODEL_REGISTRY`.

It does **not** have:

- `_MODEL_REGISTRY["MistralForCausalLM"] = LlamaForCausalLM`
- `_MODEL_TYPE_TO_REGISTRY_KEY`

So if `hf_config.architectures == ["MistralForCausalLM"]`, the JAX lookup
fails with `UnsupportedArchitectureError`.

This is the direct compatibility bug.

### Step 5: `MODEL_IMPL_TYPE=auto` makes the failure visible on TPU

In pinned `tpu-inference`:

- `get_model()` reads `envs.MODEL_IMPL_TYPE`
- for `auto`, it resolves dense models to `flax_nnx`
- it tries the JAX path first
- on `UnsupportedArchitectureError`, it logs the warning and falls back to the
  vLLM/PyTorch model

This means:

- the architecture remap itself is not fatal
- the missing translation layer in `tpu-inference` is what causes the bad
  fallback

### Step 6: RL hot-reload crashes later because the fallback runner state is a dict

Pinned `tpu-inference` then hits RL's JAX hot-reload assumption:

- `tpu_runner._sync_weights()` passes `self.state` into
  `transfer_state_with_mappings()`
- `transfer_state_with_mappings()` unconditionally calls
  `tgt_state.flat_state()`

That is valid if `self.state` is an `nnx.State`.

It is invalid if the fallback runner is PyTorch-native and stores plain dict
state.

That is the immediate reason for:

```text
AttributeError: 'dict' object has no attribute 'flat_state'
```

## Why direct HF repo IDs can differ from cached GCS artifacts

This was an important confusion in the earlier logbooks.

The direct HF repo layout for Meta Llama is:

- root `config.json`
- root `model.safetensors.index.json`
- **no** root `params.json`
- `original/params.json` exists

Pinned `vllm-tpu` only checks root `params.json` during config-format
selection. Therefore:

- direct HF repo ID can stay on HF parsing and preserve
  `LlamaForCausalLM`
- cached GCS artifact with root-level flattened `params.json` will switch to
  Mistral parsing and become `MistralForCausalLM`

This is the cleanest explanation of the HF-vs-GCS discrepancy.

It is not fundamentally "GCS streaming" as such.
It is "artifact root layout presented to `vllm-tpu`."

## Where the bug lives

The issue is split across layers.

### Marin

Marin is responsible for part of the trigger surface:

- historical `download_hf.py` definitely flattened `original/*` files into the
  artifact root for the 8B cached artifact
- Marin `main` does not currently expose `config_format="hf"` in its vLLM
  wrapper APIs
- Marin `main` only patches `Qwen2ForCausalLM` in the RL-local
  `_patch_tpu_inference_registry()` shim

Marin is **not** the layer that turns `MistralForCausalLM` into a fatal JAX
lookup miss. That happens below Marin.

### `vllm-tpu`

`vllm-tpu` is responsible for:

- preferring Mistral parsing when root `params.json` exists
- rewriting dense Mistral-format configs to `MistralForCausalLM`
- treating `MistralForCausalLM` as a Llama backend internally

This is not necessarily wrong from vLLM's own perspective.
It becomes a problem only when a lower layer expects exact JAX architecture
keys.

### `tpu-inference`

Pinned `tpu-inference==0.13.2.post6` is the main compatibility bug:

- it does not recognize `MistralForCausalLM` as Llama-equivalent
- it does not use `model_type` fallback
- it falls back to the vLLM/PyTorch runner
- that fallback makes RL hot-reload crash

So if the question is "what actually breaks runtime correctness?", the answer
is:

- pinned `tpu-inference`

If the question is "what causes the wrong architecture string to appear in the
first place?", the answer is:

- pinned `vllm-tpu` config-format auto-detection combined with hybrid artifact
  layout

## Why the earlier local-staging experiment did not disprove this

In the separate `iris_rl` worktree, local metadata staging copied:

- `config.json`
- `generation_config.json`
- tokenizer files
- `model.safetensors.index.json`
- `params.json`

That means the staged local directory still contained root `params.json`.

Therefore:

- local staging changed the transport path
- it did **not** remove the parser trigger

So the result "still resolved as Mistral even after local staging" is fully
consistent with the root-cause above.

## Existing Marin-side workaround in the RL branch

The `iris_rl` worktree already contains an RL-local workaround in:

- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`

It registers:

- `Qwen2ForCausalLM`
- `MistralForCausalLM -> LlamaForCausalLM`

using `model_loader.register_model(...)`.

This is better than direct `_MODEL_REGISTRY[...] = ...` assignment because
`register_model()` updates both:

- the JAX-side registry
- the vLLM-side compatibility wrapper registry

The `iris_rl` branch also includes a unit test in:

- `tests/rl/test_inference_ctx.py`

Important limitation:

- this workaround is RL-specific
- it does not fix every Marin call site that instantiates `vllm.LLM(...)`
  directly

It is an immediate unblock for RL, not a full-stack cleanup.

## Non-RL Marin paths have different failure surfaces

Not every Marin vLLM path is equally exposed to this exact RL crash.

Examples:

- `lib/marin/src/marin/inference/vllm_server.py` explicitly sets
  `MODEL_IMPL_TYPE=vllm`
- `lib/marin/src/marin/evaluation/evaluators/evalchemy_evaluator.py` also sets
  `MODEL_IMPL_TYPE=vllm`

Those paths may still log or resolve `MistralForCausalLM`, but they do not
necessarily hit the same JAX fallback followed by RL hot-reload crash, because
they deliberately prefer the vLLM implementation.

RL is more exposed because it needs the JAX-native runner for hot weight sync.

## Exact conclusions

1. Marin `main` pins a dependency pair:
   - `vllm-tpu==0.13.2.post6`
   - `tpu-inference==0.13.2.post6`
2. Public `0.13.3` is not a sufficient fix; the relevant alias/fallback logic
   is still absent in the public `tpu-inference==0.13.3` wheel.
3. The local `~/code/tpu-inference` fork already has the correct dependency-side
   fix in commits `b2c90c99` and `a74f6142`.
4. The direct runtime compatibility bug is in pinned `tpu-inference`, not in
   Marin.
5. The architecture remap to `MistralForCausalLM` is triggered by pinned
   `vllm-tpu` config auto-detection plus root-level `params.json`.
6. The HF-vs-GCS discrepancy is explained by layout:
   - upstream HF repos keep Meta original files under `original/`
   - cached GCS artifacts expose those files at the root
7. The 8B cached artifact root flattening is fully explained by a historical
   Marin bug in pre-2026-03-19 `download_hf.py`.
8. The 70B cached artifact shows the same flattening, but its exact producing
   path is not fully closed from the current evidence.
9. RL crashes only after the fallback because hot-reload expects `nnx.State`
   and the fallback runner stores dict state.
10. A local RL-only alias patch can unblock RL, but the principled fixes are:
    - stop presenting hybridized root layouts to `vllm-tpu`
    - or move to a dependency pair where `tpu-inference` understands the remap

## Recommended fixes

### Recommended immediate unblock: backport the RL workaround

Backport the proven RL patch from the `iris_rl` worktree:

- register `MistralForCausalLM -> LlamaForCausalLM`
- keep using `register_model(...)`
- backport the unit test too

Why:

- smallest delta
- proven against the exact hot-reload failure mode
- minimal risk to unblock RL on current pins

Why this is not enough by itself:

- RL-only
- does not remove the root trigger
- leaves non-RL paths with confusing hybrid layout semantics

### Recommended principled fix: force HF config parsing for HF-derived cached artifacts

Teach Marin's vLLM wrapper/config plumbing to pass:

- `config_format="hf"`

for cached artifacts that are meant to behave like HF checkpoints.

Why this is attractive:

- it fixes the problem at the source of the wrong architecture key
- it keeps `vllm-tpu` from switching into Mistral config parsing just because
  `params.json` is present
- it matches the behavior of direct HF repo IDs that do not have root
  `params.json`

Important nuance:

- this is a code-based recommendation inferred from the pinned `vllm-tpu` API
  and config-selection logic
- it was not live-validated in this investigation

### Recommended artifact hygiene fix: do not expose Meta original files at artifact root

For future cached model artifacts:

- do not flatten `original/params.json` to `params.json`
- do not flatten `original/consolidated*.pth` to root

Possible ways to achieve this:

1. rely on the fixed post-`3fdd982b2` relative-path logic where applicable
2. stage only HF-compatible files for vLLM consumers
3. maintain separate cache layouts for:
   - HF/vLLM consumption
   - Meta-original tools that actually need `original/*`

### Recommended dependency fix: publish or consume the forked pair

If we want the dependency layer itself to be correct:

- use forked `vllm-tpu` / `tpu-inference` builds that include:
  - `b2c90c99`
  - `a74f6142`

Why:

- that is the right layer for the compatibility rule
- it helps every caller, not just RL
- it aligns the JAX registry with what vLLM already considers acceptable

Why this is harder:

- public PyPI `0.13.3` is not enough
- `vllm-tpu` hard-pins the matching `tpu-inference` version
- we likely need a coordinated forked pair, not a one-sided bump

## Open questions

1. Why does the 70B GCS artifact still have root-level `params.json` even
   though the current mainline `download_hf.py` path logic would preserve
   `original/`?
   - likely stale producing code or another transformation path
   - not yet proven
2. Should Marin expose both:
   - `config_format`
   - `hf_config_path`
   through its vLLM wrapper APIs for better control of hybrid artifacts?
3. Do we want to normalize all cached HF model artifacts to a strict
   HF-compatible layout to avoid similar parser surprises in other tools?

## Bottom line

The fatal runtime bug is in pinned `tpu-inference`.

The reason Llama artifacts become `MistralForCausalLM` in the first place is a
combination of:

- hybrid artifact root layout
- pinned `vllm-tpu` auto-detecting Mistral format from root `params.json`

For the 8B cached artifact, Marin's historical download path flattening is the
specific reason that hybrid root layout exists.

The cleanest short-term fix is:

- backport the RL alias patch and its test

The cleanest long-term fix is:

- stop triggering Mistral parsing for HF-derived cached artifacts
- and/or move Marin onto a forked dependency pair where `tpu-inference`
  understands the remap that vLLM already performs
