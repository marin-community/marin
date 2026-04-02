# Debugging log for tpu-inference main vs pinned vLLM/Mistral mismatch

Determine whether replacing Marin's pinned `tpu-inference==0.13.2.post6` with
the local `/Users/ahmed/code/tpu-inference` `main` branch would fix the
observed RL crash chain where vLLM resolves cached Llama artifacts as
`MistralForCausalLM`, `tpu-inference` falls back to the vLLM runner, and RL
hot-reload later crashes on `dict.flat_state`.

## Initial status

- Marin currently pins `vllm-tpu==0.13.2.post6` and the matching
  `tpu-inference==0.13.2.post6`.
- The local repo `/Users/ahmed/code/tpu-inference` is on `main`.
- Local refs agree:
  - `HEAD == origin/main == upstream/main == bdc454c824038b53b282d9078e2f3d7144115f56`
- Earlier investigation identified compatibility fixes on the local `marin`
  branch, not necessarily on `main`.

## Hypothesis 1

Current `tpu-inference main` already contains the architecture compatibility
fixes, so swapping it in would avoid the fallback and fix RL.

## Changes to make

None. Inspect local branches, source, and wheel contents only.

## Results

Hypothesis was false.

Current `tpu-inference main` does include `Qwen2ForCausalLM` in
`_MODEL_REGISTRY`, so the older Marin-side Qwen2 patch is no longer needed
against current `main`.

But current `tpu-inference main` does **not** include the fixes needed for the
Mistral mismatch:

- no `_MODEL_REGISTRY["MistralForCausalLM"] = LlamaForCausalLM`
- no `_MODEL_TYPE_TO_REGISTRY_KEY` fallback
- no tests covering the Mistral alias or `model_type="llama"` fallback

Therefore, if vLLM still presents `architectures=["MistralForCausalLM"]`,
current `tpu-inference main` will still raise `UnsupportedArchitectureError`
and still fall back to the vLLM-native path.

## Hypothesis 2

The Mistral compatibility fixes exist somewhere in the local repo but are not on
`main`.

## Results

Confirmed.

The local repo contains the exact fixes on the `marin` branch
(`4abb68f4597f1cdc0ccf73d83a30dc829f2b09e9`), including tagged commits:

- `b2c90c99` `fix: add model_type fallback for vLLM architecture remapping`
- `a74f6142` `fix: register MistralForCausalLM as alias for LlamaForCausalLM`

Those commits are not contained in `main`; they are contained in `marin`.

## Hypothesis 3

Even if current `tpu-inference main` lacks the fix, perhaps current upstream
`vllm` no longer produces `MistralForCausalLM` from root-level `params.json`, so
the issue would disappear in a fully updated pair.

## Results

Hypothesis was false.

Current local `vllm` main still:

- uses `params.json` as the Mistral-format trigger in auto config detection
- rewrites dense Mistral-format configs to
  `architectures=["MistralForCausalLM"]`

So a hybrid artifact layout with root `params.json` still reaches the same
architecture string on current vLLM.

That means current `tpu-inference main` would still hit the unsupported-arch
fallback path for this exact class of artifact.

## Hypothesis 4

Even if current `tpu-inference main` does not fix the root cause, maybe it is a
safe one-sided swap with the old pinned `vllm-tpu==0.13.2.post6`.

## Results

It is not a compelling path.

- The pinned `vllm-tpu` wheel hard-pins `tpu-inference==0.13.2.post6`.
- Current `tpu-inference main` appears broadly ABI-adjacent to the old wheel
  for several symbols inspected, but it is a much newer codebase with many
  loader changes.
- Since it still lacks the actual Mistral fix, a one-sided swap would add
  compatibility risk without removing the known failure mode.

## Future Work

- [ ] If the goal is the smallest dependency-side fix, use the local
      `tpu-inference` `marin` branch or backport just `b2c90c99` and
      `a74f6142`.
- [ ] If the goal is a principled stack fix, stop presenting root-level
      `params.json` to vLLM for HF-derived cached artifacts or plumb
      `config_format="hf"`.
- [ ] Remove the Marin RL-local `Qwen2ForCausalLM` patch once the runtime stack
      is moved to a `tpu-inference` version where Qwen2 is registered by
      default.
