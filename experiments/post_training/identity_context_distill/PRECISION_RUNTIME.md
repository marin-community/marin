# Task 310 runtime precision binding

This source is not a precision-pass receipt or permission to train. The four-host/32-H100
collectives, memory, complete LR0 audit and saved-model checks remain H100 gates.

## Source metadata and measured policy are separate

`runtime_recipe.configure_precision_runtime` takes the source config/generation
**bytes**, the immutable torch rounding report bytes, its approved SHA256 and its
regional artifact path. It returns the source-derived recipe with only the
registered router policy applied, plus a schema-v2 derived receipt:

- `source_metadata.model_config.router_dot_fp32` remains **false**; the six native
  files and their hashes are not edited. `source_metadata_receipt` now explicitly
  uses schema version 2; old receipts must be regenerated and reviewed.
- `runtime_model_config.router_dot_fp32` is **true** for the measured dot-only
  trigger; `compute_dtype_overrides` is exactly `{"router_bias": "float32"}`.
- The receipt binds the approved rounding artifact path/hash and native export
  overrides. It is not an exception to the qk/context/expert architecture checks.

For the actual reference-only capture, the rounding report is
`s3://marin-rno2a/marin/experiments/snowball-identity/t310/torch-ref-queue-r1/results/t310-router-rounding-r1/routing.json`,
SHA256 `76af18ce00273d7818506a28a435648bc442078cb8afd9e6ae2caa327c655b32`.
Bias-only changes were zero; dot-only expert-set changes were 3,913 / 805,974.
The dot rule requires both the FP32 dot and native FP32 bias. It does not promote
unrounded FP32 trainable router masters: the dot upcasts the BF16 **compute view**.

## HF reload cannot discard the flag

`levanter.main.train_lm.preserve_hf_runtime_settings` runs **after** the public
converter parses native HF config. HF remains authoritative for every model
architecture field. Only the runtime router flag and pinned checkpoint/tokenizer
IO settings are copied from the requested recipe; the enabled flag requires the
exact bias override.

Stock `train_lm.main` uses this helper at its HF reload. The later
`_load_lm_model_from_configured_source` passes the resolved config explicitly;
passing `None` there would cause a second reload and lose the flag again.

`precision_probe.run_precision_probe` does not invoke stock `train_lm.main`.
It nevertheless uses the SAME post-HF-reload helper before its explicit loader
call. Its first runtime recipe check is before reload; its second is **after**
reload and restoration, before loading weights. It compares the actual loaded
model config to the approved receipt before `Trainer.initial_state`, and writes
`runtime-recipe-receipt.json`. The mandatory gate then traces **state.model** via
`router_dot_dtypes`, requiring FP32 lhs/rhs/out on that actual compute view. No
receipt or recipe flag alone replaces that witness.

The native source receipt and the expected derived runtime receipt are mandatory
arguments to the LR0 lifecycle. The lifecycle re-reads the report bytes and
validates the digest and exact recipe; it never repairs an unbound worker config.
Existing source, initial/post-LR0 master, parity/routing and full export audits
remain mandatory raising callbacks; no default pass implementation was added.

## Regression scope

`test_runtime_reload.py` uses 32 virtual CPU devices, the actual factory mesh,
public HF config conversion, the shared post-reload helper and the real loader
binding (only weight IO replaced with a tiny synthetic Snowball initializer).
It checks that the post-reload config and loaded model retain the flag, native
bias is FP32, router compute weights stay BF16, and the actual model's traced
router dot has FP32 lhs/rhs/out. It is CPU source/mechanics evidence, NOT four-host
or real-model H100 evidence.

`test_runtime_recipe.py` checks measured-policy derivation, immutable native
metadata, changed-evidence/receipt rejection, native scaling-drift rejection and
rejection of a router-master FP32 field override. No generated teacher code is
executed and no real model tensors are loaded locally.
