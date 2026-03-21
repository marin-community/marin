# Debugging log for alternating fast bootstrap weight sync

Diagnose and fix the phase-1 alternating RL fast-bootstrap failure on single-host
`v5p-8` without relying on another TPU probe to discover the tensor mismatch.

## Initial status

`ALT-TPU-002 retry6` reached phase-1 fast bootstrap and failed with:

- `ValueError: axis 2 is out of bounds for array of dimension 2`

The failure surfaced in TPU inference during `sync_weights`, after the memory
headroom issue was already resolved at `gpu_memory_utilization=0.92`.

## Hypothesis 1

The fast-bootstrap path is incorrectly reusing the live Levanter-to-vLLM weight
adapter on exported Hugging Face checkpoint tensors.

The async RL path works because it transfers live Levanter state where attention
projection weights still have articulated axes:

- `q_proj`: `(KVHeads, QHeadsPerGroup, HeadSize, Embed)`
- `k_proj` / `v_proj`: `(KVHeads, HeadSize, Embed)`
- `o_proj`: `(Embed, Heads, HeadSize)`

The alternating fast-bootstrap path instead loads exported HF safetensors where
those weights have already been flattened for torch compatibility:

- `q_proj`: `(Heads * HeadSize, Embed)`
- `k_proj` / `v_proj`: `(KVHeads * HeadSize, Embed)`
- `o_proj`: `(Embed, Heads * HeadSize)`

The current helper leaves those tensors 2D, but the TPU inference transpose map
still applies 3D transpose rules:

- `q_proj` / `k_proj` / `v_proj`: `(2, 0, 1)`
- `o_proj`: `(1, 2, 0)`

That contract mismatch explains the exact failure.

## Changes to make

- Add a dedicated HF-export checkpoint to NNX conversion path for fast bootstrap.
- Keep the existing live Levanter state conversion path unchanged for async RL.
- Add regression coverage for flattened HF attention projection weights so
  bootstrap never regresses back to 2D projection tensors.

## Future Work

- [ ] Audit Qwen attention bias handling in the new checkpoint conversion path.
- [ ] Consider renaming the old helper so it is harder to misuse with HF exports.
- [ ] Add a higher-level bootstrap regression test that exercises `_bootstrap_weights_into_engine`.

## Results

Confirmed against the real recovered checkpoint from
`gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/policies/policy_0001`:

- exported `model.layers.0.self_attn.q_proj.weight`: `(4096, 4096)`
- exported `model.layers.0.self_attn.k_proj.weight`: `(1024, 4096)`
- exported `model.layers.0.self_attn.v_proj.weight`: `(1024, 4096)`
- exported `model.layers.0.self_attn.o_proj.weight`: `(4096, 4096)`

Levanter attention code defines the native articulated shapes as:

- `q_proj`: `Out=(KVHeads, QHeadsPerGroup, HeadSize)`
- `k_proj` / `v_proj`: `Out=(KVHeads, HeadSize)`
- `o_proj`: `In=(Heads, HeadSize)`

Scanning the whole exported checkpoint against the current conversion rules found
`128` rank mismatches: all `32` layers times the four attention projection
weights. This is a systematic export/import contract bug, not a one-off tensor.

Follow-up validation on the exact blocked TPU path succeeded:

- `prepare-sampling --phase-id 1` completed on single-host `v5p-8` with
  `gpu_memory_utilization=0.92`
- live sync logs on TPU showed reconstructed attention tensors in native
  Levanter geometry before the existing transpose map ran:
  - `q_proj`: `(32, 128, 4096)`
  - `k_proj`: `(8, 128, 4096)`
  - `v_proj`: `(8, 128, 4096)`
  - `o_proj`: `(4096, 32, 128)`
- `sampling-host --phase-id 1 --host-ordinal 0` also succeeded and wrote
  `sampling/phase_0001/host_0000/status.json` with `num_train_groups=4`

To verify the fix was not just enough for bootstrap, I then resumed the
recovered controller state under the fixed image digest and let the normal
controller finish phase 1:

- materialized `batch_000000.pkl`
- resumed training from checkpoint `step-1`
- completed the final train step to `step-2`
- exported `policy_0002`
- advanced `run_state.json` to `status=completed`, `phase_id=2`,
  `policy_version=2`, `source_global_step=2`

This confirms the robust fix is the contract change itself:

- async RL remains on the live Levanter-state path
- alternating fast bootstrap now has a dedicated HF-export-to-Levanter adapter
- once the exported attention tensors are reconstructed before NNX mapping, the
  existing vLLM sync path works unchanged on real TPU hardware
