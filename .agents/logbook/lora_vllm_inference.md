# LoRA vLLM Inference: Research Logbook

## Resolved State
- Status:
  - resolved for the investigated checkpoint
- Root cause:
  - historical LoRA merged HF exports could write merged weights with the
    wrong axis order during [LoraLinear.merge](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py#L176)
- Canonical fix:
  - use the fixed merge code in
    [lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py#L176)
  - re-export from the raw LoRA checkpoint with
    [lora_vllm_investigate.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/lora_vllm_investigate.py#L225)
  - verify the export with the config-derived shape audit in
    [lora_vllm_investigate.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/lora_vllm_investigate.py#L187)
  - ensure `tokenizer_config.json` embeds `chat_template`, not just
    `chat_template.jinja` on disk
- Fallback repair:
  - if raw checkpoints are unavailable, salvage the broken HF export with
    [repair_lora_hf_export.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/repair_lora_hf_export.py#L145)
  - this may restore loadability, but is less trusted behaviorally than a
    clean re-export
- Known-good regenerated export:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`
- Known-good Bloom inference artifact:
  - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768`
- Historical export guidance:
  - treat LoRA-derived merged `hf/step-*` exports produced before the
    `LoraLinear.merge` axis-order fix as potentially tainted until they are
    shape-audited or regenerated

## Canonical Fix Procedure
1. Fix merge code in
   [lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py#L176).
2. Re-export from raw trainer state with
   [lora_vllm_investigate.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/lora_vllm_investigate.py#L225).
3. Run `shape-check` with
   [lora_vllm_investigate.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/lora_vllm_investigate.py#L187)
   and require `0` mismatches.
4. Confirm plain HF load in a Torch-capable environment.
5. Ensure `tokenizer_config.json` contains `chat_template`.
6. Run a TPU vLLM smoke test.
7. Run full Bloom-format inference if the checkpoint is intended for alignment
   eval.

## Open Follow-Up
- Produce an inventory of historical LoRA merged `hf/step-*` exports that need
  regeneration or at least shape-audit verification.

## Scope
- Goal:
  - determine why LoRA-derived Marin checkpoints fail to serve through the
    Marin vLLM inference path, and identify whether the failure is caused by
    the HF export itself, `runai_streamer`, tensor parallel sharding, or some
    combination of those
- Primary metric(s):
  - can `vllm serve` start successfully on Iris for the target checkpoint
  - if not, can we isolate the exact failing tensor or loader mode
- Constraints:
  - Iris cluster only
  - CPU-only control jobs first to validate package behavior in the real task
    image
  - TPU jobs second; no local debugging jobs because the environment differs
  - do not restart or modify the cluster
- Stop criteria:
  - we identify the exact incompatible tensor/export bug, or
  - we demonstrate a working vLLM serving path for the LoRA-derived checkpoint,
    or
  - we prove the current export format is fundamentally incompatible with the
    current Marin vLLM path and file the follow-up implementation/export work

## Baseline
- Date:
  - `2026-04-04`
- Code refs:
  - [eval_llama3_8b_alignment.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/eval_llama3_8b_alignment.py)
  - [vllm_server.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/inference/vllm_server.py)
  - [validate_bloom.md](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/.agents/logbooks/validate_bloom.md)
- Working baseline:
  - full non-LoRA HF exports serve successfully through the Marin eval path
  - Bloom-format inference succeeded for:
    - `marin-8b-instruct`
    - `beta0.01_lr5e-7_seed0`
    - `beta0.01_lr7.5e-7_seed0`
    - `beta0.1_lr5e-7_seed0`
    - `beta0.1_lr7.5e-7_seed0`
- Failing baseline:
  - prior LoRA attempts in `validate_bloom` `EXP-001` already showed
    `AssertionError` shape mismatch while loading LoRA DPO checkpoints
  - both `runai_streamer` and `load_format=auto` had failed there, at `TP=4`
    and `TP=1`
- Current target:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699`
- Current export contents:
  - present:
    - `config.json`
    - `generation_config.json`
    - `chat_template.jinja`
    - `tokenizer.json`
    - `tokenizer_config.json`
    - `special_tokens_map.json`
    - `model.safetensors.index.json`
    - `model-00001-of-00007.safetensors` through `model-00007-of-00007.safetensors`
  - the export looks like a merged full HF model, not adapter-only storage

## Current Diagnosis
- The target export is recognized by vLLM as `LlamaForCausalLM`.
- TPU bring-up succeeds on `us-central1` `v5p-8`.
- The failure is during weight loading, not prompt loading, tokenizer loading,
  TPU initialization, or OpenAI judging.
- Exact root failure from the latest Iris run:
  - `vllm/model_executor/layers/linear.py:1237`
  - `assert param_data.shape == loaded_weight.shape`
  - `AssertionError`
- Top-level symptoms:
  - `RuntimeError: vLLM server process exited before becoming ready`
  - `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`
- Best current hypothesis:
  - at least one tensor inside the tune-LoRA HF export has a shape that does
    not match the model config expected by vLLM's Llama loader

## Key Evidence
- Latest failed wrapper job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8-r2`
- Latest failed child job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8-r2/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lora1699v5p8r2-inference_8b386d48-796e5a8f`
- TPU worker:
  - `marin-tpu-v5p-8-us-central1-a-20260404-0712-ec26bb4b`
- Failing vLLM command:
  - `vllm serve gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 --trust-remote-code --host 127.0.0.1 --port 8000 --load-format runai_streamer --tensor-parallel-size 4 --max-model-len 4096 --gpu-memory-utilization 0.9`
- Load progress at failure:
  - `Loading safetensors using Runai Model Streamer: 2% Completed | 5/291`
- Export metadata:
  - `config.json` advertises a normal 32-layer 8B-class Llama architecture
  - `model.safetensors.index.json` maps standard tensor names such as
    `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj`
  - no obvious adapter-style tensor names remain in the index

## Experiment Plan

| Run | Config Change | Hypothesis |
|-----|--------------|------------|
| LVI-001 | CPU-only Iris job: `transformers` load of `step-1699` | If plain HF load fails, the export itself is malformed and vLLM is only surfacing it |
| LVI-002 | CPU-only Iris job: inspect safetensor shapes against config-derived expected shapes | We can identify the exact offending tensor without needing TPU or vLLM |
| LVI-003 | TPU Iris: `vllm serve`, `TP=1`, `load_format=auto` | If this fails too, the export is broadly incompatible with vLLM, not just streamer or TP sharding |
| LVI-004 | TPU Iris: `vllm serve`, `TP=1`, `load_format=runai_streamer` | If `TP=1 auto` works but this fails, streamer is implicated |
| LVI-005 | TPU Iris: `vllm serve`, `TP=4`, `load_format=auto` | If `TP=1` works and `TP=4` fails, tensor-parallel sharding is implicated |
| LVI-006 | TPU Iris: working non-LoRA export with the same serve harness | Confirms the harness and `v5p-8` environment are healthy under identical settings |
| LVI-007 | Inspect/export pipeline for tune-LoRA merge step | If the export is malformed, find where the bad tensor shape is introduced |

## Run Order
1. `LVI-001`
   - fastest control; answers whether this is already broken at plain HF load
2. `LVI-002`
   - if `LVI-001` fails, pinpoint the tensor and stop wasting TPU cycles
3. `LVI-006`
   - only if needed to reconfirm the serve harness against a known-good export
4. `LVI-003`
   - lowest-complexity TPU vLLM control
5. `LVI-004`
   - isolate streamer vs non-streamer
6. `LVI-005`
   - isolate TP=1 vs TP=4
7. `LVI-007`
   - export-pipeline debugging once we know whether the problem is data or loader

## Fail/Fallback Strategy
- If CPU-only `transformers` load fails:
  - skip TPU loader sweeps
  - inspect tensor names and shapes directly
  - move immediately to export-pipeline debugging
- If `transformers` load succeeds but `TP=1 auto` fails:
  - this is likely a vLLM compatibility issue, not a broken HF export
- If `TP=1 auto` succeeds and `TP=1 runai_streamer` fails:
  - file or track a `runai_streamer` incompatibility
- If both `TP=1` modes succeed and `TP=4 auto` fails:
  - focus on tensor-parallel layout or sharding assumptions

## Open Questions
- Is `step-1699` a truly merged export, or does it still embed LoRA-merged
  tensors in a shape that differs from standard Llama expectations?
- Does plain `transformers` accept the checkpoint?
- Is the failure specific to `runai_streamer`, or does it also reproduce with
  direct safetensor loading?
- Is the mismatch already present at `TP=1`, or only when vLLM shards weights?
- Which exact tensor name triggers the assertion?

## First Planned Run

### 2026-04-04 00:30 - LVI-001: CPU-Only HF Load Control

**Hypothesis**:
- If `AutoModelForCausalLM.from_pretrained()` fails in the Iris CPU image, the
  HF export itself is malformed and the current TPU/vLLM crash is downstream of
  that bad export.

**Environment**:
- Iris cluster only
- CPU-only job
- same repo/task image as the TPU jobs

**Planned command sketch**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-cpu-load-control \
  --cpu 4 --memory 32GB --disk 20GB \
  --region us-central1 \
  -- python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699"

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
print(type(tok).__name__)
print(type(model).__name__)
print(model.config)
PY
```

**Expected outcomes**:
- success:
  - export is valid enough for plain HF loading
  - continue to `LVI-003`
- failure:
  - capture exact tensor/load error
  - proceed to `LVI-002` and `LVI-007`

## Experiment Log

### 2026-04-04 00:32 - LVI-001: CPU-Only HF Load Control

**Hypothesis**:
- if plain `transformers` model loading fails in the Iris task image, the HF
  export itself is malformed and vLLM is just surfacing that bad export

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-hf-load-cpu-r1 \
  --cpu 8 --memory 64GB --disk 40GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    hf-load
```

**Status**:
- launched

### 2026-04-04 00:32 - LVI-002: CPU-Only Safetensor Shape Audit

**Hypothesis**:
- if the export contains malformed standard Llama weights, a streamed
  safetensors-header audit should identify mismatched tensor shapes without
  needing TPU or full vLLM startup

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-shape-check-cpu-r1 \
  --cpu 4 --memory 16GB --disk 20GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    shape-check
```

**Status**:
- launched

### 2026-04-04 00:00 - LVI-HIST-001: Prior LoRA Failures From `validate_bloom`

**Source**:
- [validate_bloom.md](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/.agents/logbooks/validate_bloom.md)

**Observed**:
- older LoRA DPO attempts already failed with `AssertionError` shape mismatch
- both `runai_streamer` and `load_format=auto` were reported as failing
- both `TP=4` and `TP=1` were reported as failing

**Use in this thread**:
- this is strong prior that the new `step-1699` failure is not a one-off TPU
  scheduling or eu-west problem
- the new logbook exists to isolate that broader LoRA/vLLM incompatibility more
  rigorously

### 2026-04-04 00:10 - LVI-HIST-002: Tune-LoRA `step-1699` `v6e-4` Attempt

**Observed**:
- wrapper job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v6e4`
- failure happened before model load
- root issue:
  - executor region incompatibility between `us-central1` GCS paths and the
    available `v6e-4` TPU DAG regions

**Interpretation**:
- not useful for model-format diagnosis

### 2026-04-04 00:15 - LVI-HIST-003: Tune-LoRA `step-1699` `v5p-8` Capacity Retry

**Observed**:
- first `v5p-8` child requested too much RAM and stayed pending
- the inference script was adjusted so `v5p*` uses `128g` rather than `256g`
- this resolved scheduling, not loading

**Interpretation**:
- capacity issues are now separated from the actual model-load problem

### 2026-04-04 00:20 - LVI-HIST-004: Tune-LoRA `step-1699` `v5p-8` Weight-Load Failure

**Observed**:
- wrapper job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8-r2`
- child job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8-r2/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lora1699v5p8r2-inference_8b386d48-796e5a8f`
- TPU worker:
  - `marin-tpu-v5p-8-us-central1-a-20260404-0712-ec26bb4b`
- command:
  - `vllm serve ... --load-format runai_streamer --tensor-parallel-size 4`
- root error:
  - `assert param_data.shape == loaded_weight.shape`
  - `AssertionError`

**Interpretation**:
- this is the first clean reproduction that reaches TPU and dies during actual
  weight loading
- the problem is now narrowed to export-vs-loader compatibility

### 2026-04-04 00:40 - LVI-001 Result: HF Load Control Was Inconclusive

**Observed job**:
- `/ahmed/lora-vllm-hf-load-cpu-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-hf-load-cpu-r1 \
  --cpu 8 --memory 64GB --disk 40GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    hf-load
```

**Outcome**:
- tokenizer loaded successfully:
  - `PreTrainedTokenizerFast`
- model load did not reach a checkpoint verdict because the default Iris CPU
  task image did not have PyTorch installed
- exact failure:
  - `ImportError: AutoModelForCausalLM requires the PyTorch library but it was not found in your environment`

**Interpretation**:
- this result does not tell us whether the HF export is valid
- the tokenizer side of the export is fine
- the decisive evidence comes from the safetensor shape audit instead

### 2026-04-04 00:40 - LVI-002 Result: Shape Audit Found 160 Transposed Weights

**Observed job**:
- `/ahmed/lora-vllm-shape-check-cpu-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-shape-check-cpu-r1 \
  --cpu 4 --memory 16GB --disk 20GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    shape-check
```

**Outcome**:
- checked all `291` tensors in the export
- unchecked tensors: `0`
- found `160` shape mismatches

**Mismatch pattern**:
- for every layer, the following non-square weights are transposed:
  - `mlp.down_proj.weight`: expected `(4096, 14336)`, actual `(14336, 4096)`
  - `mlp.gate_proj.weight`: expected `(14336, 4096)`, actual `(4096, 14336)`
  - `mlp.up_proj.weight`: expected `(14336, 4096)`, actual `(4096, 14336)`
  - `self_attn.k_proj.weight`: expected `(1024, 4096)`, actual `(4096, 1024)`
  - `self_attn.v_proj.weight`: expected `(1024, 4096)`, actual `(4096, 1024)`
- this accounts for `32 layers * 5 non-square matrices = 160 mismatches`

**Important inference**:
- `q_proj` and `o_proj` are also LoRA-targeted modules, but they are square in
  this model family, so a transpose would not change their shapes
- that means the bug is likely broader than the 160 mismatches suggest; the
  square LoRA-targeted matrices may also be orientation-flipped but are not
  shape-detectable

**Interpretation**:
- this is decisive evidence that the merged HF export is malformed for
  Torch/vLLM-style consumption
- the mismatch is systematic, not random corruption

### 2026-04-04 00:42 - LVI-006 Result: Known-Good DPO Export Passed The Same Audit

**Observed job**:
- `/ahmed/lora-vllm-shape-check-good-cpu-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-shape-check-good-cpu-r1 \
  --cpu 4 --memory 16GB --disk 20GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849 \
    shape-check
```

**Outcome**:
- checked all `291` tensors
- unchecked tensors: `0`
- no config-derived shape mismatches found

**Interpretation**:
- the checker itself is sound
- the failure is specific to the LoRA-derived merged export, not to the audit
  method or to all Marin HF exports

### 2026-04-04 00:45 - LVI-007 Result: Root Cause Identified In `LoraLinear.merge()`

**Code-level diagnosis**:
- [LowRankLinear.merge](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py) computes the LoRA delta as:
  - `hax.dot(self.lora_A.weight, self.lora_B.weight, axis=LORA_R)`
- for `out_first=True` linears:
  - `lora_A.weight` has axes `(LORA_R, In)`
  - `lora_B.weight` has axes `(Out, LORA_R)`
- Haliax dot output axes follow first occurrence order from the input axes, so
  that dot returns axes `(In, Out)`, not `(Out, In)`
- before the fix, [LoraLinear.merge](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py) did:
  - `self.lora.merge() + self.wrapped.weight`
- Haliax elementwise add preserves the left operand axis order, so the merged
  weight silently kept the transposed `(In, Out)` ordering

**Why this matches the artifact**:
- non-square LoRA-targeted matrices become obviously wrong in HF export
- square LoRA-targeted matrices (`q_proj`, `o_proj`) can also be transposed,
  but shape checks cannot detect that because `(4096, 4096)` is unchanged

**Fix applied**:
- updated [lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py) to explicitly rearrange the LoRA delta onto the wrapped weight axes before addition:
  - `delta = self.lora.merge().rearrange(self.wrapped.weight.axes)`
  - `weight = self.wrapped.weight + delta`

**Validation**:
- updated [test_lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/tests/test_lora.py) to assert that merged non-square LoRA weights preserve the wrapped linear's axis order
- targeted tests:
  - `uv run --package levanter pytest lib/levanter/tests/test_lora.py -k 'test_merge_lora'`
  - result: `1 passed`
  - `uv run --package levanter pytest lib/levanter/tests/test_lora.py -k 'test_lora_merged_load_in_hf'`
  - result: `skipped` in this local environment because Torch-backed HF load was unavailable
- broader regression check:
  - `uv run --package levanter pytest lib/levanter/tests/test_lora.py`
  - result: `4 passed, 2 skipped`

**Conclusion**:
- root cause solved
- the broken `step-1699` export is malformed because LoRA-merged non-square
  weights were exported with transposed axis order
- the code path now has a direct fix, but the existing GCS artifact will need
  to be re-exported with the fixed merge path before vLLM inference can work

### 2026-04-04 00:50 - LVI-008: Blast Radius Clarification

**Question**:
- does this mean LoRA training in Levanter was broken?

**Answer**:
- no evidence currently points to the LoRA training path itself being broken
- the evidence points to the merged HF export path being broken

**Why**:
- LoRA training in
  [lora_lm.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/main/lora_lm.py)
  runs on the live `loraized` model and trains only LoRA parameters
- that training path does **not** rely on `merge_lora_modules()`
- the broken code path was in
  [LoraLinear.merge](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py),
  which is used by merged-HF export helpers:
  - [save_merged_hf_model](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py)
  - [save_merged_hf_checkpoint_callback](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py)

**Likely unaffected**:
- LoRA training dynamics and optimization
- trainer metrics and loss curves
- raw trainer checkpoints under `checkpoints/`
- in-memory Levanter eval on the LoRA model itself
- PEFT adapter exports from `save_peft_pretrained()`

**Affected**:
- merged HF checkpoints exported from LoRA models through the old merge path
- downstream HF/vLLM inference from those merged exports
- tune-LoRA `hf/step-*` artifacts like the broken `step-1699` export in this
  thread

**Operational conclusion**:
- treat old LoRA-derived merged `hf/step-*` exports as potentially tainted
- do **not** assume the underlying LoRA training runs are invalid
- regenerate merged HF exports with the fixed merge path before using them for
  vLLM/HF inference

## Next Steps
1. Re-export the affected tune-LoRA checkpoint(s) with the fixed
   [LoraLinear.merge](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py) path.
2. Re-run the CPU shape audit on the regenerated export to confirm zero
   mismatches.
3. Re-run TPU vLLM inference on the regenerated export.
4. Optionally add an end-to-end Llama-style merged-HF export test so future
   LoRA regressions are caught before checkpoint publication.

### 2026-04-04 01:05 - LVI-009: Verification Plan, Recovery Plan, And Exit Criteria

**Why a new plan is needed**:
- we now have a code fix and a strong root-cause story, but we do **not** yet
  have the one piece of evidence needed to close the blast-radius question:
  a concrete proof that live Levanter LoRA behavior matches merged export
  behavior after the fix for a Llama-style model
- we also do not yet have a regenerated checkpoint on GCS, so the operational
  problem remains unresolved for downstream inference users

**Current evidence level**:
- strong evidence that old merged HF exports are broken
- strong evidence that the bad export path is
  [LoraLinear.merge](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py)
- moderate evidence that live LoRA training is healthy, because training uses
  the live `loraized` model path rather than merged export
- not yet enough evidence to say with maximal confidence that there is zero
  additional architecture-specific bug in Llama-style merged export

**Primary hypotheses to test from here**:
- `H1`:
  - live LoRA training/inference is correct; the old merged HF export path was
    the only broken component
- `H2`:
  - the fix in
    [lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py)
    is sufficient for Llama-style models, not just the smaller GPT-2-style
    merged-HF regression test already present in
    [test_lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/tests/test_lora.py)
- `H3`:
  - previously published LoRA merged `hf/step-*` exports are tainted anywhere
    they were generated with the old merge path
- `H4`:
  - regenerating the same checkpoint from raw trainer state with the fixed code
    yields a shape-clean HF export that loads in HF and serves in vLLM

**What would count as a convincing resolution**:
- to claim "LoRA training was not broken; merged export was broken", we need
  all of the following:
  1. a live-vs-merged equivalence check passes for a Llama-style model inside
     the real codebase
  2. a regenerated `step-1699` export passes the same CPU shape audit with
     `0` mismatches
  3. the regenerated export loads under HF in a Torch-capable environment
  4. the regenerated export serves through the Iris TPU vLLM path
- until all four are true, the correct wording remains:
  - "training is likely fine, but only merged export is proven broken"

**Experiment sequence from here**:

#### LVI-010 - Add a Llama-style merged-export equivalence test

**Purpose**:
- close the remaining proof gap left by the existing GPT-2 merged-export test

**Hypothesis**:
- after the axis-order fix, a Llama-style LoRA model should satisfy:
  - live Levanter LoRA logits
  - in-memory merged logits
  - HF-reloaded merged checkpoint logits
  - all agree within tolerance on the same input

**Implementation target**:
- extend
  [test_lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/tests/test_lora.py)
  with a small Llama-family configuration rather than only GPT-2 coverage

**Success criterion**:
- new test passes locally/CI and would have failed before the axis-order fix

**Failure interpretation**:
- if this fails, the bug is broader than the currently diagnosed axis-order
  mistake, and we should stop claiming the issue is confined to export only

#### LVI-011 - Inventory potentially tainted historical merged exports

**Purpose**:
- bound the operational blast radius, not just the single reproduction case

**Hypothesis**:
- any LoRA merged `hf/step-*` export produced through the old merge path is
  potentially transposed in the same systematic way

**Method**:
- identify tune-LoRA runs with published merged HF checkpoints
- run the existing `shape-check` command against a targeted list of those
  exports instead of sweeping all buckets blindly

**Success criterion**:
- produce a concrete list of affected paths to re-export or quarantine

**Failure interpretation**:
- if only some runs are affected, there may be an architecture- or
  target-module-specific condition that still needs explanation

#### LVI-012 - Re-export the canonical failing checkpoint from raw trainer state

**Purpose**:
- create the first fixed artifact, not just a source-code fix

**Target**:
- regenerate:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699`
- source of truth should be the corresponding raw trainer checkpoint under the
  same run root, not the already-bad HF export

**Hypothesis**:
- replaying merged export with the fixed
  [LoraLinear.merge](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py)
  will produce a standard-orientation HF checkpoint

**Success criterion**:
- regenerated export writes cleanly and preserves the expected tokenizer/config
  files alongside corrected model weights

**Failure interpretation**:
- if re-export still produces transposed weights, there is another axis-order
  bug downstream of `merge_lora_modules()`

#### LVI-013 - Re-run the CPU shape audit on the regenerated export

**Purpose**:
- cheaply validate the fix before spending TPU cycles

**Hypothesis**:
- the regenerated export will have `0` shape mismatches against config-derived
  expectations

**Success criterion**:
- exact result:
  - checked tensors: all
  - mismatches: `0`

**Failure interpretation**:
- any remaining mismatches mean the export path is still not safe to serve

#### LVI-014 - HF reload control in a Torch-capable runtime

**Purpose**:
- confirm that the fixed merged export is acceptable to standard HF consumers,
  not just the shape checker

**Hypothesis**:
- `AutoModelForCausalLM.from_pretrained()` succeeds on the regenerated export

**Notes**:
- the earlier Iris CPU control was blocked by missing Torch in the default task
  image, so this step needs either:
  - a Torch-capable job image on Iris, or
  - a repo-native test path that already provisions Torch

**Success criterion**:
- model and tokenizer both load; one forward pass succeeds

**Failure interpretation**:
- if HF still fails after a zero-mismatch audit, the export has a semantic
  problem beyond raw tensor orientation

#### LVI-015 - TPU vLLM smoke test on the regenerated export

**Purpose**:
- close the original serving bug

**Hypothesis**:
- the fixed merged export will bring up `vllm serve` successfully under the
  same Iris TPU harness that previously failed

**Run order**:
1. `TP=1`, `load_format=auto`
2. `TP=4`, `load_format=auto`
3. `TP=4`, `load_format=runai_streamer`

**Why this order**:
- it minimizes confounds and separates raw model validity from streamer- or
  TP-specific issues

**Success criterion**:
- vLLM serves successfully and a short generation request returns normally

**Failure interpretation**:
- if `TP=1 auto` fails:
  - export is still bad or vLLM has an independent LoRA-merged compatibility
    problem
- if `TP=1 auto` passes but `TP=4` fails:
  - look next at TP sharding
- if `TP=4 auto` passes but `runai_streamer` fails:
  - streamer remains a separate bug

#### LVI-016 - Full Bloom-format inference and judge follow-through

**Purpose**:
- verify the fixed LoRA export works in the real research workflow, not just a
  smoke test

**Hypothesis**:
- once the regenerated export passes vLLM smoke tests, it should complete the
  standard Bloom-format inference path used elsewhere in
  [validate_bloom.md](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/.agents/logbooks/validate_bloom.md)

**Success criterion**:
- full inference completes on Iris and yields the normal shard/artifact layout
- optional follow-up:
  - GPT-4.1 judge pass if we want direct adherence comparison

#### LVI-017 - Prevention and containment cleanup

**Purpose**:
- prevent recurrence and make the tainted-artifact boundary explicit

**Work items**:
- keep the new regression test in
  [test_lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/tests/test_lora.py)
- if needed, add a targeted Llama-style regression test in addition to the
  GPT-2 one
- document which historical GCS merged exports are tainted
- annotate any issue or handoff notes with the exact bad-export signature:
  - transposed non-square LoRA-targeted weights

**Priority order**:
1. `LVI-010`
2. `LVI-012`
3. `LVI-013`
4. `LVI-014`
5. `LVI-015`
6. `LVI-011`
7. `LVI-016`
8. `LVI-017`

**Why this order**:
- first prove the fix at the code/test level
- then produce a corrected artifact
- then validate the artifact from cheapest checks to most expensive checks
- then scale out the containment work

**Blocking risks**:
- we still need the exact re-export procedure from raw trainer state, not from
  the already-corrupt merged HF artifact
- the default Iris CPU image may still lack Torch for direct HF-load controls
- there may be multiple historical merged exports created with the old code,
  which makes blast-radius tracking partly an inventory problem

**Immediate next action when this thread resumes**:
- implement `LVI-010` first so the codebase has a direct Llama-style proof
  target
- in parallel, resolve the exact re-export entrypoint for `LVI-012`
- do **not** spend more TPU time on the old `step-1699` merged export; it is
  already proven bad

### 2026-04-04 01:15 - LVI-010 Result: Llama-Style Merged Export Regression Added And Passing

**Code change**:
- extended
  [test_lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/tests/test_lora.py)
  with `test_lora_merged_load_in_hf_llama`

**What the new test covers**:
- builds a tiny Llama-style model with non-square LoRA targets
- applies LoRA to:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- checks:
  - live Levanter LoRA logits
  - in-memory merged logits from `merge_lora_modules()`
  - HF-reloaded merged checkpoint logits
- all must agree within tolerance

**Local validation**:
```bash
uv run --with torch --package levanter pytest \
  lib/levanter/tests/test_lora.py -k 'test_lora_merged_load_in_hf_llama'
```

**Outcome**:
- `1 passed, 6 deselected in 14.23s`

**Interpretation**:
- this materially strengthens the claim that the axis-order fix works for a
  Llama-family model, not just the existing GPT-2-style merged-export test
- it does **not** by itself prove that all historical GCS artifacts are clean;
  it proves the fixed code path behaves correctly on a representative Llama
  configuration

### 2026-04-04 01:18 - LVI-012 Prep: Re-export Tool Added

**Code change**:
- extended
  [lora_vllm_investigate.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/lora_vllm_investigate.py)
  with `reexport-merged`

**Purpose**:
- rebuild a LoRA model tree from:
  - the broken HF export's `config.json`
  - explicit LoRA hyperparameters from the saved executor config
  - raw trainer checkpoint state from `checkpoints/step-*`
- then write a fresh merged HF export with the fixed merge path
- optionally run the existing shape audit immediately after export

**Important explicit choice**:
- raw checkpoint subpath is set to `model/policy`
- this matches the DPO policy model we actually export to HF and serve via
  vLLM; we are not trying to export the reference model

### 2026-04-04 01:19 - LVI-012 Launch: CPU Re-export Of `step-1699`

**Observed job**:
- `/ahmed/lora-vllm-reexport-step1699-cpu-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-reexport-step1699-cpu-r1 \
  --cpu 16 --memory 128GB --disk 150GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    reexport-merged \
    --raw-checkpoint-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/checkpoints/step-1699 \
    --checkpoint-subpath model/policy \
    --output-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r1/step-1699 \
    --base-model-ref marin-community/marin-8b-instruct \
    --tokenizer-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    --lora-r 64 \
    --lora-alpha 64 \
    --verify-shapes
```

**Expected result**:
- produce a clean merged HF export under `hf-reexport-r1/step-1699`
- if successful, the same job should then run the post-export shape audit and
  report `0` mismatches

**Status**:
- launched

### 2026-04-04 01:20 - LVI-011 Launch: Historical HF Shape Inventory For This Run

**Observed job**:
- `/ahmed/lora-vllm-shape-inventory-cpu-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-shape-inventory-cpu-r1 \
  --cpu 4 --memory 32GB --disk 40GB \
  --region us-central1 \
  -- bash -lc '
set -uo pipefail
for step_path in $(gcloud storage ls gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/); do
  model_path=${step_path%/}
  echo "=== ${model_path} ==="
  if python experiments/posttrain/lora_vllm_investigate.py --model-path "$model_path" shape-check; then
    echo "RESULT PASS ${model_path}"
  else
    status=$?
    echo "RESULT FAIL ${model_path} exit=${status}"
  fi
  echo
done
'
```

**Purpose**:
- determine whether the transposed-weight signature affects only `step-1699`
  or the entire published HF history for this tune-LoRA run

**Status**:
- launched

### 2026-04-04 01:21 - LVI-012 Result (`r1`): Raw Checkpoint Is Adapter-Only, Not Full Policy Model

**Observed job**:
- `/ahmed/lora-vllm-reexport-step1699-cpu-r1`

**Outcome**:
- failed immediately during checkpoint load

**Attempted assumption**:
- assumed the raw DPO checkpoint stored a full LoRA-wrapped policy tree under
  `model/policy`

**Actual failure**:
- `FileNotFoundError: Missing 28 arrays in OCDBT checkpoint`
- the missing arrays were full-model tensors such as:
  - `model/policy/transformer/layers/stacked/self_attn/q_proj/wrapped/weight`
  - `model/policy/transformer/norm/weight`
  - `model/policy/embeddings/token_embeddings/weight`
  - `model/policy/lm_head/wrapped/weight`
- but the checkpoint **did** contain:
  - LoRA adapter tensors under `model/transformer/.../lora/...`
  - optimizer state
  - training key

**Interpretation**:
- the raw trainer checkpoint is storing adapter/trainable state only
- it is **not** storing a full policy tree under `model/policy`
- therefore the correct re-export path is:
  1. load the base model from the original base checkpoint
  2. loraize that base model with the saved LoRA hyperparameters
  3. load only the adapter subtree from the raw checkpoint, under `model`
  4. combine base weights plus loaded adapters
  5. save a fresh merged HF export

**Consequence**:
- `checkpoint_subpath=model/policy` was wrong
- the re-export tool had to be updated to operate on adapter-only checkpoints

### 2026-04-04 01:22 - LVI-012 Relaunch (`r2`): Base-Model-Plus-Adapter Recovery Path

**Observed job**:
- `/ahmed/lora-vllm-reexport-step1699-cpu-r2`

**Updated command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-reexport-step1699-cpu-r2 \
  --cpu 16 --memory 128GB --disk 150GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    reexport-merged \
    --raw-checkpoint-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/checkpoints/step-1699 \
    --checkpoint-subpath model \
    --output-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r2/step-1699 \
    --base-model-ref gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c \
    --tokenizer-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    --lora-r 64 \
    --lora-alpha 64 \
    --verify-shapes
```

**Status**:
- launched
- confirmed to be reading the base 8B model shards from the regional GCS mirror

### 2026-04-04 01:25 - LVI-012 Result (`r2`): Adapter Tree Excludes `lm_head`

**Observed job**:
- `/ahmed/lora-vllm-reexport-step1699-cpu-r2`

**Outcome**:
- failed during adapter-state load, but much later and with a much narrower
  mismatch than `r1`

**Exact failure**:
- missing only:
  - `model/lm_head/lora/lora_A/weight`
  - `model/lm_head/lora/lora_B/weight`
- all transformer LoRA adapter tensors were present under `model/...`

**Interpretation**:
- the empirical checkpoint contents imply that this run did **not** LoRAize
  `lm_head`
- the correct target-module set for artifact recovery is therefore:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- this is a stronger and more useful result than the executor config alone,
  because it is derived from the actual saved checkpoint contents

### 2026-04-04 01:25 - LVI-012 Relaunch (`r3`): Transformer-Only Adapter Recovery Path

**Observed job**:
- `/ahmed/lora-vllm-reexport-step1699-cpu-r3`

**Updated command difference vs `r2`**:
- added explicit target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- this intentionally excludes `lm_head`

**Status**:
- launched

### 2026-04-04 01:25 - LVI-011 Partial Result: Historical HF Exports Are Not Just Failing At `step-1699`

**Observed job**:
- `/ahmed/lora-vllm-shape-inventory-cpu-r4`

**Confirmed failures from the inventory logs**:
- `step-1000` failed with the same `160` transposed-weight mismatches
- `step-200` failed with the same transposed-weight signature
- `step-1699` failed with the same `160` mismatches

**Interpretation**:
- the bad merged-export signature is not confined to the last checkpoint
- at minimum, multiple published HF exports across the run are tainted
- operationally, this means the whole published `hf/step-*` series for this
  tune-LoRA run should be treated as suspect pending regeneration

### 2026-04-04 01:30 - LVI-013 Launch: Plain Transformers Control On The Broken HF Export

**Purpose**:
- determine whether the old broken merged export is malformed enough to fail
  under plain Hugging Face loading, or whether the breakage is specific to
  vLLM's loader path

**Observed job**:
- `/ahmed/lora-vllm-hf-load-broken-step1699-cpu-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-hf-load-broken-step1699-cpu-r1 \
  --cpu 8 --memory 64GB --disk 80GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    hf-load
```

**Expected result**:
- if this fails, the historical HF export is malformed even for plain
  Transformers
- if this succeeds, then the old artifact may still be salvageable for HF
  consumers even though vLLM rejects it

### 2026-04-04 01:31 - LVI-013 Result (`r1`): Environment Miss, Not A Model Result

**Observed job**:
- `/ahmed/lora-vllm-hf-load-broken-step1699-cpu-r1`

**Outcome**:
- failed before model load with an environment error

**Exact failure**:
- `ImportError: AutoModelForCausalLM requires the PyTorch library but it was not found in your environment`

**Interpretation**:
- this does **not** provide evidence about whether the broken export is
  HF-loadable
- the Iris CPU image used for the control does not include PyTorch by default
- the correct retry is to keep the same experiment but launch it under
  `uv run --with torch`

### 2026-04-04 01:31 - LVI-013 Relaunch (`r2`): Broken Export HF Load With Torch Installed

**Observed job**:
- `/ahmed/lora-vllm-hf-load-broken-step1699-cpu-r2`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-hf-load-broken-step1699-cpu-r2 \
  --cpu 8 --memory 64GB --disk 80GB \
  --region us-central1 \
  -- uv run --with torch python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    hf-load
```

### 2026-04-04 01:31 - LVI-014 Launch: Known-Good Full Export HF Load Control

**Purpose**:
- confirm that the Iris CPU + `uv --with torch` environment can load a known-good
  full HF export under the same code path used for the broken LoRA export

**Observed job**:
- `/ahmed/lora-vllm-hf-load-good-dpo-cpu-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-hf-load-good-dpo-cpu-r1 \
  --cpu 8 --memory 64GB --disk 80GB \
  --region us-central1 \
  -- uv run --with torch python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849 \
    hf-load
```

### 2026-04-04 01:33 - LVI-013/LVI-014 Result: HF Load Helper Mishandled `gs://` Sharded Checkpoints

**Observed jobs**:
- `/ahmed/lora-vllm-hf-load-broken-step1699-cpu-r2`
- `/ahmed/lora-vllm-hf-load-good-dpo-cpu-r1`

**Outcome**:
- both failed with the same loader-path error before any model-specific
  conclusion could be drawn

**Exact failure pattern**:
- `AutoModelForCausalLM.from_pretrained(...)` treated the `gs://...` path like a
  Hugging Face repo id when resolving sharded safetensors
- both jobs attempted URLs of the form:
  - `https://huggingface.co/api/models/gs://.../revision/main`
- both then failed with:
  - `LocalEntryNotFoundError`
  - `OSError: We couldn't connect to 'https://huggingface.co' to load the files`

**Interpretation**:
- this is a bug in the investigation helper, not evidence about either the
  broken or the good artifact
- the correct control must first stage the remote HF directory locally, then
  call `from_pretrained(local_dir, local_files_only=True)`

### 2026-04-04 01:33 - LVI-013/LVI-014 Code Fix: Stage Remote HF Dirs Locally For Plain HF Load

**Code change**:
- updated
  [lora_vllm_investigate.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/lora_vllm_investigate.py)
  so `hf-load` now:
  1. detects non-local model paths
  2. copies the full HF directory to a temp dir
  3. calls Transformers on that local path with `local_files_only=True`

**Rationale**:
- this keeps the experiment on Iris while removing the bogus `gs://` ->
  Hugging Face Hub resolution path from the control

### 2026-04-04 01:33 - LVI-013/LVI-014 Relaunch: Corrected Plain HF Load Controls

**Observed jobs**:
- `/ahmed/lora-vllm-hf-load-broken-step1699-cpu-r3`
- `/ahmed/lora-vllm-hf-load-good-dpo-cpu-r2`

**Commands**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-hf-load-broken-step1699-cpu-r3 \
  --cpu 8 --memory 64GB --disk 80GB \
  --region us-central1 \
  -- uv run --with torch python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    hf-load
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-hf-load-good-dpo-cpu-r2 \
  --cpu 8 --memory 64GB --disk 80GB \
  --region us-central1 \
  -- uv run --with torch python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849 \
    hf-load
```

### 2026-04-04 01:34 - LVI-012 Result (`r3`): Clean Re-Export Of `step-1699`

**Observed job**:
- `/ahmed/lora-vllm-reexport-step1699-cpu-r3`

**Outcome**:
- succeeded

**Produced artifact**:
- `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`

**Observed contents**:
- complete 7-shard HF checkpoint
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `model.safetensors.index.json`

**Post-export verification from the same Iris job**:
- read all `7` safetensors headers
- checked `291` tensors
- `unchecked tensors: 0`
- `No config-derived shape mismatches found`

**Interpretation**:
- this is the first clean HF export recovered from the old LoRA trainer state
- the fixed merge path is sufficient to rebuild a structurally valid HF export
  for the previously broken run

### 2026-04-04 01:35 - LVI-015 Launch: TPU vLLM Smoke Test On The Regenerated Export

**Purpose**:
- test the actual serving path that previously failed, using the regenerated
  export instead of the old broken one

**Observed job**:
- `/ahmed/lora-vllm-smoke-reexport-step1699-v5p8-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-smoke-reexport-step1699-v5p8-r1 \
  --tpu v5p-8 \
  --cpu 4 --memory 128GB --disk 200GB \
  --region us-central1 \
  --extra tpu \
  -- python -m marin.inference.vllm_smoke_test \
    --local \
    --mode native \
    --model gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699 \
    --load-format runai_streamer \
    --max-model-len 8192 \
    --use-completions \
    --prompt "Write one short sentence about TPUs."
```

**Pass criterion**:
- server becomes ready
- one completion request returns text
- job exits successfully

### 2026-04-04 01:37 - LVI-015 Result (`r1`): Native vLLM Binary Missing From The Plain Iris Env

**Observed job**:
- `/ahmed/lora-vllm-smoke-reexport-step1699-v5p8-r1`

**Outcome**:
- failed before model load

**Exact failure**:
- `FileNotFoundError: [Errno 2] No such file or directory: 'vllm'`
- traceback originated in
  [vllm_server.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/inference/vllm_server.py)
  when native mode tried to `subprocess.Popen(["vllm", "serve", ...])`

**Interpretation**:
- this is another environment-contract miss, not a checkpoint failure
- the job launched `python -m marin.inference.vllm_smoke_test` in the repo env,
  but native vLLM expects the `vllm` CLI from `vllm-tpu` to be installed on
  `PATH`
- the correct retry is to wrap the smoke test in `uv run --with vllm-tpu`

### 2026-04-04 01:37 - LVI-015 Relaunch (`r2`): TPU Smoke With Native `vllm-tpu` Installed

**Observed job**:
- `/ahmed/lora-vllm-smoke-reexport-step1699-v5p8-r2`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-smoke-reexport-step1699-v5p8-r2 \
  --tpu v5p-8 \
  --cpu 4 --memory 128GB --disk 200GB \
  --region us-central1 \
  --extra tpu \
  -- uv run --with vllm-tpu python -m marin.inference.vllm_smoke_test \
    --local \
    --mode native \
    --model gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699 \
    --load-format runai_streamer \
    --max-model-len 8192 \
    --use-completions \
    --prompt "Write one short sentence about TPUs."
```

### 2026-04-04 01:39 - LVI-015 Result (`r2`): Regenerated Export Reaches vLLM Model Validation On TPU

**Observed job**:
- `/ahmed/lora-vllm-smoke-reexport-step1699-v5p8-r2`

**Outcome**:
- failed during vLLM API server startup, but **not** with the old merged-weight
  assertion

**What advanced successfully**:
- native `vllm` binary launched on `v5p-8`
- vLLM resolved the regenerated export as `LlamaForCausalLM`
- the old shape-mismatch failure did **not** recur

**New failure**:
- `ValidationError: User-specified max_model_len (8192) is greater than the derived max_model_len (4096)`
- vLLM recommended either:
  - lowering `max_model_len` to `4096`, or
  - setting `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` with caution

**Important contrast with the broken export**:
- the historical artifact failed immediately at weight load with transposed
  shapes
- the regenerated artifact gets past model recognition and into normal vLLM
  config validation

**Interpretation**:
- this is strong evidence that the merge/export bug is fixed
- the remaining blocker is now a serving-configuration mismatch, not corrupted
  weights

### 2026-04-04 01:39 - LVI-015 Relaunch (`r3`): TPU Smoke With `max_model_len=4096`

**Observed job**:
- `/ahmed/lora-vllm-smoke-reexport-step1699-v5p8-r3`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-smoke-reexport-step1699-v5p8-r3 \
  --tpu v5p-8 \
  --cpu 4 --memory 128GB --disk 200GB \
  --region us-central1 \
  --extra tpu \
  -- uv run --with vllm-tpu python -m marin.inference.vllm_smoke_test \
    --local \
    --mode native \
    --model gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699 \
    --load-format runai_streamer \
    --max-model-len 4096 \
    --use-completions \
    --prompt "Write one short sentence about TPUs."
```

### 2026-04-04 01:40 - LVI-013/LVI-014 Result: Broken-vs-Good HF Load Split Confirmed

**Observed jobs**:
- broken export:
  `/ahmed/lora-vllm-hf-load-broken-step1699-cpu-r3`
- known-good full export:
  `/ahmed/lora-vllm-hf-load-good-dpo-cpu-r2`

**Outcome**:
- broken export: failed
- known-good full export: succeeded

**Broken export exact failure**:
- after staging the remote HF dir locally and beginning shard load, plain
  Transformers failed with:
  - `RuntimeError: Error(s) in loading state_dict for Linear`
  - `size mismatch for weight: copying a param with shape torch.Size([14336, 4096]) from checkpoint, the shape in current model is torch.Size([4096, 14336])`

**Known-good export exact result**:
- loaded all `7` shards successfully
- resolved as `LlamaForCausalLM`
- parameter count: `8030261248`
- previewed expected tensor shapes like:
  - `model.layers.0.self_attn.k_proj.weight (1024, 4096)`
  - `model.layers.0.mlp.gate_proj.weight (14336, 4096)`
  - `model.layers.0.mlp.down_proj.weight (4096, 14336)`

**Interpretation**:
- this is decisive evidence that the old LoRA merged HF export is broken for
  plain Hugging Face consumers, not only for vLLM
- the loader failure signature matches the earlier shape-audit diagnosis exactly

### 2026-04-04 01:40 - LVI-016 Interim Conclusion: Root-Cause Loop Is Closed

**What is now established**:
- historical tune-LoRA merged HF exports are corrupted by a merge/export bug
- the corruption is the transpose of non-square linear weights
- the broken exports fail:
  - config-derived shape audit
  - plain HF load
  - vLLM load
- the fixed merge path produces a regenerated export that:
  - passes shape audit (`291` checked, `0` mismatches)
  - is accepted far enough by vLLM to resolve architecture and enter normal
    config validation

**Practical meaning**:
- the original research question ("is Levanter LoRA training broken?") now has
  a materially narrower answer:
  - the old merged HF export path was broken
  - the repaired export path fixes the corruption signature
  - the remaining TPU smoke failures are now ordinary serving-environment /
    serving-config issues, not weight corruption

### 2026-04-04 01:40 - LVI-017 Current State / Next Steps

**Active job**:
- `/ahmed/lora-vllm-smoke-reexport-step1699-v5p8-r3`

**Live status when this entry was written**:
- `JOB_STATE_RUNNING`
- on `us-central1` `v5p-8`
- launched with the corrected `--max-model-len 4096`

**Most likely next outcomes**:
1. if `r3` succeeds:
   - the end-to-end recovery is proven on TPU serving
   - regenerate additional tainted `hf/step-*` exports from the same run
2. if `r3` fails:
   - inspect the new failure; by this point it will be a post-corruption
     serving issue rather than the original broken-merge bug

**Historical remediation already implied by current evidence**:
- treat the published `hf/step-*` series under
  `bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/`
  as tainted pending regeneration

### 2026-04-04 10:27 - LVI-015 Final Result (`r3`): End-to-End TPU vLLM Smoke Passed

**Observed job**:
- `/ahmed/lora-vllm-smoke-reexport-step1699-v5p8-r3`

**Outcome**:
- succeeded

**What changed vs `r2`**:
- kept the regenerated export and native `vllm-tpu` environment
- lowered `--max-model-len` from `8192` to `4096`

**Observed successful behavior**:
- job reached the actual query stage on `us-central1` `v5p-8`
- `vllm_smoke_test` completed `[run 1/1]` in `201.6s`
- returned a non-empty completion:
  - `TPUs are specialized computer chips designed to accelerate machine learning workloads.`

**Interpretation**:
- this closes the loop on the original bug:
  - old LoRA merged HF export was corrupted
  - fixed merge path regenerated a clean HF export
  - the regenerated export serves successfully with native TPU vLLM
- the final TPU-side blockers encountered during recovery were ordinary
  environment / serving-config issues:
  - missing `vllm` binary in the plain repo env
  - too-large `max_model_len`
- neither was a model corruption problem

### 2026-04-04 10:27 - LVI-018 Final Conclusion / Operational Guidance

**Conclusion**:
- Levanter LoRA training is **not** shown to be broken by this thread
- the broken component was the old merged HF export path
- the export bug is fixed by the merge-axis correction in
  [lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py)

**Strong evidence now in hand**:
- tiny Llama regression test passes after the fix
- regenerated `step-1699` export passes shape audit
- old broken export fails plain HF load with transposed-weight mismatch
- known-good export loads under the same HF control
- regenerated export serves successfully on TPU vLLM

**Operational next steps**:
1. regenerate the tainted historical `hf/step-*` exports for this tune-LoRA run
2. mark the old `hf/` series as suspect / deprecated in any downstream usage
3. optionally add a CI/integration test that:
   - trains or constructs a small Llama LoRA model
   - exports merged HF
   - reloads with HF
   - runs a vLLM smoke test

### 2026-04-04 10:31 - LVI-019 Code Change: Direct HF Export Repair Script Added

**Code change**:
- added
  [repair_lora_hf_export.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/repair_lora_hf_export.py)

**Purpose**:
- repair a broken historical merged HF export directly, without:
  - rerunning training
  - reading the raw LoRA trainer checkpoint
  - reconstructing from the base model

**Mechanism**:
- reads each safetensors shard from the broken export
- derives expected shapes from `config.json`
- for any tensor where `actual_shape != expected_shape` but
  `transpose(actual).shape == expected_shape`, transposes the tensor
- writes repaired shards to a new HF export directory
- copies tokenizer/config metadata
- runs the same post-repair shape audit used elsewhere in this thread

**Intent**:
- this is the exact one-off recovery path needed for historical artifacts that
  already exist on GCS and should be salvaged without retraining

### 2026-04-04 10:31 - LVI-020 Launch: Direct Repair Of The Broken `step-1699` HF Export

**Observed job**:
- `/ahmed/lora-vllm-direct-repair-step1699-cpu-r1`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-direct-repair-step1699-cpu-r1 \
  --cpu 8 --memory 64GB --disk 120GB \
  --region us-central1 \
  -- python experiments/posttrain/repair_lora_hf_export.py \
    --source-model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 \
    --output-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-repair-direct-r1/step-1699
```

**Pass criterion**:
- repaired artifact written under `hf-repair-direct-r1/step-1699`
- post-repair shape audit reports `0` mismatches

### 2026-04-04 11:43 - LVI-021 Direct Repair `r1` Failure Diagnosis

**Observed**:
- `/ahmed/lora-vllm-direct-repair-step1699-cpu-r1` failed
- controller summary:
  - `Exit code 137: OOM killed (container exceeded memory limit)`
- partial output was left under:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-repair-direct-r1/step-1699`
- that path was incomplete and not safe to treat as a valid HF export

**Interpretation**:
- the direct repair approach itself was sound
- the issue was just job sizing; `64GB` memory was too small for the full
  shard-by-shard rewrite plus upload path

### 2026-04-04 11:44 - LVI-022 Reproducible Iris Submit Wrapper Added

**Code change**:
- added
  [submit_lora_hf_repair_job.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/submit_lora_hf_repair_job.py)

**Purpose**:
- provide a one-command Iris submission path for direct LoRA HF export repair
- derive a fresh `hf-repair-direct-<run-label>/step-<n>` output path
- print both the submitted job ID and the `hf_repair_path`

**Default target**:
- source:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699`
- default region:
  - `us-central1`
- default resources:
  - `8 CPU`
  - `128GB` RAM
  - `200GB` disk

### 2026-04-04 11:44 - LVI-023 Direct Repair `r2` Success

**Command**:
```bash
uv run python experiments/posttrain/submit_lora_hf_repair_job.py \
  --run-label r2 \
  --memory 128GB \
  --disk 200GB
```

**Observed job**:
- wrapper:
  - `/ahmed/lora-vllm-direct-repair-step1699-cpu-r2`

**Output path**:
- `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-repair-direct-r2/step-1699`

**Result**:
- all `7` repaired safetensor shards uploaded
- HF metadata copied:
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `chat_template.jinja`
  - `model.safetensors.index.json`
- repair summary:
  - `tensors=291`
  - `transposed=160`
  - `unchanged=131`
- post-repair shape audit:
  - `Checked 291 tensors`
  - `Unchecked tensors: 0`
  - `No config-derived shape mismatches found`
- Iris terminal state:
  - `JOB_STATE_SUCCEEDED`

**Conclusion**:
- the direct salvage path now works end to end
- no retraining is needed to recover this broken historical HF export
- the correct repaired artifact for downstream use is:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-repair-direct-r2/step-1699`

### 2026-04-04 12:00 - LVI-024 Planned: TPU vLLM Smoke Test For Directly Repaired Export

**Hypothesis**:
- the directly repaired HF export should behave like the regenerated clean
  export and bring up `vllm serve` successfully on Iris TPU without the old
  shape-mismatch crash

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-smoke-repairdirect-step1699-v5p8-r1 \
  --tpu v5p-8 \
  --cpu 4 --memory 128GB --disk 200GB \
  --region us-central1 \
  --extra tpu \
  -- uv run --with vllm-tpu python -m marin.inference.vllm_smoke_test \
    --local \
    --mode native \
    --model gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-repair-direct-r2/step-1699 \
    --load-format runai_streamer \
    --max-model-len 4096 \
    --use-completions \
    --prompt "Write one short sentence about TPUs."
```

**Pass criterion**:
- Iris job reaches `JOB_STATE_SUCCEEDED`
- no `assert param_data.shape == loaded_weight.shape` crash in vLLM startup
- smoke test returns a non-empty completion

### 2026-04-04 12:47 - LVI-024 Result: Directly Repaired Export Survives TPU vLLM Startup

**Observed job**:
- `/ahmed/lora-vllm-smoke-repairdirect-step1699-v5p8-r1`

**Outcome**:
- succeeded

**Observed behavior**:
- Iris terminal state:
  - `JOB_STATE_SUCCEEDED`
- the job ran through the full native TPU vLLM smoke path and returned a
  completion after:
  - `[run 1/1] 211.8s`
- there was **no** recurrence of the old crash signature:
  - no `assert param_data.shape == loaded_weight.shape`
  - no `Engine core initialization failed`
  - no early `vllm serve` process exit

**Returned completion**:
```text
*b, be you, typically, let, be, be, led, led, **, **, group, group, group, group, de,chi,chi,chi, de, de, cav cav cav cav cav cav cav cav cav cav cav cav cav cav cav cav cav cav cav cav cav cav wasching cav wasching cav wasching cav wasching cav wasching cav wasching cav wasching cav cav cav cav cav wasching cav wasching cav wasching cav wasching cav wasching cav wasching cav wasching cav wasching wasching wasching wasching wasching wasching and wasching and wasching
```

**Interpretation**:
- this is a meaningful improvement over the broken historical export:
  - the directly repaired artifact is now **servable** through native TPU vLLM
- however, the first smoke completion is nonsensical, unlike the clean
  `hf-reexport-r3` export which returned a sensible TPU sentence
- so the direct repair path appears to have solved the hard loader crash, but
  it is **not yet proven behaviorally equivalent** to the clean re-export path

**Updated conclusion**:
- `hf-repair-direct-r2/step-1699` is sufficient to avoid the old vLLM crash
- behavioral quality remains an open question for the direct-repair artifact
- the cleanest trusted artifact is still:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`
- the strongest next follow-up would be a side-by-side deterministic smoke or
  short prompt suite comparing:
  - broken export
  - direct repair export
  - clean re-export

### 2026-04-04 12:50 - LVI-025 Planned: Re-Run TPU vLLM Smoke On Clean Re-Export Artifact

**Decision**:
- for actual downstream use we should prefer the clean re-export artifact over
  the direct-repair artifact
- the direct-repair result was useful to prove salvageability, but the first
  completion looked behaviorally wrong
- the clean re-export already has stronger evidence:
  - passed shape audit
  - passed plain HF load
  - previously returned a sensible TPU completion

**Hypothesis**:
- rerunning the smoke test on `hf-reexport-r3/step-1699` should again produce a
  sensible non-garbage completion, confirming that this is the correct artifact
  to use

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-vllm-smoke-reexport-step1699-v5p8-r4 \
  --tpu v5p-8 \
  --cpu 4 --memory 128GB --disk 200GB \
  --region us-central1 \
  --extra tpu \
  -- uv run --with vllm-tpu python -m marin.inference.vllm_smoke_test \
    --local \
    --mode native \
    --model gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699 \
    --load-format runai_streamer \
    --max-model-len 4096 \
    --use-completions \
    --prompt "Write one short sentence about TPUs."
```

### 2026-04-04 12:58 - LVI-025 Result: Clean Re-Export Still Produces Sensible TPU Text

**Observed job**:
- `/ahmed/lora-vllm-smoke-reexport-step1699-v5p8-r4`

**Outcome**:
- succeeded

**Observed completion**:
```text
TPUs are specialized computer chips designed to accelerate machine learning workloads.

**Example:** TPUs are specialized computer chips designed to accelerate machine learning workloads.

Let me know if you’d like a longer explanation!
```

**Runtime**:
- `[run 1/1] 201.1s`

**Interpretation**:
- this reconfirms that the clean re-export artifact behaves sensibly under the
  same TPU vLLM smoke harness
- in contrast, the direct-repair artifact was merely "servable" but returned
  garbage text on the same prompt
- for downstream inference, the correct choice is the clean artifact:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`

**Updated operational guidance**:
- use `hf-reexport-r3/step-1699` for evaluation or serving
- treat `hf-repair-direct-r2/step-1699` as a useful recovery/debug artifact,
  not the preferred production artifact

### 2026-04-04 13:05 - LVI-026 Planned: Bloom Inference Sweep For Clean LoRA Re-Export

**Goal**:
- run Bloom-format inference only for the preferred clean LoRA artifact so it
  can enter the same Marin-vs-Bloom evaluation path as the other checkpoints

**Code change**:
- updated
  [eval_llama3_8b_alignment.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/eval_llama3_8b_alignment.py)
  so target `tune_lora_lr1e5_seed0_step1699` now points to:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`

**Planned command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name eval-marin-tune-lora-clean-bloom-us-central1-v5p8 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region us-central1 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-central1 \
    --tpu-type v5p-8 \
    --target tune_lora_lr1e5_seed0_step1699 \
    --run-label cleanreexportr1
```

**Pass criterion**:
- inference step reaches `JOB_STATE_SUCCEEDED`
- output artifact contains the expected Bloom eval shards and `vllm_metrics`

### 2026-04-04 13:12 - LVI-026 Result: First Bloom Inference Attempt Failed On Missing Chat Template Metadata

**Observed jobs**:
- wrapper:
  - `/ahmed/eval-marin-tune-lora-clean-bloom-us-central1-v5p8`
- child:
  - `/ahmed/eval-marin-tune-lora-clean-bloom-us-central1-v5p8/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr1-inference_05fa2505-70a64335`

**Observed output path**:
- `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr1/inference-3904ad`

**Outcome**:
- child inference step failed after native TPU vLLM startup

**Exact failure**:
- `vLLM environment ready`
- `Rendering 256 chat prompts for batched vLLM serve`
- then:
  - `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!`

**Interpretation**:
- the clean re-export fixed the broken LoRA weights, but Bloom inference still
  depends on `tokenizer.chat_template` being embedded in
  `tokenizer_config.json`
- both the historical broken export and the clean re-export had:
  - `chat_template.jinja` present on disk
  - no `chat_template` field in `tokenizer_config.json`
- this is a tokenizer metadata problem, not a vLLM loader problem

### 2026-04-04 13:14 - LVI-027 Intervention: Patch Clean Re-Export Tokenizer Metadata In Place

**Decision**:
- patch the preferred clean artifact in place instead of creating yet another
  artifact path, because the missing piece is only tokenizer metadata

**Patch performed**:
- downloaded:
  - `hf-reexport-r3/step-1699/tokenizer_config.json`
  - `hf-reexport-r3/step-1699/chat_template.jinja`
- inserted the full `chat_template.jinja` contents into the JSON field:
  - `tokenizer_config["chat_template"] = <template text>`
- uploaded the patched `tokenizer_config.json` back to:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699/tokenizer_config.json`

**Verification**:
- `gcloud storage cat .../tokenizer_config.json | rg chat_template`
  showed the field present after the patch

### 2026-04-04 13:15 - LVI-028 Planned: Re-Run Bloom Inference On Patched Clean Re-Export

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name eval-marin-tune-lora-clean-bloom-us-central1-v5p8-r2 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region us-central1 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-central1 \
    --tpu-type v5p-8 \
    --target tune_lora_lr1e5_seed0_step1699 \
    --run-label cleanreexportr2
```

### 2026-04-04 13:24 - LVI-028 Result: Bloom Inference Succeeded On Patched Clean Re-Export

**Observed jobs**:
- wrapper:
  - `/ahmed/eval-marin-tune-lora-clean-bloom-us-central1-v5p8-r2`
- child:
  - `/ahmed/eval-marin-tune-lora-clean-bloom-us-central1-v5p8-r2/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2-inference_6ccccbf4-a5522d90`

**Observed output path**:
- `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768`

**Outcome**:
- wrapper and child inference step both succeeded

**Observed progress / result**:
- native TPU vLLM startup completed normally
- Bloom-format prompt rendering succeeded after the metadata patch
- live throughput was normal for `v5p-8`:
  - roughly `21-25 items/s`
- final write:
  - `Wrote 7728 records to 2 shards`
  - `Wrote 7728 eval inference results to gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768`

**Artifact contents confirmed**:
- `.artifact`
- `.executor_info`
- `.executor_status`
- `shard_00000.jsonl.gz`
- `shard_00001.jsonl.gz`
- `artifacts/vllm_metrics.json`

**Updated conclusion**:
- the preferred clean LoRA artifact is now usable for full Bloom-format Marin
  inference, provided `tokenizer_config.json` includes the embedded
  `chat_template`
- the relevant inference artifact for downstream judging is:
  - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768`

### 2026-04-08 22:36 - LVI-029: Rollout Plan To Regenerate Historical Tune-LoRA HF Exports

**Goal**:
- regenerate the historical tune-LoRA merged `hf/step-*` exports into fresh
  sibling directories without touching the original published exports

**Scope**:
- runs under:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/`
- output convention:
  - keep old artifacts at `.../hf/step-*`
  - write new artifacts at `.../hf-fixed-<run_label>/step-*`
- current one-off submit helper:
  - [submit_lora_hf_fixed_batch.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/submit_lora_hf_fixed_batch.py)

**Key operational finding**:
- the clean re-export command is correct, but Iris controller submission is
  currently flaky under repeated `launch_job(...)` calls
- what worked:
  - direct single-job `iris job run --no-wait` submission
  - confirmed live example:
    - `/ahmed/lora-hf-fixed-lr1e6-seed0-f55cdd-step200-r1`
    - output already appeared under:
      - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd/hf-fixed-r1/step-200`
- what did not work reliably:
  - Python `IrisClient.submit(...)` fanout submitter
  - launching a second direct `iris job run` in parallel while the first was
    still running
  - failure mode:
    - `ConnectError: DEADLINE_EXCEEDED`
    - `Operation launch_job(...) failed`
    - `Request timed out`

**Recommended rollout strategy**:
1. submit one repair job at a time with direct `iris job run --no-wait`
2. wait until the job is visible on the controller before submitting the next
   job
3. prefer a single active repair job at a time until controller launch
   reliability improves
4. do not delete or overwrite any original `hf/step-*` exports
5. use `hf-fixed-r1` for the first full regeneration sweep
6. after each completed step, verify:
   - output path exists
   - `model.safetensors.index.json` exists
   - `tokenizer_config.json` embeds `chat_template`
7. only after a run’s repaired `hf-fixed-r1/step-*` artifacts are complete,
   consider downstream vLLM or Bloom inference checks

**Canonical per-step command template**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name <job-name> \
  --cpu 16 --memory 128GB --disk 150GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path <run>/hf/<step> \
    reexport-merged \
    --raw-checkpoint-path <run>/checkpoints/<step> \
    --checkpoint-subpath model \
    --output-path <run>/hf-fixed-r1/<step> \
    --base-model-ref gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c \
    --tokenizer-path <run>/hf/<step> \
    --lora-r 64 \
    --lora-alpha 64 \
    --lora-target-module q_proj \
    --lora-target-module k_proj \
    --lora-target-module v_proj \
    --lora-target-module o_proj \
    --lora-target-module gate_proj \
    --lora-target-module up_proj \
    --lora-target-module down_proj \
    --verify-shapes
```

**Why this command is the right one**:
- it uses the fixed merge logic in
  [lora.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/levanter/src/levanter/lora.py#L176)
- it regenerates from raw trainer state rather than trying to trust the broken
  merged HF shards
- it now also embeds `chat_template` into `tokenizer_config.json` as part of
  the re-export path

**Immediate next steps for the sweep**:
- continue the `lr1e6 seed0` run step-by-step after `step-200`
- if serial direct submission remains stable, expand the same pattern across:
  - `lr1e6 seed2`
  - `lr2p5e6 seed0/seed2`
  - `lr3p75e6 seed0/seed2`
  - `lr4p5e6 seed0/seed2`
  - `lr5e6 seed0/seed2`
  - `lr6p25e6 seed0/seed2`
  - `lr7p5e6 seed0/seed2`
  - `lr8p75e6 seed0/seed2`
  - `lr1e5 seed0/seed2`

**Open blocker**:
- controller-side launch flakiness is currently the limiting factor for batch
  regeneration throughput, not the re-export logic itself

### 2026-04-08 22:40 - LVI-030: Concrete Execution Plan For Full `hf-fixed-r1` Sweep

**Current observed state**:
- the first historical repair job is live and writing a complete-looking fixed
  HF directory:
  - job:
    - `/ahmed/lora-hf-fixed-lr1e6-seed0-f55cdd-step200-r1`
  - output:
    - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd/hf-fixed-r1/step-200`
- confirmed objects already present there:
  - all `model-0000*-of-00007.safetensors` shards
  - `model.safetensors.index.json`
  - `config.json`
  - `generation_config.json`
  - `chat_template.jinja`
  - `tokenizer_config.json`
- the re-export path is therefore behaving correctly on at least one historical
  checkpoint

**What I would do next**:
- regenerate every historical broken merged export into a new sibling tree
  called `hf-fixed-r1`
- keep the original `hf/step-*` trees untouched
- use only the clean raw-checkpoint re-export path; do not use the direct shard
  transpose repair path for the main sweep

**Why I would not do a broad parallel fanout yet**:
- the blocking problem is not the repair logic; it is Iris controller launch
  reliability
- repeated `launch_job(...)` submissions from the Python fanout helper timed out
  even when the underlying job definition was valid
- a second concurrent direct CLI submit also timed out and did not appear on the
  controller
- until that changes, the highest-probability path to finishing the sweep is:
  - serial controller submissions
  - one active repair job at a time
  - explicit per-step verification before launching the next

**Canonical output convention**:
- source:
  - `.../<run>/hf/step-<n>`
- raw checkpoint:
  - `.../<run>/checkpoints/step-<n>`
- fixed output:
  - `.../<run>/hf-fixed-r1/step-<n>`
- if any `hf-fixed-r1/step-<n>` is partial or ambiguous from a failed attempt,
  do not overwrite it; instead switch that run to `hf-fixed-r2`

**Execution order I would use**:
1. finish one entire run end-to-end before moving to the next run
2. within a run, repair steps in ascending order:
   - `step-200`
   - `step-400`
   - `step-600`
   - `step-800`
   - `step-1000`
   - `step-1200`
   - `step-1400`
   - `step-1600`
   - `step-1699`
3. run order:
   - `bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd`
   - `bloom_speceval_v2_marin_lr1e6_seed2_b64_v5p8-0ccb95`
   - `bloom_speceval_v2_marin_lr2p5e6_seed0_b64_v5p8-fde891`
   - `bloom_speceval_v2_marin_lr2p5e6_seed2_b64_v5p8-53c5c6`
   - `bloom_speceval_v2_marin_lr3p75e6_seed0_b64_v5p8-5dd6f9`
   - `bloom_speceval_v2_marin_lr3p75e6_seed2_b64_v5p8-a8f183`
   - `bloom_speceval_v2_marin_lr4p5e6_seed0_b64_v5p8-5777fb`
   - `bloom_speceval_v2_marin_lr4p5e6_seed2_b64_v5p8-e8649f`
   - `bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540`
   - `bloom_speceval_v2_marin_lr5e6_seed2_b64_v5p8-68378e`
   - `bloom_speceval_v2_marin_lr6p25e6_seed0_b64_v5p8-9bf4a5`
   - `bloom_speceval_v2_marin_lr6p25e6_seed2_b64_v5p8-0f0331`
   - `bloom_speceval_v2_marin_lr7p5e6_seed0_b64_v5p8-da8f07`
   - `bloom_speceval_v2_marin_lr7p5e6_seed2_b64_v5p8-981a35`
   - `bloom_speceval_v2_marin_lr8p75e6_seed0_b64_v5p8-ee2e69`
   - `bloom_speceval_v2_marin_lr8p75e6_seed2_b64_v5p8-f0636c`
   - `bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d`
   - `bloom_speceval_v2_marin_lr1e5_seed2_b64_v5p8-a73d6f`

**Per-step launch command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-hf-fixed-<short-run-name>-step<n>-r1 \
  --cpu 16 --memory 128GB --disk 150GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/<run>/hf/step-<n> \
    reexport-merged \
    --raw-checkpoint-path gs://marin-us-central1/checkpoints/dpo/tune_lora/<run>/checkpoints/step-<n> \
    --checkpoint-subpath model \
    --output-path gs://marin-us-central1/checkpoints/dpo/tune_lora/<run>/hf-fixed-r1/step-<n> \
    --base-model-ref gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c \
    --tokenizer-path gs://marin-us-central1/checkpoints/dpo/tune_lora/<run>/hf/step-<n> \
    --lora-r 64 \
    --lora-alpha 64 \
    --lora-target-module q_proj \
    --lora-target-module k_proj \
    --lora-target-module v_proj \
    --lora-target-module o_proj \
    --lora-target-module gate_proj \
    --lora-target-module up_proj \
    --lora-target-module down_proj \
    --verify-shapes
```

**Verification gate before launching the next step**:
- the new `hf-fixed-r1/step-<n>` directory must contain:
  - all model shard files referenced by `model.safetensors.index.json`
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `chat_template.jinja`
- `tokenizer_config.json` must embed `chat_template`
- the Iris job must be present on the controller and terminate successfully
- if any of those checks fail, stop the sweep for that run and record the
  failure before moving on

**Spot-check policy**:
- do not run full inference on every repaired step
- instead:
  - run one HF-load or vLLM smoke per run on the latest repaired step
  - run one full Bloom inference only on checkpoints that are likely to be used
    in downstream comparisons
- for this sweep, structural correctness is the primary goal; behavioral
  validation can be sampled

**Failure policy**:
- if the submit command times out locally but the job appears on the controller,
  treat the submission as successful and do not resubmit
- if the submit command times out and no job exists, retry exactly once
- if a retry still times out, pause that run and continue later rather than
  stacking controller load
- if GCS output is partial from a failed attempt, do not reuse that directory;
  bump the sibling output prefix to `hf-fixed-r2`

**Success condition for the full sweep**:
- every listed run has a complete sibling tree under `hf-fixed-r1/step-*` (or
  `hf-fixed-r2` where needed)
- every fixed step has passed the structural verification gate above
- at least one repaired step per run has passed an HF-load or vLLM smoke test

**Next concrete action from this plan**:
- let `/ahmed/lora-hf-fixed-lr1e6-seed0-f55cdd-step200-r1` finish
- verify `step-200`
- then launch only:
  - `/ahmed/lora-hf-fixed-lr1e6-seed0-f55cdd-step400-r1`

### 2026-04-08 22:43 - LVI-031: Plan Revision — Repair Only The Final HF Checkpoint Per Run

**User correction**:
- do **not** regenerate every historical `hf/step-*`
- only repair the **latest** merged HF checkpoint for each tune-LoRA run

**Revised goal**:
- for each listed tune-LoRA run, identify the highest shared step that exists in
  both:
  - `.../hf/step-*`
  - `.../checkpoints/step-*`
- regenerate **only that one step** into a sibling fixed directory
- keep all original `hf/step-*` trees untouched

**Why this is the right change**:
- it directly serves the practical need: one usable final merged artifact per
  run
- it avoids unnecessary controller load and GCS churn
- it reduces the sweep from a full historical backfill to a targeted recovery
  pass
- it matches the known working pattern from:
  - `hf-reexport-r3/step-1699` for
    `bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d`

**Revised output convention**:
- source final broken export:
  - `.../<run>/hf/step-<last>`
- source raw checkpoint:
  - `.../<run>/checkpoints/step-<last>`
- repaired output:
  - `.../<run>/hf-fixed-r1/step-<last>`

**Step selection rule**:
- do not hard-code `step-1699`
- for each run:
  1. list steps under `hf/`
  2. list steps under `checkpoints/`
  3. take the highest numeric step present in both
- if the highest shared step is not `1699`, repair that highest shared step

**Revised execution order**:
- still serial, one Iris CPU job at a time
- but now exactly one repair job per run:
  - `bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd`
  - `bloom_speceval_v2_marin_lr1e6_seed2_b64_v5p8-0ccb95`
  - `bloom_speceval_v2_marin_lr2p5e6_seed0_b64_v5p8-fde891`
  - `bloom_speceval_v2_marin_lr2p5e6_seed2_b64_v5p8-53c5c6`
  - `bloom_speceval_v2_marin_lr3p75e6_seed0_b64_v5p8-5dd6f9`
  - `bloom_speceval_v2_marin_lr3p75e6_seed2_b64_v5p8-a8f183`
  - `bloom_speceval_v2_marin_lr4p5e6_seed0_b64_v5p8-5777fb`
  - `bloom_speceval_v2_marin_lr4p5e6_seed2_b64_v5p8-e8649f`
  - `bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540`
  - `bloom_speceval_v2_marin_lr5e6_seed2_b64_v5p8-68378e`
  - `bloom_speceval_v2_marin_lr6p25e6_seed0_b64_v5p8-9bf4a5`
  - `bloom_speceval_v2_marin_lr6p25e6_seed2_b64_v5p8-0f0331`
  - `bloom_speceval_v2_marin_lr7p5e6_seed0_b64_v5p8-da8f07`
  - `bloom_speceval_v2_marin_lr7p5e6_seed2_b64_v5p8-981a35`
  - `bloom_speceval_v2_marin_lr8p75e6_seed0_b64_v5p8-ee2e69`
  - `bloom_speceval_v2_marin_lr8p75e6_seed2_b64_v5p8-f0636c`
  - `bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d`
  - `bloom_speceval_v2_marin_lr1e5_seed2_b64_v5p8-a73d6f`

**Revised command pattern**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-hf-fixed-<short-run-name>-step<last>-r1 \
  --cpu 16 --memory 128GB --disk 150GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/<run>/hf/step-<last> \
    reexport-merged \
    --raw-checkpoint-path gs://marin-us-central1/checkpoints/dpo/tune_lora/<run>/checkpoints/step-<last> \
    --checkpoint-subpath model \
    --output-path gs://marin-us-central1/checkpoints/dpo/tune_lora/<run>/hf-fixed-r1/step-<last> \
    --base-model-ref gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c \
    --tokenizer-path gs://marin-us-central1/checkpoints/dpo/tune_lora/<run>/hf/step-<last> \
    --lora-r 64 \
    --lora-alpha 64 \
    --lora-target-module q_proj \
    --lora-target-module k_proj \
    --lora-target-module v_proj \
    --lora-target-module o_proj \
    --lora-target-module gate_proj \
    --lora-target-module up_proj \
    --lora-target-module down_proj \
    --verify-shapes
```

**Verification gate per run**:
- repaired `hf-fixed-r1/step-<last>` exists
- `model.safetensors.index.json` exists
- all referenced shard files exist
- `tokenizer_config.json` embeds `chat_template`
- if this is a run likely to be used soon, add one HF-load or vLLM smoke on the
  repaired final step

**What changes operationally**:
- the old per-run step ladder (`200, 400, ... 1699`) is no longer the plan
- the already-running `step-200` repair is now best treated as proof that the
  mechanism works, not as the main rollout target
- the real production sweep should start from each run’s latest shared step

**Updated next action**:
- enumerate the highest shared step for each listed run
- submit only the first run’s final-step repair after that
- then continue serially run by run

### 2026-04-08 22:49 - LVI-032 Planned: Launch Serial Final-Step-Only `hf-fixed-r1` Sweep

**Enumeration result**:
- for every listed tune-LoRA run, the highest shared step present in both:
  - `.../hf/step-*`
  - `.../checkpoints/step-*`
  is:
  - `step-1699`

**Implication**:
- the revised final-step-only sweep is equivalent to:
  - one repair job per run
  - always targeting `step-1699`
- there is no need for per-run step discovery logic beyond confirming that this
  remains true if the manifest changes later

**Operational note on the already-run proof job**:
- `/ahmed/lora-hf-fixed-lr1e6-seed0-f55cdd-step200-r1` succeeded
- that job should now be treated only as proof that the mechanism works
- it is not part of the production final-step-only sweep

**Launcher chosen for the real sweep**:
- new serial babysitter:
  - [run_lora_hf_fixed_final_sweep.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/run_lora_hf_fixed_final_sweep.py)
- supporting discovery helper:
  - [submit_lora_hf_fixed_batch.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/submit_lora_hf_fixed_batch.py)

**Why this launcher exists**:
- submit exactly one Iris CPU re-export job at a time
- wait for terminal controller success
- verify the repaired `hf-fixed-r1/step-1699` export structurally
- only then launch the next run
- preserve the user-specified run order rather than alphabetical order

**Sweep target set**:
- `bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd`
- `bloom_speceval_v2_marin_lr1e6_seed2_b64_v5p8-0ccb95`
- `bloom_speceval_v2_marin_lr2p5e6_seed0_b64_v5p8-fde891`
- `bloom_speceval_v2_marin_lr2p5e6_seed2_b64_v5p8-53c5c6`
- `bloom_speceval_v2_marin_lr3p75e6_seed0_b64_v5p8-5dd6f9`
- `bloom_speceval_v2_marin_lr3p75e6_seed2_b64_v5p8-a8f183`
- `bloom_speceval_v2_marin_lr4p5e6_seed0_b64_v5p8-5777fb`
- `bloom_speceval_v2_marin_lr4p5e6_seed2_b64_v5p8-e8649f`
- `bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540`
- `bloom_speceval_v2_marin_lr5e6_seed2_b64_v5p8-68378e`
- `bloom_speceval_v2_marin_lr6p25e6_seed0_b64_v5p8-9bf4a5`
- `bloom_speceval_v2_marin_lr6p25e6_seed2_b64_v5p8-0f0331`
- `bloom_speceval_v2_marin_lr7p5e6_seed0_b64_v5p8-da8f07`
- `bloom_speceval_v2_marin_lr7p5e6_seed2_b64_v5p8-981a35`
- `bloom_speceval_v2_marin_lr8p75e6_seed0_b64_v5p8-ee2e69`
- `bloom_speceval_v2_marin_lr8p75e6_seed2_b64_v5p8-f0636c`
- `bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d`
- `bloom_speceval_v2_marin_lr1e5_seed2_b64_v5p8-a73d6f`

**Planned launch mode**:
- CPU-only Iris jobs in `us-central1`
- `16 CPU`, `128GB` RAM, `150GB` disk
- outputs under `hf-fixed-r1/step-1699`
- serial babysit until the full run list is complete or a non-transient failure
  blocks progress

### 2026-04-08 23:02 - LVI-033 Launched: Live Serial Final-Step Sweep

**Execution mode**:
- running the sweep in one live long-running local session, not as a detached
  background process
- detached `nohup` processes do not stay alive reliably in this execution
  environment, so the active monitor must remain attached to a live process

**Active launcher**:
- [run_lora_hf_fixed_final_sweep.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/run_lora_hf_fixed_final_sweep.py)

**Active state file**:
- [live_lora_hf_fixed_final_sweep_state.json](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/scratch/live_lora_hf_fixed_final_sweep_state.json)

**Current sweep status at launch**:
- `jobs_total = 18`
- `jobs_completed = 0`
- first target:
  - `bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd`
  - source:
    - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd/hf/step-1699`
  - raw checkpoint:
    - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd/checkpoints/step-1699`
  - target fixed output:
    - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd/hf-fixed-r1/step-1699`

**Current blocker observed during launch**:
- the direct Iris submit path is still hitting controller-side `LaunchJob`
  RPC retries:
  - `Operation launch_job(...) failed (attempt N/20): Request timed out`
- this is the same controller flakiness already seen in the earlier one-off
  experiments
- no evidence yet that the LoRA re-export command itself is failing

**Operating interpretation**:
- the sweep is correctly positioned on the first real final-step repair
- the limiting factor remains controller job-launch reliability
- if the controller eventually accepts the submit, the sweep will continue
  serially from there without changing the run order

### 2026-04-08 23:27 - LVI-034 Progress: First Final-Step Repair Finished, Second Running

**Completed**:
- `/ahmed/lora-hf-fixed-lr1e6-seed0-f55cdd-step1699-r1`
- state:
  - `JOB_STATE_SUCCEEDED`
- verified fixed export:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd/hf-fixed-r1/step-1699`

**Observed controller nuance**:
- the submit path returned non-zero locally due the known `LaunchJob` timeout
  behavior
- however the job did reach the controller and ran successfully
- this confirms the sweep should treat:
  - `submit returned non-zero, but controller has <job>`
  as successful submission rather than a hard failure

**Current active job**:
- `/ahmed/lora-hf-fixed-lr1e6-seed2-0ccb95-step1699-r1`
- controller state:
  - `JOB_STATE_RUNNING`
- target fixed export:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e6_seed2_b64_v5p8-0ccb95/hf-fixed-r1/step-1699`

**Current sweep totals**:
- completed:
  - `1 / 18`
- active:
  - `1`
- remaining after current:
  - `16`

### 2026-04-08 23:33 - LVI-035: Scope Reduction To Shared `seed0` Comparison Set

**User direction**:
- stop the broad historical sweep
- do **not** continue repairing unrelated LoRA runs
- instead pick one seed shared across:
  - `LoRA 10x LR`
  - `LoRA Best Eval`
  - `Full DPO`

**Seed choice**:
- shared seeds from the archived comparison HTML are:
  - `0`
  - `2`
- chosen seed:
  - `seed0`

**Why `seed0`**:
- it is shared across both LoRA groups and the full-DPO comparison group
- it is the simplest continuation because one of the two LoRA seed-0 artifacts
  is already fixed and validated

**Stopped work**:
- killed the live broad sweep process
- terminated the active off-scope repair job:
  - `/ahmed/lora-hf-fixed-lr1e6-seed2-0ccb95-step1699-r1`

**Relevant LoRA runs for the narrowed scope**:
- `LoRA 10x LR`:
  - `bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540`
- `LoRA Best Eval`:
  - `bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d`

**Current status of the two LoRA seed-0 artifacts**:
- `lr1e5_seed0`:
  - already fixed via clean re-export at:
    - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`
  - previously validated by:
    - shape audit
    - HF load
    - TPU vLLM smoke
    - Bloom-format inference
- `lr5e6_seed0`:
  - no fixed sibling export yet under:
    - `hf-fixed-r1/step-1699`
  - this is now the only repair target needed for the narrowed thread

**Next action**:
- run a single final-step repair for:
  - `bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540`
  - `hf/step-1699` -> `hf-fixed-r1/step-1699`
- babysit that one job to terminal completion

### 2026-04-08 23:37 - LVI-036: Launch Final `lr5e6_seed0` Clean Re-Export

**Hypothesis**:
- the remaining seed-0 LoRA comparison artifact can be repaired with the same
  clean raw-checkpoint re-export path that already worked for
  `lr1e5_seed0`

**Confirmed source pair**:
- broken merged HF export:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf/step-1699`
- raw trainer checkpoint:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/checkpoints/step-1699`

**Planned output**:
- `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699`

**Command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name lora-hf-fixed-lr5e6-seed0-274540-step1699-r1 \
  --cpu 16 --memory 128GB --disk 150GB \
  --region us-central1 \
  -- python experiments/posttrain/lora_vllm_investigate.py \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf/step-1699 \
    reexport-merged \
    --raw-checkpoint-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/checkpoints/step-1699 \
    --checkpoint-subpath model \
    --output-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699 \
    --base-model-ref gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c \
    --tokenizer-path gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf/step-1699 \
    --lora-r 64 \
    --lora-alpha 64 \
    --lora-target-module q_proj \
    --lora-target-module k_proj \
    --lora-target-module v_proj \
    --lora-target-module o_proj \
    --lora-target-module gate_proj \
    --lora-target-module up_proj \
    --lora-target-module down_proj \
    --verify-shapes
```

**Success criteria**:
- Iris job reaches `JOB_STATE_SUCCEEDED`
- `hf-fixed-r1/step-1699` contains a full HF export
- export includes `tokenizer_config.json` with embedded `chat_template`
- no partial-output fallback is needed

### 2026-04-08 23:48 - LVI-037: `lr5e6_seed0` Clean Re-Export Succeeded

**Submission workaround**:
- the standard `iris job run` path kept hitting the controller's `30s`
  `LaunchJob` RPC timeout
- patched
  [submit_lora_hf_fixed_batch.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/submit_lora_hf_fixed_batch.py)
  to accept `--rpc-timeout-ms` and used `300000`
- this preserved the same worker command and only lengthened the controller RPC
  deadline

**Submitted command**:
```bash
uv run python experiments/posttrain/submit_lora_hf_fixed_batch.py \
  --submit \
  --latest-only \
  --run-name bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540 \
  --run-label r1 \
  --rpc-timeout-ms 300000
```

**Job**:
- `/ahmed/lora-hf-fixed-lr5e6-seed0-274540-step1699-r1`

**Final state**:
- controller:
  - `succeeded`
- repaired export:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699`

**Artifact verification**:
- expected HF object count present:
  - `14`
- full metadata set present:
  - `chat_template.jinja`
  - `config.json`
  - `generation_config.json`
  - `model-00001-of-00007.safetensors` through `model-00007-of-00007.safetensors`
  - `model.safetensors.index.json`
  - `special_tokens_map.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
- `tokenizer_config.json` contains embedded `chat_template`
- post-export shape audit completed on the worker with:
  - `Checked 291 tensors`
  - `Unchecked tensors: 0`
  - `No config-derived shape mismatches found`

**Current narrowed-scope status**:
- shared-seed comparison set:
  - `seed0`
- `LoRA Best Eval` fixed export:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`
- `LoRA 10x LR` fixed export:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699`

**Conclusion**:
- the seed-0 LoRA pair needed for the narrowed comparison thread is now
  exported and fixed without touching the original published `hf/step-*`
  artifacts

### 2026-04-08 23:52 - LVI-038: Paired Bloom Inference Rerun For Shared `seed0` LoRA Exports

**User direction**:
- run inference over both fixed seed-0 LoRA exports
- use the original Bloom prompts/dataset/pipeline
- keep the setup as close as possible to the prior Marin eval runs
- inference only; do not launch judge jobs yet

**Decision**:
- reuse
  [eval_llama3_8b_alignment.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/eval_llama3_8b_alignment.py)
  with:
  - `PromptFormat.BLOOM`
  - the existing Bloom eval prompt split in `us-central1`
  - the same sampling config used in the earlier evals:
    - `temperature=0.7`
    - `max_tokens=1500`
    - `n=3`
  - the same LoRA-serving TPU family already validated for the clean artifact:
    - `v5p-8`
- run both LoRA seed-0 checkpoints together under a fresh `run_label` so the
  outputs land in new prefixes

**Code change**:
- added target
  `tune_lora_lr5e6_seed0_step1699`
  to
  [eval_llama3_8b_alignment.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/eval_llama3_8b_alignment.py)
  pointing at:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699`

**Planned command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name eval-marin-tune-lora-seed0pair-bloom-us-central1-v5p8-r1 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region us-central1 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-central1 \
    --tpu-type v5p-8 \
    --target tune_lora_lr1e5_seed0_step1699 \
    --target tune_lora_lr5e6_seed0_step1699 \
    --run-label seed0pairr1
```

**Expected outputs**:
- `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_seed0pairr1/...`
- `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0pairr1/...`

### 2026-04-08 23:58 - LVI-039: Move Paired Seed-0 LoRA Inference To `eu-west4` `v6e-4`

**User direction**:
- do not keep pushing the paired inference through the flaky `us-central1`
  wrapper submit path
- instead launch on `eu-west4` `v6e-4`
- copy model weights into `eu-west4` manually if needed

**Observed regional state**:
- Bloom prompt split already exists in `eu-west4`:
  - `gs://marin-eu-west4/alignment/gpt-4.1-eval-split/...`
- fixed LoRA exports do **not** yet exist in `eu-west4`:
  - `gs://marin-eu-west4/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`
  - `gs://marin-eu-west4/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699`

**Plan**:
1. mirror only the two fixed `step-1699` exports from `us-central1` into
   `marin-eu-west4`
2. reuse the same Bloom-format inference script and target keys
3. launch on:
   - region: `eu-west4`
   - TPU type: `v6e-4`
4. keep inference-only scope

**Intended eu-west launch**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name eval-marin-tune-lora-seed0pair-bloom-eu-west4-v6e4-r1 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region eu-west4 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region eu-west4 \
    --tpu-type v6e-4 \
    --target tune_lora_lr1e5_seed0_step1699 \
    --target tune_lora_lr5e6_seed0_step1699 \
    --run-label seed0paireuw4r1
```

### 2026-04-09 00:11 - LVI-040: Launch Paired Seed-0 LoRA Bloom Inference On `eu-west4` `v6e-4`

**Pre-launch verification**:
- confirmed Bloom prompt split already exists in:
  - `gs://marin-eu-west4/alignment/gpt-4.1-eval-split`
- confirmed both fixed LoRA exports now exist in `eu-west4` with the full
  expected HF object count (`14/14`):
  - `gs://marin-eu-west4/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`
  - `gs://marin-eu-west4/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699`

**Launch decision**:
- keep the inference path maximally close to the earlier Bloom reproduction:
  - reuse
    [eval_llama3_8b_alignment.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/eval_llama3_8b_alignment.py)
  - keep `PromptFormat.BLOOM`
  - keep `temperature=0.7`, `max_tokens=1500`, `n=3`
  - keep inference-only scope
- avoid the flaky 30s `iris job run` RPC path by submitting the wrapper through
  `IrisClient.remote(..., timeout_ms=300000)`
- use fresh names for both the wrapper and the output suffix:
  - wrapper: `eval-marin-tune-lora-seed0pair-bloom-eu-west4-v6e4-r2`
  - run label: `seed0paireuw4r2`

**Submitted wrapper command**:
```bash
python experiments/posttrain/eval_llama3_8b_alignment.py \
  --region eu-west4 \
  --tpu-type v6e-4 \
  --target tune_lora_lr1e5_seed0_step1699 \
  --target tune_lora_lr5e6_seed0_step1699 \
  --run-label seed0paireuw4r2
```

**Expected inference outputs**:
- `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_seed0paireuw4r2/...`
- `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/...`

### 2026-04-09 00:45 - LVI-041: Split Recovery After Mixed Paired Run Outcome

**Observed paired-run outcome**:
- wrapper:
  - `/ahmed/eval-marin-tune-lora-seed0pair-bloom-eu-west4-v6e4-r2`
- child `lr1e5` leg:
  - `/ahmed/eval-marin-tune-lora-seed0pair-bloom-eu-west4-v6e4-r2/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_seed0paireuw4r2-inference_277640f5-13a3850a`
  - executor status:
    - `FAILED`
  - controller state:
    - `JOB_STATE_KILLED`
  - no Python/vLLM stack trace surfaced in child logs
- child `lr5e6` leg:
  - `/ahmed/eval-marin-tune-lora-seed0pair-bloom-eu-west4-v6e4-r2/eval-marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2-inference_dbdcd299-22edc20a`
  - executor status:
    - `RUNNING`
  - reached:
    - `vLLM environment ready`
    - Bloom eval inference live progress
  - latest observed progress:
    - `2304 / 2576` prompts (`89.4%`) at about `18.2 items/s`

**Interpretation**:
- `eu-west4` `v6e-4` Bloom inference is viable for this thread because the
  `lr5e6` leg is running successfully on the fixed LoRA artifact
- the `lr1e5` failure is narrower than the old generalized `eu-west4 v6e-4`
  bring-up issue; current evidence is consistent with a one-off child kill,
  scheduler/preemption event, or task-level infrastructure interruption

**Next action**:
- do not throw away the healthy `lr5e6` run
- relaunch `lr1e5` by itself on a fresh eu-west4 label while the healthy leg
  continues / finishes:
  - wrapper: `eval-marin-tune-lora-lr1e5-bloom-eu-west4-v6e4-r3`
  - run label: `lr1e5euw4r3`

**Single-target rerun command**:
```bash
python experiments/posttrain/eval_llama3_8b_alignment.py \
  --region eu-west4 \
  --tpu-type v6e-4 \
  --target tune_lora_lr1e5_seed0_step1699 \
  --run-label lr1e5euw4r3
```

### 2026-04-09 00:50 - LVI-042: Repeated `lr1e5` `v6e-4` Kill Is Not A Bad Mirror; Escalate To `v6e-8`

**New evidence**:
- the focused eu-west rerun also failed with the same terminal child state:
  - wrapper:
    - `/ahmed/eval-marin-tune-lora-lr1e5-bloom-eu-west4-v6e4-r3`
  - child:
    - `/ahmed/eval-marin-tune-lora-lr1e5-bloom-eu-west4-v6e4-r3/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lr1e5euw4r3-inference_606816b8-67b861be`
  - child state:
    - `JOB_STATE_KILLED`
  - controller status details:
    - `exit_code = 0`
    - `failure_count = 0`
    - `preemption_count = 0`
    - `status_message = ""`
    - `tasks_len = 0`
- the eu-west `lr1e5` mirror is byte-identical to `us-central1`:
  - same object count (`14`)
  - matching `Content-Length`
  - matching `crc32c`
  - matching `md5`
  - `tokenizer_config.json` still contains embedded `chat_template`
- meanwhile the paired `lr5e6` eu-west run completed successfully:
  - output:
    - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/inference-abdde9`

**Interpretation**:
- repeated `lr1e5` failure on eu-west `v6e-4` is not explained by:
  - a broken HF export
  - a bad cross-region copy
  - missing tokenizer/chat-template metadata
- the failure mode is now best treated as a `v6e-4` infrastructure / task-launch
  interaction specific to this leg, not as a model-format problem

**Recovery decision**:
- keep the same region and same Bloom-format pipeline
- change only the TPU slice for `lr1e5`:
  - from `v6e-4`
  - to `v6e-8`
- rationale:
  - this stays in `eu-west4`
  - earlier TPU debugging already showed `eu-west4 v6e-8` is healthier than
    `eu-west4 v6e-4`
  - this is the smallest change that still has a good chance of finishing the
    inference mission

**Escalation command**:
```bash
python experiments/posttrain/eval_llama3_8b_alignment.py \
  --region eu-west4 \
  --tpu-type v6e-8 \
  --target tune_lora_lr1e5_seed0_step1699 \
  --run-label lr1e5euw4v6e8r1
```

### 2026-04-09 00:55 - LVI-043: Current `eu-west4` LoRA Inference State

**Confirmed successful inference artifact**:
- `lr5e6_seed0` on `eu-west4 v6e-4` finished successfully:
  - wrapper:
    - `/ahmed/eval-marin-tune-lora-seed0pair-bloom-eu-west4-v6e4-r2`
  - child:
    - `/ahmed/eval-marin-tune-lora-seed0pair-bloom-eu-west4-v6e4-r2/eval-marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2-inference_dbdcd299-22edc20a`
  - final output:
    - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/inference-abdde9`
  - artifact contents verified:
    - `.artifact`
    - `.executor_info`
    - `.executor_status`
    - `shard_00000.jsonl.gz`
    - `shard_00001.jsonl.gz`
    - `artifacts/`

**Repeated failure that remains open**:
- `lr1e5_seed0` on `eu-west4 v6e-4` has now failed twice:
  - paired child:
    - `/ahmed/eval-marin-tune-lora-seed0pair-bloom-eu-west4-v6e4-r2/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_seed0paireuw4r2-inference_277640f5-13a3850a`
  - single-target child:
    - `/ahmed/eval-marin-tune-lora-lr1e5-bloom-eu-west4-v6e4-r3/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lr1e5euw4r3-inference_606816b8-67b861be`
  - both ended as:
    - `JOB_STATE_KILLED`
  - no child task logs or controller status message explain the kill
- integrity checks rule out a bad eu-west mirror:
  - `us-central1` and `eu-west4` copies match on:
    - object count
    - `Content-Length`
    - `crc32c`
    - `md5`
  - `tokenizer_config.json` in eu-west still contains embedded `chat_template`

**Live recovery run**:
- wrapper:
  - `/ahmed/eval-marin-tune-lora-lr1e5-bloom-eu-west4-v6e8-r1`
- child:
  - `/ahmed/eval-marin-tune-lora-lr1e5-bloom-eu-west4-v6e8-r1/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lr1e5euw4v6e8r1-inference_aac2cafb-7832fc38`
- current states:
  - wrapper: `RUNNING`
  - child: `PENDING`
- current pending reason:
  - `Scheduler: Insufficient TPUs (need 8, available 0)`

**Interpretation**:
- the Bloom-format inference mission is half-complete in `eu-west4`:
  - `lr5e6` done
  - `lr1e5` waiting on the `v6e-8` fallback after repeated unexplained `v6e-4` kills

### 2026-04-09 12:19 - LVI-044: `v6e-8` Fallback Failed Before TPU Startup

**Cluster re-check**:
- `lr5e6_seed0` remains complete in `eu-west4`:
  - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/inference-abdde9`
- `lr1e5_seed0` is still not complete in `eu-west4`

**`v6e-8` fallback result**:
- wrapper:
  - `/ahmed/eval-marin-tune-lora-lr1e5-bloom-eu-west4-v6e8-r1`
- final wrapper state:
  - `JOB_STATE_FAILED`
- exact failure:
  - `RuntimeError: Failed to fetch 5160f6590c008554c0f6e43fcf7cf839a39b4d763ff0a237ae8fd1ded0a349fa: timed out`
- stack location:
  - `iris.cluster.bundle.BundleStore._fetch_from_controller`
  - `iris.cluster.worker.task_attempt._download_bundle`
- interpretation:
  - this failure happened during Iris bundle staging, before the TPU child could
    start model inference
  - it is an infrastructure/workspace-transfer failure, not a model-format or
    vLLM-load failure

**Current state**:
- `lr5e6_seed0`:
  - inference done in `eu-west4`
- `lr1e5_seed0`:
  - has a valid completed inference artifact in `us-central1`
  - does **not** yet have a completed inference artifact in `eu-west4`

### 2026-04-09 12:37 - LVI-045: Start Batch-64 Full-DPO Seed0 Comparison Inference

**User correction accepted**:
- the intended non-LoRA baseline for the paired seed-0 comparison is **not** the
  older batch-128 Marin-instruct DPO run
- the correct run comes from the archived comparison selection:
  - `bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963`

**Recovered checkpoint paths**:
- W&B config points to:
  - `gs://marin-us-central1/checkpoints/dpo/compare_lora/bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963/checkpoints`
  - `gs://marin-us-central1/checkpoints/dpo/compare_lora/bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963/hf`
- latest shared HF checkpoint selected for inference:
  - `gs://marin-us-central1/checkpoints/dpo/compare_lora/bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963/hf/step-1699`

**Pre-launch validation**:
- source `hf/step-1699` exists in `us-central1`
- no completed inference artifact was found yet for this run
- `tokenizer_config.json` is missing embedded `chat_template`, even though
  `chat_template.jinja` exists beside it

**Execution plan**:
- mirror only `hf/step-1699` from `us-central1` to `eu-west4`
- patch only the mirrored `tokenizer_config.json` to embed `chat_template`
- add the checkpoint as a first-class target in
  `experiments/posttrain/eval_llama3_8b_alignment.py`
- launch the same Bloom-format inference pipeline used for the completed LoRA
  runs:
  - region: `eu-west4`
  - TPU: `v6e-4`
  - target: `compare_lora_beta01_seed0_b64_step1699`
  - run label: `seed0fullb64euw4r1`

### 2026-04-09 12:43 - LVI-046: Mirrored `step-1699`, Patched Chat Template, Launched `eu-west4` Inference

**Mirror result**:
- copied `hf/step-1699` from:
  - `gs://marin-us-central1/checkpoints/dpo/compare_lora/bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963/hf/step-1699`
- to:
  - `gs://marin-eu-west4/checkpoints/dpo/compare_lora/bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963/hf/step-1699`
- verified final mirrored object count:
  - `13`
- verified mirrored objects include:
  - `model-00001-of-00007.safetensors`
  - `...`
  - `model-00007-of-00007.safetensors`
  - `tokenizer_config.json`
  - `chat_template.jinja`

**Metadata patch**:
- source export was missing embedded `chat_template` inside
  `tokenizer_config.json`
- patched only the mirrored tokenizer metadata in `eu-west4`
- verified:
  - `gcloud storage cat .../tokenizer_config.json | rg chat_template`
    returns the embedded template

**Code change for launch path**:
- added target `compare_lora_beta01_seed0_b64_step1699` to
  `experiments/posttrain/eval_llama3_8b_alignment.py`
- target name:
  - `marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval`
- regional model path now resolves to the mirrored `eu-west4` checkpoint

**Launch command**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name eval-marin-full-dpo-b64-seed0-bloom-eu-west4-v6e4-r1 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region europe-west4 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region eu-west4 \
    --tpu-type v6e-4 \
    --target compare_lora_beta01_seed0_b64_step1699 \
    --run-label seed0fullb64euw4r1
```

**Submitted wrapper job**:
- `/ahmed/eval-marin-full-dpo-b64-seed0-bloom-eu-west4-v6e4-r1`

**Immediate status after submission**:
- wrapper state:
  - `pending`
- scheduler message:
  - `Pending scheduler feedback`

**Current state**:
- mirror ready in `eu-west4`
- tokenizer metadata patched
- wrapper submitted successfully
- still waiting for the executor to emit the child inference step / output prefix

### 2026-04-09 12:45 - LVI-047: Wrapper Running; Inference Step Emitted With Real Output Prefix

**Controller state**:
- wrapper job:
  - `/ahmed/eval-marin-full-dpo-b64-seed0-bloom-eu-west4-v6e4-r1`
- wrapper advanced from `pending` to `running`

**Wrapper logs confirm executor launch**:
- emitted step:
  - `eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference_f08fb3e3`
- emitted output path:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2`

**Current interpretation**:
- the new batch-64 full-DPO seed0 baseline is now on the same Bloom-format
  inference pipeline as the paired LoRA runs
- no model-format or tokenizer metadata error appeared at wrapper startup
- the run is live; remaining work is ordinary babysitting until the child
  inference finishes or fails

### 2026-04-09 13:16 - LVI-048: Batch-64 Full-DPO Seed0 Bloom Inference Succeeded In `eu-west4`

**Terminal wrapper state**:
- wrapper:
  - `/ahmed/eval-marin-full-dpo-b64-seed0-bloom-eu-west4-v6e4-r1`
- final state:
  - `JOB_STATE_SUCCEEDED`
- finished at:
  - `1775765774062` epoch ms

**Child inference result**:
- child:
  - `/ahmed/eval-marin-full-dpo-b64-seed0-bloom-eu-west4-v6e4-r1/eval-marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1-inference_f08fb3e3-92347911`
- output prefix:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2`

**Final artifact contents verified**:
- `.artifact`
- `.executor_info`
- `.executor_status`
- `shard_00000.jsonl.gz`
- `shard_00001.jsonl.gz`
- `artifacts/`

**Final writeout lines**:
- `Wrote 7728 records to 2 shards`
- `Wrote 7728 eval inference results`

**Runtime**:
- executor runtime:
  - `1898.51s`

**Interpretation**:
- the exact batch-64 full-DPO seed0 baseline requested by the user is now
  finished on the same Bloom-format inference pipeline used for the paired LoRA
  runs
- comparison set is now complete at the inference stage for seed 0:
  - LoRA best-eval:
    - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768`
  - LoRA 10x-LR:
    - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/inference-abdde9`
  - full DPO batch-64:
    - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2`

### 2026-04-09 13:20 - LVI-049: Plan Bloom-Compatible LM-as-Judge Sweep For The 3 Seed-0 Comparison Models

**Goal**:
- run the exact Bloom-compatible GPT-4.1 judge path on the three completed
  seed-0 comparison inference artifacts so the LoRA-vs-full-DPO comparison uses
  the same LM-as-judge protocol as the earlier Bloom adherence runs

**Judge implementation to use**:
- script:
  - `experiments/posttrain/run_bloom_judge.py`
- reason:
  - uses the Bloom-compatible system prompt and compliance judge prompt
  - reads the saved Bloom-format inference rows directly
  - computes the same prompt-collapsed summary used by Bloom's `adherence.py`

**Shared judge settings**:
- judge model:
  - `openai/gpt-4.1-2025-04-14`
- max tokens:
  - `4000`
- concurrency:
  - `128`
- spec:
  - `experiments/posttrain/specs/openai_model_spec.jsonl`
- execution mode:
  - Iris CPU-only job

**Three target judge runs**:
- LoRA best-eval (`lr1e5_seed0`):
  - inference:
    - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768`
  - planned judge output:
    - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/judge-gpt41`
- LoRA 10x-LR (`lr5e6_seed0`):
  - inference:
    - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/inference-abdde9`
  - planned judge output:
    - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/judge-gpt41`
- full DPO batch-64 (`beta0.1_seed0_b64_step1699`):
  - inference:
    - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2`
  - planned judge output:
    - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/judge-gpt41`

**Recommended launch shape**:
- launch as Iris CPU-only jobs, not local jobs
- use `16 CPU`, `32GB` RAM, `10GB` disk per judge job
- preferred regions:
  - `us-central1` for the `lr1e5` artifact
  - `europe-west4` for the two `eu-west4` artifacts

**Recommended execution order**:
- either:
  - launch all three in parallel if OpenAI rate limits are acceptable
- or:
  - run `lr1e5` first as a control, then the remaining two in parallel

**Exact judge commands to submit on Iris**:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name judge-bloom-gpt41-lora-lr1e5-seed0-us-central1 \
  --cpu 16 --memory 32GB --disk 10GB \
  --region us-central1 \
  -e OPENAI_API_KEY "$OPENAI_API_KEY" \
  -- python experiments/posttrain/run_bloom_judge.py \
    --inference-path gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768 \
    --spec-path experiments/posttrain/specs/openai_model_spec.jsonl \
    --output-path gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/judge-gpt41 \
    --judge-model openai/gpt-4.1-2025-04-14 \
    --max-tokens 4000 \
    --concurrency 128
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name judge-bloom-gpt41-lora-lr5e6-seed0-euw4 \
  --cpu 16 --memory 32GB --disk 10GB \
  --region europe-west4 \
  -e OPENAI_API_KEY "$OPENAI_API_KEY" \
  -- python experiments/posttrain/run_bloom_judge.py \
    --inference-path gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/inference-abdde9 \
    --spec-path experiments/posttrain/specs/openai_model_spec.jsonl \
    --output-path gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/judge-gpt41 \
    --judge-model openai/gpt-4.1-2025-04-14 \
    --max-tokens 4000 \
    --concurrency 128
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name judge-bloom-gpt41-fulldpo-b64-seed0-euw4 \
  --cpu 16 --memory 32GB --disk 10GB \
  --region europe-west4 \
  -e OPENAI_API_KEY "$OPENAI_API_KEY" \
  -- python experiments/posttrain/run_bloom_judge.py \
    --inference-path gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2 \
    --spec-path experiments/posttrain/specs/openai_model_spec.jsonl \
    --output-path gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/judge-gpt41 \
    --judge-model openai/gpt-4.1-2025-04-14 \
    --max-tokens 4000 \
    --concurrency 128
```

**Planned deliverables after judge sweep**:
- three `judge-gpt41/summary.json` artifacts
- prompt-collapsed means, std, sem, and CI for all three models
- direct seed-0 comparison:
  - full DPO batch-64 vs LoRA best-eval
  - full DPO batch-64 vs LoRA 10x-LR
  - LoRA best-eval vs LoRA 10x-LR

### 2026-04-09 13:27 - LVI-050: Launch Only The Full-DPO Batch-64 GPT-4.1 Judge Job

**User direction**:
- run LM-as-judge only for the full DPO checkpoint
- do not launch the two LoRA judge jobs yet

**Selected target**:
- inference:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2`
- judge output:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/judge-gpt41`

**Judge launch command**:
```bash
source .env
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name judge-bloom-gpt41-fulldpo-b64-seed0-euw4 \
  --cpu 16 --memory 32GB --disk 10GB \
  --region europe-west4 \
  -e OPENAI_API_KEY "$OPENAI_API_KEY" \
  -- python experiments/posttrain/run_bloom_judge.py \
    --inference-path gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2 \
    --spec-path experiments/posttrain/specs/openai_model_spec.jsonl \
    --output-path gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/judge-gpt41 \
    --judge-model openai/gpt-4.1-2025-04-14 \
    --max-tokens 4000 \
    --concurrency 128
```

### 2026-04-09 13:27 - LVI-051: Full-DPO Batch-64 Judge Job Submitted And Running

**Submitted job**:
- `/ahmed/judge-bloom-gpt41-fulldpo-b64-seed0-euw4`

**Controller state immediately after launch**:
- `running`
- resources:
  - `16cpu, 32 GiB, 10 GiB disk`
- region:
  - `europe-west4`

**Target artifacts**:
- inference input:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2`
- judge output root:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/judge-gpt41`

**Current state**:
- job is live on Iris
- output prefix did not yet exist at first post-submit check, which is normal
  before the worker writes the first artifacts

### 2026-04-09 13:34 - LVI-052: Full-DPO Batch-64 Judge Succeeded

**Terminal state**:
- job:
  - `/ahmed/judge-bloom-gpt41-fulldpo-b64-seed0-euw4`
- final state:
  - `JOB_STATE_SUCCEEDED`
- no recovery / restart was needed

**Judge artifacts**:
- output root:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/judge-gpt41`
- verified files:
  - `judged_results.jsonl`
  - `summary.json`

**Prompt-collapsed summary**:
- prompts:
  - `2566`
- judged responses:
  - `7678`
- errors:
  - `20`
- skipped no-rubric:
  - `30`
- overall mean:
  - `8.4443361912185`
- compliance:
  - `0.8211223694466095`
- std:
  - `2.0616362741975993`
- sem:
  - `0.04069899835743221`
- ci95:
  - `0.07977003678056713`

**Notable statement-level results**:
- strongest:
  - `protect_privacy = 9.9247`
  - `be_kind = 9.8428`
  - `avoid_abuse = 9.7296`
- weakest:
  - `refusal_style = 3.6984`
  - `avoid_targeted_political_manipulation = 5.4155`
  - `support_programmatic_use = 6.3774`
  - `formatting = 6.5975`

**Operational note**:
- the `32GB` RAM request was conservative but unnecessary in hindsight
- the job was API-bound, healthy, and never showed memory pressure
- future single-model judge reruns can likely use less RAM without changing behavior

### 2026-04-09 14:02 - LVI-053: Plan TPU-Only Batch-64 Adherence Plot And Fill Missing Judge Artifacts

**User direction**:
- add a new TPU-only adherence plot
- make the figure explicitly about the batch-64 comparison set
- keep the comparison aligned with the original Bloom-format inference and judge pipeline

**Plot scope**:
- seed-0 full DPO, batch 64
- seed-0 LoRA best-eval (`lr=1e-5`)
- seed-0 LoRA 10x-LR (`lr=5e-6`)

**Current artifact state before plotting**:
- full DPO batch-64 seed0:
  - inference done
  - `judge-gpt41/summary.json` done
- LoRA `lr=1e-5` seed0:
  - inference done
  - judge not yet run
- LoRA `lr=5e-6` seed0:
  - inference done
  - judge not yet run

**Next actions**:
1. Launch the two missing Bloom-compatible GPT-4.1 judge jobs on Iris CPU workers.
2. Wait for both `judge-gpt41/summary.json` artifacts to complete.
3. Add a dedicated TPU-only plotting script in `experiments/posttrain/`.
4. Generate batch-64 comparison outputs under `plot/output/`.
5. Append exact means / CIs and figure paths back into this logbook.

### 2026-04-09 14:08 - LVI-054: Launch Two LoRA Judge Jobs, One Immediate Env Propagation Miss

**Launched**:
- `/ahmed/judge-bloom-gpt41-lora-lr1e5-seed0-us-central1`
- `/ahmed/judge-bloom-gpt41-lora-lr5e6-seed0-euw4`

**Command shape**:
- Iris CPU-only jobs
- `run_bloom_judge.py`
- `gpt-4.1-2025-04-14`
- `max_tokens=4000`
- `concurrency=128`

**Immediate status check**:
- `lr1e5_seed0` (`us-central1`): worker reached `running user command`
- `lr5e6_seed0` (`europe-west4`): failed immediately with
  - `ERROR: OPENAI_API_KEY not set`

**Cause**:
- local submission typo propagated the wrong shell variable name for the eu-west4 job
- this is a launch-wrapper error, not a problem with the inference artifact or judge script

**Next action**:
- relaunch the eu-west4 LoRA `lr=5e-6` judge job with the correct `OPENAI_API_KEY` passthrough
- continue babysitting both active judge runs until terminal state

### 2026-04-09 13:44 - LVI-055: TPU-Only Full-DPO Plot Using Existing Judge Summaries

**User correction**:
- do not launch more LM-as-judge work
- use the already-finished full DPO batch-64 run and the older full fine-tuning DPO runs
- just make the plot

**Action taken**:
- added [tpu_adhere.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/tpu_adhere.py)
- reused the existing prompt-collapsed Marin summary loader from
  [plot_bloom_gpu_vs_marin_tpu_adherence.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/plot_bloom_gpu_vs_marin_tpu_adherence.py)
- plotted only TPU-side full-DPO judge summaries
- made batch size explicit in every label and highlighted the new batch-64 run

**Included runs**:
- `β=0.01, lr=5e-7, batch=128`
- `β=0.01, lr=7.5e-7, batch=128`
- `β=0.1, lr=5e-7, batch=128`
- `β=0.1, lr=5e-7, batch=64 (new)`
- `β=0.1, lr=7.5e-7, batch=128`

**Generated outputs**:
- [tpu_full_dpo_adherence.png](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/plot/output/tpu_full_dpo_adherence.png)
- [tpu_full_dpo_adherence.pdf](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/plot/output/tpu_full_dpo_adherence.pdf)
- [tpu_full_dpo_adherence.json](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/plot/output/tpu_full_dpo_adherence.json)

**Prompt-collapsed means used in the plot**:
- `β=0.01, lr=5e-7, batch=128`: `8.7651 +/- 0.0711`
- `β=0.01, lr=7.5e-7, batch=128`: `8.7280 +/- 0.0708`
- `β=0.1, lr=5e-7, batch=128`: `8.4061 +/- 0.0802`
- `β=0.1, lr=5e-7, batch=64 (new)`: `8.4443 +/- 0.0798`
- `β=0.1, lr=7.5e-7, batch=128`: `8.4274 +/- 0.0808`

**Immediate read**:
- the new `β=0.1, lr=5e-7, batch=64` full-DPO run sits slightly above the older
  `β=0.1, lr=5e-7, batch=128` run
- it is also effectively tied with the older `β=0.1, lr=7.5e-7, batch=128` run
  at this confidence scale

### 2026-04-09 13:52 - LVI-056: Sequential LoRA Judge Plan

**User direction**:
- run LM-as-judge for the two LoRA runs
- do it one after the other

**Pre-flight check**:
- no active `/ahmed/judge-bloom-gpt41-lora-*` jobs remain on Iris
- neither LoRA inference root currently has a `judge-gpt41/` directory
- safe to use the canonical `judge-gpt41` output roots without stale partials

**Planned execution order**:
1. `lr1e5_seed0` in `us-central1`
2. `lr5e6_seed0` in `europe-west4`

**Shared judge config**:
- script: `experiments/posttrain/run_bloom_judge.py`
- model: `openai/gpt-4.1-2025-04-14`
- `max_tokens=4000`
- `concurrency=128`
- Iris CPU-only jobs

**Target outputs**:
- `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/judge-gpt41`
- `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/judge-gpt41`

### 2026-04-09 13:57 - LVI-057: LoRA `lr=1e-5` Judge Completed, Launching `lr=5e-6` Next

**Completed first sequential judge run**:
- job:
  - `/ahmed/judge-bloom-gpt41-lora-lr1e5-seed0-us-central1-r2`
- final state:
  - `JOB_STATE_SUCCEEDED`
- output root:
  - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/judge-gpt41`

**Prompt-collapsed summary**:
- mean:
  - `8.553065731358787`
- ci95:
  - `0.07992071399607277`
- prompts:
  - `2566`
- judged responses:
  - `7697`
- errors:
  - `1`
- skipped no-rubric:
  - `30`

**Immediate read**:
- this LoRA run sits above the new batch-64 full-DPO run (`8.4443`) on the same
  Bloom-compatible judge

**Next action**:
- launch the second sequential LoRA judge for `lr=5e-6` in `europe-west4`

### 2026-04-09 14:02 - LVI-058: LoRA `lr=5e-6` Judge Completed, Sequential Sweep Done

**Completed second sequential judge run**:
- job:
  - `/ahmed/judge-bloom-gpt41-lora-lr5e6-seed0-euw4-r3`
- final state:
  - `JOB_STATE_SUCCEEDED`
- output root:
  - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/judge-gpt41`

**Prompt-collapsed summary**:
- mean:
  - `8.50396206806962`
- ci95:
  - `0.08107546321527889`
- prompts:
  - `2566`
- judged responses:
  - `7690`
- errors:
  - `8`
- skipped no-rubric:
  - `30`

**Selected statement-level weak spots**:
- `refusal_style`:
  - `2.873015873015872`
- `support_programmatic_use`:
  - `6.320754716981133`
- `avoid_targeted_political_manipulation`:
  - `5.378995433789954`

**Sequential sweep status**:
- `lr=1e-5` LoRA judge:
  - done
  - mean `8.5531 +/- 0.0799`
- `lr=5e-6` LoRA judge:
  - done
  - mean `8.5040 +/- 0.0811`

**Immediate read**:
- both LoRA seed-0 runs score above the new batch-64 full-DPO run (`8.4443 +/- 0.0798`)
- `lr=1e-5` remains slightly ahead of `lr=5e-6` on this judge

### 2026-04-09 14:08 - LVI-059: Matched Batch-64 Plot For Full DPO Vs LoRA

**User direction**:
- make a plot showing the full-DPO batch-64 run versus the two LoRA runs with the same batch size

**Action taken**:
- updated [tpu_adhere.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/tpu_adhere.py)
  to emit a second figure focused only on the three matched batch-64 runs

**Generated outputs**:
- [tpu_batch64_matchup.png](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/plot/output/tpu_batch64_matchup.png)
- [tpu_batch64_matchup.pdf](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/plot/output/tpu_batch64_matchup.pdf)
- [tpu_batch64_matchup.json](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/plot/output/tpu_batch64_matchup.json)

**Matched batch-64 values in the plot**:
- full DPO:
  - `8.4443361912185 +/- 0.07977003678056713`
- LoRA `lr=5e-6`:
  - `8.50396206806962 +/- 0.08107546321527889`
- LoRA `lr=1e-5`:
  - `8.553065731358787 +/- 0.07992071399607277`

**Immediate read**:
- both LoRA runs sit slightly above the full-DPO batch-64 run
- `lr=1e-5` remains the highest of the three on this judge

### 2026-04-09 14:21 - LVI-060: One-Off Bloom Inference Launch For LoRA Smoke Export Step 4

**User direction**:
- launch a one-off Bloom-format inference job on `us-central1` compute for:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/lora_smoke_export_marin_8b_instruct_v5p8_5step-e97bb1/hf/step-4`

**Action taken**:
- added [eval_bloom_oneoff_model.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/eval_bloom_oneoff_model.py)
- kept the same executor path and Bloom-format prompt setup as the existing inference runs
- verified the target export has:
  - `chat_template.jinja`
  - `generation_config.json`
  - `tokenizer_config.json` with embedded `chat_template`

**Launch command**:
```bash
source .env
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name eval-lora-smoke-export-step4-us-central1 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region us-central1 \
  -- python experiments/posttrain/eval_bloom_oneoff_model.py \
    --region us-central1 \
    --model-path gs://marin-us-central1/checkpoints/dpo/tune_lora/lora_smoke_export_marin_8b_instruct_v5p8_5step-e97bb1/hf/step-4 \
    --name marin_dpo_lora_smoke_export_step4_bloom_speceval \
    --description "Marin LoRA smoke export step4 on 2,576 Bloom eval prompts (46 statements)" \
    --run-label r1
```

**Submitted wrapper job**:
- `/ahmed/eval-lora-smoke-export-step4-us-central1`

**Current state at launch**:
- accepted by the Iris controller
- still in initial submission / worker bring-up phase

### 2026-04-09 14:27 - LVI-061: Fixed One-Off Step-4 Inference Launch By Switching To `v5p-8`

**Observed failure on first launch**:
- wrapper:
  - `/ahmed/eval-lora-smoke-export-step4-us-central1`
- error:
  - `Executor step 'eval/marin_dpo_lora_smoke_export_step4_bloom_speceval_r1/inference' has no overlap between GCS regions ['us-central1'] and TPU-capable DAG regions ['europe-west4', 'us-east1', 'us-east5']`

**Root cause**:
- the one-off script defaulted to `v6e-4`
- for this cluster state, `v6e` TPU-capable DAG regions did not include `us-central1`
- the model and prompt paths were both pinned to `us-central1`, so the executor correctly rejected the placement

**Fix applied**:
- relaunched the same one-off script with:
  - `--tpu-type v5p-8`

**Relaunched wrapper**:
- `/ahmed/eval-lora-smoke-export-step4-us-central1-v5p8-r2`

**Child inference step**:
- `/ahmed/eval-lora-smoke-export-step4-us-central1-v5p8-r2/eval-marin_dpo_lora_smoke_export_step4_bloom_speceval_r2-inference_ea4c878c-9dfcefc9`

**Confirmed running signal**:
- child reached:
  - `Loaded 2576 eval prompts (bloom format)`
  - `Running eval inference: 2576 prompts, n=3`
  - `Starting vLLM native server`

**Inference output root**:
- `gs://marin-us-central1/eval/marin_dpo_lora_smoke_export_step4_bloom_speceval_r2/inference-6f1fa3`

### 2026-04-09 15:22 - LVI-062: One-Off Step-4 Bloom Inference Eventually Succeeded

**Target**:
- `gs://marin-us-central1/checkpoints/dpo/tune_lora/lora_smoke_export_marin_8b_instruct_v5p8_5step-e97bb1/hf/step-4`

**Wrapper job**:
- `/ahmed/eval-lora-smoke-export-step4-us-central1-v5p8-r2`

**Observed degraded behavior during babysit**:
- the wrapper stayed `RUNNING` while the child inference task appeared to lose
  its worker and restart multiple times
- direct `iris task exec` temporarily returned:
  - `Task ... not assigned to a worker`
- despite that degraded mid-run state, the job eventually recovered and closed
  successfully without a manual resubmit

**Final result**:
- wrapper state: `JOB_STATE_SUCCEEDED`
- output:
  - `gs://marin-us-central1/eval/marin_dpo_lora_smoke_export_step4_bloom_speceval_r2/inference-6f1fa3`
- verified artifacts:
  - `.artifact`
  - `.executor_info`
  - `.executor_status`
  - `shard_00000.jsonl.gz`
  - `shard_00001.jsonl.gz`
  - `artifacts/vllm_metrics.json`
- final log lines:
  - `Wrote 7728 records to 2 shards`
  - `Step eval/marin_dpo_lora_smoke_export_step4_bloom_speceval_r2/inference_ea4c878c succeeded`

### 2026-04-09 15:23 - LVI-063: Added Non-Preemptible Option For One-Off TPU Inference

**Code changes**:
- added `preemptible: bool` to
  [inference_config.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/inference_config.py)
  so `VLLMConfig` can request non-preemptible workers when needed
- added `--non-preemptible` to
  [eval_bloom_oneoff_model.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/posttrain/eval_bloom_oneoff_model.py)

**Reason**:
- this step-4 run showed repeated child-worker loss on the default preemptible
  path
- although `r2` eventually succeeded, future one-off inference retries should
  be able to pin to non-preemptible TPU workers directly
