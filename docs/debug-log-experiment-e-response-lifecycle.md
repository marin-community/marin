# Debugging log for experiment-e-response-lifecycle

Investigate why the first heterogeneous Experiment F smoke failed after Experiment E changed `align()` to combine local chosen/rejected generation into one worker-local step.

## Initial status

The first post-Experiment-E end-to-end smoke run launched as:
- `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored`

Observed status:
- `spec` succeeded
- `prompts` succeeded
- combined `responses` failed

The failing child was:
- `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored/align-debug_vllm_70b_mixtral_rejected_smoke-responses_d2195adc-f5228fbb`

## Hypothesis 1

The new combined local-local response step is correct for same-model reuse, but starting a second TPU `vllm serve` process for a different local model in the same worker is not reliable.

## Changes to make

- Confirm from logs whether chosen generation finished before the failure.
- If so, narrow Experiment E:
  - keep the combined local-local response step only when `teacher_model == rejected_model`
  - for different local models, keep separate `chosen` and `rejected` steps and serialize them with an explicit executor dependency from rejected to chosen
- Add tests covering:
  - same-model local-local => combined `/responses`
  - different-model local-local => separate `/chosen` then `/rejected`

## Future Work

- [ ] Investigate whether TPU `vllm serve` can ever support multiple sequential different-model startups in one worker with additional teardown or environment cleanup.
- [ ] If executor-level explicit dependencies become common, consider a more principled dependency field instead of per-config `dependency_path` fields.
- [ ] Capture response-step timing separately for chosen and rejected after the serialized fallback is validated.

## Results

Confirmed from the failed child logs:
- chosen Llama generation completed far enough to send a batched `/v1/completions` request for `67` prompts
- the second startup, for Mixtral, failed before server readiness
- the failing command was:
  - `vllm serve gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230 ...`
- the deepest visible failure was inside TPU mesh init during the second server startup:
  - `AttributeError` from `tpu_inference.utils.make_optimized_mesh`, after JAX warned it was falling back to CPU

Interpretation:
- the bug is not in the combined output schema
- the bug is specifically the lifecycle assumption that a second different-model TPU `vllm serve` can be started cleanly in the same worker after the first one exits
- same-model reuse is still useful and should be kept
- different-model local-local orchestration should use two fresh child jobs with an explicit dependency instead
