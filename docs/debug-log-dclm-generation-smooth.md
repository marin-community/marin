# Debugging log for DCLM generation-smooth eval failures

Goal: complete the 300M DCLM Core v2 “some smooth scalar for every task” gap-fill job by fixing deterministic generation-smooth failures before launching a reviewed recovery.

## Initial status

The live Iris parent `/calvinxu/dm-300m-dclm-smooth-all22-gapfill-retry3-20260613-0650` is running, but the child fanout is unhealthy. At the 2026-06-13 07:11 PT poll it had 65 succeeded children, 60 running children, 235 pending children, and 88 failed children. Pending children are mostly waiting on east5 `v5p-8` capacity. Failed children are dominated by deterministic generation-smooth failures: 83 `TypeError: can only concatenate str (not "list") to str` crashes, plus empty-continuation `PromptCompletion` failures, one SQuAD resource-mapping failure, and two SIGSEGV/add-port failures.

## Hypothesis 1

The `generation_smooth` mode converts generation tasks to Levanter loglikelihood tasks by changing `output_type` and metrics, but some lm-eval generation tasks still emit singleton list targets. Levanter’s loglikelihood tokenizer concatenates `context + completion`, so a list-valued completion crashes.

## Changes to make

Normalize loglikelihood context/completion values to strings when they are singleton string lists or tuples. Keep multi-reference targets explicit by raising a clear error, because a single loglikelihood score is ambiguous for multiple valid continuations.

## Hypothesis 2

Some converted generation tasks produce an empty continuation or a continuation that is removed by tokenization/truncation. `PromptCompletion` correctly rejects packed examples with no response tokens, but loglikelihood should preserve lm-eval response cardinality by returning loglikelihood 0 and `greedy=True` for those empty continuations.

## Changes to make

Teach the loglikelihood packing path to return skipped request IDs for tokenized empty continuations, mark them as covered before dispatch, and preserve explicit segment IDs in greedy packing so filtered requests still map back to original request indices.

## Results

Focused local validation passed:

- `uv run python -m py_compile lib/levanter/src/levanter/eval_harness.py lib/levanter/src/levanter/data/packing.py lib/levanter/tests/test_eval_harness_loglikelihood_preprocessing.py lib/levanter/tests/test_packing.py experiments/domain_phase_mix/launch_300m_dclm_core_evals.py tests/test_300m_dclm_core_evals.py`
- `uv run --package marin --extra cpu python -m pytest -o addopts='' lib/levanter/tests/test_eval_harness_loglikelihood_preprocessing.py lib/levanter/tests/test_packing.py::test_greedy_pack_prompt_completions_preserves_explicit_segment_id -q`
- `uv run --package marin --extra cpu python -m pytest -o addopts='' tests/test_300m_dclm_core_evals.py -q`

The SQuAD registered `squad_completion` task ignores inline `output_type=loglikelihood` and remains `generate_until`. The launcher now uses a dedicated inline `squad_smooth_loglikelihood` task for `squad_10shot`, with the same `hazyresearch/based-squad` `{{text}}` prompt field and `{{value}}` target field. A local `lm_eval.tasks.get_task_dict` check confirms this loads as `loglikelihood`.

Executor retry semantics support reusing the same executor prefix: `SUCCESS` steps are skipped, `FAILED` steps are force-run by default, and active `RUNNING` locks are waited on. Therefore the recovery should keep the same `--name-prefix`, `--executor-prefix`, state CSV, and eval keys as retry3 while changing only the parent Iris job name and bundled code.

Claude Code review through `ctc` and Opus 4.8 returned Conditional GO. The hard gates were:

- Confirm downstream loglikelihood results scatter by segment ID. Code inspection shows `_eval_loglikelihood` returns IDs from `per_segment_loss` / `per_segment_correct`, and `LevanterHarnessLM.loglikelihood` scatters by those IDs. Added `test_per_segment_loss_preserves_explicit_noncontiguous_segment_ids`, which passed along with the focused preprocessing tests.
- Confirm retry3 is drained before retry4. At the 2026-06-13 07:53 PT Iris poll, retry3 was terminal failed with 201 succeeded children, 248 failed children, 0 running children, and 0 pending children.

## Future work

- [ ] Separately classify the SQuAD `No resource mapping found` path if it persists after splitting generation-smooth task aliases.
- [ ] Treat SIGSEGV/add-port failures as retryable infrastructure faults unless they reproduce under the patched single-task retry.

## Hypothesis 3

Retry4 exposed list-valued completions with multiple references, not only singleton lists. The previous patch intentionally raised on this case, but generation QA tasks such as CoQA/SQuAD can legitimately provide multiple acceptable target strings. For the smooth proxy, a principled scalar is the maximum loglikelihood over references for each original request.

## Changes to make

Expand multi-reference completions into separate packed loglikelihood segments with unique segment IDs, keep a segment-to-original-request map, and reduce results by max loglikelihood per original request. Keep greedy correctness as an OR over references. Skip only requests whose every reference has no continuation tokens after tokenization/truncation.

## Results

Fresh focused validation after the multi-reference patch passed:

- `uv run python -m py_compile lib/levanter/src/levanter/eval_harness.py lib/levanter/src/levanter/data/packing.py lib/levanter/tests/test_eval_harness_loglikelihood_preprocessing.py lib/levanter/tests/test_packing.py experiments/domain_phase_mix/launch_300m_dclm_core_evals.py tests/test_300m_dclm_core_evals.py`
- `uv run --package marin --extra cpu python -m pytest -o addopts='' lib/levanter/tests/test_eval_harness_loglikelihood_preprocessing.py lib/levanter/tests/test_packing.py::test_greedy_pack_prompt_completions_preserves_explicit_segment_id lib/levanter/tests/test_packing.py::test_per_segment_loss_preserves_explicit_noncontiguous_segment_ids -q`
- `uv run --package marin --extra cpu python -m pytest -o addopts='' tests/test_300m_dclm_core_evals.py -q`

A local lm-eval task inspection confirmed that the list-valued generation-smooth targets are CoQA alternative references, not continuation fragments. Among the first 500 CoQA validation docs, every target was a list with 1-4 acceptable answer strings; examples include quoted and unquoted variants such as `"Thank you"` and `Thank you`. The other sampled generation-smooth tasks used scalar string targets in the first 500 docs. This satisfies the remaining CC condition for interpreting list completions as alternative references and reducing by maximum reference loglikelihood.
