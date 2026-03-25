# Debugging log for experiment-c-judge-path

Fix the Experiment C fixed-pair judge validation so the judge step reads spec/chosen/rejected artifacts from the prepare step output on GCS instead of serializing unresolved executor placeholders into local file paths.

## Initial status

The first Experiment C run failed before judging. The judge worker raised `FileNotFoundError` for a literal path containing `InputName(step=ExecutorStep(...))/spec/spec.jsonl`, which indicates the experiment script stringified `output_path_of(prepare_step)` instead of preserving it as an executor path reference.

## Hypothesis 1

The fixed-pair experiment used Python f-strings with `output_path_of(prepare_step)`, which forced `InputName.__str__`/repr-like serialization too early. The executor therefore passed a plain string into the worker config, and the worker attempted to open that literal text as a local file.

## Changes to make

- Patch `experiments/judge_llama_3_3_70b_fixed_pairs_refactored.py` to use `output_path_of(prepare_step) / "chosen"`, `... / "rejected"`, and `... / "spec" / "spec.jsonl"`.
- Relaunch Experiment C with a fresh job name after killing the failed/pending root.
- Append the diagnosis and retry to the research logbook.

## Future Work

- [ ] Add a small regression test or helper pattern for executor-path composition in experiment scripts.
- [ ] Audit any other experiment scripts using f-strings around `output_path_of(...)`.
- [ ] Consider a lint rule or helper to discourage string interpolation on `InputName`.

## Results

- Confirmed the bug from Iris metadata and logs:
  - prepare step succeeded
  - judge step failed with `FileNotFoundError` on a literal path containing `InputName(step=ExecutorStep(...))/spec/spec.jsonl`
- Root cause matches Hypothesis 1.
- Patched `experiments/judge_llama_3_3_70b_fixed_pairs_refactored.py` to preserve executor path references using:
  - `output_path_of(prepare_step) / "chosen"`
  - `output_path_of(prepare_step) / "rejected"`
  - `output_path_of(prepare_step) / "spec" / "spec.jsonl"`
- Relaunched Experiment C as `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry1`.
- The retry cleared the original bug:
  - worker logs showed `Loaded 2 chosen, 2 rejected responses`
  - then `Processing 2 prompt pairs via local batched judge`
  - then `Starting vLLM environment`
- The next blocker was unrelated:
  - the first retry child was killed with `Parent task preempted`
  - the root job restarted, reacquired the step lock, and resubmitted the judge child
- Current interpretation:
  - the file-path bug is fixed
  - any remaining issue on this retry is operational scheduling/preemption, not missing GCS artifacts
- Follow-up operational fix:
  - relaunched the experiment with a CPU-only root as `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry2-cpu-root`
  - the new root holds only CPU/memory/disk, while the judge child alone requests `v5p-8`
- Final outcome:
  - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry2-cpu-root` succeeded
  - the judge child loaded the fixed artifacts, started `vllm serve`, sent a batched `/v1/completions` request for 4 judge prompts, and wrote 2 preference pairs to:
    - `gs://marin-us-central1/align/debug_local_judge_llama_3_3_70b_fixed_pairs/preference_pairs-01dd5c`
  - verified row count in `shard_00000.jsonl.gz`: `2`
