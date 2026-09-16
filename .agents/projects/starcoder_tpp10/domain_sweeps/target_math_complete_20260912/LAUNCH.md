The final evaluation reuses the eleven verified first-stage measurements and scores the four remaining FineMath target checkpoints. It uses the same central1 CPU parent, v5p-8 child, scoring protocol, tokenizer, and batch size as the first stage.

Run these commands from the repository root, after training finishes:

```bash
uv run python -m experiments.domain_phase_mix.evaluate_tpp10_target_math_complete --build
bash .agents/projects/starcoder_tpp10/domain_sweeps/target_math_complete_20260912/preflight.sh
```

The build refuses incomplete first-stage evaluations or missing final permanent checkpoints and verifies the original training runtime. The preflight validates the exact launch command, builds the native Iris workspace zip locally, and checks its frozen source, tokenizer, and spec contents. It writes local receipts only.

The prepared launch command is in `submit.sh`. Run it only after the two checks above pass.

Before submission, register the release spec and a submitting evaluation job in Fieldbook, linked to the existing experiment and runs; afterward, record the returned Iris parent ID.

```bash
bash .agents/projects/starcoder_tpp10/domain_sweeps/target_math_complete_20260912/submit.sh
```

The expected parent is `/calvinxu/tpp10-target-math-completion`; the child is `target-math-completion`. Both use central1, and all experiment state remains in `gs://marin-us-central1`. The CPU parent has explicit non-preemptible placement and zero automatic retries. Durable verified results are reused if the evaluator is resubmitted.

The bundle explicitly includes only the completed spec and both existing asset directories. The first-stage spec, original training plan, and proxy scoring reference are embedded in the completed spec, so no additional `.agents` files are needed in the worker bundle. The native bundler includes repository Python modules and its generated runtime artifacts.

To check only the prepared command before the final spec exists:

```bash
bash .agents/projects/starcoder_tpp10/domain_sweeps/target_math_complete_20260912/preflight.sh --command-only
```

This command does not certify checkpoint completion or the final bundle. After the evaluation finishes, refresh the full plot with:

```bash
uv run python -m experiments.domain_phase_mix.analyze_tpp10_target_math \
  --complete-spec .agents/projects/starcoder_tpp10/domain_sweeps/target_math_complete_20260912/spec.json \
  --require-complete
```
