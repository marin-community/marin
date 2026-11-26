# Fifty-step Mochi data campaign

`experiments.post_training.long_loop.campaign` varies the source mix of the
[.131 recipe snapshot](mochi131.yaml), from commit
`fe4695fbcd27e365284dccab649a40d6caa9fb2b`, with its retired rack exclusion
removed. Every source has weight `1/k`.
The initializer, four-proposal DSpark draft, optimizer, batch, context,
drain/rolling policy, staleness, TP1/BF16 and CUDA graphs stay fixed. The actual
request is 128 GB200: 48 trainer and 80 rollout GPUs, at production priority
on `cw-us-east-08a`, with all racks eligible.

```bash
uv run python -m experiments.post_training.long_loop.campaign \
  --mix-id v2-anchor-exp5-01 --max-steps 50 --version 2026.11.26.501 \
  --validation-output campaign-validation
```

Without `--run`, this constructs the graph and runs the frozen native CPU
recipe validator on the existing .131 input bindings. This validates recipe
and role geometry against the existing model binding. It does not claim that
an unmaterialized candidate has passed data or agent admission. With `--run`,
the graph prepares the candidate pools on an
08a CPU worker, checks every published parquet SHA and population count, lowers
the unchanged TaskSpecs through the native adapter, and removes a random
200-prompt holdout (seed 20261020). The curriculum records the selected tasks
and the equal source weights. Cross-source holdout leakage fails closed.
The actual candidate's native real-agent and input checks must pass before
GPU allocation.

The trainer stops at 50 optimizer updates. The shared 43,200-second learning
callback is explicitly disabled for this campaign; other long-loop rounds
retain it. The inherited 48-hour Iris task timeout remains an operational
failure bound. A stable coordinator name makes repeated submission of the same
mix/version adopt its existing coordinator. Use a fresh calendar version for
each new experiment.

Intermediates use rolling temporary storage. After training commits its best
HF export, an 08a CPU stage verifies the export, copies and verifies the final
native checkpoint into permanent storage, then removes the run's temporary
native checkpoint root. It records kept URIs and deleted bytes in
`storage-receipt.json`. Frozen Micro210 evaluates the best HF export, publishes
through the existing EvalDash path, and preserves trial archives. This campaign
keeps the final native checkpoint without making a second final HF export or
evaluating it.

The CSV snapshot is configuration metadata copied byte-for-byte from
`experiments/human/long-loop-data-ablation/mix-candidates/mix_candidates_v{2,3}.csv`.
`source_pins.json` binds its sources to the pinned Atlas
publication registry, including full parquet hashes, counts, licenses and
source-route evidence. SWE-smith binds its existing retained release under the
canonical catalog's user-assumed-admission override; its QC status is recorded
without changing the source bytes. Raw converters for
`SWE-rebench:ibragim-bad/SWE-rebench-V2.1@repaired-20261119` and
`AllenAI:Dolci-Think-SFT-Clean-v2/precise-if` still require published TaskSpecs.
Affected mixes fail closed. A re-sealed grader release needs a fresh reviewed
catalog binding before use.

The coordinator owns the rated queue and watchdog. Take candidates in
`experiments/human/long-loop-data-ablation/campaign/queue.json` order, keep three
experiments active once the queue is ready, and record actual job IDs, metrics,
best export, EvalDash URL and storage receipts in its `runs.csv`. The example
above is a dry-run candidate, not a queue selection. Never infer completion,
coverage or a score from rollout progress.
