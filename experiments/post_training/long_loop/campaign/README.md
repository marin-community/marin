# Fifty-step Mochi data campaign

`experiments.post_training.long_loop.campaign` varies the source mix of the
[.131 recipe snapshot](mochi131.yaml), from commit
`fe4695fbcd27e365284dccab649a40d6caa9fb2b`, with its retired rack exclusion
removed. Every source has weight `1/k`.
The initializer, four-proposal DSpark draft, optimizer, batch, context,
drain/rolling policy, staleness, TP1/BF16 and CUDA graphs stay fixed. The actual
request is 144 GB200: 64 trainer and 80 rollout GPUs, at production priority
on `cw-us-east-08a`, with all racks eligible.
The campaign uses PP16/CP4/TP1/EP4 for training after backward activation
allocations exhausted memory with PP12. Serving geometry is unchanged.

```bash
uv run python -m experiments.post_training.long_loop.campaign \
  --mix-id v2-mix-032 --max-steps 50 --version 2026.11.26.501 \
  --validation-output campaign-validation
```

Without `--run`, this constructs the graph and runs the frozen native CPU
recipe validator on the existing .131 input bindings. This validates recipe
and role geometry against the existing model binding. It does not claim that
an unmaterialized candidate has passed data or agent admission. Before invoking
`--run`, the operator builds the graph's canonical pool dependency in a CPU-only
08a job. The RL graph then reuses that completed artifact. Pool preparation
checks every published parquet SHA and population count, lowers
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
`source_pins.json` is an immutable configuration readback of Atlas load
`54405be9b4db4c2d9828549469abe759`, snapshot `assembly-r40-cut-v1`, served by Atlas
revision 39. The CSV supplies only source IDs and k. The catalog supplies each
release, pipeline version, admission status and measured actor route. Shared
`curriculum_entries` and `weighted_curriculum` bind these entries and their holdout;
the user-authorized admission divergence permits retained Pending and Excluded
pins and records each difference in the manifest. Never substitute a population.
An unbound source or an unresolved task-environment or grader defect holds its
mix while the next runnable candidate fills the slot. Qualified re-seals and
re-pins return held mixes to their rated queue positions.

HF blob SHA-256 metadata is checked against the exact catalog-pinned revision in
region before reading TaskSpecs. The published literal TaskSpec byte hashes and
catalog population counts are verified without reserializing source rows. Each
pool manifest records its actual input readbacks. A changed catalog admission or
release requires a fresh readback and artifact version.
Preparation reads and lowers payloads in bounded batches, retaining task identities
for the holdout draw. Each mix must finish its regional CPU pool build before its
RL coordinator is submitted.

The coordinator owns the rated queue and watchdog. Take candidates in
`experiments/human/long-loop-data-ablation/campaign/queue.json` order, keep three
experiments active once the queue is ready, and record actual job IDs, metrics,
best export, EvalDash URL and storage receipts in the run table of
[marin-logbooks issue91](https://github.com/marin-community/marin-logbooks/issues/91),
using `campaign-runs set`. The CSV run tracker is retired. The example
above is a dry-run candidate, not a queue selection. Never infer completion,
coverage or a score from rollout progress.

An OOM in any pool or training worker halts new submissions and refills until
the coordinator explicitly authorizes resumption. Record the stage, rank, memory
and sequence evidence; other active runs continue. The reset authorizes cancellation
of .529 only. Other running jobs require separate explicit permission.
