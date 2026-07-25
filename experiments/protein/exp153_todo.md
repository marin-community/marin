# Exp 153 — What We Know, What's Left

A running reminder for the contacts-v1 **6B** sweep on CoreWeave
([#153](https://github.com/Open-Athena/MarinFold/issues/153), parent
[#154](https://github.com/Open-Athena/MarinFold/issues/154)).

Notes to ourselves. The operating rules, paths, commands, and measured numbers live in
`exp153_policy.md` — when the two disagree, the policy wins.

## Settled

**The setup runs end to end.** Tokenization, a 1-node smoke, and a 2-node smoke all passed.
Token counts match #117 exactly, so this is the same corpus the 1.5B and 3B rungs used.

**Storage and speed are known.** ~34 s/step on one node, ~17 on two — near-linear. An
8-epoch trial is about a week on two nodes and leaves ~0.59 TiB behind; the shorter rungs
are ~0.09 TiB each. The bucket already holds ~360 TB, so even a full grid is a few percent
of it.

**Batch priority just works.** `--priority batch` on the driver covers every child job — the
scheduler walks the parent chain. MarinFold #108 had to bypass the executor to get this; we
don't.

**The cluster is placement, not identity.** All CoreWeave clusters share one bucket, so a
trial can move between them and resume. That deleted the entire cross-region-restart concept
the TPU policy needed.

**Runs are not bitwise reproducible across node counts.** Different accumulation depth means
a different reduction order; 1-node and 2-node runs of the same point diverge at step 2.
Statistically equivalent, so this is a caveat rather than a problem — but don't chase a small
objective difference between placements as if it were a bug.

**Checkpoints stay permanent for now.** Staging them in `tmp/` with a TTL and promoting
winners later is appealing and the storage side does support it — the bucket has enabled
30-day lifecycle rules, verified against S3 directly. What is not yet understood is the code
that decides a checkpoint's lifetime: temp paths nest inside one another, the TTL constant is
hardcoded, and the config resolving inside a CoreWeave pod is not the one the filenames
suggest. Not worth staking real checkpoints on until traced. See the policy's retention
section.

## Left to figure out

1. **Calibrate GB200.** H100 is done: capacity is 8 sequences per GPU, measured end to end
   (train + eval + checkpoint survive at 8; 16 dies in the allocator). One number per GPU
   type extrapolates to every gang size, so no H100 work remains. GB200 has no measurement
   and no smoke yet.
2. **Reconcile the policy with `run-adaptive-sweep`.** Its tools expect `max_inflight_chips`
   and a recovery triple ending in `cross_region_restart_timeout`; our policy uses nodes and
   has no cross-region concept. Either add a mapping note or adapt the tooling. **Nothing
   launches until this is resolved.**
3. **Decide the node ceiling for real.** Currently 3, which caps a gang at 2 nodes and puts
   the 8-epoch rung at ~7 days per point. Fine for early reps, probably not for the full
   grid.
4. **GB200 is unproven.** `cw-us-east-08a` has by far the most capacity, but no levanter
   dense-LM training has ever run on Blackwell in this repo. Its target list stays empty
   until a smoke passes.
5. **Write the pruning script.** Keep top-N per rung, drop the rest to final-only. The
   delete procedure is proven (dry-run, read it, delete, verify counts); what's missing is
   the part that picks winners from the W&B objective.
6. **Retest 8-node gangs** — only matters if the ceiling goes up. #108 hit a JAX bootstrap
   abort at 8 nodes; the timeout that likely caused it was raised afterwards.
7. **Profile for wall-clock wins** (deferred by request). #108 measured ~15% MFU and suspects
   non-fused attention plus f32 FSDP parameter gathers.
8. **Delete the smoke checkpoints** once they stop being useful (~96 GB each).

## Worth upstreaming

- Zephyr's default coordinator is too small for any Kubernetes backend, and its docstring
  claims a different default than the code uses. We worked around it in the tokenize path;
  the default itself is still wrong.
- marin#7013 (GPU attention falling back to a non-fused kernel) is still open, and is why
  `JAX_FLASH` has to be set by hand.
