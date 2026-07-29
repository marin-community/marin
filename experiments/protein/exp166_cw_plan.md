# Exp 166 on CoreWeave — plan

Port of the contacts-v1 amino-acid augmentation ablation
([MarinFold #166](https://github.com/Open-Athena/MarinFold/issues/166)) from TRC TPUs to
CoreWeave GPUs. The augmentation question is unchanged; almost everything else in the
original harness existed to survive TRC capacity and does not carry over.

Branch `eac/plm-exp166-cw`, cut from `eac/plm-exp153-cw`. The exp166 TPU tree
(`eac-plm-exp166`) is read-only reference and is never modified.

The only data movement this plan authorizes is one 16.44 GiB pull of the published exp117
seed into S3. Nothing is uploaded, and nothing leaves CoreWeave or GCS, without explicit
approval first.

## What the experiment asks

Does re-permuting the `<pN> <AA>` sequence statements during training improve
contacts-v1 val loss? The augmentation is a dataset wrapper over the tokenized stream, so
no corpus is regenerated and nothing is re-tokenized.

## Runs

Eight. Six train from random weights with augmentation, one per exp117 configuration.
Two continue from an exp117 checkpoint and differ only in whether augmentation is on.

| # | initial weights | augmentation | lr | wd | batch | run id suffix |
|---|---|---|---|---|---|---|
| 1 | random | yes | 3.1623e-3 | 0.2 | 64 | `lr3p162e-3-wd0p2-bs64-scratch-aug` |
| 2 | random | yes | 3.1623e-4 | 1.6 | 64 | `lr3p162e-4-wd1p6-bs64-scratch-aug` |
| 3 | random | yes | 3.1623e-3 | 0.1 | 128 | `lr3p162e-3-wd0p1-bs128-scratch-aug` |
| 4 | random | yes | 1e-3 | 0.8 | 128 | `lr1e-3-wd0p8-bs128-scratch-aug` |
| 5 | random | yes | 3.1623e-4 | 1.6 | 128 | `lr3p162e-4-wd1p6-bs128-scratch-aug` |
| 6 | random | yes | 1e-3 | 0.2 | 64 | `lr1e-3-wd0p2-bs64-scratch-aug` |
| 7 | exp117 `lr3p162e-3-wd0p2-bs64` | yes | 3.1623e-3 | 0.2 | 64 | `lr3p162e-3-wd0p2-bs64-exp117init-aug` |
| 8 | exp117 `lr3p162e-3-wd0p2-bs64` | no | 3.1623e-3 | 0.2 | 64 | `lr3p162e-3-wd0p2-bs64-exp117init-noaug` |

Run IDs are `prot-exp166cw-cv1-aaaug-1_5b-e8-<suffix>`, W&B group
`exp166cw-contacts-v1-aa-augmentation`, smoke group `exp166cw-aaaug-smoke`. The `exp166cw`
prefix keeps these clear of the 28 exp166 runs the TPU attempt already left in W&B, whose
names all end in a region. The suffix carries both variables that distinguish runs 7 and 8.
Names are identity only: lr, weight decay, batch, epochs and initialization mode live in
W&B tags and config and are never parsed back out of the string.

Runs 1-6 are each compared against the exp117 result for the same settings: same config,
same 8 epochs, augmentation the only difference.

Runs 7 and 8 are a pair. Both start from exp117's finished weights and train 8 more
epochs. Run 8 carries no augmentation, so it measures what the extra 8 epochs buy on
their own. The original 12-run TPU plan had six run-7s and no run-8, so a gain there
could not be separated from additional training.

All eight are 1.5B Qwen3, seq 8192, 8 epochs, cosine with 10% warmup, `data_seed=0`,
`pack=True`, 2 evals per epoch. Objective is `eval/tokenized/contacts-v1-val/loss` at the
final step.

Continuation covers one configuration. Whether it is worth extending is a question for
after runs 7 and 8 report.

## What CoreWeave removes

Every CoreWeave cluster reads the same bucket, so a run keeps its checkpoint wherever it
lands. That deletes the entire regional apparatus the TPU version was built around.

| exp166 on TRC | on CoreWeave |
|---|---|
| `regional_race_width: 3`, up to 36 dispatches for 12 trials | one dispatch per run |
| region in run ID and checkpoint path; Region Locality rules | no region anywhere |
| winner promotion, `race_lost`, duplicate artifacts | nothing to arbitrate |
| `relocate_after: 4d`, relocation restarts at step 0 | relocation keeps the checkpoint |
| seed fan-out: 24 copies, 296 GiB egress, 394 GiB stored | one copy per seed |
| descending reslice rotation | node count is a wall-clock knob only |
| probe floor, race floor, follow-the-grants placement | capacity is steady enough to ignore |

Training is deterministic at a fixed gang shape, so repeated runs of one configuration
add nothing. It is not bitwise reproducible across node counts, and a small objective
difference between two placements is not a defect.

## Placement

GB200 (`cw-us-east-08a`) first: 208 schedulable nodes against the H100 fleet's 36, the
exp153 smoke passed there, and the parity runs put it slightly ahead of v6e-4. H100
(`cw-us-east-02a`) is the fallback and needs no code difference. Gangs are whole nodes;
node counts are powers of two. The peer report does not break out GPU type, so how much
of that 208 is GB200 and free is unknown until we submit — submitting is the only
measurement.

**Four nodes for every run**, which is also the gang ceiling. Every run moves the same
8 x 4.677B = 37.4B tokens, and calibration measured the two batch sizes 1.6% apart, so
wall clock depends on GPU count and almost nothing else. Equal gangs therefore finish
together; the earlier plan of four nodes for bs64 and two for bs128 would have made the
bs128 runs take about twice as long.

| batch | runs | nodes | GPUs | seqs/device |
|---|---|---|---|---|
| 64 | 1, 2, 6, 7, 8 | 4 | 16 | 4 |
| 128 | 3, 4, 5 | 4 | 16 | 8 |

32 nodes if all eight run at once, about 26 h each.

Measured on the first three production gangs (2026-07-29), which replaces the one-node
extrapolation:

| batch | tok/s at 4 nodes | vs one node | ETA |
|---|---|---|---|
| 64 | 354,923 | 3.23x | 28.9 h |
| 64 | 372,683 | 3.39x | 27.5 h |
| 128 | 407,948 | 3.71x | 25.1 h |

81-93% scaling efficiency, and bs128 is about 10% faster per token than bs64 at the same
gang — consistent with the 1.6% single-node gap widening slightly under communication.

### Launch failure modes seen 2026-07-29

Three distinct ones, all contention rather than defects. The same trial that failed runs
unchanged on a later attempt, and a retry keeps the run id, so identity and any checkpoint
carry over — only the Iris job name changes.

| symptom | cause | response |
|---|---|---|
| `Unschedulable: 0/208 nodes are available`, `SchedulingGated` | Kueue admits a gang all-or-nothing and the nodes were not free | resubmit when capacity frees |
| `ValueError: num_devices (16) must be divisible by num_slices (3)` at `trainer.py:942` | the gang asked for four nodes and only three registered with the JAX coordinator, so the mesh cannot be built | resubmit; a partial gang is unrecoverable in place |
| `ConnectError` on `register_endpoint`, or a driver that fails with exit 0 and no log output | transient controller RPC, or a driver pod that never reached Python | resubmit |

Read a Python traceback in the *training* task as a real fault; the three above all occur
before or outside training and are indistinguishable from preemption.

**Thirty-two nodes is not reliably available.** Submitting all eight at once on 2026-07-29
admitted three and failed five within 90 seconds: Kueue gang admission is all-or-nothing,
so a gang that cannot be placed fails immediately rather than queueing
(`Unschedulable: 0/208 nodes are available`, `SchedulingGated`). The cluster is shared and
other users held most of it. Submit in waves sized to what is free, and treat a
gang-admission failure as capacity rather than a fault: the same trial that failed will
run unchanged once nodes free up. Node count is a pure wall-clock knob, so a wave that
cannot get four nodes per run can take two.

## Measured 2026-07-29

Three one-node GB200 smokes, 20 steps each, with evals and a final checkpoint.

| run | batch | seqs/device | tok/s | final eval |
|---|---|---|---|---|
| scratch, augmented | 128 | 32 | 110,030 | 6.2489 |
| scratch, augmented | 64 | 16 | 108,290 | 6.3303 |
| exp117-init, unaugmented | 64 | 16 | 107,059 | 5.8976 |

`MAX_SEQS_PER_DEVICE["GB200"] = 32`. That is the largest microbatch one node can be asked
for at batch 128, not an observed ceiling; the real limit is higher and unmeasured. H100
stays unmeasured and fails loudly.

**The two batch sizes run 1.6% apart.** Every run moves the same 37.4B tokens, so the
extra 35,680 optimizer steps a bs64 run pays cost about that much — not the multiple the
asymmetric gang allocation assumes. At bs64 on four nodes and bs128 on two, the bs128 runs
get half the GPUs for the same token budget and take roughly twice as long, which is the
opposite of evening out completion times. Four nodes for all eight runs gives 32 nodes and
about 26 h each, finishing together. Multi-node scaling is extrapolated from one node and
has not been measured.

The seed loads: the training job logs `Loading checkpoint from .../exp166-init/`, and the
exp117-init trajectory sits below scratch at every step. It does **not** start near
exp117's 2.71 because a 20-step smoke compresses warmup to 2 steps, so the model takes two
full-learning-rate updates before the first logged point. A production run warms up over
7,136 steps.

## Calibration

Two short runs on `tmp/ttl=1d/`, one per GPU type, before anything else is launched.
They fix max sequences per device for the 1.5B and give a measured tokens/s at 1, 2 and 4
nodes for both batch sizes. exp153's H100 value of 8 was measured on the 6B and does not
transfer. An unmeasured GPU type fails loudly rather than guessing, because guessing wrong
costs a multi-day run.

Watch the augmentation's cost during calibration: it does a `device_get` and
`device_put` per example inside `get_batch`, so at batch 128 that is 128 host round trips
per step. It was acceptable on TPU and has not been profiled on GPU.

## Storage

Everything this experiment writes goes under its own prefix,
`s3://marin-us-east-02a/MarinFold/exp166cw_qwen_contacts_v1`, which is what `MARIN_PREFIX`
is set to on both driver and gang.

| what | path under the prefix |
|---|---|
| run outputs | `checkpoints/{run_id}/2026.07.29.01/{checkpoints,hf}` |
| init seed | `checkpoints/exp166-init/prot-exp117-cv1-s02-1_5b-e8-lr3p162e-3-wd0p2-bs64-europe-west4/2026.07.13.02/checkpoints/step-71359/` |

Checkpoint cadence:

| kind | cadence | size |
|---|---|---|
| resumption checkpoint | every 10 minutes, rolling; each replaces the last | 16.44 GiB transient |
| permanent Levanter | final step only | 16.44 GiB per run |
| HF export | final step only | 5.48 GiB per run |

The 10-minute rolling save is marin's `_RESUMPTION_INTERVAL` and caps what a preemption
costs. It is written inside the run's own directory rather than the bucket-root
`tmp/ttl=` path, following exp153's `temporary_base_path=None`, so nothing escapes the
prefix. No per-epoch permanents: the objective is the final val loss, and the
R-precision-vs-tokens curve that motivated per-epoch keeps belongs to #117.

Total footprint ~192 GiB: 16.4 GiB seed plus ~21.9 GiB for each of eight completed runs.

Nothing under `MarinFold/` expires on its own and the bucket has no object versioning, so
every delete is permanent.

### Token cache

The cache is **read in place, not copied**. `tokenized(pin=...)` takes an absolute
location:

```
s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1/2026.07.25
s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25
```

`ArtifactStep.path()` returns a pin verbatim when it carries a URL scheme instead of
joining it onto the prefix (`lazy.py:229-233`), and a pinned step records no provenance
(`lazy.py:334`), so exp154 is read-only from here. That saves copying 6.2 GiB and 820
objects, and avoids rewriting the absolute `output_path` embedded in the cache's
`.executor_info` and `.artifact.json`.

**Consequence: `exp154_qwen_contacts_v1/tokenized/` must not be pruned.** That prefix is
the one 614 GiB of exp153 checkpoints were deleted from; the cache under it is now a live
dependency of this experiment.

Before the first real run, byte-compare the pinned cache against the GCS original. It is
the one confound that would invalidate every number here, and it is the check that ruled
data out of the TPU-vs-GPU investigation.

## Seeds

Runs 7 and 8 initialize from a single S3 copy of the exp117 final checkpoint for
`lr3.1623e-3 / wd0.2 / bs64`, pulled once from
[open-athena/marinfold-exp117](https://huggingface.co/open-athena/marinfold-exp117) in
Levanter format, then read-only. It is the only exp117 checkpoint published there, which
is why continuation covers that one configuration.

exp166 pairs each configuration with the checkpoint from the exp117 run that used the same
hyperparameters, so a configuration without a published checkpoint cannot have a
continuation run.

## Recovery

Two clocks, both measured from when a run's W&B step last advanced. A run that has never
produced a step is stalled from its first dispatch.

| stall exceeds | action |
|---|---|
| 6h | resubmit on the same cluster and gang |
| 12h | resubmit on a different cluster or gang size |

No relocation clock: moving a run here keeps its checkpoint, so relocation is an ordinary
re-dispatch. W&B state plus a moving step is the only liveness signal; Iris `running` is a
scheduling gate and never evidence that training started.

One hard invariant: at most one live dispatch per run. Two gangs writing one checkpoint
corrupt it.

## Where the code lives

| what | where |
|---|---|
| worktree | `/home/exedev/repos/marin-br/eac-plm-exp153-cw/.claude/worktrees/exp166-cw` |
| branch | `eac/plm-exp166-cw`, cut from `eac/plm-exp153-cw` at `92a601707`, local only |
| files written | `experiments/protein/exp166_cw_sweep.py`, `experiments/protein/exp166_cw_plan.md` |
| read-only reference | `/home/exedev/repos/marin-br/eac-plm-exp166` (branch `eac/plm-exp166`) |

Nothing outside that worktree is edited. The exp153 branch, the parity branch
(`eac/plm-exp153-parity`), the exp166 TPU clone, and `main` are all untouched.

### Isolation

| resource | exp166cw | shared with |
|---|---|---|
| S3 writes | `MarinFold/exp166cw_qwen_contacts_v1/` (new, empty) | nothing |
| S3 reads | `exp154_.../tokenized/`, `MarinFold/data/document_structures/` | exp153, #117 — read-only |
| W&B | group `exp166cw-*`, run ids `prot-exp166cw-*` | nothing; zero name collisions |
| Iris | job names derived from the run ids | nothing |

The prefixes `exp108_*`, `exp112_*`, `exp154_*/checkpoints`, `exp163`, `exp167_eval` and
`exp169_eval` are never written to. A pinned cache step records no provenance, so the
exp154 read cannot write back.

## Code

New `experiments/protein/exp166_cw_sweep.py` on this branch, exp153's structure with
exp166's augmentation dropped in.

- Run identity is configuration plus initialization mode. No region.
- `attn_backend=AttentionBackend.JAX_FLASH` on the model config. Levanter's GPU default is
  Transformer Engine, absent from the `gpu` extra, and it falls back to an O(seq^2) kernel
  that OOMs at seq 8192.
- GPU batch calibration in place of the TPU per-family correction factors.
- Seed resolved from a single S3 path.
- Token caches declared with an absolute `pin=` rather than resolved against the prefix.
- `MARIN_PREFIX` forwarded to the gang, which does not inherit the driver's environment.
- `--priority batch`; driver runs inside the target cluster and must outlive its gang.
- Driver needs at least 3 GB; Kubernetes enforces memory requests as hard limits.

The augmentation code, the six configurations, and their exp117 losses port unchanged.

## Order

1. Copy the one published exp117 seed from HF into S3 (16.44 GiB).
2. Byte-compare the pinned token cache against the GCS original.
3. Write the sweep script.
4. Calibrate on GB200, then H100.
5. Smoke one run end to end, checkpoint and eval, on temp storage.
6. Launch runs 1-8.
