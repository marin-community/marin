# NEST-BURN-001: 24-hour fixed25 burn-in

## Question

Does the fixed E16 ⊂ E128 ⊂ E256 treatment retain or improve full-model quality
at the d768 compute-optimal point, without material optimizer-step overhead?

## Preregistered comparison

- Control: d768/L8, E256, top-4, every training row eligible for all experts.
- Treatment: identical model, with 25% of rows restricted. Restricted rows
  alternate between the fixed E128 and fixed E16 subsets; unrestricted rows use
  E256. Eligibility-QB maintains separate router-bias state per routing mode.
- Data: canonical datakit store `8ac06c74`, phase-0 weights for the first 80%
  and phase-1 weights for the last 20%, simulated against the production 10.37T
  token budget. Both arms use seed 0 and therefore the same sampled sequences.
- Scale: compute budget 4.14e18, hidden dimension 768, and heuristic target
  steps 2^15. The May heuristic derives L8, six query heads, one KV head,
  384-wide experts, 4.41449B tokens, sequence length 8192, global batch 32,
  and 16,840 updates. Each arm uses 16 GB200s with full FSDP and expert axis 1;
  capacity factor remains 1.25. Sixteen devices are the largest balanced
  topology for the literal E16 prefix without splitting one global example.
- Optimizer: derived from the same base inputs, without hand overrides:
  MuonH and AdamH-group LR 0.00837984, plain-Adam LR 0.00193381, beta1 0.9062,
  beta2 0.998001, epsilon 1.25564e-15, 1% warmup, linear decay to
  min_lr_ratio 0.05, and no gradient clipping.
- Precision and kernels: fp32 parameters and bf16 compute/output. The
  sequence-8192 FA4 THD executable reproducibly freezes at first dispatch, so
  the matched promotion uses cuDNN fused attention and full FSDP. Datakit
  remains pack=1; the redundant segment mask is disabled because every packed
  example contains at most one document. Padding remains excluded from loss.
  Any further fallback is applied to both arms and starts new run identities.

## Gates

1. Liveness, through update 1,000 (262.144M tokens): finite loss and gradients;
   no sustained routing overflow above 1%; treatment median step time no more
   than 10% slower; cross-region reads do not dominate step time. A kernel or
   precision failure contains no quality observation and permits a matched
   fallback.
2. Main phase, through the last evaluation before 80% of training: both arms remain
   finite and improve on training and held-out losses. Stop an arm for a
   treatment-specific loss regression above 0.10 nats at two consecutive
   evaluation points, sustained overflow above 5%, or more than 25% step-time
   overhead.
3. Datakit cooldown, final 20%: compare full E256 Paloma, uncheatable, and
   datakit validation curves at matched tokens and analytic FLOPs. The fixed25
   arm additionally reports E128 and E16 losses.
4. Post-training: initialize matched full-model SFT jobs from each final
   checkpoint. Run one packed epoch of pinned WildChat followed by one packed
   epoch of pinned canonical-think data, with the same full-E256 SFT recipe.
   Export BF16 HF checkpoints, then run matched NLP/chat evaluations and
   `tb2-lite` plus `swebench-lite`. Agentic scores may be zero at this scale;
   they are pipeline and relative-quality measurements, not a leaderboard claim.

## Decision rule

Promote fixed25 if its final full-model Paloma and broad downstream aggregate
are not worse than control, its nested E128/E16 modes remain usable, and its
measured training-time surcharge is below 10%. A full-model quality gain at
matched tokens and nearly matched elapsed optimizer cost is the preferred
outcome.

## Time budget

- Preflight, compilation, and first gate: 1 hour.
- Pretraining: hard stop at 15 elapsed hours. After 200 steady-state updates,
  extrapolate the finish time; do not change the compute-optimal cell after it
  has produced an optimization observation.
- SFT and export: 4 hours.
- Evaluation: 3 hours.
- Analysis and report: 1 hour.

The 4.414B-token target is expected to fit well inside the pretraining window.
The schedule tolerates periodic preemption through 10-minute checkpoints and
Iris retries; exit 137 is treated as preemption unless logs show otherwise.

## NEST-BURN-002: 100B-token long-horizon gate

The compute-optimal cell completed too quickly to measure whether the fixed25
effect persists through a production-length optimizer trajectory. NEST-BURN-002
starts both arms from scratch and holds the architecture, data order, seed,
precision, capacity factor, and reference attention implementation fixed. It
changes the global batch from 32 to 128 and derives a fresh optimizer for the
99,999,547,392-token horizon. Resuming NEST-BURN-001 would cross a batch-size
and learning-rate-schedule discontinuity.

Each arm runs 95,367 updates at sequence length 8,192 and global batch 128 on
64 GB200s. The mesh has a four-way replica axis and a 16-way data axis, so each
device still processes two sequences per update. The E256 control routes every
row across all experts. The fixed25 treatment restricts 25% of rows,
alternating the fixed E128 and E16 prefixes. Evaluations run every 2,500
updates in full-E256 mode for both arms and in E128/E16 mode for fixed25.

The 100B-token heuristic gives MuonH/AdamH LR 0.00488646, plain-Adam LR
0.00112764, beta1 0.9062, beta2 0.992028, epsilon 2.98810e-15, 1% warmup,
linear decay to 5% of peak LR, and no gradient clipping. This run is about
23 times beyond the d768 cell's compute-optimal token count. It tests
long-horizon treatment behavior and cost, not compute-optimal scaling quality.

Promote the arm through the long-horizon gate if both runs remain finite,
fixed25 full-mode Paloma stays within 0.05 nats of E256 at two consecutive
evaluations, and fixed25 median steady-state update time remains within 10% of
E256. Stop fixed25 for a treatment-specific regression above 0.10 nats at two
consecutive evaluations, sustained routing overflow above 5%, or update-time
overhead above 25%. A single kernel, node, data-loader, or collective failure
does not count as a quality observation; retry both arms from their latest
checkpoints on the same code snapshot.

The NEST-BURN-001 medians project 9.76 optimizer hours for E256 and 9.83 hours
for fixed25 before compilation, evaluation, and checkpoint overhead. The pair
uses about 1,254 optimizer GPU-hours. Checkpoint both arms every ten minutes,
treat exit 137 as preemption unless logs identify another cause, and post
matched-token loss, routing, and timing updates to PR #7667 at least every six
hours.
