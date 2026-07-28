1. **Claim** — “The MFU-versus-EP-degree curve has never been measured” and “MXFP8 expert GEMMs … do not exist on the EP64 stack at all.”

   **Problem** — The brief omits a matched, end-to-end EP64 test of the MXFP8 expert-MLP port. At d5120/i1280, 8-of-256, EP64, QB-on, cf1.0, 48 layers, and 120 steps, MXFP8 lost 2.582 p50-MFU points with matched drops. This resolves the sign of the current expert-only port at the main EP64 operating point; it is not merely an unimplemented dependency. A later d6144/i3072 EP4 screen also stayed negative. The hybrid grouped+dense recipe still lacks a complete EP-degree curve, but the brief incorrectly generalizes that narrower unknown to the already-tested expert-only port.

   **Evidence** — Local-only commit `24d411b38` on `agent/ep25-d2-bakeoff` says: “BF16 control … p50 22.345” and “MXFP8 treatment … p50 19.763,” with drop fractions 0.0885 and 0.0847; “Treatment - control … -2.582.” Local commit `fac261215e` records the fatter-shape check: “bf16 p50 9.067% … vs MXFP8 p50 8.754% … = -0.313pp, bands non-overlapping.”

   **Severity** — `blocking`

   **Fix** — Add a separate expert-only EP64 entry for the matched negative, change “does not exist” to “a local prototype exists and measured negative,” and move the current expert-only port to the sealed/negative section. Restrict the open question to materially different mechanisms such as fused quantization epilogues or the full hybrid recipe at other shapes.

2. **Claim** — “Leg-batching and QB are each measured alone and never stacked” and derisking D-3: “Compose leg-batched expert GEMMs with QB-on.”

   **Problem** — A QB-on composition was already tested. The available grouped reconstruction (`G=2`) regressed 3.66pp in a matched 120-step A/B with matched drops and loss. Full batching (`G=4`) failed to produce a step in two attempts, and the original rav patch that produced 25.39% is not committed. The exact original patch remains unverified under QB, but “never stacked” suppresses the only composition evidence and makes D-3 look like a clean two-run experiment instead of a runtime/reconstruction investigation.

   **Evidence** — Local-only commit `081450952f` on `agent/ep25-d1-adjoint` records control p50 22.66 versus grouped-batching p50 19.00, “batching G=2 = -3.66pp,” drops 0.088 versus 0.092, and loss 5.614 versus 5.643. It also states that full batching “never produced a step” and that the original rav patch was uncommitted.

   **Severity** — `blocking`

   **Fix** — Replace “never stacked” with the tested-result matrix. Keep a follow-up only if the exact rav implementation can be recovered; require it to clear an EP64 runtime/memory gate before another rack A/B. Do not project the unmatched +1.35pp into a compliant configuration.

3. **Claim** — “Sender-local router balancing” is “the only unprobed direction aimed at the hypothesised cause,” and derisking D-7 proposes sender-local bias, DeepSeek-style integral accumulation, and damped gain as new work.

   **Problem** — Both actionable hypotheses were already tested and were negative. Sender-local QB tracked global QB and slightly worsened the tail drop rate, falsifying the persistent sender-hotspot diagnosis in live training. Two integral-controller gains also remained at 46–61% drops. The terminal diagnosis in the local record is batch-stochastic within-step burstiness, which favors same-step spill or receiver pooling, not another delayed-bias controller.

   **Evidence** — Local-only commit `6ac4bbeee` reports sender-QB tail-100 drops 0.0856 versus global-QB 0.0732 and says the trajectories are “statistically IDENTICAL”; it concludes the residual is batch-stochastic. Local-only commit `a48a8a9e3` reports integral gain 0.001 ending near 0.606 drops and gain 0.01 near 0.461, versus the global baseline tail-100 0.073, and closes the integral family. The g=0.5 run completed but its metrics were lost, so the damped arm is not itself measured.

   **Severity** — `blocking`

   **Fix** — Move sender-local QB and both integral arms to the sealed table, retain the g=0.5 result as unavailable rather than measured, replace the hotspot hypothesis with the measured batch-stochastic diagnosis, and remove these arms from D-7.

4. **Claim** — Ledger item 13 assigns Receiver-ECHO the “best compliant EP64 point on record: 24.15% at 2.77% (20-step) / 22.30% at 1.71% (120-step).”

   **Problem** — The 24.153% number is not a Receiver-ECHO on/off result. It is the treatment arm of the padded-Muon A/B; both arms already use Receiver-ECHO, so the number measures item 10, not item 13. The brief also omits the only matched transport isolation in #7670: within the same clone core, ECHO-ragged reduced drops from 1.32% to 0.02% while the source reports about a 1pp MFU cost. That is the measured ECHO transport tradeoff an engineer can act on. The source’s MFU/tok/s rows should be reconciled before copying them: it reports 19.98% and 18.99% but rounds both to 279K tok/s.

   **Evidence** — The [24.153% source](https://github.com/marin-community/marin/issues/7201#issuecomment-5088824573) says: “The matched rack A/B changed only `SCALE_MUON_PAD_NONEXPERT=0` to `1`,” and lists 22.374% versus 24.153%. [Issue #7670](https://github.com/marin-community/marin/issues/7670) says the NCCL and ECHO-ragged arms “differ only in the transport flag” and reports NCCL 19.98%/1.32% drops versus ECHO-ragged 18.99%/0.02% drops.

   **Severity** — `blocking`

   **Fix** — Mark Receiver-ECHO’s 24.153% and 22.299% figures as absolute stack points with no on/off causal delta. Add a separate ECHO-ragged card with the #7670 matched speed/drop tradeoff and reconcile its MFU/tok/s inconsistency. Do not rank the 725-line Receiver-ECHO change by the padded-Muon treatment’s absolute MFU.

5. **Claim** — “Neither top-4 candidate reaches a 3% bar by spill alone at cf1.0; they need capacity headroom on top, at roughly −0.58pp per +0.05 of capacity factor.”

   **Problem** — The local experiment explicitly retracted the linear price. The measured curve has a 1.179pp cliff from cf1.0 to cf1.05, then is nearly flat above cf1.05. Pricing candidate configurations at −0.58pp per 0.05 understates the first increment by about 2× and invents a slope across a discontinuity.

   **Evidence** — Local-only commit `528ef5765` records cf1.0/m3 21.849% to cf1.05/m3 20.670% (“-1.179pp”), cf1.05 to cf1.0625 “+0.038pp,” and says: “the penalty is paid ONCE on leaving cf1.0 and is not a function of capacity thereafter.” It also falsifies the proposed 128-row alignment explanation.

   **Severity** — `major`

   **Fix** — Replace the linear estimate with the measured points and state that the cliff’s mechanism is unknown. Use 20.670% at cf1.05 and 20.708% at cf1.0625 directly in projections.

6. **Claim** — “Items 17 and 18 are small and worth folding in,” and sequence Phase F includes the two-shared-expert and Muon shape-grouping changes as commits with +0.29pp and +0.09pp benefits.

   **Problem** — These sub-0.3pp increments are presented as established FSDP wins and sequenced for implementation, but no repeated matched A/B is cited. The source explicitly says only the later PGLE result was reproduced across two runs. The brief’s own protocol says margins below about 2pp require repeated placement draws, and derisking D-6 still asks for these A/Bs under EP. The recommendation outruns the evidence.

   **Evidence** — [The cited source](https://github.com/marin-community/marin/issues/7201#issuecomment-5076593050) gives the stacked 23.17→23.46→23.55 progression, but its replication statement is specifically: “PGLE … Reproduced across two runs (24.99%, 24.77%).” No analogous replication is supplied for +0.29pp or +0.09pp.

   **Severity** — `major`

   **Fix** — Label both increments unreplicated screens, remove “worth folding in,” and keep them out of the commit series until repeated matched FSDP A/Bs establish the sign and EP A/Bs establish transfer.

7. **Claim** — “Dependencies come before dependents,” while sequence Phase B orders “Add the `sonic_cute` … backend” before “Add the `quack-kernels[cu13]` dependency”; ledger item 13 depends only on items 7 and 11.

   **Problem** — The order violates the stated rule twice. `sonic_cute` imports QuACK and therefore depends on the dependency commit. Receiver-ECHO’s measured stack uses “SM100 grouped expert GEMMs,” so item 13 also depends on the `sonic_cute`/QuACK substrate currently placed after it.

   **Evidence** — Commit `5cf76b64a` adds `quack_moe_cute.py` and the backend; commit `538381606` adds its package dependency. The [Receiver-ECHO source](https://github.com/marin-community/marin/issues/7201#issuecomment-5088824573) lists “SM100 grouped expert GEMMs and grouped weight gradients” in the kernel path.

   **Severity** — `major`

   **Fix** — Put the extracted QuACK dependency before the backend, move the shared `sonic_cute` substrate ahead of Receiver-ECHO, and add it to item 13’s dependency list.

8. **Claim** — “4-of-256 at EP64 does not fit one rack.”

   **Problem** — This is missing the model shape and is false as a general 4-of-256 claim: the brief’s own d5120/i2048 4-of-256 Receiver-ECHO run completed on one rack. The OOM result is specifically the d6144/i3072, roughly 707B candidate.

   **Evidence** — Local commit `fac261215e` labels the result “707B / 4-of-256” and decomposes tensors with hidden 6144 and intermediate 3072. The [d5120/i2048 4-of-256 run](https://github.com/marin-community/marin/issues/7201#issuecomment-5088824573) completed 20/20 steps on 64 GB200.

   **Severity** — `major`

   **Fix** — Change every occurrence to “d6144/i3072 4-of-256 at EP64 does not fit one rack.”

9. **Claim** — The branch diagram says shared base → `chunk-moe-fsdp` is “+13,” `chunk-moe-fsdp` → `b200-minimal` is “+21,” and shared base → `codex/per-layer-kv-heads-static-fa4` is “+1.”

   **Problem** — The ancestry is right but the commit counts are wrong.

   **Evidence** — After `git fetch origin`, `git rev-list --count` gives 17, 19, and 2 commits for those three edges. The tips are `c241f31d7`, `8823246ef`, and `46fbff3173`; all have the claimed ancestry from `696eb370d`.

   **Severity** — `minor`

   **Fix** — Replace the three edge labels with +17, +19, and +2.

## Verified

- `origin/main` is `1c631c4c05b6b56f20e5fda7b0a38f5d0ac27353`; the listed unmerged tokens (`sonic_cute`, `Pfsdp`, `_embedding_gather`, `_newtonschulz_4d_distributed`, `SCALE_A2A_CHUNKS`, and the other named launcher knobs) are absent from it.
- Gather dispatch matches the cited 17.552%→20.558% p50 A/B (+3.006pp), and the custom adjoint matches 20.61%→24.04% (+3.43pp); both source A/Bs are QB-off and the brief correctly flags that fidelity regime.
- Same-step spill matches the cited 22.062%/7.10% to 21.849%/3.66% result and the 20.708%/1.44% compliant point. The correction about comparing equal fractions of the LR schedule is carried correctly.
- The MuonH sharding fix matches 20.22%/14.84s/208 warnings to 22.02%/13.63s/zero warnings.
- The padded-Muon A/B matches 22.374% to 24.153%, with drops moving 2.918% to 2.765%.
- The MXFP8 quality gate matches 31,474 steps, 66.006B tokens, +7.220% throughput, +0.0562% aggregate eval, +0.1105% Paloma, +0.2088% uncheatable, and BF16 winning all 32 aggregate-eval pairs.
- The FP8 dispatch-wire issue’s terminal three-draw values are correctly quoted as 1.286× forward and 1.144× forward+backward, with bit-exact weight gradients and the layer-only caveat.
- The cited implementation commits exist locally, and the principal LOC claims for `45ce02d20`, `c9e30f848`, `497423bc6`, `1224ccb02`, `24ee86090`, `bdf61d7ed`, and `a33e16ced` match their numstats.

## Not checked

- I did not independently recompute metrics from W&B histories or Iris/XProf artifacts; I checked them against the GitHub comments and local commit records that the brief cites.
- The original rav leg-batching implementation behind the 25.39% run is uncommitted. Its exact code and its behavior under QB-on remain unverifiable; only the committed grouped reconstruction and its failed/full-batch attempts are inspectable.
- The local result commits cited above are not on any remote branch. Their contents are verifiable in this checkout, but their durability and raw job artifacts are not.
