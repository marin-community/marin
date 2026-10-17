# Score-centering readiness evidence

This experiment qualifies the optional PPO/TIS implementation in [MarinSkyRL #968](https://github.com/marin-community/MarinSkyRL/pull/968).
It does not compare learning quality or model-parameter gradient changes. Research [#996](https://github.com/marin-community/MarinSkyRL/pull/996) is outside the merge candidate.

## Source and workload

The product candidate is `2fb64c1c2b5e8692f996942704383be68a242587`, incorporating main `4f40ed137f3c040558f6b864fdf11e3c70780b7e`.
The [diagnostic runtime](https://github.com/marin-community/MarinSkyRL/tree/fe6c3a9300129af2126e8f81422f728f4d6eda52) includes that candidate plus the finite [worker probe](https://github.com/marin-community/MarinSkyRL/blob/fe6c3a9300129af2126e8f81422f728f4d6eda52/skyrl-train/skyrl_train/entrypoints/score_centering_readiness.py).
That scaffolding is excluded from the production PR. The runtime lock, actual package versions, parallelism and sampling settings are retained in each result.

The submitted YAML files preserve exact model, tokenizer, data, resource and artifact identities. Their client-config paths refer to the devbox checkout used for submission; equivalent launches must point that field at their own validated Iris client configuration. `prepare` validates without submission. `detach` submits one finite job through the native SkyRL launcher and returns; it does not schedule another attempt.

Snowball uses the 67B/A2B SFT checkpoint, four 8-H100 learner nodes and one 8-H100 inference node in `cw-us-east-02a`, at BATCH priority. Megatron has PP2, EP8, DP16 and TP/CP/SP1; inference has DP8/EP8. The batch has 32 prompt groups with four responses each. Fixed response lengths 64/256/1024/4096 keep tensor shapes matched. EOS stopping is disabled for those windows; no minimum-token processor changes the distribution. Prompt and response padding, mask holes and an entirely masked row exercise the existing collator. A final native task/admission/GRPO update also exercises the production trajectory path with ordinary stopping.

## Protocol

[protocol.json](protocol.json) preserves the finite windows, deadlines, retry limit, its one Qwen startup amendment and failed starts. The four modes all retain tokenwise TIS with upper cap 2:

| Mode | Head capture and old/current rescoring | SC | Existing training telemetry |
| --- | --- | --- | --- |
| ordinary | none | off | off |
| telemetry | none | off | on |
| capture | natural K128 | off | on |
| sc | natural K128 | on | on |

The telemetry toggle is `trainer.training_metrics`, covering chosen-token mismatch and token age. Existing worker PPO/TIS metrics are collected in every mode.

Each Snowball mode has one warmup and three measured windows. Mode order rotates by repetition. Timing includes generation, CPU collation, old-policy scoring, current-policy optimizer execution and the serialized cycle. Fixed signed advantages qualify finite applied optimizer updates. The sampler stays at its verified initial weights during the windows; learner updates continue, and their count is retained separately from the sampler's source tag. Stale-source checks replay identical tokens before and after those updates. A fresh full weight sync precedes the final native task update. The Snowball plan has 18 applied updates in total: 16 timing windows, one sampled-diagnostic update and one native update.

The probe independently checks chosen and selected-head log probabilities against full-vocabulary normalization of actual Megatron logits at causal positions. Actual vLLM prefixes and selected IDs are replayed, with a one-token-shift negative control, at temperatures 1.0 and0.7. Matching-weight TIM is reported as a residual, rather than requiring equality between inference and trainer engines.

Peak memory includes reset Torch allocator peaks and whole-device NVML samples every 0.1 seconds, uniformly in all modes. Short device-memory spikes can fall between samples. Reserved memory uses the allocation's existing cache after warmup. Repeated timing intervals describe this single allocation; they do not estimate variation between allocations. New sampled diagnostics stay opt-in.

## Qwen startup qualification

Job `/romain/sc-readiness-qwen-a4-01a146a3` succeeded with nine applied updates, no skips and finite positive gradient norms. Its eight windows had identical response token hashes, mixed lengths 8/24/64/128 and the same masks as the Snowball probe. Both temperatures passed independent normalizer checks (maximum error below 2e-6) and correct-prefix replay beat the shifted-prefix control. After eight updates, trainer log probabilities on identical tokens changed by an absolute mean 0.08907 and maximum 3.40807; the unchanged sampler still served source tag 1. Qwen has only one measured window per mode, so it provides startup evidence and no Snowball overhead conclusion.

See [the compact Qwen result](results/qwen-a4.summary.json) and [every timing window](results/qwen-a4.windows.json).
The full artifact, including exact prefix/token/support samples, is `s3://marin-us-east-02a/marin/users/romain/score-centering-readiness-01a146a3/sc-readiness-qwen-a4-01a146a3/readiness.json`.
Task milestones are retained in Echo under project `marinskyrl:score-centering-readiness` (work-log 4714).

## Snowball qualification and report recovery

The executed runtime was `fe6c3a9300129af2126e8f81422f728f4d6eda52`. Independent actual-logit checks covered 381 token positions at each temperature, with maximum chosen/head normalization errors below 2e-6. At matching weights, chosen-token absolute TIM had mean 0.03853, p99 0.35906 and maximum 3.37347 over 168386 valid tokens. The 21553408 head entries had absolute mean 0.16161 and maximum 11.18716; sampled p99 was 0.95795. Correct-prefix replay had absolute mean gap 0.15908 versus 6.31592 for the shifted-prefix control. These residuals do not establish agreement between different engines.

All 16 timing windows persisted before the diagnostic report writer failed on an optional `native.metadata["uids"]` lookup. The job terminal state is FAILED, and the raw report's complete flag remains false. [Recovery evidence](results/snowball-a0.recovery-evidence.json) separates observations from deductions from the executed source and exact traceback. The raw artifact remains unchanged at `s3://marin-us-east-02a/marin/users/romain/score-centering-readiness-01a146a3/sc-readiness-snowball-a0-01a146a3/readiness.json`.

[Native worker status](results/snowball-a0.native-status.json) and [the relevant driver events](results/snowball-a0.events.json) show that the final production task/admission/GRPO step admitted 32 groups and 128 trajectories (128567 response tokens), consumed tokens with maximum age 0, applied one update, skipped none and had finite positive gradient norm 0.53223. The observed sampled current/behavior retained mass was 0.995352/0.995688 and the mean absolute scalar SC loss was 0.001632. The PPO current/old log-ratio mean was 0, while sampled current/behavior absolute log ratio was 0.06311. These are loss/probability diagnostics, without a model-parameter gradient comparison.

Reaching the failing UID lookup proves that the earlier assertions passed: update 17 applied with no skip and finite positive gradient norm; trainer probabilities on identical tokens changed; the stale sampler replay beat its shifted-prefix control and still returned source tag 1; and the native update passed its update/skip/gradient/token-age checks. Source versions 0, 17 and 18 appear in the completed full weight-sync logs. This supports 18 applied updates in total. The numerical same-token drift and stale-replay summaries were not persisted, and are explicitly unknown. The 17th update's sampled numbers were also lost; the independently logged native 18 numbers are used instead. Native exact tensor shape, loss-token count and the full trainer metrics/timings dictionary were not recovered.

The [writer repair](https://github.com/marin-community/MarinSkyRL/commit/fad05d5aa) drops the unused UID field and saves the stale checks before native execution. Its report expression and file serialization passed a CPU regression with absent UID metadata. It changes no production code. No GPU allocation was repeated solely for that export repair.

## Snowball cost

Each measured window has 174080 response tokens, 168386 loss tokens and a 128×4549 sequence tensor. Each mode has three measured windows after its warmup. See [all windows](results/snowball-a0.windows.json), [the compact summary](results/snowball-a0.summary.json) and [the analysis](analyze.py).

| Mode | Cycle seconds | Response tokens/s | Old forward / update seconds | Learner peak allocated / sampled device GiB |
| --- | ---: | ---: | ---: | ---: |
| ordinary, telemetry off | 61.97 | 2809 | 0.96 / 6.45 | 54.66 / 79.29 |
| ordinary, telemetry on | 61.27 | 2841 | 0.98 / 6.19 | 54.66 / 78.27 |
| K128 capture/rescore, SC off | 74.86 | 2326 | 3.26 / 10.08 | 54.73 / 78.34 |
| K128 capture/rescore, SC on | 74.33 | 2342 | 3.17 / 9.96 | 55.70 / 79.27 |

SC's full adoption cycle cost against ordinary TIS with existing telemetry is +21.29%, with a descriptive 95% paired interval of +15.77% to +27.07%. Head capture/rescoring alone costs +22.17% (+19.28% to +25.12%). SC versus matched capture is −0.72% (−5.62% to +4.44%); that does not establish a speedup or a zero-cost correction. Existing trainer telemetry on/off is −1.13% (−3.35% to +1.14%), which does not establish the 1% target. New sampled diagnostics remain default 0; their one diagnostic/native update is a correctness check, not a steady-state telemetry benchmark.

Intervals use paired log ratios by repetition with Student-t critical value 4.30265 (n=3, df=2). They assume independent, approximately normal log differences. There is one allocation, rotating mode order, persistent caches and sequential learner updates; there is no estimate of between-allocation variation. Generated tokens differ across Snowball windows despite matched prompts, lengths, masks and shapes.

Learner peak reserved memory is 62.38/61.37/61.44/62.50 GiB in table order. Inference peaks are 57.57 allocated, 58.87 reserved and 64.40 sampled device GiB in every mode. Device peaks use 0.1-second NVML samples and may miss shorter spikes; allocator caches persist across modes. These are maxima per GPU across the measured windows, not sums or rank averages.

The initial weight sync completed 541.5 seconds after admission, including 9.16 seconds in the initial weight-sync call. Setup is measured once for the shared allocation, rather than separately for each mode. [The complete allocation ledger](results/allocation-ledger.json) accounts for all six jobs and 15 attempts, including both failed Qwen starts, the successful Qwen run, Snowball's full allocation, setup/idle/teardown and both unallocated withdrawals. Controller task runtime totals 25.186 H100-hours (Snowball 20.966); a conservative provider-admission-to-terminal-observation bound is 27.474 H100-hours. These are allocation estimates, not invoices. [Every job](results/jobs.json) and [attempt](results/attempts.json) is terminal; [the provider check](results/provider-stop.json) confirms owned pods are absent.

## Limitations

The Snowball windows share prompts, response lengths, masks and tensor shapes; generated token content can vary between windows, and every response hash is retained. Fixed lengths and serialized generation/training measure a controlled engineering workload. They do not establish quality, sample efficiency or production asynchronous throughput. Prefixes and probabilities retain actual source stamps; a step label alone is not evidence of weight freshness. Small omitted mass and scalar SC loss do not bound parameter-gradient error. Sampled loss/mass telemetry uses existing microbatch/rank averages, including zeros for empty microbatches; it is not a pooled token mean.
