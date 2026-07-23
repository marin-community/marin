# Snowball 67B rank sentinel: vLLM rank and run variation on one long prompt

Date: 2026-07-21 · Cluster: `cw-us-east-02a` (8×H100) · Marin tree `371e00e64` · Related: [#7354](https://github.com/marin-community/marin/issues/7354), [#7183](https://github.com/marin-community/marin/issues/7183), [#7314](https://github.com/marin-community/marin/issues/7314)

Status: exploratory, and partly superseded. Every result below comes from one prompt and one or two launches per configuration. Nothing here is replicated across prompts, and no result should be treated as a stable backend property.

**Superseded:** this report concludes that no root cause was identified. A follow-up investigation found one — see [Root cause of vLLM per-rank logprob variation](snowball-vllm-rank-variance-root-cause.md). Two statements below are also corrected there: the run-to-run instability is discrete rather than continuous (the same per-rank values recur bit-exactly across jobs and days), and the chunked-versus-unchunked comparison remains confounded with launch state, which is now known to be large enough to account for the difference on its own.

## TL;DR

- `test_snowball_export_matches_representative_goldens[vllm-gpu]` fails on the rank sentinel. The failure is pre-existing and reproduces on an untouched export, so the config rename in [#7458](https://github.com/marin-community/marin/pull/7458) did not cause it.
- The failure is confined to the rank sentinel. [#7354](https://github.com/marin-community/marin/issues/7354) reports all 64 representative cases passing for both backends in a paired run and in five additional cold vLLM runs. The sentinel takes the same prompt and replays it once per data-parallel rank, then scores each replay separately.
- On that prompt, vLLM's per-rank probability for the golden's top token spans 0.394602 to 0.564495 within a single launch. The golden is 0.557848 and Levanter-GPU is 0.502516.
- Across launches the same rank and length can move much further. At 8,192 tokens, rank 0 unchunked was 0.621766 in one launch and 0.343003 in another: 0.278763 apart under identical static configuration.
- Levanter-GPU was bit-identical across seven batch rows, which occupy seven of eight data shards on the mesh. It shows no comparable variation, though this is one run.
- Changing the gate's reference from the frozen golden to Levanter-GPU would not fix it. The worst vLLM rank sits 0.107914 from Levanter-GPU, still above the 0.075 bound.
- No root cause was identified for the rank and run variation. Chunked prefill could not be isolated, and no hardware cause was measured.

Terms: the gate serves the export with vLLM data parallelism (DP) of 8 plus expert parallelism (EP), so eight worker processes each hold a slice of the experts. A request is pinned to one worker with the `X-data-parallel-rank` header. The rank sentinel is the 15,025-token case `knowledge-longbench-02` replayed once per rank after the normal 64-case wave.

## Setup

Model: Snowball June 67B-A2B (MoE, ~2B active), checkpoint step 42150, bf16 HF export at
`s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/`.

Attention recipe: `sliding_window=2048`; long (full-attention) layers every 4th plus the last, giving layers {3,7,11,15,19,23,25} of 26; NoPE on long layers; half-RoPE on short layers.

| | vLLM arm | Levanter arm |
|---|---|---|
| build | marin fork `afb2671946` | in-repo `lib/levanter` |
| parallelism | `--data-parallel-size 8 --enable-expert-parallel`, `--max-num-seqs 1` | one JAX process, mesh `data=8, model=1`, expert axis size 1 |
| kernels | `--attention-backend FLASH_ATTN`, FusedMoE | `moe_implementation=sonic`, `attention_implementation=gpu_fa4_cute` |
| determinism | `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_FLASHINFER_SAMPLER=0` | `XLA_FLAGS=--xla_gpu_deterministic_ops=true` |
| prompt shape | exact token ids per request | batch of 8 rows right-padded to 16384 |

The two arms differ in more than framework: parallelism strategy, MoE dispatch, attention kernel, and prompt padding all differ. Any comparison between them is a comparison of two whole stacks.

Reference: `tests/cluster/vllm/resources/june_tpu_67b_a2b_step_42150_representative_eval_golden.json`, 64 cases, top-25 logprobs per case. The file contains a single `cases` key: no model URI, commit, hardware, or generation command.

Gate contract (`tests/cluster/vllm/backend_parity.py:30-34`), three assertions per observation: the greedy token must appear in the golden top 25; the golden probability gap to the greedy token must be at most twice the observed maximum error; and `max_probability_error`, the max over the golden's 25 tokens of `|p_backend − p_golden|`, must be at most `MAX_PROBABILITY_ERROR = 0.075`. The sentinel failure under discussion is the third assertion. Each observation is scored separately; the gate never averages ranks.

## Experiment A: structure of the sentinel failure

Job `snowball-experiment-a-86c579b6`, one server launch, gate-identical flags.

| arm | result |
|---|---|
| A1: rank 0, 8× sequential | first request p(423)=0.461436; the following seven bit-identical at 0.464546 |
| A2: each rank once, sequential | spread 0.077538 |
| A3: all 8 ranks concurrent, 2 rounds | round 0 and round 1 bit-identical on all 8 ranks; spread 0.076250 |

Per-rank p(423) in A2:

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| p | 0.464546 | 0.530301 | 0.492024 | 0.517343 | 0.484222 | 0.536996 | 0.542084 | 0.538992 |
| err vs golden | 0.093303 | 0.027547 | 0.065824 | 0.040505 | 0.073626 | 0.020852 | 0.018087 | 0.028779 |

All 32 observations fell below the golden, but they are repeated dependent measurements of one prompt in one process, not 32 independent cases. Within this process the output repeats exactly for a given rank and concurrency pattern, so rank 0 exceeded 0.075 on every request. The first request differs from the following seven; this run does not isolate why.

## Experiment B: length ladder and head-to-head

Jobs `snowball-experiment-b-vllm-bd7f6354` (both settings), `snowball-experiment-b-vllm-de845a00` (unchunked only, an earlier partial run), `snowball-experiment-b-levanter-9f283c4b`. Three 8×H100 jobs, roughly 20 node-minutes, about 160 H100-minutes.

The sentinel was left-truncated to 128, 512, 1024, 2040, 2048, 2056, 3072, 4096, 8192 and 15025 tokens. Truncation keeps the prediction position at the end of the same suffix; it does not keep the model's prediction fixed, and the greedy token does change across rungs. Each rung also removes prefix content and shifts retained positions, so content and length vary together. vLLM ran every length on every rank sequentially, under `max_num_batched_tokens=512` (gate-identical, 30 chunks at full length) and `max_num_batched_tokens=16384` (single-chunk prefill). Each setting needs its own vLLM server, so setting, process, and request history are confounded.

The gate sends the sentinel to all eight ranks concurrently; Experiment B sends requests sequentially. These are different request patterns and Experiment A shows the pattern changes values.

### Full-length results, all runs

At 15,025 tokens, on token 423. Source job named for every row:

| source | setting | mean of 8 ranks | min | max | spread | mean − Levanter |
|---|---|---:|---:|---:|---:|---:|
| golden | — | 0.557848 | — | — | — | — |
| `levanter-9f283c4b` | Levanter GPU | 0.502516 | — | — | — | — |
| `vllm-bd7f6354` | chunked | 0.502833 | 0.394602 | 0.564495 | 0.169893 | +0.000316 |
| `vllm-bd7f6354` | unchunked | 0.523631 | 0.487727 | 0.556032 | 0.068305 | +0.021115 |
| `vllm-de845a00` | unchunked | 0.511117 | 0.472349 | 0.542084 | 0.069735 | +0.008600 |

The eight chunked rank values are 0.494724, 0.515144, 0.473948, 0.531029, 0.394602, 0.514611, 0.534107, 0.564495.

The near-coincidence between the chunked mean (0.502833) and Levanter-GPU (0.502516) is not backend agreement. The gate scores each rank, and the largest rank-to-Levanter difference in that launch is 0.107914. The two unchunked replicates put the mean 0.0086 and 0.0211 from Levanter, so the mean itself moves by more than the chunked coincidence across a relaunch.

### Run-to-run instability

The two unchunked launches disagree far beyond the full-length picture. At 8,192 tokens, rank 0: 0.621766 in `bd7f6354` against 0.343003 in `de845a00`, a difference of 0.278763 under identical static configuration. Any single-launch statement about a particular length or rank should be read against that.

Rank spread of p(423) across the 8 ranks, by length:

| tokens | 128 | 512 | 1024 | 2040 | 2048 | 2056 | 3072 | 4096 | 8192 | 15025 |
|---|---|---|---|---|---|---|---|---|---|---|
| `bd7f6354` chunked | 0.068 | 0.029 | 0.073 | 0.088 | 0.104 | 0.103 | 0.121 | 0.096 | 0.174 | 0.170 |
| `bd7f6354` unchunked | 0.043 | 0.055 | 0.104 | 0.146 | 0.084 | 0.117 | 0.181 | 0.059 | 0.283 | 0.068 |
| `de845a00` unchunked | 0.049 | 0.043 | 0.106 | 0.126 | 0.050 | 0.075 | 0.188 | 0.125 | 0.190 | 0.070 |

Spread is present at every tested length, including 128 tokens. It is larger at the long end of this ladder, but with one prompt and unreplicated rungs this does not establish that spread grows with context length.

Where the golden and Levanter-GPU differ on the top tokens:

| token | golden | Levanter GPU | delta |
|---|---|---|---|
| 423 | 0.557848 | 0.502516 | −0.055332 |
| 426 | 0.211735 | 0.233692 | +0.021957 |
| 578 | 0.142152 | 0.175027 | +0.032875 |

Tokens 426 and 578 recover 99.095% of token 423's loss, and the net delta over all 25 golden tokens is +0.000347. The top three are flatter on Levanter-GPU with the same greedy winner. This is a description of the difference, not a verdict on which is correct.

Levanter control: seven rows carrying identical 15,025-token input produced identical persisted maps (the golden 25 tokens plus that row's top 200, not the full 128,256-token vector). With mesh `data=8`, those rows occupy seven of eight data shards, so this is evidence of no row/shard-dependent difference in this run rather than a structural guarantee.

## Checks performed

Selected architecture settings match between Levanter and the pinned vLLM fork at source level:

| property | Levanter | fork | match |
|---|---|---|---|
| long-layer schedule | `((idx % 4) == 3) \| (idx == n-1)` | `(i+1) % 4 == 0 or i == n-1` | yes |
| NoPE on long layers | `disable_rope=True` | `use_rope=not (is_long and disable_long_rope)` | yes |
| half-RoPE on short layers | rotary on first `head_dim//2` | `partial_rotary_factor: 0.5` | yes |
| sliding window | `k >= q - (W-1)` | passes `per_layer_sliding_window=W` | yes |
| router | top-(K+1), drop last, sigmoid on unbiased logits, renorm to 2.5 with eps 1e-9 | identical | yes |
| shared expert | always on, added to routed output | always on | yes |

This is a check of selected high-level settings, not a semantic equivalence proof. It says nothing about kernel-level behavior, all-to-all dispatch, reduction order, casts, loader mappings, or how the pinned FlashAttention backend interprets window boundaries.

Expert-capacity token dropping does not occur in this configuration. Capacity clipping lives only in the branch taken when the expert axis is larger than 1 (`lib/levanter/src/levanter/grug/grug_moe.py:181-252`), and the gate runs Levanter with expert axis size 1.

Chunked prefill was not isolated. At 128 and 512 tokens both settings prefill in a single chunk, yet only 2/8 and 0/8 ranks matched bit-for-bit, so chunking is not necessary for the observed differences. That is the only supported statement. Changing `max_num_batched_tokens` requires a fresh server, so every comparison of the two settings is also a comparison of two launches, and launch differences are large. Estimating a chunking effect needs repeated matched fresh-process pairs on one node with identical request order and alternating setting order.

## What this does not show

No hardware cause was measured. The golden's provenance is its filename. A stale golden, a different export, a different graph, or a different runtime policy is observationally equivalent to a TPU/GPU difference in this data. [#7183](https://github.com/marin-community/marin/issues/7183) found two *TPU* execution paths on June-67B that also differed, which further weakens any hardware attribution.

The vLLM arm and the Levanter arm differ in parallelism, MoE dispatch, attention kernel, and padding, so this is not a hardware-controlled comparison of one implementation.

## Errors in this analysis

Two mistakes were made and corrected during review; both are recorded because they affected published numbers.

The first pass sign-tested across all 25 golden tokens and concluded the offset was absent in Levanter-GPU. The corrected reason for rejecting that test is not "mass conservation forces balanced counts" — conservation constrains the sum of signed deltas over the full vocabulary, not the count of positive and negative ones. The count is simply not the gate statistic, covers 25 of 128,256 tokens, and its terms are dependent under normalization.

The second pass paired the chunked setting from one job with the unchunked setting from an earlier job, because the analysis ran before the later job finished writing its unchunked half. That understated the chunking shift and produced an exactly-equal average rank spread across settings that was an artifact of mixing runs. Caught by Gemini in review; the omitted replicate and the mean-versus-per-rank error were caught by Codex.

## Where this leaves the gate

Three separable questions are worth keeping apart, because they have different answers:

1. Representative correctness. Already passing: 64/64 cases for both backends in the paired run and five additional cold vLLM runs ([#7354](https://github.com/marin-community/marin/issues/7354)).
2. Same-prompt rank consistency. This is what fails. Within one launch the sentinel spans 0.169893 across ranks; across launches a single rank/length moved 0.278763.
3. Golden provenance. Unresolved and unmeasured.

Switching the sentinel's reference to Levanter-GPU does not fix question 2: the worst rank residual against Levanter-GPU is 0.107914, above the 0.075 bound. If the intent is rank consistency, comparing ranks to each other (range or pairwise) tests it directly and does not require a golden at all. Any candidate contract should be evaluated against every rank and all 64 cases before a threshold is chosen.

Raising the tolerance to pass the observed worst case would need roughly 0.16. This work did not measure what regression size such a bound would miss, so it cannot say how much detection power that costs.

## Limitations

- One prompt, one checkpoint, one token for the headline numbers. Nothing is replicated across the other 63 representative cases.
- One Levanter load, one chunked vLLM launch, two unchunked launches. That does not estimate backend means or variances.
- Setting, process, job allocation, request history, and setting order are confounded in every vLLM comparison.
- Experiment B is sequential; the gate's sentinel wave is concurrent.
- Rank observations are fixed workers in one server, not independent samples, so a mean over them has no established inferential meaning.
- The ladder varies content and position along with length.
- The Levanter row control covers the persisted token subset in one run, not full vectors or cross-launch behavior.
- The long/short attention schedule is hardcoded independently in Levanter and the fork and is not carried in `config.json`; `grugmoe_attention_mode` is the opaque string `"production"`. They agree today, and a future divergence would be invisible on any prompt under 2048 tokens because the sliding-window and full-attention masks coincide inside the window.
- Prior evidence in [#7354](https://github.com/marin-community/marin/issues/7354) covers five cold runs and should be treated as the larger evidence base for run variation.

## Reproduction

Reproduction: the analysis code and reducers are committed under `tests/cluster/vllm/` (`_experiment_ab_reduce_fixture.py` and the Experiment C tooling). The raw result blocks live in finelog and are retrievable with `iris job logs /romain/<job>` for the jobs named in each section. The reduced fixtures those scripts produce are measurement data, so they are hosted (or regenerated from the raw logs on demand) rather than committed; the small summarized tables above are inlined here.

One implementation note that does transfer: remote entrypoints take plain token ids and are defined in `__main__` so cloudpickle ships them by value, avoiding a `tests.*` import in the worker. `draccus` 0.11.6 ships a top-level regular package named `tests`, which shadows the repository's `tests/` namespace package regardless of `sys.path` order; pytest escapes this with `--import-mode=importlib`, a plain callable entrypoint does not. Operational notes on Iris log truncation are in `scratch/LOG-sharp-edges.md`.
