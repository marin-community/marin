# Training–inference mismatch and score-centering cost on Snowball

On one four-node H100 allocation, score centering increased mean full-cycle
wall time by 8.61% over ordinary tokenwise truncated importance sampling (TIS).
Six matched repetitions gave a within-allocation 95% interval of 7.92–9.39%.
Collecting a probe every cycle had no resolved added latency: −0.24%
[−0.83, +0.29%]. Keep interval 10 as an adjustable starting point. These
measurements establish neither a general 1% ceiling nor a learning benefit.

Training–inference mismatch (TIM) is disagreement between sampler and trainer
probabilities. Score centering (SC) adds a detached expectation correction to
PPO's score coefficient. A hypothetical probe evaluates that correction without
applying it. Its K128 head contains the 128 most probable token IDs and their
full-vocabulary probabilities. SC stays off by default; the
[usage guide](../references/score-centering.md) explains the metrics and settings.

## Matched cost

The model was `models/snowball-67b-a2b-sft-s2-thinking@2026.08.30:c6168770`.
Three learner nodes used eight H100s each, pipeline parallelism 3, data
parallelism 8 and expert parallelism 8, with 26 layers split 9/9/8. One inference
node used eight data/expert-parallel workers. Tensor, context and sequence
parallelism were one. SC and head capture require local vLLM and unpacked
Megatron with these three parallel sizes one; data/pipeline parallelism is
supported. Samples were unpacked and BF16. Temperature 1, top-p 1,
top-k -1 and min-p 0 left sampling probabilities intact.

All arms used the original supervised fine-tuning (SFT) weights and AdamW with
learning rate zero. Gradients and optimizer moments still advanced. Signed
synthetic advantages supplied nonzero gradients; these controls test runtime.
Each arm had one excluded warmup and six repetitions, with rotating arm order.
Intervals resample the six paired repetitions 10,000 times. They exclude
between-allocation variability.

| Window | Arm | Mean [min, max], seconds | Change from TIS [95% interval] |
| --- | --- | ---: | ---: |
| Trainer only | TIS; probes off | 9.995 [9.754, 10.194] | baseline |
| Trainer only | TIS; compact probe scoring | 10.313 [10.046, 10.615] | +3.19% [+0.98, +5.87] |
| Trainer only | TIS + SC | 13.030 [12.877, 13.308] | +30.37% [+27.35, +33.66] |
| Full cycle | TIS; probes off | 62.743 [62.211, 63.478] | baseline |
| Full cycle | TIS; probe every cycle | 62.591 [62.335, 62.776] | −0.24% [−0.83, +0.29] |
| Full cycle | K128 capture/scoring; SC off | 67.821 [67.605, 68.281] | +8.09% [+7.50, +8.72] |
| Full cycle | TIS + SC | 68.148 [67.517, 68.523] | +8.61% [+7.92, +9.39] |

Full capture retains K128 heads across all 128 responses. Compact probes retain
and score at most 32 loss-eligible positions per response in one four-response
group; collection still captures every generated position in that group.

Trainer-only arms reused identical tokens and loss masks: 128 responses, 32
each of lengths 64, 256, 1,024 and 4,096. Newline tokens extended the longest
prefixes to 4,096 attended tokens, giving a 128 × 8,192 input tensor. The probe
scored 32 positions in four long responses. Timers included forward and
optimizer compute; optimizer offload was outside this window.

Full cycles used 32 retained prompt groups with four responses each, natural
prompt lengths, fixed seeds and forced response lengths 64/256/1,024/4,096.
EOS stopping was disabled. Every 31st token and one response were masked
consistently. All cycles had 174,080 response tokens and 164,484 loss tokens.
Their tensor hashes differed: only trainer-only arms establish token-for-token
identity. Full cycles hold the input workload fixed and include collection,
grading, rollout storage, training/offload, weight publication and normal metric
emission. Context creation was outside the timer. These windows are not additive.

## Probe cadence and coverage

Keep `interval: 10`, K128 and 32 positions per response. The controlled sequence
consumed probes at steps 29, 39 and 49, with spacing 10 and 10: 288 positions
from nine 4,096-token responses, all fresh. A later mask removed one response
per selected group, leaving 96 positions per observation, or 0.0584% of that
batch's loss tokens. Step 29 was in warmup; the other two observations fell in the 20 post-warmup
cycles. This
sequence kept its buffer context; the repeated calibration recreated contexts.
Their absolute times do not form a paired comparison.

The current-main async check started from the original SFT weights and used
graded-response advantages, learning rate 1e-6, maximum staleness 4 and 64
in-flight prompt groups. It applied 25 ordinary TIS updates and four SC updates.
Every update had nonzero advantages, finite nonzero gradients, one applied
optimizer step and zero skipped steps. SC raw gradient norms were 0.2666–0.3914.

Probes arrived at steps 1, 15 and 25, with spacing 14 and 10. They covered 384
positions in 12 responses: 128 fresh, 256 stale by four versions, none unknown.
Coverage was 0.1065–0.1525% of each observed batch's loss tokens. Across all
3,712 responses, lengths ranged from 1 to 4,096, with mean 801.7 and median 249.
Although 504 responses reached 2,048 tokens, none of the 12 probed responses
did. Natural long-probe coverage remains a sampling gap; controlled checks
supply long-response execution evidence. Changing async weights and trajectories
do not isolate overhead or learning benefit.

Fresh means the generating-policy stamp matches the pre-update trainer snapshot.
Stale disagreement also includes weight drift. Unknown stamps remain unknown.
Requests can be rejected or delayed, so use consumed observations and their
spacing. One prompt group is not an unbiased sample of tasks or lengths. The
export is bounded, but vLLM collects heads at every generated position in the
selected group; other requests in its decode batch can also compute top K.
Increase the interval to reduce work, then check observation and length coverage.

## Remaining cost and implemented reduction

Capture/scoring accounts for most of the observed SC gap. It includes collection,
representation, transfer and selected-head scoring. Applying SC added a mean
0.327 seconds over capture-only; that point difference does not isolate a GPU
kernel. Logical SC input/returned head tensors were 512/256 MiB, versus about
4.05/2 MiB for a compact probe. Padding is included; these are not network bytes.

An isolated numeric Ray round trip for four real 4,096-token responses had
median 9.06 ms with chosen scores and 49.84 ms with heads, with pickle payloads
of 229,108 and 17,006,768 bytes. It excludes generation, rollout storage and
learner transfer. Excluded warmup profiles found at most 0.459 seconds of head
selection per inference frontend and 0.036 seconds in the SC driver collator.
Concurrent host profiles do not identify the remaining GPU kernel cost.

Dense alignment now validates arrays in bulk and reuses contiguous heads.
Masked/tool positions retain padding and exact-token checks. A shared-host CPU
microbenchmark used prebuilt int32/FP32 K128 arrays, two warmups and 20 alternating
samples of five calls. At 4,096 tokens, dense median time fell from 4.903 to
0.536 ms; masked alignment fell from 5.010 to 3.355 ms. The dense operation was
89% shorter; no full-cycle gain was measured for this change. Earlier numeric
transport already removed public chat objects from the internal head path.

The [authors' implementation](https://github.com/martin-marek/score-centering/tree/7c56e9ee2972aa57f446cf564de1a1658d14b321)
uses GPU arrays and a chunked buffer in JAX. The
[paper's](https://arxiv.org/html/2609.20807v1) roughly 1% result does not bound
our pinned vLLM, host-array, Ray and rollout-storage path.

## Added logging cost

A one-thread CPU comparison used prebuilt inputs, three warmups and alternating
calls. Normal summaries on 128 responses and 168,386 loss tokens took
22.775 → 31.579 ms (+8.804 ms). Four-response, 32-position probe summaries took
1.697 → 1.905 ms (+0.208 ms). Existing pooled/age statistics matched within
1e-10; probe scalars matched exactly. Position windows now count actual response
positions across masked holes. The [logging data](data/score-centering-tim-20261022/logging-cpu.json)
records serialization and payload growth.

Across 24 measured full cycles, normal stdout/tracker emission averaged
10.15 ms, range 9.44–13.40 ms. This includes existing and new fields; it does
not isolate incremental logging cost. Full-cycle timers include this emission.
No GPU synchronization was added for timing. Host stages overlap and include
ordinary work, so their durations cannot be summed to recover added cycle time.

## Correctness

Megatron's response scorer was compared with an independent full-vocabulary
FP32 log-softmax at 381 causal positions, including first/middle/last positions,
masks, 4,096-token responses and the 8,192-token input. Maximum log-probability
error was 9.54e-7 for chosen tokens and 1.91e-6 for head tokens. Twenty-four same-engine prefix replays had mean absolute decode/prefill
disagreement 0.2291 log-probability units; a deliberate one-token shift gave
5.7193. Matching policy stamps and the larger shift control support numeric
engine disagreement on these samples. They do not exclude every alignment error.

The behavior tail is the sampler probability mass outside the retained K128
head. SC floors tail masses at 1e-6; `tail_floor_fraction` counts sampled
positions where either trainer or sampler tail reaches that floor.
An exported 96-position probe at current=old matched an independent analytic
scalar equation using worker FP32 probability/tail terms. Maximum per-position
error was 3.06e-7; exported mean and absolute-mean errors were at most 2.85e-8.
A double-precision mass replay differed by 2.58e-5 in those summaries. With the
same Torch 2.13 build and inputs, an AVX512 worker and AVX2 host differed by one
5.96e-8 rounding step in a behavior tail near 1e-6. SC mathematics and its 1e-6
floor are unchanged. Small tails remain sensitive just above the floor, even
when `tail_floor_fraction` is zero.

CPU equation, gradient, padding, transport and data/pipeline-parallel
normalization references pass. Qwen3 0.6B startup checks preceded Snowball.
Full parameter-gradient comparisons and Snowball learning benefit remain untested.

## Reproduction and provenance

The [matched-cost source](https://github.com/yonromai/marin/tree/29b2b26d84bfc439daa0bfed48697d23f40d1751)
and [current-main acceptance source](https://github.com/yonromai/marin/tree/eccf60b65feb3e50b23e3989090c30f3cc5e86a6)
include the measurement harness; the latter includes main `eac6c5b695c3`.
Fourteen critical runtime AST/config checks match. The upstream retention
default changed; both launches retain one checkpoint and disable checkpoint/export
intervals. Current Iris/Rigging credential delivery was checked by the fresh
async run; the matched timings do not isolate that integration change.

Both measurement checkouts overlay the separate grader repair
[431b16264571](https://github.com/marin-community/marin/commit/431b16264571c546f774586de373cf4577f5dcdd)
from [#10394](https://github.com/marin-community/marin/pull/10394). Installed-worker
checks verified it, editable Marin libraries and vLLM `39e62869693c` on all four
nodes. Deployment must include that repair; editable dependencies need no package
publication. SkyRL uses its independent 503-package lock and standard Iris
workspace bundle. The CPU extra uses Torch 2.11; GPU workers use Torch 2.13/cu132.

For matched controls, check out `29b2b26d84bf`; for natural coverage, use
`eccf60b65feb`. Both are linked above. Use a fresh calendar version:

```bash
MARIN_PREFIX=s3://marin-us-east-02a/marin MARIN_TEMP_PREFIX=s3://marin-us-east-08a \
  marin-env uv run --frozen python -m experiments.post_training.score_centering_tim \
  --stage controls --version YYYY.MM.DD.N --max-concurrent 1 --run
```

`--stage coverage` runs the natural-reward check. Explicit prefixes retain the
matched storage locations. The [manifest](data/score-centering-tim-20261022/manifest.json)
records tokenizer/input hashes and the 192 retained source indices' provenance.
The [numeric controls](data/score-centering-tim-20261022/controls.json.gz),
[summary](data/score-centering-tim-20261022/controls-summary.json) and
[launch](data/score-centering-tim-20261022/controls-launch.yaml), plus
[natural coverage](data/score-centering-tim-20261022/coverage.json.gz),
[summary](data/score-centering-tim-20261022/coverage-summary.json) and
[launch](data/score-centering-tim-20261022/coverage-launch.yaml), preserve the results.
[Worker identity](data/score-centering-tim-20261022/worker-provenance.json),
[source comparison](data/score-centering-tim-20261022/source-reuse.json) and
[scalar replay](data/score-centering-tim-20261022/precision.json) record their scopes.
Files ending in `.json.gz` use gzip.

## Earlier results, attempts and resources

[Earlier controls](https://github.com/yonromai/marin/tree/3b9f0183badb2c20d4c9bd4468489a442f9222e2/experiments/post_training/score_centering_integration)
reported 9.27% SC overhead with only two repetitions. Their async trajectories
had different lengths and did not isolate overhead. The Countdown arithmetic-task
learning campaign had wide paired intervals crossing zero and is stopped. Qwen 8,192-token
training-window and Snowball learning comparisons remain incomplete;
[SkyRL #996](https://github.com/marin-community/MarinSkyRL/pull/996) remains research.

The first Snowball allocation split layers 10/8/8 and ran out of GPU memory in
SC warmup after one ordinary update initialized optimizer state. Balancing to
9/9/8 addressed that stage imbalance. A Qwen probe configuration failed before
GPU allocation; its corrected retry completed.

An exploratory chosen-token-only vLLM formatting fast path saved 54 updates,
including all six cost
repetitions, before operator cancellation. A 622.808-second probe-tagged cycle
included 604.107 seconds of rollout waiting; selected-group collection took
43.723 seconds. The cause is unresolved. The run recovered before cancellation,
which used an outdated stall assessment. A 5.242-second CPU replay overlapped
its final SC timing sample; its periodic sequence ended after five of 22 cycles.
That formatting fast path was reverted. Its [data](data/score-centering-tim-20261022/exploratory-controls.json.gz)
and [timing limits](data/score-centering-tim-20261022/exploratory-limits.json) are
excluded from the final overhead estimate; the separate scalar replay remains useful.

The first current-main coverage attempt failed during parallel S3 model loading
before any update. Bounded native credential and 64 MiB reads passed; the cause
remains unresolved. Its successful retry restored East storage paths, allowed
60-second S3 stalls and disabled model-load retries. These changes do not isolate
the startup failure's cause.

All eight root runs and their descendants are terminal. [Accounting](data/score-centering-tim-20261022/resource-accounting.json)
totals 114.91 requested H100-hours across seven GPU allocations, including
startup, failures and cancellation. This is requested device count times Iris
attempt wall time; it excludes queue wait and is not provider billing.
After consumers stopped, 7,855,304,576 bytes were removed. Fresh
[cleanup checks](data/score-centering-tim-20261022/cleanup.json) found all eight exact
TTL roots empty and zero active owned GPU tasks on both clusters. Compact
configurations, metrics, source identities and reports remain. Original SFT files
were outside every deleted prefix; retained input hashes still match.
