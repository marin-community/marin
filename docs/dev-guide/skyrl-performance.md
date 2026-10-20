# Choose a Snowball or Hero performance path

This guide preserves the dated measurements in [MarinSkyRL #873](https://github.com/marin-community/MarinSkyRL/pull/873). Rates and qualification limits describe those source revisions. Use the current [Marin SkyRL artifact launcher](skyrl-migration.md) and validate its resolved recipe before running a historical layout.

Choose a BF16 layout for CoreWeave H100 or GB200/B200, measure the bottleneck, then change one setting. Evidence was checked on 30 September 2026. Match a historical run's workload and numerical settings before using its rate as a target.

| Your task | Write down | Starting path | First check |
| --- | --- | --- | --- |
| Interactive serving | Prompt/output lengths, burst concurrency, p95 first-token and response-time limits | [Standalone vLLM](skyrl-performance-vllm.md#standalone-serving) | Queue time and p95 latency |
| Bulk inference | Input/output lengths, concurrency, deadline, acceptable output behavior | [Measured vLLM layouts](skyrl-performance-vllm.md#measured-serving-examples) | Useful completed output tokens per GPU-hour, errors and tails |
| Short single-turn RL | Responses/group, groups/update, response cap, permitted policy age | [Learner layouts](skyrl-performance-megatron.md#choose-a-learner) → [Snowball H100 example](skyrl-performance-megatron.md#worked-task-short-snowball-math-rl-on-h100) | Learner wait and server activity together |
| Multi-turn coding RL | Turns, generated tokens, tool time/text, peak context, group completion tails | [Full-loop diagnosis](skyrl-performance-megatron.md#find-the-full-loop-limit) and [context limits](skyrl-performance-vllm.md#context-and-qualification) | Usable completed groups; tool time versus model time |

Record the model/export revision, GPU type/count and total context cap. For latency, set a time limit. For GPU efficiency, define useful work and a deadline. RL also requires numerical alignment and usable answers.

**Hero learning is qualified only for non-agentic GSM8K at 4,096 total tokens.** The tested settings are TP1, non-eager batch-invariant vLLM, captured expert-route replay, frozen query bias and fresh MuonH/AdamH/Adam state or a full RL restore. Longer-context serving probes measured fit and short outputs. Multi-turn Hero RL remains provisional. The 64-update learning curves used staleness at most one; the staleness-two trial measured short-run performance. [Learning audit](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5860256429).

## Estimate an unfamiliar task cheaply

Tokenize 16–32 representative prompts or trajectories, including the longest known cases, with the checkpoint's tokenizer and chat template. This gives a planning estimate, not a reliable p99. Reuse a recorded smoke for unknown output lengths before planning a new GPU sweep.

| Shape | Count | Why it matters |
| --- | --- | --- |
| Input | Prompt tokens per call, including system text and tool results | Large prefills compete with decode and delay the first token |
| Output | Tokens per response and trajectory; median, sample maximum, truncations | Long outputs retain KV and delay group completion |
| Context | Peak input + continuation per call | Must fit model, rotary, engine and trainer limits; tool text counts |
| Concurrency | Simultaneous calls; burst arrivals within one response duration | Sets client/scheduler caps; tool waits can leave servers idle |
| Groups | Responses/prompt and complete groups/update | Eight responses × 16 groups = 128 trajectories; the slowest response can hold a group |
| Tool work | Wall seconds, returned tokens, failures/timeouts | More serving GPUs cannot remove tool waits |

Estimate supply demand as `groups/second × responses/group × mean generated tokens`. Compare it with a server measurement at similar lengths and concurrency. Mark the result **estimated**: admission, pauses, scoring, rejected groups and tails reduce what reaches the learner. Check running requests before adding workers or servers.

## Keep the three rates separate

| Rate | Numerator | Denominator and use |
| --- | --- | --- |
| Learner-only | Unpadded prompt + response sequence tokens | Warm training seconds × learner GPUs; choose a learner layout |
| Supplied serving | Native generated decode tokens | Supplied-window seconds × serving GPUs; inspect capacity |
| Whole RL | Accepted, loss-masked response tokens | Complete ordinary cycle seconds × all learner and serving GPUs; choose the fleet |

Multiply tokens/GPU-second by 3,600 for tokens/GPU-hour. For short jobs, include setup, evaluation, saving and teardown in task-running GPU-hours; report reservation/billing time separately. Faster fixed-update runs can consume more tokens or yield worse answers. [Historical Snowball study](https://github.com/marin-community/marin/issues/8936#issuecomment-5581680479).

## Choose a starting layout

For short Snowball H100 math RL, evaluate the historical **32 learner + 8 serving GPU** [example](skyrl-performance-megatron.md#worked-task-short-snowball-math-rl-on-h100). If warm learner wait is negligible, keep eight serving GPUs and inspect training, scoring and publication. If wait grows with queues or preemptions, use the [vLLM diagnosis table](skyrl-performance-vllm.md#read-the-signals-together).

**The source example describes a historical launcher incompatibility.** Current Marin artifacts use the vendored standard runtime and `trainer.rollout_buffer`. The historical 256 × eight task still requires an explicitly authored, validated recipe and matching role allocation. [Launcher gap](skyrl-performance-megatron.md#worked-task-short-snowball-math-rl-on-h100).

For other tasks, use the closest measured row. Unmatched cases are **unknown**. Any proposed probe should name the decision, use a small fixed bank and include startup cost.
