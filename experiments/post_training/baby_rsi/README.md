# Baby RSI

Baby RSI is the recursive self-improvement loop of
[Post-training: Boxes/Systems](https://docs.google.com/document/d/1qRs0UIuYIzMOdIchHy6OItT2VUKJVQFK4nJ0L2zinZ4)
with real boxes. The design document writes the loop as:

```python
task_prompts = seed_tasks(...)
while True:
    tasks = generate_tasks(task_prompts, glm_inference, task_filter)
    model = train(model, tasks)
    eval_scores, rollouts = eval_policy(model)
    task_prompts = analyze_failed_runs(rollouts)
```

`loop.py` writes the same loop as streams. Each box is a function from a stream to a stream, and the data drives
the loop, so it has no `while`:

```python
def rsi_loop(seed_tasks, boxes, config):
    rollouts = Broadcast[TaskRollouts]()
    failures = failed_runs(rollouts.listen(), config.max_pass_rate)
    ideas = boxes.analyze_failed_runs(batched(failures, config.failures_per_analysis))
    new_tasks = boxes.generate_tasks(ideas)
    tasks = mix(Source(seed_tasks, 1 - config.synthesized_share), Source(new_tasks, config.synthesized_share))
    return rollouts.publish(boxes.train(tasks))
```

The tasks flow into training, training yields graded rollouts, the failed rollouts flow into the failed-run
analysis, its capability ideas flow into task synthesis, and the new tasks flow back into training beside the seed
tasks. `batched` makes a box wait for enough data, and `mix` never lets a slow box (synthesis) stop a fast one (the
seed tasks). The control arm is `boxes.train(seed_tasks)`: the same training on the seed tasks only.

## Boxes

| Box | Module | What it does |
|---|---|---|
| Seed | `seed.py` | Samples the RL task pool of Marin-Snowball-67B-A2B-v0.1 (the Smartie run's `train.parquet`, 69,487 tasks) in an order whose every prefix keeps the pool's strata weights: structural closure 82.6%, competitive coding 10.9%, Reasoning Gym 2.8%, STEM multiple choice 2.7%, calendar 1.1%. |
| Train | `skyrl_box.py`, `skyrl_launch.py` | A SkyRL job (full-parameter GRPO with Megatron, vLLM engines on their own node, staleness 1). The box writes the tasks into the job's feed and yields the graded groups that the job writes back. The job serves the live policy on an OpenAI-compatible endpoint, which the Taskforge gate uses. |
| Analyze | `feedback.py` | Keeps one failed attempt of each task that the policy passes at most 25% of the time, and runs `analyze_failed_runs` (#10265) on each batch. It yields each new capability ID once. |
| Generate | `feedback.py` | One agentic Taskforge session (`taskforge.agentic`) for each idea, with bubblewrap solver machines. The live policy is the solver of the gate, so an accepted task has a policy pass rate from 0.1 to 0.8. The build program of each accepted task makes more training instances and held-out probe instances. `submit` refuses a program whose 8 test instances share one reference answer more than 25% of the time: in the first run, a family with answers 3, 4 or 5 taught the policy to guess. |
| Evaluate | `evaluate.py`, `run.py evaluate` | The base policy (twice) and the HF exports of the trained policies on held-out pool tasks (the Smartie validation file) and on the probe instances, which no arm trained on. |

The feed is an object-store prefix that the loop and the SkyRL job share: the loop writes task chunks, and the job
writes one graded group (or the reason that it skipped a task) for each task. The box keeps a few training batches
of tasks ahead, so the shares of `mix` apply to each batch. In the job, a structural-closure task continues its
assistant prefix, a task with a shell gets a bubblewrap machine, and a script grader runs in bubblewrap.
`policy.py` serves a policy for the gate and the evaluation: `VllmPolicy` is a `rolloutengine` model that renders
the chat template itself and keeps the exact sampled tokens.

Two conversions run the pool tasks on any policy and without Daytona. A structural-closure row holds prompt token
IDs of the Snowball tokenizer; `structure_closure_task` turns it into a conversation that ends with the unfinished
assistant text. The other strata grade with a script in a Docker image; the seed tasks grade with the host's Python
environment (which has `reasoning_gym`, `verifyit`, numpy and pandas) in a bubblewrap sandbox.

## Run

An arm is a SkyRL job and a loop job with the same feed. Start the SkyRL job with its launch module, then the loop
in a privileged CPU task (the Taskforge agent and its solvers run in bubblewrap):

```bash
uv run python -m experiments.post_training.baby_rsi.skyrl_launch --version <version> --model qwen3-4b --arm rsi \
  --feed-root s3://<bucket>/<prefix>/rsi/feed --steps 60 --train-batch 32 --lr 2e-6 --export-every 30 --run
uv run iris --cluster=marin job run --no-wait --target-cluster cw-us-east-02a --cpu 32 --memory 128GB \
  --sync-package marin-taskforge --container-profile container_profile_privileged --job-name <name> \
  -e HF_TOKEN "$HF_TOKEN" -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" -e PARALLEL_API_KEY "$PARALLEL_API_KEY" \
  -- .venv/bin/python -m experiments.post_training.baby_rsi.run skyrl-rsi --model Qwen/Qwen3-4B-Instruct-2507 \
  --feed-root s3://<bucket>/<prefix>/rsi/feed --output-dir s3://<bucket>/<prefix>/rsi --steps 60 --tasks-per-step 32
```

`skyrl-control` takes the same options without the feedback options. The SkyRL job resumes from its checkpoints and
its feed position; a restarted loop continues the feed, does not write its pool tasks again, gives back the
synthesized instances that are not in the feed, and does not synthesize the same idea again (`resume.py`).

`evaluate` runs in an Iris GPU task through `job.sh`, which installs vLLM in its own environment and selects the
task's own GPUs. It takes `--checkpoint NAME=URL` for each HF export and `--probes` for each RSI output directory,
and `report.py` compares the policies with paired bootstrap intervals over tasks.

## Results of the LoRA trainer (2026-10-21)

These runs used an in-process LoRA trainer beside one vLLM server, before SkyRL became the train box.
Qwen3-4B-Instruct-2507, LoRA rank 32, 120 GRPO steps of 32 tasks × 8 attempts, learning rate 4e-5, one H100 for
each arm, two seeds. The RSI arm mixes the pool's tasks 50/50 with the synthesized tasks once they come; the
control trains on the pool's tasks only. The evaluation uses 400 held-out tasks of the pool's validation file and
the held-out probes of each accepted task (8 instances that its build program makes from new task IDs, minus any
instance with the content of a training instance), 8 attempts each. Deltas are paired over tasks, with 95%
bootstrap intervals; both seeds pooled:

| Task set | Tasks | Base | Control | RSI | RSI − control | RSI − base |
|---|---|---|---|---|---|---|
| Held-out probes | 358 | 0.390 | 0.278 | 0.455 | +0.177 [+0.134, +0.218] | +0.065 [+0.020, +0.110] |
| Held-out pool tasks | 800 | 0.232 | 0.393 | 0.375 | −0.018 [−0.036, +0.000] | +0.143 [+0.118, +0.168] |

Of the 104 probes that the base never passes in 8 attempts, the RSI arm passes 49% at least once, the control
21%, and a second evaluation of the base 24% (chance). Seed 0 had its best probe score at step 60 (RSI − control
+0.162) and lost some of it by step 120, when both arms gave shorter answers; seed 1 gained after step 60 (RSI −
control +0.251 at step 120). An independent audit by DeepSeek V4.1 Flash (`audit.py`) found no problem in 11 of
25 and 8 of 20 accepted tasks; the GLM reviewer had accepted all of them.

With Qwen3-1.7B the gate accepted tasks in 1 of 17 sessions: the policy often writes tool calls without the
`<tool_call>` tag. One accepted family had the reference answers 3, 4 or 5, and the policy learned to answer a
guess without the work (0.06 against 0.51 for the control on that family's probes), which led to the
answer-diversity rule of `submit`.
