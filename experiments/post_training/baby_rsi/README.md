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
| Train | `trainer.py` | On-policy GRPO with a LoRA adapter. Each step runs 8 attempts of each task with the served policy, yields the groups, and makes one update. The new adapter loads into the vLLM server at once, so the other boxes use the new policy. |
| Analyze | `feedback.py` | Keeps one failed attempt of each task that the policy passes at most 25% of the time, and runs `analyze_failed_runs` (#10265) on each batch. It yields each new capability ID once. |
| Generate | `feedback.py` | One agentic Taskforge session (`taskforge.agentic`) for each idea. The served policy is the solver of the gate, so an accepted task has a policy pass rate from 0.1 to 0.8. The build program of each accepted task makes more training instances and held-out probe instances. `submit` refuses a program whose 8 test instances share one reference answer more than 25% of the time: in the first run, a family with answers 3, 4 or 5 taught the policy to guess. |
| Evaluate | `evaluate.py`, `run.py evaluate` | The base policy (twice) and trained adapters on held-out pool tasks (the Smartie validation file) and on the probe instances, which no arm trained on. |

`policy.py` serves the policy: a `vllm serve` process with runtime LoRA loading, and `VllmPolicy`, a
`rolloutengine` model that renders the chat template itself and keeps the exact sampled tokens and log
probabilities. `rollouts.py` gives every task the same `RolloutEngine`: a pool task answers in one turn, a
structural-closure task continues its assistant prefix, and a synthesized task gets a shell on a ShellSim machine.

Two conversions run the pool tasks on any policy and without Daytona. A structural-closure row holds prompt token
IDs of the Snowball tokenizer; `structure_closure_task` turns it into a conversation that ends with the unfinished
assistant text. The other strata grade with a script in a Docker image; the seed tasks grade with the host's Python
environment (which has `reasoning_gym`, `verifyit`, numpy and pandas) in a bubblewrap sandbox.

## Run

The commands run in an Iris GPU task through `job.sh`, which installs vLLM in its own environment, adds the
trainer's packages, and selects the task's own GPU. A privileged CoreWeave task sees every GPU of its node and
shares the host network, so `job.sh` reads the assigned GPU from the device plugin, and the vLLM server binds a free
port.

```bash
uv run iris --cluster=marin job run --no-wait --target-cluster cw-us-east-02a \
  --enable-extra-resources --gpu H100x1 --cpu 16 --memory 128GB --disk 150GB \
  --sync-package marin-taskforge --container-profile container_profile_privileged \
  --job-name <name> -e HF_TOKEN "$HF_TOKEN" -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" \
  -e PARALLEL_API_KEY "$PARALLEL_API_KEY" \
  -- bash experiments/post_training/baby_rsi/job.sh -m experiments.post_training.baby_rsi.run \
  rsi --model Qwen/Qwen3-4B-Instruct-2507 --gpu-memory-utilization 0.4 --output-dir s3://<bucket>/<prefix>/rsi
```

The defaults are the settings of the results below; the vLLM server and the trainer share the one GPU.

A preempted run continues from what it uploaded (`resume.py`): the adapter and optimizer state of every tenth step,
the records of the groups that it trained, and the accepted tasks of its finished synthesis sessions. It skips the
pool tasks that it trained, gives back the synthesized instances that it did not train on, and does not synthesize
the same idea again.

`control` takes the same training options without the feedback options. `evaluate` takes `--adapter
NAME=URL` for each trained adapter and `--probes` for each `rsi` output directory, and `report.py` compares the
policies with paired bootstrap intervals over tasks.

## Results (2026-10-21)

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
