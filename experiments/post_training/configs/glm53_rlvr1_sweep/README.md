# Datakit RLVR loop and length sweep

Each arm starts from
[`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21).
The six YAML files in this directory fix the training and validation dataset,
seed, 128-GPU allocation, optimizer settings, 65,536-token request window,
16,384-token per-turn output cap, and grader runtime. They run for 16 optimizer
steps with the same 100 validation prompts at baseline and every two steps.
Validation uses temperature zero. Checkpoints are saved every four steps.

| Arm | Loop penalty per charged token | Loop penalty cap | Maximum length penalty |
| --- | ---: | ---: | ---: |
| `loop1_length0` | 1 | 32,768 | 0 |
| `loop1_length02` | 1 | 32,768 | 0.2 |
| `loop2_length0` | 2 | 65,536 | 0 |
| `loop2_length02` | 2 | 65,536 | 0.2 |
| `loop4_length0` | 4 | 131,072 | 0 |
| `loop4_length02` | 4 | 131,072 | 0.2 |

The length penalty begins after 8,192 generated tokens across a trajectory's
turns and increases linearly to its listed maximum at 16,384 tokens. It
changes the optimization reward, not the verifier verdict. The loop detector
checks the final trainable response segment for an exact repeated suffix
with a period of at most 256 tokens and at least eight repetitions. It
subtracts the listed penalty from normalized advantages on tokens after the
first seven repetitions. The cap scales with the per-token penalty so long
loops retain the intended strength.

Launch one arm from the targeted-SFT workspace with:

```bash
cd /Users/benfeuer/Documents/experiments/active/targeted-sft
RL_SWEEP_ARM=loop1_length02 bash configs/launch_glm53_rlvr1_marin.sh
```

Replace `loop1_length02` with any arm name in the table. Each arm writes to
`users/benfeuer/checkpoints/glm53-rlvr1-sweep-<arm>@2026.09.29.19` and
has the W&B run name in its YAML. The launcher pins MarinSkyRL commit
`cce1d4969e8a40cd3809c72aa74459d7a689241c`.

Compare holdout pass@1 and average score at matched steps. Record the number
of responses stopped by the token limit, responses ending in a tool call, and
retained trajectories with a detected repeated suffix. The fixed holdout
supports paired trace review; its 100 prompts make small pass-rate differences
uncertain.
