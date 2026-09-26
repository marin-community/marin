# Curriculum SFT

`generate_curriculum_sft(capability_id, ...)` selects a capability from the pinned
`TASK_CURRICULUM` artifact. Each GLM 5.3 request generates one self-contained task and a complete
text-only user–assistant conversation in one structured response. The generator does not read
evaluation examples or call a judge or task environment.

The generation artifact contains `tasks/` audit Parquet, `chat/` Datakit chat Parquet, exact
`raw-responses.jsonl`, and a manifest. It checks structure, completion, and duplicate task text.
The answers are synthetic and unverified. Do not treat a simulated tool observation or external
claim as a successful environment rollout; this version asks GLM to avoid both.

Bind the generation step in an experiment pipeline:

```python
from experiments.post_training.curriculum_sft.generation import generate_curriculum_sft

generation = generate_curriculum_sft(
    "d27.reporting.analysis",
    version="2026.09.24",
    requested_examples=80,
    accepted_examples=64,
    seed=17,
    max_completion_tokens=4096,
    task_specification="Use self-contained fictional tasks with checkable answers.",
)
```

The step needs an Iris client, the GLM relay, and `GLM_BULK_TOKEN` at execution time. The earlier
finance pilot used oracle-derived arithmetic conversations; this GLM recipe produces unverified
answers.

`pipeline.py` creates one generation and chat-preparation step per capability. The prepared
Parquet artifacts carry canonical OpenAI `messages` and feed `sft_step` through
`ArtifactDatasetSpec`. The shared launcher tokenizes the chat, masks user turns from the loss,
and trains `SnowballLMHeadModel` from a cached, weights-only Levanter conversion of the pinned
Hugging Face checkpoint. The conversion is a separate artifact dependency with a fixed version,
so SFT retries and task variants can reuse it without loading the 39 HF shards again. Conversations
stay separate (`pack=False`) because Snowball does not consume packed-document attention masks.
The launcher saves a sharded bfloat16 Hugging Face checkpoint at the final training step under
the training artifact's `hf/` directory.

`math_trial.py` declares a bounded first loop: deterministic OlympiadBench and Math500 on the
pinned September 20 HF model, three algebra capabilities, four Snowball updates, and
the same evaluations after training. The deterministic OlympiadBench variant uses Minerva/SymPy
equivalence without an LLM judge. Its scores are not comparable to historical judge-backed
OlympiadBench runs. The existing GLM conversations are read from the east-region `ttl=30d`
source prefix. New preparation, training, and evaluation artifacts go under the
`marin_temp_bucket(ttl_days=7)` trial prefix, including the converted checkpoint. Run `train` and `after` on RNO2A with
`MARIN_PREFIX` set to `S3_TRIAL_PREFIX` in `math_trial.py`. Both stages read the pinned
catalog and generated Parquet from east-region S3.
Generation, prepared chat, and HF conversion use fixed versions; the SFT and its re-evaluation use
the CLI `--version`, so a failed SFT can be retried without regenerating or retokenizing data or
reconverting the model.
`--stage full` binds both
evaluations and training in one graph after generation has completed.

This trial has not established an improvement. Its GLM answers are structurally checked but not
oracle-verified, and it has no matched task-only control. A before/after change would measure the
whole synthetic-SFT recipe, not the curriculum's independent contribution. The standard Levanter
Adam optimizer does not reproduce Will's special-token learning-rate or frozen-router-bias changes;
it starts a fresh optimizer from the published September 20 weights.
