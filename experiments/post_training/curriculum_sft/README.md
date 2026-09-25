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
)
```

The step needs an Iris client, the GLM relay, and `GLM_BULK_TOKEN` at execution time. Training and
evaluation are separate consumers of its chat Parquet. This generation recipe has not yet been run;
the earlier finance pilot used oracle-derived arithmetic conversations.
