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

The step needs an Iris client, the GLM relay, and `GLM_BULK_TOKEN` at execution time. This
generation recipe has not yet been run; the earlier finance pilot used oracle-derived arithmetic
conversations.

`curriculum_grug_sft(...)` in `grug_pipeline.py` connects this artifact to native Grug SFT. It
renders the Datakit chat Parquet with the Marin template, tokenizes the rendered Parquet through
Levanter's text cache, and packs complete conversations into training sequences with attention
blocked across conversation boundaries. Pass an adopted native Grug checkpoint, its matching
tokenizer, optimizer, resources, and an explicit training budget. The module includes the
September 20 checkpoint handle and tokenizer path. The native `step-158000` checkpoint metadata
and an export directory were found under the corrected `-cpfix` GCS run prefix.

The current Grug adapter supports weights-only initialization from a native checkpoint, not an HF
export. It does not include Will's special-token learning-rate or frozen-router-bias changes, and
it has not been trained or evaluated with this generated data. Keep the first run in `us-central2`,
where the native checkpoint resides; copying it to another region is a separate large transfer.
The synthetic answers remain unverified, so compare a baseline and a matched control before
attributing any evaluation change to curriculum guidance.
