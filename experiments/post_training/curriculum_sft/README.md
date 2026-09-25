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

`curriculum_grug_sft(curriculum_ids=[...], ...)` in `grug_pipeline.py` creates one generation and
render step per capability, then mixes their Parquet sources at equal weight. Levanter packs the
rendered conversations with attention blocked across conversation boundaries. Pass a native Grug
checkpoint, its matching tokenizer, optimizer, resources, and an explicit training budget.

`math_trial.py` declares a bounded first loop: deterministic OlympiadBench and Math500 on the
pinned September 20 HF model, three algebra capabilities, four Grug updates, an HF export, and
the same evaluations after training. The deterministic OlympiadBench variant uses Minerva/SymPy
equivalence without an LLM judge. Its scores are not comparable to historical judge-backed
OlympiadBench runs. The HF importer materializes native weights on RNO2A, so the trial does not
transfer the large `us-central2` checkpoint across regions. Run `--stage generate` on
`cw-us-east-08a`, where the GLM relay is registered, with `MARIN_PREFIX` set to the
`S3_TRIAL_PREFIX` in `math_trial.py`. Then run `train`, `export`, and `after` on RNO2A with the
same prefix. Both clusters read the pinned catalog and generated Parquet from east-region S3.
`--stage full` binds both evaluations and training in one graph after generation has completed.

This trial has not established an improvement. Its GLM answers are structurally checked but not
oracle-verified, and it has no matched task-only control. A before/after change would measure the
whole synthetic-SFT recipe, not the curriculum's independent contribution. The native Grug adapter
also does not reproduce Will's special-token learning-rate or frozen-router-bias changes; it is a
weights-only continuation from the published September 20 model.
