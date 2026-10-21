# Pi scenarios in vendored SkyRL

`vendor/harbor` captures Pi model calls and executed tool observations in memory, with a validated trial ID and policy step. `vendor/skyrl` consumes that capture through its shared RolloutEngine task worker. Neither worker logs nor saved sandbox files supply missing trajectory evidence.

The nested SkyRL lock installs `harbor` from the checked-in sibling tree through `vendor/skyrl/harbor`. The launcher includes Harbor's source and package metadata in the hash-verified runtime bundle. CPU and GPU workers therefore install the same edited Harbor package. Do not replace the local lock source with a stale upstream Harbor revision: that bypasses Pi capture even when both vendored directories exist.

The reviewed scenario selectors are `teacher_demo_pi_base`, `teacher_demo_pi_verified_base`, `teacher_distribution_pi_base`, and `verifier_grpo_base`. The launch recipes under `vendor/skyrl/cloud/iris/configs/pi_*_smoke.yaml` still require fresh job, attempt, and output identities. They allow four model turns: the vendored RolloutEngine cancels Pi at the configured turn cap, so a two-turn limit can stop a tool follow-up before validated capture is returned. Per-turn distribution and GRPO training use the exact served prompt and sampled response. Distribution teacher requests use unique turn IDs while group admission retains the physical task and repetition IDs. The teacher-trace path keeps private teacher labels apart from student samples; grades come from the task verifier. Worker-owned Pi sandbox attempts stream fixed-schema phase journals. Pi attempts do not upload raw sandbox artifacts by default.

The standalone work completed one checkpointed teacher-trace update, one verifier-GRPO update, and one top-K teacher-distribution update. See [the standalone smoke record](https://github.com/marin-community/MarinSkyRL/blob/1fce8f26/docs/pi-scenario-smokes.md) for trial counts and limits. Those jobs ran the standalone revision. Vendored results and remaining gates are below. AI-judged GRPO, DAgger, and critique-to-revision still need their own live training gates; their ETL contracts alone do not establish GPU training.

## Vendored GPU smoke, 2026-10-21

The clean runs used Marin revision `760acbff0`, runtime digest `556aeef9aada29115cd95f90f7476d7042885cb78f8b6b6469211b36f473805f`, Daytona, and GB200s on CoreWeave. Each attempt had a new Iris job and output prefix under `s3://marin-us-east-02a/MarinFold/pi-scenarios-vendored-smoke-20261021/output/`. The receipts are under each attempt's `diagnostics/` directory. Only step-1 checkpoints, tokenizer files, schema-checked bounded phase journals, and content-free receipts were retained from successful runs; raw/unchecked worker and trajectory objects were removed.

- `distribution-attempt-53-vendored-retention-fix`: **succeeded**. Two independently graded physical student trials supplied four exact served-turn rows. A separately pinned Qwen3-1.7B teacher scored 1,594 response tokens (top-20 sparse forward KL; retained mass minimum 0.9958). Verifier-score coverage was 1, policy distillation loss 15.7498, and raw gradient norm 225.87. Optimizer step 1, step-1 weight sync, a step-1 checkpoint with tokenizer, and trajectory publication all completed. No checkpoint restart or quality improvement was tested.
- `trace-attempt-55-vendored-two-tasks`: **succeeded** for independent-teacher SFT (policy loss 3.1712, raw gradient norm 77.21), step-1 weight sync, checkpoint, and trajectory publication. It used `teacher_demo_pi_base`, which disables verifier grading; this is **not** an independently graded teacher-trace gate.
- `grpo-attempt-54-vendored-4groups`: **completed** eight independently graded physical trials in four groups and checkpointed optimizer step 1, but all verifier scores were 1. The informative-group fraction, policy loss, and raw gradient norm were **zero**. Weight sync occurred, but the step is **not** an objective-connected GRPO update and says nothing about AI-judged GRPO.

Follow-ups did not pass: `grpo-attempt-56-vendored-seed1337` (same tasks, predeclared different seed) and `trace-attempt-57-vendored-verified` stopped before any optimizer update at the Pi capture-or-grade guard. In the former, one of eight agents reported a `ValueError`; the latter's two agents ran to completion, but neither trial was admitted. The current guard conflates missing capture with a failed rollout/grade, so neither outcome warrants guessing at the underlying cause. These failed attempts retain only bounded journals and failure receipts.

The first vendored distribution job failed during worker setup because Iris excluded a nested Harbor `docs/` file included in the source hash. A source-bundle test now exercises Iris's actual file collector. A later distribution attempt was cut off after two model turns during a tool follow-up; the recipes now allow four. The first step-1 distribution job then exposed a post-admission `TaskInput.lowered` retention error; the port now reads `TaskInput.spec`. The subsequent clean attempt above passed the retention gate. No vendored checkpoint restart or matched checkpoint evaluation has been run.

The isolated CPU gates run with the nested lock:

```bash
uv run --project vendor/harbor --frozen --group test pytest -q vendor/harbor/tests/unit/agents/installed/test_pi.py
uv run --project vendor/skyrl --frozen --extra cpu --group harbor-test pytest -q \
  vendor/skyrl/cloud/iris/tests/test_pi_scenario_composition.py \
  vendor/skyrl/skyrl-train/tests/cpu/rollouts/test_pi_scenario_etl.py \
  vendor/skyrl/skyrl-train/tests/cpu/rollouts/test_pi_full_trajectory_examples.py
uv run --no-project python -m scripts.skyrl_migration check-vendor
```

For a vendored GPU acceptance run, generate a new runtime source digest after every edit, preflight the recipe with the nested launcher, and use a new co-located artifact prefix. Inspect completed physical trials, independent grades, teacher provenance where applicable, optimizer updates, weight sync, and the saved checkpoint as separate outcomes. Do not reuse a standalone job's success as vendored-runtime evidence.
