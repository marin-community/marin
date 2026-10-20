# Pi scenarios in vendored SkyRL

`vendor/harbor` captures Pi model calls and executed tool observations in memory, with a validated trial ID and policy step. `vendor/skyrl` consumes that capture through its shared RolloutEngine task worker. Neither worker logs nor saved sandbox files supply missing trajectory evidence.

The nested SkyRL lock installs `harbor` from the checked-in sibling tree through `vendor/skyrl/harbor`. The launcher includes Harbor's source and package metadata in the hash-verified runtime bundle. CPU and GPU workers therefore install the same edited Harbor package. Do not replace the local lock source with a stale upstream Harbor revision: that bypasses Pi capture even when both vendored directories exist.

The reviewed scenario selectors are `teacher_demo_pi_base`, `teacher_demo_pi_verified_base`, `teacher_distribution_pi_base`, and `verifier_grpo_base`. The launch recipes under `vendor/skyrl/cloud/iris/configs/pi_*_smoke.yaml` still require fresh job, attempt, and output identities. Per-turn distribution and GRPO training use the exact served prompt and sampled response. Distribution teacher requests use unique turn IDs while group admission retains the physical task and repetition IDs. The teacher-trace path keeps private teacher labels apart from student samples; grades come from the task verifier. Worker-owned Pi sandbox attempts stream fixed-schema phase journals. Pi attempts do not upload raw sandbox artifacts by default.

The standalone work completed one checkpointed teacher-trace update, one verifier-GRPO update, and one top-K teacher-distribution update. See [the standalone smoke record](https://github.com/marin-community/MarinSkyRL/blob/1fce8f26/docs/pi-scenario-smokes.md) for trial counts and limits. **Those jobs ran the standalone revision. No vendored Pi GPU update or checkpoint restart has been run.** AI-judged GRPO, DAgger, and critique-to-revision still need their own live training gates; their ETL contracts alone do not establish GPU training.

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
