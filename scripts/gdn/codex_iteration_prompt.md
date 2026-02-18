You are running one unattended hill-climb iteration for Gated DeltaNet TPU kernels.

Iteration metadata:
- Iteration: {{ITERATION}} / {{TOTAL_ITERATIONS}}
- Starting commit: {{HEAD_SHA}}

Repo context:
- Kernel implementation: `lib/levanter/src/levanter/layers/gated_deltanet.py`
- Correctness tests: `lib/levanter/tests/test_gdn_kernels.py`, `lib/levanter/tests/test_gdn_layer.py`
- Optimization recipe: `docs/recipes/optimize_gdn_pallas_tpu.md`
- Running log: `lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Infra CLI: `scripts/gdn/gdnctl.py`

Required behavior for this iteration:
1. Propose exactly one concrete TPU optimization aimed at increasing MFU.
2. Implement the change in code.
3. Validate correctness on TPU by running GDN tests.
4. Launch a lightweight profiled training run on TPU.
5. Pull the resulting profile trace (HF or W&B artifact path), analyze bottlenecks, and update the running log.
6. If validation passes, commit exactly one commit describing the optimization and evidence.

Constraints:
- TPU-only optimization target.
- No backward-compatibility shims/fallback hacks.
- Do not relax test tolerances.
- Keep edits minimal and focused on one optimization.
- If blocked on infra/transient cluster issues, document the blocker in the running log and stop without speculative code changes.

Preferred commands:
- `uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both`
- `uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --no-wait`
- `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --no-sync`
- `uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 <job_id> --show-logs --tail 400`
- `uv run python scripts/gdn/gdnctl.py hf-download-trace ...`

Definition of done:
- One optimization committed, tests green, one profiled run completed, running log updated with measurements + next hypothesis.
