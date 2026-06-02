# GrugMoE vLLM TPU Logbook

Issue: https://github.com/marin-community/marin/issues/6106

## Branches

- Marin: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/marin/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-support`
  - Seed artifact commit: [`999ae16e8`](https://github.com/marin-community/marin/commit/999ae16e859a39d3a4ebd5d5ee74d20c0ee16630)
- vLLM: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/vllm/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm`
  - Commit: [`86c81da58`](https://github.com/marin-community/vllm/commit/86c81da58c53065d463390b2f4778f01aecc5862)
- tpu-inference: [`grugmoe-vllm-tpu-support`](https://github.com/marin-community/tpu-inference/tree/grugmoe-vllm-tpu-support) in `/home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference`
  - Commit: [`80a576bb`](https://github.com/marin-community/tpu-inference/commit/80a576bb5a1cd733dbe1e7ac1c79c102fbe9a30a)

## Milestones

- 2026-06-02: Refreshed `/home/romain/dev/marin` with `git -C /home/romain/dev/marin pull origin main`.
- 2026-06-02: Created the Marin worktree and cloned vLLM/tpu-inference on matching branches.
- 2026-06-02: Opened Marin experiment issue #6106 with `experiment` and `agent-generated` labels.
- 2026-06-02: Added vLLM `GrugMoeForCausalLM` as a correctness-first, unfused PyTorch implementation.
- 2026-06-02: Added a vLLM MoE unit test for QB-bias expert selection and unbiased sigmoid combine weights.
- 2026-06-02: Added tpu-inference auto-resolution support for `GrugMoeForCausalLM`.
- 2026-06-02: Added this logbook, the reference note, and the tiny parity harness.
- 2026-06-02: Ran component and composed parity successfully against the vLLM branch with the Marin harness.
- 2026-06-02: Attempted Iris `v6e-4` validation. First attempt failed because `~/.ssh/google_compute_engine` was absent; created the standard keypair. Second attempt hung while establishing the Iris controller tunnel and never returned a job id, so no TPU job was submitted.

## Verification So Far

Commands that passed:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm
python -m py_compile vllm/model_executor/models/grugmoe.py tests/models/test_grugmoe.py

cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference
python -m py_compile tpu_inference/models/common/model_loader.py tests/models/common/test_model_loader.py

cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
python -m py_compile experiments/grug/moe/vllm_tpu_parity.py

cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run --with ruff ruff check experiments/grug/moe/vllm_tpu_parity.py

cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m experiments.grug.moe.vllm_tpu_parity \
  --vllm-root ../grugmoe-vllm-tpu-vllm

cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm
uv run --no-project \
  --with-requirements requirements/common.txt \
  --with pytest \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m pytest --confcutdir=tests/models tests/models/test_grugmoe.py -q

cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm
uv run --no-project \
  --with-requirements requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python - <<'PY'
from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS
print(_TEXT_GENERATION_MODELS["GrugMoeForCausalLM"])
PY

cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference
PYTHONPATH=/home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm \
uv run --no-project \
  --with-requirements requirements.txt \
  --with-requirements /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm/requirements/common.txt \
  --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m pytest tests/models/common/test_model_loader.py::TestGetModel::test_get_model_auto_resolves_to_vllm_for_grug_moe -q
```

Final parity output:

```text
component: GrugMoeMLP matches Levanter moe_mlp
full: GrugMoeModel hidden states match Levanter Transformer
```

Focused repo-test results:

- vLLM MoE unit test: `1 passed, 18 warnings in 8.78s`
- vLLM registry lookup: `('grugmoe', 'GrugMoeForCausalLM')`
- tpu-inference resolver test: `1 passed, 20 warnings in 46.59s`

Commands attempted but blocked by local environment:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm
python -m pytest tests/models/test_grugmoe.py -q
```

Result: default Python has no `pytest` or `torch`.

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-vllm
../issue-5510-investigate-logs/.venv/bin/python -m pytest tests/models/test_grugmoe.py -q
```

Result: existing PyTorch venv lacks vLLM deps, first missing module `cbor2`.

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference
python -m pytest tests/models/common/test_model_loader.py::TestGetModel::test_get_model_auto_resolves_to_vllm_for_grug_moe -q
```

Result: default Python has no `pytest`.

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-inference
../issue-5510-investigate-logs/.venv/bin/python -m pytest tests/models/common/test_model_loader.py::TestGetModel::test_get_model_auto_resolves_to_vllm_for_grug_moe -q
```

Result: existing PyTorch venv does not have `vllm` importable.

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run iris --cluster=marin job run \
  --no-wait \
  --enable-extra-resources \
  --tpu v6e-4 \
  --region europe-west4 \
  --priority interactive \
  --timeout 1800 \
  --job-name grugmoe-vllm-tpu-parity \
  -- bash -lc 'git clone --depth 1 --branch grugmoe-vllm-tpu-support https://github.com/marin-community/vllm.git /tmp/grugmoe-vllm-tpu-vllm && uv run --with "torch==2.10.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu python -m experiments.grug.moe.vllm_tpu_parity --vllm-root /tmp/grugmoe-vllm-tpu-vllm'
```

Result: first attempt failed with `SSH key not found at /home/romain/.ssh/google_compute_engine`. After creating the keypair, the retry hung at `Establishing SSH tunnel to iris-controller-marin (zone=us-central1-a)` for several minutes. The hung local Iris/GCloud tunnel process was terminated. No job id was emitted.

## Gate Status

- Gate 1, repo map: complete. GrugMoE reference, vLLM registry/model extension points, tpu-inference model resolver, and closest MoE implementations were identified.
- Gate 2, component parity: passed locally with the Marin harness.
- Gate 3, composed parity: passed locally with the Marin harness.
- Gate 4, TPU validation: blocked. Iris submission did not reach job creation because the controller SSH tunnel hung after key setup. The exact command is recorded above.
- Gate 5, handoff: complete for local work. Branches are pushed, issue exists, docs/focused tests/repro commands are recorded. TPU validation remains blocked as Gate 4.

## Next Commands

Create or reuse a compatible local environment, then run:

```bash
cd /home/romain/dev/marin-wt/grugmoe-vllm-tpu-support
uv run --with 'torch==2.10.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m experiments.grug.moe.vllm_tpu_parity \
  --vllm-root ../grugmoe-vllm-tpu-vllm
```

Then run the focused repo tests listed in `vllm_tpu_reference.md`.
