# CatCount CPU starting policy

The CPU canary downloads the calibrated 529,792-parameter FP32 policy from
`s3://marin-us-east-02a/marin/rl-canaries/cat-count/cpu/pretrained/llama-530k/v1/`.
Its manifest SHA256 is `7a5f1047648a90262514a663168580610b9f6bbe1522b2c38b7578bbc5eb82ae`;
each downloaded file must match the manifest's byte count and SHA256.
CatCount CI steps supply the repository's CoreWeave credentials and S3 endpoint
through the shared remote I/O factory. A successful download prints
`CAT_COUNT_POLICY source=s3`. Tests skip when credentials are absent, as on fork
PRs; download and integrity errors with credentials present fail the test.
CI never pretrains the policy.

PR CI runs ten normal training steps on seed 0 with the positive assertions.
Nightly checks normal/reversed pairs on seeds 0 and 1 and asynchronous resume,
staleness and optimizer updates. At some resumed evaluation the mean train and
held-out reward must be at least 0.65 and at least 0.3 above its initial value.
The resume run uses 100 steps.

Regenerate from the repository root in the frozen CPU environment:

```bash
uv sync --frozen --group dev --group harbor-test --extra cpu --extra telemetry
uv run --frozen python skyrl-train/examples/cat_count/cpu_canary.py \
  --out /tmp/cat-count-policy --steps 3000 --lr 3e-4 \
  --width 128 --layers 2 --seed 0
sha256sum /tmp/cat-count-policy/model.safetensors
```

Pretraining uses one CPU thread, counts other words and teaches `cat` only at
N=1. The retained `pretrain()` implementation produces the fixture. Byte-level
reproduction depends on the frozen dependencies and CPU numerical operations.
Review regenerated weights and rerun both CI selections before replacing them.

Weights SHA256: `5f1774fdaf2faa6f7b53cb932df09ffc31ff6cd7a7358eabac0939e67a51474d`.
The immutable S3 manifest records byte counts and SHA256 for every file.
