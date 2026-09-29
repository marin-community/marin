# Ortet GLM 5.3 science conversion

The RNO2A CPU producer resolves `glm-5.3` from the Iris relay job
`/muchanem/glm53-relay-rno2a`. It sends groups of at most 64 chat requests
to the relay's `/bulk/v1` batch API with `priority=batch` and a
`GLM_BULK_TOKEN`. It does not launch a serving pool. Never print or save the
token; the operator's `/Users/benfeuer/Documents/secrets.env` exports it.

The 17 producer tasks retain the original source-stratified schedule and
output paths. Completed MiniMax batches remain in
`s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v3/outputs/main/`.
The cutover inventory is
`/Users/benfeuer/Documents/experiments/active/targeted-sft/artifacts/science-sft-ortet-cutover-minimax-files.txt`.
The worker skips committed batches and resumes unfinished ones. Its output
records use `science-forward/minimax-m3-plus-glm53-2026.09.29` as the source
identity. The inventory distinguishes the retained MiniMax batches from
later GLM batches when auditing model provenance.

From a clean checkout of the conversion branch, submit the producer with a
new job name and pass the batch token as an Iris environment variable:

```bash
source /Users/benfeuer/Documents/secrets.env
uv run iris --cluster=cw-rno2a job run \
  --job-name science-sft-conversion-ortet-<suffix> \
  --priority batch --enable-extra-resources \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu \
  --replicas 17 --max-retries 8 --timeout 0 --no-wait \
  -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" -- \
  python -m experiments.datakit.science_sft_conversion.conversion \
    --relay-job /muchanem/glm53-relay-rno2a \
    --concurrency 112 --concurrent-batches 2 \
    --batch-size 64 --batch-workers 2
```

Before a full restart, run `probe` with `--samples-per-source 1` and inspect
all 17 validated chat records. Then use `--max-items 1` on a short 17-task
canary to verify atomic Parquet commits and source coverage. The handoff
waits for the replacement producer, audits all expected files and lineage,
builds a fresh token store, and launches the downstream SFT. It no longer
stops a dedicated serving job. Use a new store path and SFT version for the
mixed-model corpus.
