# Science-forward text-to-chat conversion

`sources.json` pins the 17 non-chat source artifacts used by the science-forward
SFT mix. They contain 862 Parquet shards in the CoreWeave copy of the source
pool. The Nemotron Math Proofs, Nemotron Science v2, and TextbookReasoning
components are already structured chat and are excluded here.

The worker reads each source row and splits long text at nearby paragraph or
line boundaries, retaining every source character across chunks. Each chunk
gets one of six answer formats chosen from a stable source ID hash. MiniMax
generates a user request, an assistant `reasoning_content` span, and a final
answer. The worker validates the three fields, the requested final format, and
Datakit's Harmony message structure before writing `CHAT_SCHEMA` Parquet.
Output files live under
`s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v1/chat/`.
Each file covers one 16-row source batch; completed batches are skipped on
restart. A failed batch remains unwritten and fails the Iris task after four
request attempts. The source artifact stays unchanged.

The conversion uses the shared `/benfeuer/minimax-m3-science-sft` Iris endpoint
on `cw-rno2a`. Start one small batch before the full run:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive \
  --job-name science-sft-conversion-smoke-20260927 \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.conversion \
    --max-items 1 --max-batches 1
```

After checking the first generated records and model response, submit the
full resumable workload:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive \
  --job-name science-sft-conversion-20260927 --replicas 16 \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.conversion
```

The conversion is complete only when every expected source row and chunk has
a validated output record. Count source and output rows before registering
the converted chat source in Datakit or using it for SFT.
