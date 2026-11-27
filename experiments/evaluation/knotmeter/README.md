# Snowball 64K AgentX benchmark

The recipe pins Snowball's model and tokenizer revisions, the released AgentX
harness, and the Weka trace revision and checksum. Prepare the replay once on
CPU. Run it against both servers using the same receipt and accelerator type.

```bash
uv run python -m experiments.evaluation.knotmeter.workload \
  --series experiments/evaluation/knotmeter/snowball-v01-tpu-64k.yaml \
  --output /tmp/agentx-prepare

uv run python -m experiments.evaluation.knotmeter.profile \
  --workload /tmp/agentx-prepare/workload.json \
  --url http://127.0.0.1:38197 \
  --concurrency 128 \
  --output /tmp/agentx-profile
```

The preparation command defaults to the active region's temporary bucket. Run
preparation and profiling in the same storage region. The profiler verifies the
archive checksum and requires an AgentX native cache hit. Run it on the server's
host with the checkpoint revision in the recipe, a 65,536-token context, and
equivalent CPU allocations for both implementations. Use the same three-minute
measurement and compare output tokens per second, p90 time to first token, and
the request error rate in `aiperf_artifacts/profile_export_aiperf.json`.
