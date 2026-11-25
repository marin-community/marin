# Qwen3.5-35B-A3B-Base policy-config evaluation

This isolated evaluation uses the campaign's exact benchmark configurations and
codebase pins without adding the base checkpoint to the campaign tracker. The
model is pinned to Hugging Face revision
`0f0813072d2358973511097385626f21fcb6d422`.

Validate and submit the cache gate first:

```bash
./launch.sh --phase cache
./launch.sh --phase cache --submit --wait
```

After the cache gate passes model serving startup, submit the remaining 23
benchmarks. The 17 nonagentic benchmarks share one model server, and the six
agentic benchmarks share another.

```bash
./launch.sh --phase remaining
./launch.sh --phase remaining --submit
```

The launcher's local config snapshots and every durable result record retain the
effective inputs and Marin revision. These launches do
not set a policy label.
