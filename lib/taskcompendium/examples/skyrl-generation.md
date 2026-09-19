# Harbor-backed SkyRL generation

`skyrl_generate.py` demonstrates TaskCompendium lowerings through the pinned
MarinSkyRL `TrajectoryRunner.run` interface. It requires a runtime with both
MarinSkyRL (revision `74f70956fa5000a6592165db2165e0d8344f8314`) and
TaskCompendium's pinned Harbor extra installed. It does not start training,
inference engines, or Ray workers.

Supply a JSON array containing a unique `uid`, absolute `task_dir`, and resolved
Harbor `execution` for each lowering. The execution is consumer-owned: use
`resolve_harbor_execution`, or copy an exported `reference-execution.json` for
a deliberate local reference run. Different rows can select different agents,
environments, endpoints, and generation budgets.

```json
[
  {
    "uid": "example/plain",
    "task_dir": "/absolute/path/to/lowering",
    "execution": {
      "environment": {"import_path": "taskcompendium.harbor.environments:NoToolEnvironment"},
      "agent": {
        "import_path": "taskcompendium.harbor.agents:DirectChatAgent",
        "model_name": "Qwen3.8-27B-FP8",
        "kwargs": {"api_base": "http://spark-0143.local:8001/v1", "max_tokens": 4096}
      },
      "verifier": {"import_path": "taskcompendium.harbor.verifier:SemanticVerifier"}
    }
  }
]
```

The tokenizer locator and revision are explicit and separate from the served
model alias. With that runtime active:

```bash
python lib/taskcompendium/examples/skyrl_generate.py \
  --requests /absolute/path/to/requests.json \
  --tokenizer /absolute/path/to/tokenizer \
  --tokenizer-revision local \
  --output /absolute/path/to/new-generation \
  --concurrency 1 --repetitions 2
```

Each call writes an `attempts-*.jsonl` archive and Harbor trial artifacts.
The CLI writes `batch.json` when every attempt was graded and its messages can
be recovered and tokenized. A wrong answer can
carry reward zero; malformed submissions and infrastructure failures carry
null reward and cause `UngradedBatchError` after archival. The adapter does not
filter rows or change their instance/repetition identities.

The TaskCompendium chat, shell-tool, provider, and replay adapters retain the
required trace. A native Terminus-2 launch must set `store_all_messages: true`
in its agent kwargs. Native mini-SWE-agent trace conversion is not implemented;
its semantic attempt can be archived, but cannot yet become a token batch.

Token IDs are reconstructed from retained messages and tool definitions.
User and tool observations have zero loss mask; assistant spans are active.
No served token IDs, rollout logprobs, or exact transport alignment are claimed.
This validates the consumer contract, not model quality or training readiness.

The lower-level `taskcompendium.harbor.generation.generate_attempts` API returns
all attempts without requiring SkyRL or a tokenizer. `write_attempts` stores
them independently of numeric trainer batches.

Run the consumer checks in the same pinned runtime:

```bash
PYTHONPATH=lib/taskcompendium/examples \
  python -m pytest lib/taskcompendium/examples/test_skyrl_generate.py -q
```

The tests use a local tokenizer and scripted model actions through real Harbor
trials. Production runner selection, workload isolation, admission of partial
batches, and inference-engine token capture remain follow-up work.
