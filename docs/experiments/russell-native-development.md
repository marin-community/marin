# Russell native DEVELOPMENT comparison

`experiments.post_training.russell_rsi.launch_agentic_development` builds three stages: CPU qualification, two checkpoint evaluations, and a paired report. The report measures an adapted SWE-bench Verified DEVELOPMENT comparison. It does not change the Russell promotion gates.

The cohort contains eight fixed IDs, in this order:

1. `django__django-16493`
2. `sphinx-doc__sphinx-8265`
3. `sphinx-doc__sphinx-7985`
4. `django__django-16429`
5. `matplotlib__matplotlib-22719`
6. `django__django-13810`
7. `django__django-15277`
8. `pydata__xarray-7229`.

The canonical task source is commit `86723674f04e4209ac479d0fb75d9d9f44b4377e`. Harbor is commit `2666d6526477ae3e46030a8dc4f3f2c68fd7a84f`. The worker installs mini-swe-agent 2.1.0 and LiteLLM 1.104.0. This experiment pin does not change the shared Harbor dependency.

## Frozen inputs

The JSON config contains `evaluation_version` and the fields of `DevelopmentPlan`. Each `PinnedFile` contains `uri` and `sha256`.

| Field | Content |
| --- | --- |
| `source_manifest` | Pinned metadata with `task_source_commit`, `harbor_commit`, and ordered `tasks`. Each task has `task_id`, `dockerfile_sha256`, and `source_files`. |
| `image_manifest` | Pinned `prepared-images.json`, either a list or an object with `images`. Each image binds the task ID, guest archive, data manifest, derived image, and recipe. |
| `runtime_bundle` | The shared QEMU `RuntimeBundle`. Its existing archive and extraction limits stay unchanged. |
| `task_archive`, `task_archive_size` | Pinned regional gzip tar archive and exact compressed size. Its layout is `tasks/<task_id>/<canonical-relative-path>`. |
| `native_configs` | Ordered YAML pins with the canonical mini config first and explicit overrides after it. |
| `resolved_native_config` | The exact result of native `recursive_merge` for those YAML files. Include temperature, maximum output tokens, step limit, and `model.model_kwargs.num_retries: 0`. |
| `source_pins` | Entries with `module` and a `file` pin. The worker compares each pin with the loaded module bytes. |
| `producers` | Exactly two distinct `FrozenProducer` records. No checkpoint selection occurs in this driver. |
| `journal_path` | Durable storage for all sixteen reserved evaluation slots. |
| `worker_cluster` | Explicit Iris peer cluster near the pinned regional task and image archives. |

The task archive contains exactly the files in the task source metadata. The worker checks the inventory, sizes, and Git blob hashes. It does not add or replace IDs.

Each prepared image entry contains `task_id`, `bundle_uri`, `bundle_sha256`, `bundle_size`, `data_manifest_uri`, `data_manifest_sha256`, `derived_manifest_digest`, `dockerfile_sha256`, `derived_recipe_sha256`, and `image`. The data manifest binds these fields, the canonical source files, and the shared runtime archive hash. It also gives the extracted file inventory and hashes. The root filesystem retains its logical 10 GiB capacity. The guest has 4096 MiB of memory and no network.

Each producer contains `identity`, `export_uri`, `record`, `export_manifest`, `tokenizer`, `tokenizer_revision`, and `chat_template`. The canonical producer record binds the identity and export URI. The export manifest binds `producer_identity`, `export_uri`, `tokenizer`, `tokenizer_revision`, and `chat_template_sha256`.

Required module pins include the evaluation worker, attempt journal, native adapter, native environment, QEMU environment, QEMU machine, runtime installer, inference server, Harbor trial and verifier, and native InteractiveAgent and LiteLLM model. Branch modules must load from the packaged branch. The remote worker supplies the branch `PYTHONPATH` before imports.

Use regional image and task archives near the worker. Model exports remain at their frozen regional locations. Task instructions, tests, solutions, and evaluation traces do not belong in task-builder inputs.

## Qualification and execution

Use a coordinator environment with the experiment's pinned Harbor, mini-swe-agent, and LiteLLM packages. Preview the graph with a pinned config:

```bash
uv run python -m experiments.post_training.russell_rsi.launch_agentic_development \
  --config-uri CONFIG_URI --config-sha256 CONFIG_SHA256
```

The CPU stage uses a fresh QEMU guest for each baseline, reference, and native command qualification. The baseline uses Harbor NopAgent and must receive canonical reward 0. The reference uses Harbor OracleAgent and must receive canonical reward 1. Native command qualification uses fixed localhost tool responses. It checks command execution, cwd, environment, merged output, exit code, timeout output, and native completion. It does not call a model.

A failed check holds the whole cohort. No task replacement occurs. Local unit tests use the guest command loop and actual Harbor/native packages. Those tests do not qualify the prepared images or real QEMU runtime.

Evaluation uses one attempt for each task and producer. The two checkpoint stages run in sequence. Each stage permits four concurrent trials. The eight-H100 server has context length 32768, automatic tool choice, and the Hermes tool parser. Harbor records the immutable producer identity as `model_name`. The native client routes requests through `model_alias`.

The setup, agent, and verifier timeouts are 360, 1800, and 300 seconds. Harbor retries are absent. Fray failure and preemption retries are zero. The qualification job deadline is six hours. Each checkpoint job deadline is two hours. Native retry attempts are one, and LiteLLM `num_retries` is zero.

## Durable evidence and results

The journal binds all sixteen slots to the task, verifier source, guest bundle, runtime, two producers, source pins, and native settings. It reserves each slot before `Trial.create`. A completed slot returns its saved result without inference. An incomplete reservation refuses execution. Changed pins refuse journal reuse.

At `VERIFICATION_START`, the worker saves the available native trajectory and controller log. It also records the workspace patch, Git inventory, and command errors before the canonical verifier starts. Damage to `.git` does not suppress a canonical grade. A patch is not a full workspace snapshot. This driver does not permit automatic grade-only recovery.

After the trial, the worker saves partial agent records, canonical `TrialResult`, and verifier records before the terminal journal result. Native controller EOF uses Harbor's nonzero-agent error contract, so Harbor can still execute the verifier. Canonical rewards remain valid when the agent reaches a limit.

`development-comparison.json` contains exactly sixteen distinct slots. It reports valid grades, infrastructure errors, agent errors, model errors, comparable pairs, wins, losses, and paired net gain. Error counts include agent or model errors that still have valid canonical grades. A missing slot causes an error. It does not become reward zero.
