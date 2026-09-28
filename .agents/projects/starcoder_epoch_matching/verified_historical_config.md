# Historical target configuration verified

On 2026-09-08, the representative C40 interior run's persisted configuration, worker configuration, endpoint metrics and runtime package metadata were recovered. Its model, optimizer, tied weights, seven source-cache paths, horizon, batch size, sequence length, seeds, support cap and block-shuffle settings agree with the reconstruction. The actual native endpoint is step 28259 with BPB 0.7880429029464722. No training was submitted.

The sequence-index mapping check also passed at the actual packed-cache length, including a direct replay of the archived block-shuffle class under JAX 0.10.1. This audit does not create the reuse receipt. The exact launch Git commit remains unavailable in the recovered metadata.

## Downloaded evidence

The representative run is `dss_r3d28260_m100_c109_s0711` at p=0.7. Its GCS base is:

```text
gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_dense_support_surfaces_20260808/dss_r3d28260_m100_c109_s0711/2026.07.11
```

The [download manifest](historical_metadata/download_manifest.json) contains the exact generation-qualified URI, local file, byte count and SHA-256 for every retrieved object. Only small metadata files were downloaded, about 235 KB in total.

| File | GCS generation | SHA-256 |
| --- | --- | --- |
| [Materialized artifact](historical_metadata/target_artifact.json) | 1786382221812759 | b9e047c49742fa6d73672d62f9890d70fc95eb5f855a47c6f23f94654ced013e |
| [Executor sidecar](historical_metadata/target_executor_info.json) | 1786200989872035 | 8b443a338526f0eb53e58a976ba6163fd919cedc7205e9d2492061ab574b4461 |
| [Native evaluation log](historical_metadata/target_eval_metrics.jsonl) | 1786382159488476 | f02e5b25e7ffbf0cbadd0d86b07ef75f2a0daeb8f9b14a63e89a7ecfde35094c |
| [Worker tracker mirror](historical_metadata/target_tracker_metrics.jsonl) | 1786382172669779 | See download manifest |

The artifact contains a full materialized `TrainLmOnPodConfig`. Its `.executor_info` is a `step_runner` stub carrying fingerprint attributes and was not used as scientific configuration evidence. The worker tracker mirror supplies the configuration after runtime defaults and output paths were resolved.

Three small files were also retrieved from [the original W&B run](https://wandb.ai/marin-community/marin/runs/dss_r3d28260_m100_c109_s0711): [config.yaml](historical_metadata/wandb/config.yaml), [requirements.txt](historical_metadata/wandb/requirements.txt), and [wandb-metadata.json](historical_metadata/wandb/wandb-metadata.json). Their exact hashes are in the download manifest.

## Scientific configuration comparison

The [machine-readable comparison](historical_metadata/config_comparison.json) records every assertion and the parent-versus-worker configuration differences.

- The entire persisted model configuration equals the current `CompletedAdamHHeuristic()._build_model_config(640, seq_len=2048)` configuration: Qwen3, hidden dimension 640, MLP dimension 2560, seven layers, five query and KV heads, untied embeddings, and the same rotary-embedding parameters and model flags.
- The entire persisted optimizer equals `base._optimizer(7408189440)`, including Muon LR 0.02, Adam LR 0.008, weight decay 0.1, warmup 282, decay 5652, cosine schedule, momentum, numerical epsilons and clipping.
- The persisted target uses 28,260 steps, 128 sequences per batch, sequence length 2048, trainer seed 20260711 and data seed 20260711. Its p=0.7 weights are identical at steps 0 and 22,608. The six broad-component relative weights match the current historical helper exactly.
- All seven training cache paths match the pinned dataset handles. StarCoder uses `tokenized/dolma/starcoder-8b6089` in `gs://marin-us-central1`. Only its `max_train_batches` is capped, at 1068. Global experiment/target budgets and simulated-epoch subset seed are unset.
- Block shuffle is `(io_block_size=256, window_blocks=512, perm_type=feistel)`. Restart sampling, the mixture block size 2048, EOS handling and cross-document attention blocking match the historical configuration. There is no validation batch limit.

The worker resolves BF16 compute/output and FP32 parameters, 32 train/evaluation sequences per device, accelerator requirement enabled and automatic cache building disabled. The remaining differences from the parent record are expected output/checkpoint paths and equivalent dtype/duration serialization. They are enumerated in the comparison artifact, rather than silently discarded.

The native metric key is `eval/paloma/dolma_100_programing_languages-llama3/bpb`. The 30-row persisted evaluation log ends at step **28259**, value **0.7880429029464722**, exactly matching the archived CSV. This directly verifies the collector's `num_train_steps - 1` convention for the historical target. The current trainer's final-hook path also reuses the final `StepInfo`; it does not advance the step before forced final evaluation.

## Ordering and serialization

The canonical artifact and tracker files sort dictionary keys, so their printed component order cannot be used to recover runtime order. The execution path was checked separately:

1. [The lazy runner](/Users/calvinxu/Projects/Work/Marin/marin/lib/marin/src/marin/execution/lazy.py:380) builds the actual dataclass and executes `_handle.run(config)` before converting it to canonical JSON for its record.
2. [The training dispatcher](/Users/calvinxu/Projects/Work/Marin/marin/lib/marin/src/marin/experiment/train.py:88) passes the dataclass to Fray directly.
3. [Fray remote](/Users/calvinxu/Projects/Work/Marin/marin/lib/marin/src/marin/execution/remote.py:79) captures the function arguments in a callable; [Iris serialization](/Users/calvinxu/Projects/Work/Marin/marin/lib/iris/src/iris/cluster/types.py:976) cloudpickles that callable and its arguments. This preserves Python dictionary insertion order. There is no intervening JSON/YAML reconstruction.

The contemporaneous August 8 source at commit `eafa4d49f7c55fbf2abb26b5d92c1ac7d093f9fb` has the same dispatch and cloudpickle transport. The actual W&B metadata names `/app/_callable_runner.py`, consistent with that route. The original launch command identifies the dense-support launcher and r3 coverage block; the recovered metadata contain no contrary component-order override.

Together with the preserved launcher, this supports the six-Nemotron-then-StarCoder interior order and legacy StarCoder key `[898005854, 446240491]`. This is a source-and-transport reconstruction; the sorted record is not direct observational evidence of dictionary insertion order. The p=1 zero-weight-filter exception in [the reuse audit](target_reuse_audit.md) remains applicable. No change to the corrected named-key contract is indicated by the recovered configuration.

## Runtime and outstanding provenance

The original child `requirements.txt` directly records JAX/JAXlib 0.10.1, NumPy 2.3.5, LibTPU 0.0.41, Cloudpickle 3.1.2, Equinox 0.13.2 and Optax 0.2.6. W&B metadata record Python 3.12.13 and a four-device v5p worker. This confirms the intended historical JAX/NumPy versions from the actual training child, rather than only the planning manifest. The persisted trainer also enables `jax_threefry_partitionable` and `jax_softmax_custom_jvp`; retain these when reconstructing the runtime.

Neither `.artifact.json` nor W&B metadata recovered the exact launch Git hash: the artifact's `base_commit` and `tree_hash` are empty and the W&B metadata have no Git section. The temporary launch checkout is no longer present locally. The child package versions and full model/optimizer/data configuration are recovered; a complete original source-tree identity is still unavailable.

## Archived shuffle replay

The [metadata-only cache audit](historical_metadata/cache_audit.json) established 105,745,752 packed training sequences. The [new-loader index audit](historical_metadata/indices_training_runtime/index_audit.json) compared all 136,704 parent positions and the nested 5,120-position subset under JAX/JAXlib 0.10.1 and NumPy 2.3.5.

To check whether algorithm changes could invalidate that comparison, the `BlockShufflingDataset` class was loaded directly from the contemporaneous August 8 source and executed on an index-valued dataset of the actual cache length. Every parent position equals both the current class's output and the new loader's stored parent array. See [source comparison](historical_metadata/legacy_shuffle_source_parity.json), [training-runtime parity](historical_metadata/legacy_shuffle_training_runtime_parity.json), and the retained [archived source](historical_metadata/legacy_dataset_source.py).

The PRP module is byte-identical between the archived commit and current checkout (SHA-256 `ff43ef12f9a19bb4fd7dd41b2280d0b6d1ce3f9879befad4f8630724abe68f66`); the CPU key-placement and fold-in helpers are AST-identical. The block-shuffle class changed its cache ownership and vectorized batch mapping, but the complete parent-position replay establishes parity for this experiment. These checks read source-index mappings, not corpus tokens.

The resulting ordered-array SHA-256 values are:

- Parent: `24c5bdd4bd466de18bc56ba06ea92053e329ad749aadea354a983fb38c61c4c1`.
- Matched subset: `6c8ac23d240d4c99f88b2fa854cb7f7415b19e5387a68cf634f03dd73d2706c3`.

Historical scientific configuration, native endpoint, runtime package versions, parent mapping and nested-subset mapping are verified. The unrecovered exact launch Git identity remains an explicit provenance limitation. A passing receipt can reference the reviewed configuration and mapping artifacts; it must not claim an exact original-checkout replay.
