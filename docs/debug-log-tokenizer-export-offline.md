# Debugging log for offline tokenizer export

Make scratch-model training and Hugging Face checkpoint export use the tokenizer already resolved by the data pipeline, without depending on a second Hub download.

## Initial status

The first 3e18 canary failed before optimizer step 0 because scratch Qwen configuration inferred a tokenizer from its remote reference checkpoint. Passing the resolved data tokenizer into the converter fixed training: the recovery canary completed step 3006 and wrote both the native checkpoint and HF model shard. Metadata export then failed at `Saving tokenizer` because `HfMarinTokenizer.as_hf_tokenizer()` reconstructed `marin-community/marin-tokenizer` from its public Hub identifier. The resulting Xet signed-URL authentication failure made an otherwise complete run fail.

## Hypothesis 1

Scratch model conversion must receive the already-resolved data tokenizer instead of inferring a tokenizer from the model reference.

## Changes to make

Build the scratch converter through `converter_from_hf_compat_config(config.model, tokenizer=tokenizer)` and cover the full one-step scratch Qwen training path while blocking remote tokenizer construction.

## Results

The focused test passed. The recovery canary reached optimizer step 3006 and wrote the final native checkpoint plus HF model shard, proving that training no longer depends on the model reference tokenizer.

## Hypothesis 2

`load_tokenizer()` stages complete tokenizer files locally but replaces the token object's source path with the public identifier. Preserving both the public identity and staged directory will let checkpoint export reconstruct an exact HF tokenizer with `local_files_only=True`.

## Changes to make

Store the staged directory separately on `HfMarinTokenizer`, use it for HF conversion, and add a tokenizer-export round-trip whose public name is deliberately nonlocal.

## Results

The offline tokenizer export round-trip, the complete HF-export test file, the HF utility tests, the scratch-Qwen training regression, and the tokenizer security regression pass. Targeted formatting and lint checks pass; project-wide Pyrefly still reports eight pre-existing errors outside these changes. A separate-process `HF_HUB_OFFLINE=1` smoke for `marin-community/marin-tokenizer` preserved its 128,256-token vocabulary, BOS/EOS IDs, chat template, and sample encodings. CC approved the recovery canary with no blocker.

The live retry resumed the completed step-3006 checkpoint, wrote the 1.43 GB model shard, saved tokenizer metadata without Hub access, finished the HF-compatible checkpoint, synced W&B, and exited successfully.

## Hypothesis 3

The first native Table-9 child failed before loading the checkpoint because the evaluator called the removed top-level `levanter.initialize` export. Levanter's package initializer deliberately exports no submodule APIs, so the evaluator must import initialization and tracker operations from their defining modules.

## Changes to make

Import `initialize` from `levanter.trainer` and `log`/`current_tracker` from `levanter.tracker`, then exercise `olmo_base_eval()` locally through initialization with a no-op tracker.

## Results

The 44 focused OLMoBaseEval tests pass. A local dry invocation completes `initialize(config)` and reaches the expected downstream missing-checkpoint error. Targeted lint passes; Pyrefly reports only eight unrelated pre-existing errors. CC found no blocker. Retry 2 skips the successful training/export checkpoint and reruns only native Table-9 evaluation.

## Hypothesis 4

Retry 2 initialized JAX, W&B, the tokenizer, and all 88,592 requests, then remained silent for more than an hour. An Iris thread dump localized the stall to `Qwen3Config.hf_checkpoint_converter()` inferring the public Qwen tokenizer through Xet. Both model-config probing and the shared `load_hf_checkpoint()` helper constructed converters before binding the checkpoint-local tokenizer.

## Changes to make

Parse the checkpoint's local `config.json` directly into the registered Hugging Face and Levanter config classes. In the shared loader, construct the converter with `converter_from_hf_compat_config`, binding the already-loaded tokenizer before converter construction.

## Results

A full tiny-Qwen save/load regression now fails on any incidental tokenizer load and passes through `load_hf_checkpoint`; all 45 OLMoBaseEval tests pass. Targeted lint and formatting checks pass; Pyrefly still reports the same eight unrelated pre-existing errors. CC approved killing the Xet-stalled child and rerunning the cached canary. Retry 3 skipped cached training and export, loaded the 1.43 GB shard, scored all 88,592 requests, wrote Table-9 macro BPB 1.087263, and succeeded end to end.

## Future work

- [x] Confirm the recovery canary exports all tokenizer metadata.
- [x] Confirm the recovery canary completes native Table-9 evaluation.
- [x] Release the 22-candidate GRP validation panel only after that end-to-end gate passes.
