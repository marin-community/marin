# SFT cold-start reproduction reference

This branch preserves the code used for the Snowball/Grug SFT reproduction, checkpoint exports, data and trace audits, and corrected evaluation cohort. It is a historical reference, not a merge candidate or a tested port of the training run to current main.

The reference branch starts from main at `671c7bcf78`. Training used base `8577d258214dd0dcb992c04458ee4eb10496d4e5` plus the recorded working-tree patch. Corrected evaluations used `2a174c5fff27ecbcf2db4a300441e247ea18ebbd`. Those original worktrees and their running evaluations remain unchanged.

`reference/manifest.json` maps every stored source/config to its original campaign path, restoration path, byte count, and SHA-256. Historical Python, shell, template, and configuration files have a `.source` suffix to prevent accidental imports or test collection. Large generated configurations are losslessly gzipped. These snapshots have not been mechanically reformatted or validated against current main; their exact bytes are the reproduction evidence. Upstream dependencies are represented by commit pins rather than copied vendor trees. Raw model weights, eval traces, credentials, and logs are excluded.

Restore all snapshots into a new directory:

```bash
uv run --no-project python experiments/sft_cold_start/restore_reference.py /tmp/sft-cold-start-reference
```

The restore command verifies every digest and refuses an existing destination. `training-worktree.patch.source.gz` is the complete tracked change from the training base. The restored `training-worktree/` directory contains the changed Python files and campaign modules; `artifacts/` contains the scripts and immutable launch configs. Existing absolute local paths and cluster/object-store paths are preserved as historical inputs. Do not execute restored launch, cleanup, upload, or watcher scripts without reviewing their pinned targets and the experiment policy.

Separate compact changes should carry reusable fixes to main after focused review and validation. In particular, the reproduced optimizer and precision settings differ from the original 10-05 run; this reference does not establish numerical or quality equivalence.
