# Review: exact-checkpoint Uncheatable sidecar for the Delphi TPP40 regional bridge

Please perform a read-only, blocking scientific and operational review. The high-level goal is to decide whether an isolated four-row East5-v5p versus Europe-v6e bridge is safe to use as the acceptance gate before splitting the 280-row TPP40 production swarm across regions. Do not edit files.

## Frozen scientific contract

- Acceptance contract: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v2.json`
- Contract SHA-256: `c4cc0d476270bd2cfdceec87b59ce3d8c92ee2fa5844f6778f1e3920b6ecff36`
- Frozen rows: `2,120,240,260`
- Exact checkpoints: phase boundary `21855`, endpoint `27335`
- Both sides must materialize the seven English Uncheatable components at both checkpoints and native Table-9 at endpoint.
- Production remains blocked until all completion, numerical-tolerance, and unchanged-idempotent-rerun gates pass.

The Europe and East5 training bridge parents are already running or queued under an identical workspace bundle. Europe uses v6e-8; East5 uses v5p-8. The new sidecar deliberately evaluates both regions on v6e-8 so evaluator hardware does not confound the training-deployment comparison.

## Files to review

- Implementation: `experiments/domain_phase_mix/launch_delphi_tpp40_bridge_uncheatable_eval.py`
- Focused tests: `tests/test_delphi_tpp40_bridge_uncheatable_eval.py`
- Training launcher reconstructed by the sidecar: `experiments/domain_phase_mix/launch_delphi_augmented_swarm_tpp40.py`
- Shared training graph: `experiments/domain_phase_mix/launch_delphi_augmented_swarm_3e18.py`
- East5 exact command: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/east5_bridge_uncheatable_command_v1.txt`
- Europe exact command: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/europe_bridge_uncheatable_command_v1.txt`
- East5 training preflight: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/east5_bridge_preflight_v1.json`

## Local validation already completed

- Both regional dry runs succeeded and reconstructed exactly 8 cells: 4 rows times 2 checkpoints.
- East5 dry manifest SHA-256: `432f557859c68cb6528314c016fce96ca624ca0fdaee4fad08bce810547f8878`
- Europe dry manifest SHA-256: `41c7470cddc2919a9878e17303dab308f1a7f19b3488ace435e6bfdea2867a04`
- `uv run pytest -q tests/test_delphi_tpp40_bridge_uncheatable_eval.py tests/test_delphi_augmented_swarm_tpp40.py`: `27 passed`
- Pyrefly on the implementation and tests: `0 errors`
- Repository pre-commit checks on the implementation and tests: pass.
- East5 launch-safety: pass for parent `us-east5-a`, child `us-east5-b`, and `gs://marin-us-east5`.
- Europe launch-safety: pass for parent `europe-west4-b`, child `europe-west4-a`, and `gs://marin-eu-west4`.

## Questions requiring an explicit verdict

1. Does `_original_training_paths` reconstruct the exact already-running training outputs, or is there any version-affecting field missing or changed relative to the original launcher commands?
2. Does the evaluator restore exactly `checkpoints/step-21855` and `checkpoints/step-27335`, with adequate completeness and identity checks and no possibility of silently selecting another checkpoint?
3. Does the `LmDataConfig` evaluate exactly the seven frozen English Uncheatable validation caches, without training-data leakage or implicit cache construction?
4. Is computing the acceptance macro as the arithmetic mean of the seven component BPBs consistent with the frozen contract and Levanter's hierarchical `macro_bpb`?
5. Is evaluation comparable across sides: same model config, tokenizer, caches modulo region-local payload-equivalent paths, precision, batch size, mesh semantics, and v6e-8 hardware?
6. Is ready-only scheduling safe if phase-boundary evaluations launch while endpoint training is still in progress?
7. Is the idempotence contract sound: successful cells validate durable result identity and completeness; interrupted cells rerun; no completed cell can be mistaken for another row/checkpoint/region?
8. Do the exact Iris commands include every transitive source and frozen-contract file excluded by the workspace filters, and are their parent/child/data placements safe?
9. Identify any blocker that must be fixed before submitting either sidecar. Separate blockers from non-blocking polish.

Return `GO` only if both exact commands are safe to submit as ready-only acceptance sidecars. Otherwise return `NO-GO` and enumerate the minimal required fixes.
