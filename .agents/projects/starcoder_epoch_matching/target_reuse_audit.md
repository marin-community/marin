# Reusing the existing target curve

Audit date: 2026-09-08. Read-only source and archived-outcome audit; no jobs launched.

Follow-up: GCS access was restored later in this preparation session. The configuration, child environment, native final metric, actual packed-cache length and archived/current index mappings are now verified in [verified_historical_config.md](verified_historical_config.md). Its findings supersede the unresolved live checks recorded below. The exact launch Git SHA remains unavailable.

C40 can provide 25 of its 26 observations for a constructed finite-corpus target. Its interior points share a StarCoder support of 279,969,792 tokens. Its pure-StarCoder endpoint uses a different shuffle key and therefore a different support. Reusing the complete curve requires replacing that endpoint or explicitly restricting the target grid. The current support-seed API cannot reproduce the interior support by setting its seed to 20260711: it uses a different key derivation.

This corrects the earlier informal statement that every C40 point shared one pool. A common training seed was insufficient to establish that claim.

## Evidence and limits

The target is `dense_replay__r3_increase_d_h0640_s28260__m100__endpoint`, labeled C40 by the September 2 atlas. Its 26 observed values agree with the archived coverage CSV and frozen design. Every primary row uses data seed 20260711, tied mixture weights in both phases, and final metric step 28259. Two nonwrapping rows reuse full-pool runs. The table below records the actual training identities.

The frozen design pins StarCoder to `gs://marin-us-central1/tokenized/dolma/starcoder-8b6089`: 206,640,114 documents, 49 shards, consolidated layout, and Llama-3.1 tokenizer metadata. It records 216,567,300,822 source tokens as historical registry provenance. This audit did not recount that source. The finite support is exactly 1068 initial batches, or 136,704 packed sequences of 2048 tokens. See the [frozen design](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json) and [cache validation](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py:160).

The observed curve informed this choice of target regime. It is development evidence, not an independently selected prospective target. Its minimum at p=0.7 is a grid minimum from one primary seed. The primary metric is `eval/paloma/dolma_100_programing_languages-llama3/bpb`. Paloma and Uncheatable are validation handles assigned zero training weight; this configuration fact does not establish corpus-level absence of overlap. See [validation handles](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py:550) and [weight assignment](/Users/calvinxu/Projects/Work/Marin/marin/lib/marin/src/marin/experiment/data.py:308).

A single small request for the persisted executor config failed because GCloud credentials require reauthentication. No alternate authentication or repeated fetch was attempted. The local archives establish the outcomes; a fresh durable-config and source-cache check remains necessary before reconstructing the target inputs.

## Training configuration

| Quantity | Value |
| --- | --- |
| Model | Qwen3 configuration with Llama-3.1 tokenizer |
| Total / nonembedding parameters | 210,052,480 / 45,884,800 |
| Hidden / MLP dimensions | 640 / 2560 |
| Layers / attention heads / KV heads | 7 / 5 / 5 |
| Sequence length / batch size | 2048 / 128 sequences |
| Steps / materialized tokens | 28,260 / 7,408,189,440 |
| Phase boundary | Step 22,608; tied mixture weights |
| WSD warmup / decay | 282 / 5652 steps; cosine decay, minimum LR ratio 0 |
| Muon / Adam learning rates | 0.02 / 0.008 |
| Weight decay / clipping norm | 0.1 / 1.0 |
| Momentum / Adam beta1 / Adam beta2 | 0.95 / 0.9 / 0.98 |
| Adam epsilon / Muon epsilon | 1e-15 / 1e-5 |
| Muon backend / Nesterov | 5 steps, quintic coefficients / enabled |
| Trainer and data seed | 20260711 |
| StarCoder parent P | 279,969,792 tokens |
| Nominal target epochs at p=1, D/P | 26.4606741573 |
| Broad data | Full Nemotron caches, six fixed relative component weights |

Model and optimizer values were instantiated from current source and compared with the frozen manifest where recorded. They have not all been checked against the persisted training config. The [model builder](/Users/calvinxu/Projects/Work/Marin/marin/experiments/scaling_law_sweeps/completed_adamh.py:133) produces a seven-layer model; the [launcher](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py:508) validates both parameter counts. See [optimizer construction](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/launch_starcoder_wsd_80_20_surface.py:202). The frozen manifest records training JAX 0.10.1, NumPy 2.3.5, Threefry2x32 and x64 disabled. Its planning environment used JAX 0.11.0. Current-source instantiation does not prove equality to every launch-time default.

## Support identity

The historical path sets `data_key=PRNGKey(20260711)`, splits it into mixture and dataset-shuffle keys, and assigns successive `key_iterator` keys to the active datasets. Dataset construction removes components with zero weight in all phases before assigning those keys. See [data seed override](/Users/calvinxu/Projects/Work/Marin/marin/lib/levanter/src/levanter/main/train_lm.py:281), [active component filter](/Users/calvinxu/Projects/Work/Marin/marin/lib/levanter/src/levanter/data/text/datasets.py:1135), [dataset split](/Users/calvinxu/Projects/Work/Marin/marin/lib/levanter/src/levanter/data/text/datasets.py:1152), and [key iterator](/Users/calvinxu/Projects/Work/Marin/marin/lib/levanter/src/levanter/utils/jax_utils.py:333).

For every interior point, six Nemotron components precede StarCoder. StarCoder receives the seventh key. At p=1 the broad components disappear, and StarCoder receives the first key. The contemporaneous commit `eafa4d49f7c55fbf2abb26b5d92c1ac7d093f9fb` has the same zero-weight filtering and shuffle-before-slice order in `lib/levanter/src/levanter/data/text/datasets.py` (lines 705–708 and 782–846). This is a contemporaneous source comparison, not recovery of the precise training checkout.

| Component | Cache suffix below gs://marin-us-central1 | Interior shuffle key |
| --- | --- | --- |
| nemotron_cc/hq_actual-llama3 | tokenized/nemotron_cc/hq_actual-5af4cc | [4181282030, 3509741613] |
| nemotron_cc/hq_synth-llama3 | tokenized/nemotron_cc/hq_synth-3525e2 | [1692708717, 814235673] |
| nemotron_cc/medium_high-llama3 | tokenized/nemotron_cc/medium_high-d21701 | [838510724, 1386979079] |
| nemotron_cc/medium-llama3 | tokenized/nemotron_cc/medium-d86506 | [3430063960, 3470476155] |
| nemotron_cc/medium_low-llama3 | tokenized/nemotron_cc/medium_low-0fdb07 | [34974657, 2948256024] |
| nemotron_cc/low_actual-llama3 | tokenized/nemotron_cc/low_actual-cb3f2c | [1427242644, 385010521] |
| dolma/starcoder | tokenized/dolma/starcoder-8b6089 | [898005854, 446240491] |

At p=1 StarCoder instead receives `[4181282030, 3509741613]`. At p=0 there is no StarCoder stream. The first positive point p=0.0364194347695976 is a valid no-wrap alias to a full-pool run with the interior seven-component ordering.

The shuffle is `BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")` in both inspected versions. The finite support is the first 136,704 positions after shuffling the complete packed-sequence dataset. Full dataset length, packed-sequence construction, component order, key, block-shuffle mapping and prefix length all define its identity. This is a fixed sequence subset, not a new independent document sample.

Calibration seeds 20260811–20260813 change the support together with trainer and data-order randomness ([launcher](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py:492)). They are not trainer-only replicates on the reference parent.

## Construction needed for reuse

1. Freeze the interior legacy StarCoder support with its explicit uint32 key, exact source cache, and 136,704 sequence positions. New training seeds must not change its membership. Keep the six Nemotron caches and their relative weights fixed.
2. Both proxies use that parent. The unmodified proxy uses all parent sequences; the matched proxy uses a nested subset with size approximately `(D_proxy / D_target) * P`. Select membership independently of mixture p and apply new run-order randomization only after membership is fixed.
3. Require `D_proxy <= P` to avoid exhausting StarCoder at every p, including p=1. Audit actual integer mixture counts because nominal p times D need not equal consumed source tokens exactly.
4. `max_train_batches_subset_seed=20260711` defines a different pool. It uses a [name-based folded key](/Users/calvinxu/Projects/Work/Marin/marin/lib/levanter/src/levanter/data/text/datasets.py:648), then a separate [run-order shuffle](/Users/calvinxu/Projects/Work/Marin/marin/lib/levanter/src/levanter/data/text/datasets.py:1290). Reuse requires a preselected dataset, an explicit legacy-key fixture, or a narrowly implemented explicit-key selection path.
5. Compare source sequence-index mappings from the old and new constructions before launching. Include prefix ends, block boundaries, all matched-subset members, p=0 and p=1, and multiple new training seeds. Matching keys alone does not verify dataset identity. Obtain packed dataset length from the actual cache, not solely the registry token count.
6. Replace the p=1 target observation under the interior parent, or omit it and state the retained target range. Existing p=0.9 already exceeds the p=0.7 loss by 0.0158483 BPB. The old endpoint should not enter fixed-parent target regret.
7. Verify a persisted target training config, source-cache metadata and original build identity when authentication is restored. This concerns reconstruction of inputs and runtime; archived endpoint outcomes are already available locally.

Batch-rounded supports cannot generally match epochs exactly here. Target steps/support batches are 28260/1068, with gcd 12. The smallest positive integer matching pair is 2355 proxy steps and 89 support batches, exceeding the no-repeat proxy limit of 1068 steps at batch size 128. Record rounding error or use a sequence-level support bound. Do not label rounded ratios exact.

Keeping N fixed makes this a token-horizon transfer experiment; the proxy has lower tokens per parameter. Existing curves already show optimum movement with D under matched nominal epochs. Better target selection is testable; identical optima or coincident absolute-loss curves are not assured.

## Archived observations

These are observed BPB, not surrogate predictions. Durable final metrics follow `gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_dense_support_surfaces_20260808/<actual_run>/2026.07.11/checkpoints/eval_metrics.jsonl`.

| p | BPB | Actual training run | Same interior parent |
| --- | --- | --- | --- |
| 0 | 1.573216438 | dss_r3d28260_full_c000_s0711 | Yes |
| 0.0364194347695976 | 1.021322966 | dss_r3d28260_full_c028_s0711 | Yes |
| 0.05 | 0.991811931 | dss_r3d28260_m100_c030_s0711 | Yes |
| 0.1 | 0.913295865 | dss_r3d28260_m100_c032_s0711 | Yes |
| 0.140704042401089 | 0.883577168 | dss_r3d28260_m100_c036_s0711 | Yes |
| 0.15 | 0.876994610 | dss_r3d28260_m100_c038_s0711 | Yes |
| 0.17005 | 0.869808197 | dss_r3d28260_m100_c039_s0711 | Yes |
| 0.18 | 0.865430176 | dss_r3d28260_m100_c041_s0711 | Yes |
| 0.2 | 0.851812661 | dss_r3d28260_m100_c043_s0711 | Yes |
| 0.2405 | 0.837174237 | dss_r3d28260_m100_c048_s0711 | Yes |
| 0.25 | 0.837680519 | dss_r3d28260_m100_c049_s0711 | Yes |
| 0.26 | 0.834467411 | dss_r3d28260_m100_c050_s0711 | Yes |
| 0.3 | 0.821392596 | dss_r3d28260_m100_c057_s0711 | Yes |
| 0.35 | 0.813608587 | dss_r3d28260_m100_c073_s0711 | Yes |
| 0.4 | 0.803918839 | dss_r3d28260_m100_c079_s0711 | Yes |
| 0.46 | 0.798272848 | dss_r3d28260_m100_c085_s0711 | Yes |
| 0.5 | 0.798245668 | dss_r3d28260_m100_c088_s0711 | Yes |
| 0.54 | 0.794552743 | dss_r3d28260_m100_c094_s0711 | Yes |
| 0.55 | 0.792759299 | dss_r3d28260_m100_c096_s0711 | Yes |
| 0.6 | 0.792351246 | dss_r3d28260_m100_c100_s0711 | Yes |
| 0.65 | 0.789860845 | dss_r3d28260_m100_c103_s0711 | Yes |
| 0.7 | 0.788042903 | dss_r3d28260_m100_c109_s0711 | Yes |
| 0.75 | 0.792149603 | dss_r3d28260_m100_c112_s0711 | Yes |
| 0.8 | 0.793210328 | dss_r3d28260_m100_c116_s0711 | Yes |
| 0.9 | 0.803891242 | dss_r3d28260_m100_c119_s0711 | Yes |
| 1 | 0.834336638 | dss_r3d28260_m100_c124_s0711 | No: endpoint key differs |

## Input identities

Local SHA-256 values identify the files inspected; they do not certify remote object generations or the original launch checkout.

| Input | SHA-256 |
| --- | --- |
| `experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json` | `ca06420ec7c46379463091bdd55c5f720910ac38b46a0f37f08545ea9966ecbe` |
| `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_all_tied_curves_canonical_dsp_20260902/predictions.csv` | `a7b8a767ff4b2a1bcbc72d49806eabc53d8599673599455dec73fd6919b2636b` |
| `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_dense_support_calibration_results_20260813/coverage_with_calibration_weights.csv` | `949d0dbc41a3826acbfc185987f41fecdfe74ea7d5003411f56a1974bab1ae79` |
| `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_single_phase_curve_inventory_20260902/curve_memberships.csv` | `97bbe6afc52858a670fee89af66b4edbf7a82e8528b270a01019087f5d7e54ba` |
| `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_single_phase_curve_inventory_20260902/target_observations.csv` | `024a06335507f4322fb840f5969b430bf863f8cb9cc7f3459271567c51417dda` |
| `experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py` | `c42f3919079f13f72bb1181afa5e72b9c5acc0834a30a1b65322a034f36f02fc` |
| `experiments/domain_phase_mix/launch_starcoder_wsd_80_20_surface.py` | `4334978b7c6fbe8c33c6af34806d33621655ca2082a4f1c08533ab8a7bc1d468` |
| `experiments/scaling_law_sweeps/completed_adamh.py` | `f3b5d4f59de0887e13ab51e98894d253c2811dd77ff7c6b7f591555be93536fb` |
| `lib/levanter/src/levanter/data/text/datasets.py` | `c3ea1ea1f03ac5b96f81c208ce81fd50d5ebd9a74c57cb31ec02619fb4188948` |
| `lib/levanter/src/levanter/main/train_lm.py` | `3d1fd8e1238b2acb829649a193f4bc9e3c4eccc10cc2cbc0bef2fbc6428e8f76` |
| `lib/levanter/src/levanter/utils/jax_utils.py` | `fadbfc228b71898d477268cde932cc0e6ef6b10e03b30d9fd01d46e20a62021a` |
| `lib/marin/src/marin/experiment/data.py` | `0617d49b9209a1dc76afe0de36af8d8c8f6693dfd3730c09c548461a9678c023` |

Frozen design canonical self-hash: `d4ffb9079f969af808230c623555315262cb314434a21db6d36e9651b747cd48`. Audit checkout HEAD: `5548443ef6c67c0c9c23c6c7f088e205ee03bfde`. Working files may contain concurrent changes; per-file hashes identify this audit's inputs.
