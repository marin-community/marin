# Delphi TPP40 Europe bridge: corrected phase-0 noise review

Review this as an independent scientific and launch-gate assessment. Do not edit files.

## Goal

We want to use otherwise idle Europe `v6e-8` capacity for the 280-row Delphi TPP40 swarm while preserving a defensible comparison with the canonical East5 `v5p-8` swarm. The user is willing to treat one passing matched trajectory as operationally sufficient, but we must not select a passing pair after observing outcomes or silently weaken the frozen gate.

## Frozen row-2 result

The preregistered primary bridge row is run order 2. At its phase-0 boundary:

- East5 unweighted seven-component Uncheatable macro: `1.0538958311080933`
- Europe macro: `1.0575664043426514`
- Europe minus East5: `+0.0036705732345581055`
- Frozen mean absolute delta threshold: `0.002`
- Frozen any-row threshold: `0.005`

Thus row 2 fails the binding `0.002` gate while remaining below `0.005`. The Europe endpoint is still training.

## Corrected noise evidence

The earlier review provisionally used the Table-9 run-noise estimate (`0.003419`) for Uncheatable. That assumption is false. The new deterministic audit is:

- Script: `experiments/domain_phase_mix/audit_delphi_tpp40_bridge_noise.py`
- Artifact: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_uncheatable_noise_audit_v1.json`
- Script SHA-256: `3a68d4084a9247d11c17af4e7c39d3282a96c3c800aab293d4148b2efec86fbd`
- Artifact SHA-256: `37b2da29ea745b8662fa87fc260508e0fee7e034fe6cbb47c2aa4029bc7fa23e`

The audit uses 1,864 clean historical Delphi endpoint rows after the existing 93-row data-quality exclusion. It groups rows sharing both 39-bucket phase mixtures exactly to 8 decimals and differing only by seed. There are 21 repeated-coordinate groups with sizes `{2: 7, 3: 9, 4: 2, 8: 1, 26: 2}`. The estimator is the median within-group sample standard deviation.

Measured single-run standard deviations:

- Exact seven-component bridge macro: `0.0006676525864158786`
- Token-weighted Uncheatable macro: `0.0006794914916926336`
- Table-9 macro: `0.003419356552672383`

Under independent trajectories, the bridge macro paired SD is `0.0009442033426628104`; the observed row-2 phase-0 delta is `3.8875` paired SDs. Under a zero systematic shift, the frozen `+-0.002` gate passes with probability about `0.9658`. Under a true Europe shift of `+0.003`, it passes with probability about `0.1448`.

The large macro discrepancy is concentrated in BBC News (`+0.01222`) and Wikipedia English (`+0.00881`); code components are much closer. Exact component values and historical SDs are in the artifact.

## Supplementary rows

The original four-row bridge contract selected run orders `2,120,240,260` before outcomes. Europe phase-0 checkpoints already exist for 120 and 260; row 240 has resumable temporary state but no committed boundary checkpoint. The old four-row parent was cancelled after row 2 reached phase 0, freeing three Europe slices. A dry-run-validated resume command for the three supplementary rows exists at:

`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/europe_bridge_supplementary_resume_command_v1.txt`

SHA-256: `85293ca4957c1ac5e461570ea5bebe0a6ebfe1314f361f67b093eace33a2a878`

It passed Europe launch safety and reconstructs exactly run orders 120, 240, and 260. It has not been submitted. No additional cross-region checkpoint transfer is authorized or planned.

## Questions

1. Given the corrected Uncheatable noise, should row 2's phase-0 failure be treated as evidence of real cross-accelerator training drift rather than an underpowered gate?
2. Is it scientifically defensible to restore the three originally selected supplementary rows and evaluate a pooled four-row contract, or does the row-2 primary failure already make Europe production a no-go?
3. If supplementary rows are run, specify the decision rule that avoids selecting whichever pair happens to pass. Should the original pooled signed mean and any-row thresholds remain binding?
4. Is there any defensible condition under which production can launch before supplementary evidence completes, given the user's goal of maximizing available compute?
5. Identify any flaw in the new noise audit, especially the repeat grouping, macro definition, use of a median within-group SD, independence assumption, or interpretation of a 3.89 paired-SD discrepancy.

Return a concise verdict with blockers, recommended next action, and the exact evidence that would authorize or reject the Europe production shard.
