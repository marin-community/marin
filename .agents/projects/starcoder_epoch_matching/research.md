# Research inputs for the finite-pool experiment

Effort: targeted source audit, 8 September 2026. The search stopped once the parent-pool identity, reusable observations, and required loader change were established. No training was run.

## Existing evidence

Current Figure 2A holds model size and token budget fixed while varying available StarCoder support. It therefore does not directly compare a small proxy with an observed larger target. The dense horizon-by-support experiment provides the larger-scale measurements needed for that comparison. See [the detailed reuse audit](target_reuse_audit.md) for pinned file hashes and each target observation.

The 7.408B-token curves with 1.12B and 2.24B StarCoder pools still minimize at p=1. Natural repetition alone does not guarantee a U-shape. C40 instead uses a 279,969,792-token pool and has its lowest interior observation at p=0.70. Choosing that existing curve is an exploratory design decision informed by previous outcomes, not a new prospective confirmation.

The earliest reuse proposal assumed the common data seed implied a common pool at all p. Source inspection disproved that at p=1: zero-weight components disappear before sequential shuffle keys are assigned. Interior StarCoder uses the seventh component key, whereas the old endpoint uses the first. A fixed named-component shuffle-key override preserves all interior data streams and fixes the new endpoint. The old endpoint is retained in the manifest only as an explicitly ineligible observation.

The parent corpus can be defined by the historical cache, chunking, shuffle algorithm, key and 1,068-batch prefix; it need not be copied into a new token cache. The matched proxy uses a 40-batch prefix of that ordered corpus. The newer name-based subset-seed API cannot recreate the historical pool simply by reusing the integer data seed. No new subset seed is used in this experiment.

## Prior work and limitations

[Zhou et al., Repetition Mismatch](https://arxiv.org/abs/2606.07597v2) compares shorter and target training horizons with and without repetition-controlled subsampling. It motivates evaluating mixture selection rather than expecting equal loss levels across scales. Its smaller-model results also show that repetition control can fail when the proxy is too weak. Our experiment is a controlled demonstration for the paper's setting; the epoch-aware surrogate remains a separate contribution.

The existing fixed-model ladder shows that increasing token budget can shift the optimum even after nominal epoch matching. The proposed matched proxy therefore has no guaranteed ordering relative to the unmatched proxy. The common parent and nested subset remove an avoidable source-identity difference, but conditioning on one subset leaves subset-selection bias unmeasured.

Echo search returned HTTP 403 during the preceding investigation. Local source, archived manifests and measurements were used instead. GCS authentication initially failed and was then restored during preparation. The subsequent live configuration, cache-offset and source-index checks are recorded in [review_packet.md](review_packet.md) and `historical_metadata/`.

## Reused code

- `experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py`: cache contract, historical model and support caps.
- `experiments/domain_phase_mix/launch_starcoder_wsd_80_20_surface.py`: source proportions, optimizer and phase alignment.
- `lib/levanter/src/levanter/data/text/datasets.py`: active-component filtering, shuffle-before-slice behavior, new named shuffle-key override.
- `experiments/domain_phase_mix/starcoder_epoch_matching.py`: self-contained, checksum-validated design and target eligibility.

The audit records source hashes because parts of the current working tree are uncommitted. Review is local; no PR or external message has been published.
