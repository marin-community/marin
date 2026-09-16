# StarCoder WSD80 gradient plot completion v5 review brief

V4 passed source review but was never launched. Its mandatory historical-worktree smoke test exposed a provenance error before TPU spend:

- Fieldbook recorded the v10 jobs at clean commit `7efb96842624a2e8cbab36c9a9aa6b1cb68c4922` with `code_dirty=true`.
- The hash-pinned v6 freeze calls `materialized_config(..., artifact_cache=...)`, but clean `7efb` predates that API and cannot reconstruct any full configuration.
- The relevant dirty root/library changes were later committed as `377ad16d816a1726cc97396355607594910e9f0a` (`Support resumable state-probe continuations`). A detached worktree at `377ad`, overlaid with the hash-pinned experiment modules and frozen design artifacts, reconstructs all 256 frozen full configurations and passes all 5,888 field comparisons in `config_provenance.csv`.
- V4's GCS result root does not exist. `PRELAUNCH_PROVENANCE_FAILURE.md` records that it was never submitted and is non-consumable.

V5 changes only provenance anchoring and fresh identities:

1. Historical root lockfiles and `lib/*` now come from `377ad`, which retains JAX/JAXLIB 0.10.1 in the TPU lock and contains the exact config-materialization API required by the frozen v6 implementation.
2. V5 has a distinct release directory, release/schema/artifact versions, result root, table directory, plot directory, and CC-review path.
3. The immutable v4 release and its failure marker are hash-pinned by v5 alongside the existing v3 runtime-canary failure.
4. The analysis contract states that `7efb` was the recorded dirty-worktree commit and that `377ad` is the later committed form of the required library state. It does not claim that experiment overlays come from that commit.
5. Worker-only Python/JAX/JAXLIB/libtpu gates, complete package inventory, source-only final-state reproduction, staged environment baseline, materialization overlap A/B, endpoint blindness, and all 288 frozen row identities remain unchanged.

Please adversarially verify whether `377ad` is now the defensible executable provenance anchor, whether the 256-config/5,888-field reconstruction is sufficient prelaunch evidence, whether any numerical source changes between `7efb` and `377ad` create a new concern, and whether the fresh v5 release correctly makes v3 and v4 non-consumable. Inspect all v5 implementation files and the supersession markers. Return `PASS_AFTER_BLOCKERS_RESOLVED` as the exact first line only if no launch blocker remains; otherwise return `BLOCK` and enumerate blockers.
