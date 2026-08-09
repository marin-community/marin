# Grug GPU reverse-region recovery diagnostic

This artifact captures the actual `PRE_SCHEDULER` GPU HLO from an ordinary
one-layer Grug training step on one GB200. The run used Shuttle revision
`9779a6a9f1f12dc088977987606f5063d245fdf1`, JAX/JAXLIB 0.11.0, CUDA toolkit
13.3, and NVIDIA driver 595.71.05.

The original shared-input pair-Contract matcher found 68 Contracts and zero
pair-Map regions, so the typed-FFI replacement did not run. Inspection of the
captured HLO showed that XLA combined the saved gate/up projections into one
larger Contract. The reverse graph still contains a generic region with one
cotangent Contract followed by a two-output scalar Map. The generic
`recover_multi_output_contract_map_regions` analysis recovers that region
without consulting instruction names or frontend metadata.

This is a diagnostic failure artifact, not execution or performance evidence.

