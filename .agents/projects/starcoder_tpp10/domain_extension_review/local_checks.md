# Local checks for the domain-sweep review

11 September 2026. These checks support the scientific review; they do not certify new training code or completed caches.

- Recomputed N, D, TPP, available tokens, maximum epochs and training FLOPs from the frozen StarCoder design. `evidence.json` records exact values, archive rows and source hashes.
- Extracted all focal-domain rows for Wikipedia, FineMath-3+ and synthetic math from the corrected 39-bucket archive. Synthetic math's 14.486-epoch observed minimum belongs to OlmoBaseEval; its Uncheatable observed minimum is 7.243 epochs.
- Read GCS metadata only for the two Wikipedia shards and 128 FineMath-3+ parquet files in central1. `source_inventory.json` records the complete FineMath inventory and 16 prespecified selected files. No raw training payload was downloaded locally.
- Verified FineMath's existing download metadata names `HuggingFaceTB/finemath` at revision 8f233cf. `finemath_source_provenance.json` retains the small metadata documents and generations.
- Confirmed the FineMath catalog's tokenized handle always pins the legacy Llama3 cache. A fresh 32k-tokenizer preparation must use a new identity and validate tokenizer metadata. The new domain preparer and launcher are not implemented.
- Counted the existing pilot and refinement plan rows. The pilot has 7 target, 14 unmatched and 36 matched runs. The refinement adds 5 target, 10 unmatched and 30 matched runs. Two StarCoder matched points are missing from the proposed common grid; the initial maximum is therefore 16 new proxy runs.

A separate cold-reader pass read only `SPEC.md` and identified seven clarity gaps. The current specification defines realized epochs and the relative-mismatch denominator, supplies background weights and exact code/evaluation references, names the three focus domains explicitly, defines immediate-neighbor refinement and the final target grid, explains the historical archive labels, and expands the Figure 5 run-count accounting. `SPEC.reviewed_v1.md` preserves the initial draft sent to CC.

The cold-reader pass did not audit source code or certify scientific adequacy. CC's independent review is recorded separately. No new training or regional preparation was submitted during these checks.
