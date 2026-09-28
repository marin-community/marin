## Fix check

The segment-ID fix is right. `make_example` now passes an all-zero `segment_ids` (`evaluate_tpp10_finemath_math.py:94`), which routes through `LmExample.causal`'s `elif segment_ids is not None` branch so both populations carry the same `attn_mask` tree structure, and a single segment adds no attention boundary — the new test materializes it against `np.tril` and the 106.0/114.0 expectations correctly isolate per-tag weighted means across both the dataset boundary and the partial final batch. Tail padding stays inert: it shares segment 0, but causal ordering keeps it unreachable from every scored position.

The other five items land as described: diagnostic receipt written before the PALOMA gate (`:316-319`), pins extended to `dataset.py`/`loader.py`/`attention_mask.py`, the p0 control selected by unique name with a `len(controls) != 1` guard (`:156-159`), `mask[0] != 0` rejected at ingest (`:119`), and `boundary_policy` recorded in the spec with the test now asserting `decode(selected) == solution` plus the leading `▁` rather than hiding it behind `.strip()`. Only leftovers: `--build` still reads regional objects without `require_central1()`, and `code_pins` still uses `.resolve()` where `original.code_pins()` does not. Both minor; the collector is already in flight.

## Corrections accepted

I was wrong that the 14.486 minima were censored — the erratum CSV carries 28.971 dose points, and all three triples turn up there, so those are genuine interior optima on table9 macro. And Nemotron-Math-Textbooks is a plain `text` column (1M rows, 2.39 GB in part_000000), so it needs a prefix allowlist edit in `parquet_documents`, not a format redesign. I withdraw the p30 prediction for FineMath-4+.

## Shortlist

The two axes now point different ways, which is the useful part.

**Under common Uncheatable**, nothing archived exceeds 7.243, and the one domain with both archived and in-design numbers moved 7.243 → 4.77 (FineMath-3+). A single point, but the only cross-design anchor, and it reads *lower* here. So >10 on Uncheatable (p≥70) is unsupported for every candidate, and I would not let it drive selection. Separated optima are achievable; >10 should be a nice-to-have that gets labelled honestly if it fails to appear.

**Under MATH-500/GSM8K solution likelihood**, the ordering should be near-inverted, and that disagreement is the illustration.

1. **Nemotron Math Textbooks** — best prospect on the math diagnostic (textbook prose with worked solutions), now cheap to plumb. Uncheatable prior is indirect (Dolmino synth_math, different corpus), but the interior 14.486 optimum is real evidence that curated synthetic math is a high-tolerance class on at least one common eval.
2. **Stack-Edu Python** — highest archived Uncheatable tier and by far the lowest level, and expected to be the weak member on math likelihood. Caveat unchanged: 2/7 Uncheatable components are GitHub and the retained validation is PALOMA programming languages, so its high optimum is partly eval composition.
3. **ProofPile-2 arXiv** — ample payload; archived 7.243; eval-aligned via the two arXiv components. Solid backup, but theorem prose rather than worked solutions makes it a likely mid performer on the math diagnostic, so it separates less.

Recommend proposing **Nemotron × Stack-Edu Python** on a stated rule — one math-native, one code-native, both at the archive's top Uncheatable tier — which is 14 proxy-only points reusing p0. Waiting for the FineMath math results first is right: if the diagnostic is flat in p, the pair is not worth running.
