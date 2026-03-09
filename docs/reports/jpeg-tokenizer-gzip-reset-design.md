# Gzip Reset Experiment Design

This note captures a narrow follow-up experiment for the JPEG tokenizer thread:

> if gzip is a bad tokenizer, is the core failure mode hidden rolling codec state?

The point is not to rescue gzip. The point is to make the failure mechanism more precise.

## Claim

The working claim is:

1. vanilla gzip bytes are a poor autoregressive token stream
2. a major reason is that byte semantics depend on hidden adaptive compression state
3. shortening that hidden state with frequent resets should make the stream more modelable
4. if that helps, it strengthens the argument against vanilla gzip as a tokenizer

That is a stronger claim than just "gzip loss is bad."

## Why Resets Matter

In a long gzip or deflate stream, local output bytes depend on a rolling compression history.
That means:

- the same local content can map to different bytes depending on distant prefix history
- small prediction errors can move the model off the compressor's implicit state manifold
- token meaning is not stable under prefix perturbation

Frequent resets should reduce that instability by making each region depend on a shorter prefix.
The tradeoff is worse compression.

That tradeoff is exactly the point of the experiment:

- if reset-heavy gzip becomes easier to model, then hidden adaptive state is doing much of the damage
- if it does not, then the problem is more intrinsic to compressed-byte tokenization

## Experimental Variants

Use one canonical image corpus and one canonical preprocessing recipe, exactly as in the JPEG runs.

The gzip ladder should be:

1. `gzip/vanilla`
   - one gzip member per image
   - no forced internal resets
2. `gzip/reset_<N>`
   - split the canonical image bytes into fixed chunks of `N` bytes
   - compress each chunk independently as its own gzip member
   - concatenate members into one valid gzip stream
3. `gzip/raw_chunks_<N>` optional control
   - uncompressed fixed-size byte chunks with EOS/PAD
   - useful for separating "compressed bytes are hard" from "chunking itself helps"

The practical way to implement resets is multi-member gzip, not trying to coerce the stock `gzip` CLI into periodic
dictionary flushes.

## Minimal Sweep

Keep the sweep small:

- `vanilla`
- `reset_256`
- `reset_1024`
- `reset_4096`

Those numbers are intended as chunk sizes in uncompressed source bytes before each fresh gzip member.

This is enough to test the qualitative mechanism:

- very frequent resets
- moderate resets
- weak resets
- no resets

## Comparison Targets

The gzip reset sweep should be compared against:

- canonical JPEG bytes
- `K=8` reference coefficients
- `K=8` exact libjpeg coefficients, if that baseline is healthy

That gives three distinct failure/success modes:

- raw adaptive compressed bytes
- container bytes with stable codec structure
- explicit transform-domain structure

## Metrics

The most useful metrics are:

- eval loss
- sequence-level bits per image
- rollout stability under autoregressive sampling
- prefix sensitivity under controlled perturbations

For this question, rollout and prefix sensitivity matter more than reconstruction.

The critical analysis is not "which stream compresses best?" It is:

- how stable is token meaning under small prefix changes?
- how much does shortening hidden codec state help?

## Expected Outcomes

The most informative possible outcomes are:

1. reset-heavy gzip improves substantially over vanilla gzip
   - supports the hidden-state-entanglement explanation
2. reset-heavy gzip improves a little but still trails JPEG bytes badly
   - suggests hidden state matters, but compressed bytes remain poor tokens even after mitigation
3. reset-heavy gzip barely changes
   - suggests the main issue is not long-range adaptive state alone

Outcome 1 is the cleanest support for the current thesis.

## Implementation Notes

Keep this intentionally simple:

- reuse the existing byte-window token-store machinery where possible
- version gzip variants explicitly in artifact paths and run names
- do not add a general-purpose compression-tokenizer framework

The smallest clean implementation is:

1. canonicalize image to bytes
2. compress canonical bytes with one of the gzip member policies above
3. tokenize resulting bytes with the existing fixed-window byte path

## Recommendation

Do not prioritize this ahead of the current byte and exact-JPEG baselines.

Once those baselines are stable, run the small gzip reset sweep as a mechanism test for the claim:

> gzip is a bad tokenizer because hidden rolling codec state makes token semantics unstable

If the reset sweep behaves as expected, it gives a much better explanation than just reporting that gzip loss is high.
