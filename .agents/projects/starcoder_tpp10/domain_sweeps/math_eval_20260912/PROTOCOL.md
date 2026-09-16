# FineMath proxy math likelihood

Evaluate the completed FineMath proxy curve before selecting more domains. This release scores eight existing small checkpoints and contains no training.

MATH-500 is primary and GSM8K secondary. Score every original reference solution conditioned on its problem, using the frozen TinyLlama32k tokenizer and plain `Question: ...\nAnswer: ...` format. Compute perplexity as the exponential of total solution-token negative log likelihood divided by the number of solution tokens. Keep the two evaluations separate. The prompt, BOS and padding receive zero loss; no EOS is appended. These are likelihood measurements, not generation accuracy.

Joint tokenization scores the token that spans the final prompt space and first solution character. Original GSM8K calculator markup (`<<...>>`) and the `####` answer delimiter are retained, so the secondary score includes those formatting tokens. No few-shot examples are used.

The full populations contain 500 MATH-500 and 1,319 GSM8K problems, with 112,697 and 173,832 scored solution tokens. The longest problem plus solution has 1,614 and 575 tokens respectively, so no example is truncated or windowed. The implementation also tests overlapping-window token coverage. No benchmark decontamination audit of the FineMath subset has been performed.

All seven FineMath matched proxies use N=16,587,008 parameters, D=165,937,152 training tokens, the same 10,485,760-token focus pool, one trainer seed and one subset. The shared zero-FineMath proxy supplies p=0; the remaining fractions are 5%,10%,20%,30%,50%,70%,100%. Use the archived allocator's realized epochs. The current upper range is 15.825 epochs. Report all points, the observed minimum and adjacent losses. A still-improving p=100% endpoint does not establish an interior optimum.

One regional CPU coordinator dispatches one TPU worker that evaluates the eight checkpoints sequentially and skips verified completed receipts on retry. Each checkpoint must reproduce its saved PALOMA token loss within 5e-5 before its math metrics are accepted. Source generations, compressed-content hashes, complete tokenized populations, code, tokenizer files and final checkpoint identities are recorded in spec.json. Checkpoint reads and computation stay in central1. Building the specification reads the two bounded evaluation objects locally (1.57MB compressed) to verify their actual schemas and scored populations.

The first evaluation `/calvinxu/tpp10-finemath-math-eval` failed because math examples and native PALOMA examples had different attention-mask tree structures. No result was accepted. The retry `/calvinxu/tpp10-finemath-math-eval-retry` gives each math example a single explicit segment, preserving causal attention while allowing native mixed batches. It adds unverified diagnostic receipts before the PALOMA gate. Checkpoints, scored token populations and evaluation objects are unchanged; attempt1/ preserves the submitted source and specification.

Eight focused tests pass, including native evaluation across a math/control batch boundary and a partial final batch; type checking, lint, regional launch safety and bundle verification pass. The broad shared-checkout test run reports 2,985 passed, 23 failed and 13 collection errors outside the new test module. Its output is `/tmp/tpp10_math_safe_tests.txt`.

Collect with `uv run python -m experiments.domain_phase_mix.analyze_tpp10_finemath_math --include-uncheatable --require-complete`. The collector retains checkpoint and Fieldbook identities, all raw metrics, observed minimum, boundary flag, second-best point and neighboring gaps. Results live under results/<spec_sha256>/; partial receipts cannot be mistaken for a complete grid.

Further proxy training requires discussion with CC and the user, followed by explicit approval. Target training remains deferred. Existing target cancellations and preserved checkpoints remain in force.
