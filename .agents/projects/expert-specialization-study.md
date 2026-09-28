# Expert specialization study (Ladder Climb v1, #9451)

Requested by the project lead on 2026-09-28: characterize what routed experts specialize in, measure how much
specialization can be seeded, then test whether tying a router column to a token embedding from step 1 produces
the expected specialization.

## Setup (tractable model)

- Candidate 9 recipe at d512 with **top-1 of 8 routed experts per layer** (`num_experts=8`,
  `num_experts_per_token=1`). The shared expert and everything else stay as in candidate 9.
- **Expert width = 8 × 384 = 3072**, so active routed compute matches candidate 9. Total routed params drop 8×.
- This deliberately breaks the "8 active experts" rule: it is an analysis model, not a candidate.
- 2817 steps, 2 seeds, so specialization found on both seeds can be told apart from seed noise.

## Phase 1: measurement tooling

`routing_stats` dump: at chosen steps (e.g. 250, 1000, 2817), run the model forward on a fixed set of held-out
batches and accumulate `counts[layer, token_id, expert]`: the number of times the current token was routed to
each expert. Also `counts_prev[layer, prev_token_id, expert]`, since routing may key on context rather than
the current token. Save as `.npz` next to the run output, then analyze locally:

- **Per layer:**
  - mutual information I(token; expert), normalized by H(expert);
  - expert load;
  - per-token routing entropy.
- **Per expert:**
  - the top tokens by lift, P(expert | token) / P(expert), decoded with the 16k tokenizer;
  - coverage by token class (whitespace/punctuation, digits, subword continuations vs word starts, common
    function words).
- **Across seeds and over time:**
  - whether the specializations match across seeds (Hungarian-matched expert similarity over token
    distributions);
  - when specialization emerges (the dump steps).

## Phase 2: characterization runs

- Top-1-of-8 baseline, seeds 0 and 1, with dumps at 250 / 1000 / 2817.
- Scale up after trends are visible: top-1 of 32, then top-2 of 64, then candidate 9 itself (top-8 of 512), to
  see which trends survive.

## Phase 3: seeding

- **Router–embedding tie:** force router column `e` (layer `l`, or all layers) to equal
  `α · token_embed[v]` for a chosen vocab token `v`, from step 1. It is the same parameter, so gradients flow
  into the embedding. Choose `v` from the Phase 1 specializations (e.g. an expert that already loves digits:
  tie it to a digit token) and also an arbitrary token as a control.
- Measure whether expert `e`'s P(e | v) and its specialization class grow relative to the untied baseline, and
  whether the loss changes.
- **Follow-ons:**
  - tie several experts to several tokens;
  - tie to class centroids (mean embedding of a token class);
  - seed with router-bias priors instead of ties;
  - release the tie after N steps and see whether the specialization persists.

## Side task (same request)

Gather-based per-expert read slices: each expert gathers its own 256 channels into a real `[E, 256, I]` `w_up`
with a per-slice learnable norm (the fair version of batch 104, whose masked implementation cost ≈ +0.016 by
itself in the batch 104 control).
