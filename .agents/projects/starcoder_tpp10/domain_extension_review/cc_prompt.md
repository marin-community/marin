Review this proposed experiment before training is submitted. Work read-only; do not edit files, launch jobs, poll running experiments, or spawn agents.

The complete scientific specification is:
/Users/calvinxu/Projects/Work/Marin/marin/.agents/projects/starcoder_tpp10/domain_extension_review/SPEC.md

Read that file first, then its sibling evidence.json. Check the claims against these existing implementation references as needed:
- experiments/domain_phase_mix/starcoder_tpp10_assets/design.json
- experiments/domain_phase_mix/starcoder_tpp10.py
- experiments/domain_phase_mix/prepare_starcoder_tpp10.py
- experiments/domain_phase_mix/launch_starcoder_tpp10.py
- experiments/domain_phase_mix/evaluate_starcoder_tpp10_uncheatable.py
- experiments/datasets/midtraining.py
- .agents/projects/starcoder_tpp10/design.md
- .agents/projects/starcoder_tpp10/zero_weight_stream_audit.json
- .agents/projects/starcoder_tpp10/live/pilot_plan.json

All paths above are relative to /Users/calvinxu/Projects/Work/Marin/marin unless absolute. This is a scientific setup and feasibility review. The Wikipedia/FineMath builders and generalized launcher are not implemented yet, so distinguish sound design from readiness to submit.

User requirements:
1. Cheap initial sweeps for Wikipedia and FineMath-3+, comparable to the existing StarCoder matched proxy.
2. Matched tokens per TOTAL trainable parameter and matched actual focus-domain epochs between proxy and eventual target.
3. An optional unmatched proxy that does not repeat its finite focus pool at any swept fraction.
4. All focus-domain curves optimize the same Uncheatable evaluation.
5. Freeze prior downsampling and the eventual target now, so later target expansion requires no scientific redesign or changed pool membership.
6. Assess whether this provides the intended evidence concerning different buckets' repetition preferences and the limits of a fixed epoch cap. Distinguish an upper cap from a shared preferred epoch count. Do not approve an unsupported claim merely because the specification already cautions about it.

Independently examine scientific identifiability, loss tradeoffs from replacing web tokens, model capability at 16.6M/TPP10, range and grid, subset/source sampling, tokenizer and legacy-cache hazards, evaluation population, fair reuse of StarCoder replicates and p=0, nominal versus actual allocator exposure, missing target-grid costs, and decision rules for flat or boundary curves. Keep suggestions proportionate to an illustrative one-seed pilot. Domain choice used exploratory historical data; do not treat the archived optima as guarantees in this setting.

Return a complete review with:
- Verdict: whether the setup answers the stated narrow question, whether it supports the intended cap claim, and whether it is ready for training submission.
- Blocking scientific changes, if any, with concrete reasons and exact fixes.
- Implementation gates that must be closed before submission, separately from scientific flaws.
- Optional improvements ranked by value and cost.
- Source paths and line numbers for concrete code findings; check the numeric accounting.

Do not demand new expensive target sweeps solely to approve an exploratory proxy stage. Identify what evidence would be needed for stronger claims. Prefer specific falsifiable checks over generic checklists. Do not write or submit anything. Your final answer will be saved as the review artifact.
