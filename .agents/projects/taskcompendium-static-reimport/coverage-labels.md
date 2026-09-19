# Static NeMo coverage labels

These labels use the existing `coverage-tags.json` vocabulary. They describe the
semantic task in each `responses_create_params.input`; verifier answers,
expected fields, and other evaluator-visible metadata were not used.

- `nemo/mcqa/abb2b851-363c-54fa-890c-44d67c37c9b8`: A difficult multiple-choice legal question about disposition of remains and competing next-of-kin rights.
- `nemo/mcqa/0dfcf12c-bc99-535f-9c8c-6687a545f35d`: A biomedical multiple-choice question about implant-material biocompatibility; it requires specialized medical-science recall.
- `nemo/open_math/0`: A very long-digit divisibility construction problem, so number-theoretic reasoning and a numeric answer are central.
- `nemo/open_math/1`: A high-index radical recurrence requiring a nontrivial mathematical transformation and a formula response.
- `nemo/stack_math/0`: A geometric area/intersection problem involving a circular field and a rope length; the required result is a geometric formula.
- `nemo/stack_math/1`: A discrete counting problem with a convergent sequence of timed removals/additions; it is a moderate calculation task.
- `nemo/open_qa/ae045132-7beb-5f97-bf65-a27d7d86b998`: An open chemistry/instrumentation question interpreting heteronuclear NMR splitting and asking for an explanation.
- `nemo/open_qa/bd761808-13a7-5faf-9852-961422532fbe`: An open constitutional-criminal-procedure question requiring legal doctrine and an explanatory response.
- `nemo/science/c0d8dc56ce5d485bb989483047fa661c`: An elementary classical-mechanics question asking for the maximum-height formula for vertical motion.
- `nemo/reasoning_gym/d06d0d00-6b32-4ed1-9a73-1ee720d4aa42`: A simple needle-in-a-haystack lookup over many statements, with a name as the answer.
- `nemo/reasoning_gym/cf876b7f-0007-40a4-ab3a-f49fe8cdd8e2`: A small friendship-graph counting question; the input supports an easy relational/counting inference.
