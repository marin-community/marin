# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared contract criteria; individual blend leaves add their source-specific scope."""

FAMILY_CRITERIA = {
    "swe-repo": (
        (
            "The public repository/environment reference supplies code context; a pinned checkout and "
            "supplied issue can be coherent without inline repository files."
        ),
        (
            "Compare expected_action, ref_patch, issue, historical observations and environment to "
            "detect unrelated hidden repair requirements. Preserve SWE-Gym versus SWE-rebench "
            "attribution; a shared agent selector is not proof of source equivalence."
        ),
    ),
    "agentic-safety": (
        (
            "Preserve trusted instructions, tool schemas, full tool observations, initial environment, "
            "and attacker injection boundaries."
        ),
        (
            "Injected instructions are intentional untrusted observations; compare the requested "
            "legitimate objective with verifier_config without treating the injection as authoritative "
            "or exposing hidden evaluator goals."
        ),
    ),
    "qa-abstention": (
        "Check whether the actual question is answerable, and whether the private answer is correct.",
        (
            "Abstention policy and any [IDK] output requirements belong to the source contract; do not "
            "substitute exact-only matching for its semantic evaluator."
        ),
    ),
    "instruction-following": (
        (
            "Identify the underlying content request and verify that all supplied formal constraints "
            "and semantic rubric requirements can hold together."
        ),
        (
            "Preserve every conversation turn. Public schemas/examples are legitimate context. "
            "Distinguish factual extraction from authorized arbitrary schema generation, and compare "
            "private constraints with public instructions."
        ),
    ),
    "competitive-programming": (
        ("Check complete input/output definitions, boundaries, examples, and consistency with " "retained unit_tests."),
        (
            "Special judges, alternative valid constructions, and function versus stdio delivery must "
            "retain their source contracts. No reference solution or unavailable execution alone is a "
            "quality defect."
        ),
    ),
    "safety": (
        (
            "Judge whether the actual public request and source response_policy_mapped define a "
            "coherent response objective."
        ),
        (
            "Adversarial or jailbreak text is intentional task input; assess policy/reference "
            "contradictions and impossible instructions rather than treating adversarial wording itself"
            " as corruption."
        ),
    ),
    "math-proof": (
        (
            "Check that the complete Lean header, formal_statement, imports, and holes to be filled are"
            " present or available through the stated environment."
        ),
        (
            "Hard proofs and absent reference proofs alone are not defects. Verify the formal target "
            "agrees with informal text and preserve exact Lean/toolchain requirements."
        ),
    ),
    "math-answer": (
        (
            "Verify complete mathematical inputs and agreement of expected_answer with the actual "
            "public problem; difficulty alone is not a defect."
        ),
        (
            "The source can require symbolic, approximate, or judge-assisted scoring. A single stored "
            "expression is evidence, not authority to reject equivalent answers. Unresolved external "
            "question placeholders are acquisition gaps."
        ),
    ),
    "arc-agi": (
        (
            "Check that all training grids, public test inputs, and private expected_output match the "
            "stated grid transformation and dimensions."
        ),
        (
            "Inductive variants require producing a reusable transformation program; transductive "
            "variants request the output grid. Do not replace one grading contract with the other or "
            "expose hidden outputs."
        ),
    ),
    "chemistry": (
        (
            "Check public molecular inputs and requested properties against retained target/validator "
            "fields and their units or format."
        ),
        (
            "RDKit validity, stereochemistry, equivalence, and numerical tolerances belong to the "
            "original evaluator. Its absent runtime is distinct from a malformed or contradictory "
            "chemistry task."
        ),
    ),
    "reasoning-gym": (
        (
            "Compare the complete question and private answer/metadata, checking cheap contradictions "
            "and missing puzzle context."
        ),
        (
            "The source_dataset can determine scoring, aliases and partial credit. A hard puzzle or "
            "several surface forms of the same answer is not automatically a defect."
        ),
    ),
    "qa-multiple-choice": (
        "Check option labels and answer encoding against the complete public choices and expected_answer.",
        (
            "Knowledge questions can use ordinary external knowledge. Missing referenced passages, "
            "images, or material contradictions are defects; source labels must not be exposed as "
            "public hints."
        ),
    ),
    "tool-use": (
        (
            "Check the complete role/tool sequence and advertised schemas against expected_action, "
            "scenario, and source environment state."
        ),
        (
            "Historical observations are public context. Expected future tool calls and reward state "
            "are private evidence; multiple valid actions require the original comparison policy rather"
            " than invented exact matching."
        ),
    ),
}
