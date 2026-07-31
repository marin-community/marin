# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

from experiments.rollout_data.generate_identity_conversation_seeds import (
    MARIN_IDENTITY_PROFILE,
    Facet,
    IdentityProfile,
    generate_seeds,
    main,
)


def test_generate_seeds_is_deterministic_and_balances_facets():
    first = list(generate_seeds(count=200, random_seed=17, identity_profile=MARIN_IDENTITY_PROFILE))
    second = list(generate_seeds(count=200, random_seed=17, identity_profile=MARIN_IDENTITY_PROFILE))

    assert first == second
    assert {seed.facet for seed in first} == set(Facet)
    assert len({seed.seed_id for seed in first}) == 200


def test_generate_seeds_separates_known_and_unknown_identity_facts():
    seeds = list(generate_seeds(count=400, random_seed=9, identity_profile=MARIN_IDENTITY_PROFILE))

    known = [seed for seed in seeds if seed.facet == Facet.TRAINED_BY]
    unknown = [seed for seed in seeds if seed.facet == Facet.UNKNOWN_FACT]
    open_weight_data = [seed for seed in seeds if seed.facet == Facet.OPEN_WEIGHT_DATA]
    false_premises = [seed for seed in seeds if seed.facet == Facet.FALSE_PREMISE]

    assert known
    assert all("Marin Community" in seed.canonical_answer for seed in known)
    assert unknown
    assert all(seed.truth_status == "unknown" for seed in unknown)
    assert all("not known" in seed.canonical_answer for seed in unknown)
    assert open_weight_data
    assert all("DeepSeek" in seed.canonical_answer for seed in open_weight_data)
    assert all("Kimi" in seed.canonical_answer for seed in open_weight_data)
    assert all("GLM" in seed.canonical_answer for seed in open_weight_data)
    assert false_premises
    assert all(seed.distractor_model for seed in false_premises)
    assert all(
        seed.distractor_model is not None and seed.distractor_model in seed.generation_prompt for seed in false_premises
    )


def test_generate_seeds_uses_only_facts_from_custom_identity_profile():
    profile = IdentityProfile(
        profile_id="example",
        model_name="Example Model",
        developer="Example Developers",
        trained_by="Example Trainers",
        maintained_by="Example Maintainers",
        ecosystem_contributors=("Example Upstream",),
        training_data_models=("Example Teacher",),
        unknown_facts=("the exact release date",),
    )

    seeds = list(generate_seeds(count=400, random_seed=11, identity_profile=profile))
    profile_prompts = [seed.generation_prompt for seed in seeds if seed.facet != Facet.FALSE_PREMISE]

    assert all("Marin" not in prompt for prompt in profile_prompts)
    assert all("Stanford" not in prompt for prompt in profile_prompts)
    assert all("NVIDIA" not in prompt for prompt in profile_prompts)
    assert all("DeepSeek" not in prompt for prompt in profile_prompts)
    assert all(seed.profile_id == "example" for seed in seeds)
    assert all(
        seed.canonical_answer == "That information is not known, so I should not guess."
        for seed in seeds
        if seed.facet == Facet.UNKNOWN_FACT
    )


def test_generation_axes_have_systematic_coverage():
    seeds = list(generate_seeds(count=500, random_seed=3, identity_profile=MARIN_IDENTITY_PROFILE))

    assert len({seed.entropy.register for seed in seeds}) >= 5
    assert len({seed.entropy.turn_pattern for seed in seeds}) == 2
    assert len({seed.entropy.question_strategy for seed in seeds}) >= 8
    assert all("Return only the conversation" in seed.generation_prompt for seed in seeds)
    assert all(f'Assistant: "{seed.canonical_answer}"' in seed.generation_prompt for seed in seeds)
    assert all("animal" not in seed.generation_prompt.lower() for seed in seeds)
    assert all("checkpoint" not in seed.generation_prompt.lower() for seed in seeds)


def test_cli_uses_retrieved_human_phrasing(tmp_path):
    phrasing_path = tmp_path / "phrasing.jsonl"
    profile_path = tmp_path / "profile.json"
    output_path = tmp_path / "seeds.jsonl"
    phrasing_path.write_text(
        "\n".join(
            [
                json.dumps({"response": "Wait, who made you again?", "source": "realitytest"}),
                json.dumps({"text": "Be honest: are you actually Claude?", "source": "wildchat"}),
            ]
        )
        + "\n"
    )
    profile_path.write_text(json.dumps(dataclasses.asdict(MARIN_IDENTITY_PROFILE)))

    main(
        [
            "--output",
            str(output_path),
            "--count",
            "20",
            "--random-seed",
            "4",
            "--phrasing-input",
            str(phrasing_path),
            "--identity-profile",
            str(profile_path),
        ]
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 20
    assert {row["human_phrasing"]["source"] for row in rows} == {"realitytest", "wildchat"}
    assert all(row["human_phrasing"]["text"] in row["generation_prompt"] for row in rows)
