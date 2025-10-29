"""Simple sanity check for experiments.defaults.default_validation_sets."""

from experiments.defaults import default_validation_sets
from experiments.paloma import PALOMA_DATASETS_TO_DIR
from experiments.evals.exp1600_uncheatable_evals import ACTIVE_DATASETS


def main() -> None:
    tokenizer = "meta-llama/Meta-Llama-3.1-8B"
    validation_sets = default_validation_sets(tokenizer=tokenizer)
    available_keys = set(validation_sets.keys())

    expected_paloma = {f"paloma/{name}" for name in PALOMA_DATASETS_TO_DIR}
    expected_uncheatable = {f"uncheatable_eval/{name}" for name in ACTIVE_DATASETS}
    expected_keys = expected_paloma | expected_uncheatable

    missing = sorted(expected_keys - available_keys)
    unexpected = sorted(available_keys - expected_keys)

    print(f"default_validation_sets returned {len(validation_sets)} entries for tokenizer '{tokenizer}'.")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    if not missing and not unexpected:
        print("All expected validation sets are present.")


if __name__ == "__main__":
    main()
