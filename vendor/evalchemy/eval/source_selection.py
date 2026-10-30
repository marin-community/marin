"""Select immutable source identities after parent prompt and seed resolution."""

from collections.abc import Iterable

from eval.contracts.eval_spec import EvalSpec
from eval.contracts.sample_manifest import SampleCoverageError, canonical_json_identity


def source_key(spec: EvalSpec) -> str:
    """Return the canonical namespace/source identity, shared by all repeats."""
    return canonical_json_identity({"namespace": spec.namespace, "source_id": spec.source_id})


def selected_specs(sources: tuple[Iterable[EvalSpec], ...], keys: tuple[str, ...]) -> tuple[EvalSpec, ...]:
    """Select already-rendered items without changing prompts, seeds, ordinals, or repeats."""
    # Render the full parent source before filtering. Few-shot RNG progression
    # and longest-prompt output budgeting must be identical to a full run.
    specs = tuple(spec for source in sources for spec in source)
    wanted = set(keys)
    selected = []
    found = set()
    for spec in specs:
        key = source_key(spec)
        if key in wanted:
            selected.append(spec)
            found.add(key)
    if missing := wanted - found:
        raise SampleCoverageError(f"selected source keys absent from parent dataset: {sorted(missing)}")
    return tuple(selected)

