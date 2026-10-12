# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select verified closing-only supervision without retokenizing model history."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from rolloutengine.structure_closure import ClosureState, PrefixStatus


class ClosureTargetKind(StrEnum):
    WRONG_TOKEN = "wrong_token"
    MISSING_SUFFIX = "missing_suffix"


@dataclass(frozen=True)
class ClosureTarget:
    kind: ClosureTargetKind
    response_position: int
    repair_token_ids: tuple[int, ...]


def verified_closure_target(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    *,
    structure_prefix: str,
    inspect: Callable[[str], ClosureState],
    verify_complete: Callable[[str], bool],
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
) -> ClosureTarget | None:
    """Select the first illegal token or a verified repair at generation end.

    Inputs describe one completed model turn, not an infrastructure failure or
    a stitched conversation. The structure starts at the end of the prompt.
    ``verify_complete`` must check required content, not merely syntax. A
    syntactically valid but premature close is not repaired by this selector.
    Text after a completed outer structure is not a closure error, including
    when the closing sequence and trailing text share one token.

    For FTPO, retain history through ``response_position`` exclusively. For
    missing-suffix SFT, append the repair there and mask all preceding tokens.
    No sampled history is re-encoded. Incomplete Unicode boundaries are not
    supervision positions; rewrites of stable decoded history raise.
    """
    prompt = list(prompt_ids)
    response = list(response_ids)
    prompt_text = decode(prompt)
    if not prompt_text.endswith(structure_prefix):
        raise ValueError("Structure prefix does not match the exact prompt tokens")
    previous_text = prompt_text
    structure = structure_prefix
    state = inspect(structure)
    if state.status != PrefixStatus.EXTENDABLE:
        return None
    position = len(response)
    kind = ClosureTargetKind.MISSING_SUFFIX
    for index in range(len(response)):
        text = decode(prompt + response[: index + 1])
        if not text.startswith(previous_text):
            raise ValueError("Decoded response rewrites its conditioning prefix")
        # Byte-split characters decode to a temporary replacement character.
        # Wait for a stable boundary without changing the sampled token history.
        if text.endswith("\ufffd"):
            continue
        next_structure = structure_prefix + text[len(prompt_text) :]
        next_state = inspect(next_structure)
        if next_state.completion_offset is not None:
            if not next_state.completion_guard:
                return None
        elif state.completion_guard or next_state.status == PrefixStatus.INVALID:
            position = index
            kind = ClosureTargetKind.WRONG_TOKEN
            break
        structure, state, previous_text = next_structure, next_state, text
    if kind == ClosureTargetKind.MISSING_SUFFIX and state.completion_offset is not None:
        return None
    suffix = state.completion_guard or state.closing_suffix
    if (state.status != PrefixStatus.EXTENDABLE and not state.completion_guard) or not suffix:
        return None
    completed = structure + suffix
    if inspect(completed).status != PrefixStatus.COMPLETE or not verify_complete(completed):
        return None
    repair = encode(suffix)
    conditioning = prompt + response[:position]
    conditioning_text = decode(conditioning)
    if conditioning_text != previous_text:
        return None
    if not repair or decode(conditioning + repair) != conditioning_text + suffix:
        raise ValueError("Repair tokens do not preserve history and verified suffix")
    for stop in range(1, len(repair) + 1):
        partial = decode(conditioning + repair[:stop])
        if not partial.startswith(conditioning_text):
            raise ValueError("Intermediate repair token changes conditioning history")
        if not (conditioning_text + suffix).startswith(partial):
            raise ValueError("Intermediate repair token does not extend the verified suffix")
    return ClosureTarget(kind, position, tuple(repair))


def verified_closure_candidates(
    target: ClosureTarget,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    candidate_ids: Sequence[int],
    *,
    decode: Callable[[list[int]], str],
) -> tuple[int, ...]:
    """Keep FTPO alternatives that advance the verified suffix in exact context.

    Candidates need not use the encoder's canonical token segmentation. Each
    must emit a nonempty prefix of the verified repair and leave history intact.
    A multi-token repair is not replaced by an arbitrary single closing token.
    """
    if target.kind != ClosureTargetKind.WRONG_TOKEN:
        return ()
    history = list(prompt_ids) + list(response_ids[: target.response_position])
    before = decode(history)
    repaired = decode(history + list(target.repair_token_ids))
    rejected = response_ids[target.response_position]
    chosen = []
    for candidate in dict.fromkeys(candidate_ids):
        if candidate < 0 or candidate == rejected:
            continue
        text = decode([*history, candidate])
        if text.startswith(before) and len(text) > len(before) and repaired.startswith(text):
            chosen.append(candidate)
    return tuple(chosen)
