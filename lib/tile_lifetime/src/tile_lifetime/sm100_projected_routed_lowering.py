# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose projected Selection and routed streaming SM100 candidates."""

from dataclasses import dataclass
from itertools import product

from tile_lifetime.msa_recovery import NaturalProjectedRoutedAttentionCompilation
from tile_lifetime.sm100_routed_lowering import (
    SM100RoutedStreamingLowering,
    default_sm100_routed_schedules,
    lower_sm100_routed_streaming_program,
)
from tile_lifetime.sm100_selection_lowering import (
    SM100ProjectedSelectionLowering,
    default_sm100_selection_schedules,
    lower_sm100_projected_selection,
)


@dataclass(frozen=True)
class LoweredAffineIndexDomain:
    """Physical binding of generic left/right positions to an index predicate."""

    left_axis: str
    right_axis: str
    predicate: str
    left_count: int
    right_count: int
    left_position_offset: int
    right_position_offset: int

    @property
    def bottom_right_aligned(self) -> bool:
        """Whether the final left and right positions are identical."""
        left_stop = self.left_position_offset + self.left_count
        right_stop = self.right_position_offset + self.right_count
        return left_stop == right_stop

    def left_position(self, left_index: int) -> int:
        """Map a left-domain index to its absolute position."""
        if left_index < 0 or left_index >= self.left_count:
            raise ValueError(f"left index {left_index} is outside [0, {self.left_count})")
        return self.left_position_offset + left_index

    def right_position(self, right_index: int) -> int:
        """Map a right-domain index to its absolute position."""
        if right_index < 0 or right_index >= self.right_count:
            raise ValueError(f"right index {right_index} is outside [0, {self.right_count})")
        return self.right_position_offset + right_index

    def allows(self, left_index: int, right_index: int) -> bool:
        """Evaluate the lowered predicate for one logical index pair."""
        if self.predicate != "left_greater_equal_right":
            raise ValueError(f"unsupported index predicate {self.predicate!r}")
        return self.left_position(left_index) >= self.right_position(right_index)


@dataclass(frozen=True)
class SM100ProjectedRoutedCandidate:
    """One cross-product point over generic Selection and streaming skeletons."""

    selection: SM100ProjectedSelectionLowering
    streaming: SM100RoutedStreamingLowering
    position_domain: LoweredAffineIndexDomain


@dataclass(frozen=True)
class SM100ProjectedRoutedCandidateSet:
    """Bounded physical candidates derived from one natural compilation."""

    source: NaturalProjectedRoutedAttentionCompilation
    candidates: tuple[SM100ProjectedRoutedCandidate, ...]


def lower_sm100_projected_routed_candidates(
    compilation: NaturalProjectedRoutedAttentionCompilation,
) -> SM100ProjectedRoutedCandidateSet:
    """Enumerate generic projected-Selection and routed-streaming candidates."""
    selection_program = compilation.recovered.relation_selection
    selection_candidates = tuple(
        lower_sm100_projected_selection(selection_program, schedule) for schedule in default_sm100_selection_schedules()
    )
    streaming_candidates = tuple(
        lower_sm100_routed_streaming_program(compilation.streaming_program, compilation.relation, schedule)
        for schedule in default_sm100_routed_schedules()
    )
    position_domain = LoweredAffineIndexDomain(
        left_axis=selection_program.token_restriction.left_axis,
        right_axis=selection_program.token_restriction.right_axis,
        predicate=selection_program.token_restriction.predicate,
        left_count=selection_program.source_count,
        right_count=selection_program.resolved_right_count,
        left_position_offset=selection_program.left_position_offset,
        right_position_offset=selection_program.right_position_offset,
    )
    _validate_composition(selection_candidates, streaming_candidates, position_domain)
    candidates = tuple(
        SM100ProjectedRoutedCandidate(selection=selection, streaming=streaming, position_domain=position_domain)
        for selection, streaming in product(selection_candidates, streaming_candidates)
    )
    return SM100ProjectedRoutedCandidateSet(source=compilation, candidates=candidates)


def _validate_composition(
    selection_candidates: tuple[SM100ProjectedSelectionLowering, ...],
    streaming_candidates: tuple[SM100RoutedStreamingLowering, ...],
    position_domain: LoweredAffineIndexDomain,
) -> None:
    if not selection_candidates or not streaming_candidates:
        raise ValueError("projected routed lowering requires nonempty candidate families")
    selection = selection_candidates[0]
    streaming = streaming_candidates[0]
    if selection.program.source_count != streaming.query_length:
        raise ValueError("projected Selection left domain does not match streaming query domain")
    if selection.program.resolved_right_count != streaming.key_length:
        raise ValueError("projected Selection right domain does not match streaming key domain")
    if selection.program.group_count != streaming.key_value_heads:
        raise ValueError("projected Selection groups do not match streaming key/value heads")
    if selection.program.selected_count != streaming.selected_count:
        raise ValueError("projected Selection width does not match routed streaming edge slots")
    if not streaming.score_map.causal:
        raise ValueError("the projected causal relation requires a causal streaming DomainRestriction")
    if position_domain.predicate != "left_greater_equal_right":
        raise ValueError("the initial SM100 composition requires left-position >= right-position")
    if not position_domain.bottom_right_aligned:
        raise ValueError("the initial asymmetric causal composition requires bottom-right alignment")
