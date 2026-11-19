# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed publication and derived-artifact guards for the private data tier.

Every shared publish helper calls ``require_public_data`` (or
``require_public_source``) before bytes leave for any upload, release or image
registry, so a ``PRIVATE_ALWAYS`` source is refused at the helper, not per
caller. Derived populations must keep the strongest restriction of their
inputs: ``require_no_restriction_downgrade`` refuses a flipped or weakened tag.
"""

from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from infra.marina.apps.rl_data_catalog.server.restrictions import require_public_data


def require_public_source(source: RlDataSource, destination: str = "") -> None:
    """Refuse publishing anything derived from a PRIVATE_ALWAYS source."""
    require_public_data(source.info.data_restriction, source.info.id, destination)


def require_no_restriction_downgrade(original: SourceInfo, derived: SourceInfo, relation: str) -> None:
    """Refuse a derived population whose restriction is weaker than its input's."""
    if original.data_restriction is DataRestriction.PRIVATE_ALWAYS and (
        derived.data_restriction is not DataRestriction.PRIVATE_ALWAYS
    ):
        raise ValueError(
            f"Restriction downgrade refused: {relation} of PRIVATE_ALWAYS source {original.id} "
            f"declares {derived.data_restriction.value}"
        )


def configured_restriction(source_id: str) -> DataRestriction:
    """The tier a source's code declaration states; an unknown source fails closed."""
    source = _configured_source(source_id)
    if source is None:
        raise ValueError(f"Unknown source declares no restriction tier: {source_id}")
    return source.info.data_restriction


def _configured_source(source_id: str):
    """The code-configured RlDataSource for one id, or None.

    The local import breaks the cycle with ``sources``, which binds these guards
    into the source graph. The restriction tier is a code declaration, so the
    lookup reads the configured populations directly: assembling the full estate
    would demand registry documents just to answer a tier question, which no
    offline unit test can provide. Both sides match through the canonical
    identity, since selection inputs carry original spellings.
    """
    from experiments.post_training.task_curation.sources import configured_sources  # noqa: PLC0415
    from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id  # noqa: PLC0415

    canonical = canonical_source_id(source_id)
    return next(
        (source for source in configured_sources() if canonical_source_id(source.info.id) == canonical),
        None,
    )
