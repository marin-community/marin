# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from marin.datakit.download.openstax import (
    OPENSTAX_BOOKS,
    OPENSTAX_PHYSICS_ARCHIVE_URL,
    OPENSTAX_PHYSICS_MANIFEST,
    OpenStaxStageConfig,
    cnxml_to_markdown,
    stage_openstax_cnxml,
)
from marin.datakit.ingestion_manifest import UsagePolicy


def test_cnxml_to_markdown_preserves_scientific_content_and_structure():
    payload = b"""\
<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML">
  <title>Kinematics</title>
  <metadata xmlns:md="http://cnx.rice.edu/mdml">
    <md:title>Metadata title must not become body text</md:title>
  </metadata>
  <content>
    <section>
      <title>Constant acceleration</title>
      <para>For constant acceleration, <m:math><m:mi>v</m:mi><m:mo>=</m:mo><m:mi>u</m:mi>
      <m:mo>+</m:mo><m:mi>a</m:mi><m:mi>t</m:mi></m:math>.</para>
      <figure><caption>Velocity changes linearly with time.</caption></figure>
      <code>v = u + a * t</code>
    </section>
  </content>
</document>
"""

    title, text = cnxml_to_markdown(payload)

    assert title == "Kinematics"
    assert text == "\n\n".join(
        (
            "# Kinematics",
            "## Constant acceleration",
            "For constant acceleration, v = u + a t .",
            "Velocity changes linearly with time.",
            "```\nv = u + a * t\n```",
        )
    )
    assert "Metadata title" not in text


def test_stage_openstax_rejects_non_training_policy_before_download(tmp_path: Path):
    blocked_policy = OPENSTAX_PHYSICS_MANIFEST.policy.model_copy(update={"usage_policy": UsagePolicy.BLOCKED})
    blocked_manifest = OPENSTAX_PHYSICS_MANIFEST.model_copy(
        update={
            "policy": blocked_policy,
        }
    )
    config = OpenStaxStageConfig(
        manifest=blocked_manifest,
        archive_url=OPENSTAX_PHYSICS_ARCHIVE_URL,
        output_path=str(tmp_path),
        revision=OPENSTAX_BOOKS["openstax/physics"].revision,
    )

    with pytest.raises(ValueError, match="is not approved for training"):
        stage_openstax_cnxml(config)

    assert list(tmp_path.iterdir()) == []
