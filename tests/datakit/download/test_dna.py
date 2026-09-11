# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.dna import dna_document_text


def test_dna_document_text_uses_natural_language_region_header():
    assert dna_document_text("ACGT", "promoter") == "[DNA]\n[Region: promoter]\nACGT"
