# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot migrations over the eval archive fleet.

FineStore keeps immutable data objects while manifests control logical visibility. Migrations make
an extra region-local 30-day snapshot before removing a table from the active manifest, and may
archive superseded evaluator-native files after validating their replacement.
"""
