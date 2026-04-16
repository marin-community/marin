# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Institutional Books 1.0 dataset."""

from experiments.defaults import default_download

institutional_books_raw = default_download(
    name="raw/institutional-books-1.0",
    hf_dataset_id="institutional/institutional-books-1.0",
    revision="d2f504a",
    override_output_path="raw/institutional-books-d2f504a",
)
