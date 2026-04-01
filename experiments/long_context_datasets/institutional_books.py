# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Institutional Books 1.0 dataset."""

from marin.datakit.download.institutional_books import download_institutional_books_step

institutional_books_raw = download_institutional_books_step().as_executor_step().as_input_name()
