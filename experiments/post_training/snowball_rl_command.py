# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import click
from marin.rl.rl_progress import Verdict
from rigging.filesystem.s3_compat import configure_coreweave_s3

EXIT_CODES = {Verdict.PASS: 0, Verdict.FAIL: 1, Verdict.INVALID: 2, None: 3}
REFUSAL_EXIT_CODE = 4


class SnowballCommand(click.Command):
    def make_context(self, *args, **kwargs):
        try:
            return super().make_context(*args, **kwargs)
        except click.UsageError as error:
            error.exit_code = REFUSAL_EXIT_CODE
            raise

    def invoke(self, context):
        configure_coreweave_s3()
        try:
            return super().invoke(context)
        except click.UsageError as error:
            error.exit_code = REFUSAL_EXIT_CODE
            raise
