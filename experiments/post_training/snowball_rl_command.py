# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import traceback

import click
from marin.rl.rl_progress import Verdict
from marin.rl.skyrl import SkyRLReceiptLookupError
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.snowball_rl_scoring import InvalidComparison

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
        try:
            configure_coreweave_s3()
            return super().invoke(context)
        except click.UsageError as error:
            error.exit_code = REFUSAL_EXIT_CODE
            raise
        except (InvalidComparison, SkyRLReceiptLookupError, FileNotFoundError) as error:
            click.echo(
                json.dumps(
                    {"preset": context.params["preset_name"], "verdict": Verdict.INVALID.value, "reason": str(error)}
                )
            )
            raise click.exceptions.Exit(REFUSAL_EXIT_CODE) from None
        except click.exceptions.Exit:
            raise
        except Exception:
            traceback.print_exc()
            raise click.exceptions.Exit(EXIT_CODES[None]) from None
