# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reformat a pinned prompt/response dataset into a separate chat variant."""

import argparse
from functools import partial

from marin.datakit.sft_conversion import ConversionPlan, convert_chat_variant
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = ConversionPlan.model_validate_json(StoragePath(args.plan).read_bytes())
    step = StepSpec(
        name=f"datakit/conversion/{plan.variant_id}",
        hash_attrs=plan.model_dump(mode="json"),
        fn=partial(convert_chat_variant, plan),
        output_path_prefix=args.output_prefix,
    )
    print(step.output_path)
    if args.run:
        StepRunner().run([step], max_concurrent=1)


if __name__ == "__main__":
    main()
