# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate model metadata and independent tokenizer staging without accelerators."""

import argparse
import contextlib
import json
import sys

from cloud.iris.task_runtime import prepare_policy_model, prepare_policy_tokenizer
from marinskyrl.resource_locator import RolloutModelLoading
from rigging.filesystem.s3_compat import configure_coreweave_s3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--tokenizer-uri", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    inputs = parser.parse_args()
    configure_coreweave_s3()
    args = argparse.Namespace(
        model_source_uri=inputs.model_uri,
        model_source_identity=inputs.model_identity,
        runtime_profile=inputs.runtime_profile,
        rollout_model_loading=RolloutModelLoading.STREAM,
        policy_tokenizer=inputs.tokenizer_uri,
        policy_tokenizer_revision=inputs.tokenizer_revision,
    )
    with contextlib.redirect_stdout(sys.stderr):
        model = prepare_policy_model(args)
        tokenizer = prepare_policy_tokenizer(args)
    assert model is not None and tokenizer is not None
    print(
        json.dumps(
            {
                "model_uri": model.source_uri,
                "model_identity": model.source_identity,
                "tokenizer_uri": inputs.tokenizer_uri,
                "tokenizer_revision": inputs.tokenizer_revision,
                "model_metadata_staged": True,
                "tokenizer_staged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
