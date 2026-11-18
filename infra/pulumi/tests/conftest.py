# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks


class RecordingMocks(Mocks):
    def __init__(self) -> None:
        self.resources: list[MockResourceArgs] = []

    def new_resource(self, args: MockResourceArgs):
        self.resources.append(args)
        return args.name, args.inputs

    def call(self, args: MockCallArgs):
        return args.args


@pytest.fixture
def recording_mocks():
    return RecordingMocks()
