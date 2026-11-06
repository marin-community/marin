# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from forge.model.workpiece import Submission


@dataclass(frozen=True, slots=True)
class ValidateRequest:
    submission: Submission
