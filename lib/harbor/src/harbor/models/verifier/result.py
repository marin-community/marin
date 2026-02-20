# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class VerifierResult(BaseModel):
    rewards: dict[str, float | int] | None = None
