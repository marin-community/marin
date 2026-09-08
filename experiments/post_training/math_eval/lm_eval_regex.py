# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GSM8K comparison channels matching the pinned lm-eval regex configuration."""

import re

SOURCE_REVISION = "b954108c9baaaa934b4ad842033b31a97ee30816"
CONFIG_SHA256 = "7c9dd2561fe334a55f62bdc6a876f045e6dc7f53714c7a9db506b89c6de6bc96"
FILTER_SHA256 = "eb976bb36d5d3e93faafe6d065678e24b4474e78cd4fce1e730e2b8f49b9fda0"
STRICT = r"The answer is (\-?[0-9\.\,]+)."
FLEXIBLE = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"


def normalize(text: str) -> str:
    for pattern in (",", r"\$", r"(?s).*#### ", r"\.$"):
        text = re.sub(pattern, "", text)
    return text.lower().strip()


def scores(response: str, gold: str) -> tuple[float, float]:
    """Keep first strict and last flexible matches, using exact string comparison."""
    strict = re.findall(STRICT, response)
    flexible = re.findall(FLEXIBLE, response)
    predictions = (
        strict[0] if strict else "[invalid]",
        next((part for part in flexible[-1] if part), "[invalid]") if flexible else "[invalid]",
    )
    return float(normalize(predictions[0]) == normalize(gold)), float(normalize(predictions[1]) == normalize(gold))
