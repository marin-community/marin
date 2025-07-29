"""
Filter configurations for the Tulu dataset (allenai/tulu-3-sft-mixture).

Based on analysis, there are approximately 40 examples out of 900K+ that contain
identity branding patterns like "My name is Tülu" and "I am trained by Ai2 researchers".

This module provides three different filtering strategies:
1. TULU_REMOVE_FILTER: Remove examples entirely
2. TULU_REPLACE_FILTER: Replace with neutral language
3. TULU_OBFUSCATE_FILTER: Replace with generic placeholders
"""

from marin.processing.data_filter import DataFilter, FilterPattern

TULU_REMOVE_FILTER = DataFilter(
    strategy="remove",
    patterns=[
        FilterPattern(r"My name is T[üu]lu", ""),
        FilterPattern(r"I am T[üu]lu", ""),
        FilterPattern(r"I'm T[üu]lu", ""),
        FilterPattern(r"called T[üu]lu", ""),
        FilterPattern(r"trained by Ai2 researchers", ""),
        FilterPattern(r"My creators at Ai2", ""),
        FilterPattern(r"created by Allen Institute", ""),
        FilterPattern(r"developed by AllenAI", ""),
        FilterPattern(r"built by AI2", ""),
    ],
)

TULU_REPLACE_FILTER = DataFilter(
    strategy="replace",
    patterns=[
        FilterPattern(r"\bMy name is T[üu]lu\b", "I am an AI assistant"),
        FilterPattern(r"\bI am T[üu]lu\b", "I am an AI assistant"),
        FilterPattern(r"\bI'm T[üu]lu\b", "I'm an AI assistant"),
        FilterPattern(r"\bcalled T[üu]lu\b", "called an AI assistant"),
        FilterPattern(r"\bT[üu]lu\b", "an AI assistant"),
        FilterPattern(r"(trained by|developed by|created by|built by)\s+Ai2\s+researchers", r"\1 my developers"),
        FilterPattern(r"(trained by|developed by|created by|built by)\s+Ai2\b", r"\1 my developers"),
        FilterPattern(r"\bMy creators at Ai2\b", "My developers"),
        FilterPattern(r"(created by|developed by)\s+Allen Institute\b", r"\1 my developers"),
        FilterPattern(r"(created by|developed by)\s+AllenAI\b", r"\1 my developers"),
        FilterPattern(r"(trained by|developed by|created by|built by)\s+AI2\b", r"\1 my developers"),
        FilterPattern(r"\bAi2\s+researchers\b", "my developers"),
        FilterPattern(r"\bAllen\s+Institute\b", "my developers"),
        FilterPattern(r"\bAllenAI\b(?=\s|[.,!?]|$)", "my developers"),
        FilterPattern(r"\bAI2\b(?=\s|[.,!?]|$)(?![A-Za-z0-9+/=]{10,})", "my developers"),
        FilterPattern(r"\bAi2\b(?=\s|[.,!?]|$)(?![A-Za-z0-9+/=]{10,})", "my developers"),
    ],
)

TULU_OBFUSCATE_FILTER = DataFilter(
    strategy="replace",
    patterns=[
        FilterPattern(r"T[üu]lu", "X"),
        FilterPattern(r"Ai2", "Y"),
        FilterPattern(r"AI2", "Y"),
        FilterPattern(r"Allen Institute", "Y"),
        FilterPattern(r"AllenAI", "Y"),
    ],
)

DEFAULT_TULU_FILTER = TULU_REPLACE_FILTER
