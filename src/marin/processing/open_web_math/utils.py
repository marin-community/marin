#!/usr/bin/env python3
"""
This file was taken from https://github.com/keirp/OpenWebMath
"""
import re

import numpy as np
import yaml


def has_style(style, styles):
    """Does the style string contain any of the styles?
    This function is robust to variations in the spaces between the styles.
    """
    # Remove any spaces.
    style = style.replace(" ", "")
    styles = [s.replace(" ", "") for s in styles]
    for s in styles:
        if s in style:
            return True
    return False


def word_wrap(text, char_width=20):
    """Wrap text to a given width, not breaking words."""
    if not text:
        return ""

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if len(" ".join([*current_line, word])) <= char_width:
            current_line.append(word)
        else:
            if current_line:  # Check if current_line is not empty
                lines.append(" ".join(current_line))
            current_line = [word]

            # Handle the case when the word is longer than the character width
            while len(current_line[0]) > char_width:
                lines.append(current_line[0][:char_width])
                current_line[0] = current_line[0][char_width:]

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


class ReplacementManager:
    """This replacement manager simply adds tags next to the instances of the text.
    It contains a method to remove these tags."""

    def __init__(self):
        self.tags = []

    def add_replacement(self, text, tag="default"):
        self.tags.append(tag)
        return f"§§{tag}§§" + text

    def remove_tags(self, text):
        tag_regex = "|".join(f"§§{tag}§§" for tag in self.tags)
        return re.sub(tag_regex, "", text)

    def has_tag(self, text, tag):
        return f"§§{tag}§§" in text


class Config:
    """A simple config object that loads a config from a YAML file and
    presents as a dictionary"""

    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def sample_from_list(self, input_list):
        """Sample from a list of (probability, value) tuples."""
        probabilities = [p for p, _ in input_list]
        values = [v for _, v in input_list]
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return np.random.choice(values, p=probabilities)  # noqa: NPY002

    def _sample(self, config):
        # For every value that has a type of list, first check it is in the format of:
        # - (probability, value)
        # - (probability, value)
        # - ...
        # And the probabilities sum to 1.
        # Then sample from the list.
        sampled_config = {}
        for key, value in config.items():
            # print the type of the value
            if isinstance(value, list):
                # Check the format of the list.
                # Check the probabilities sum to 1.
                # Sample from the list.
                sampled_config[key] = self.sample_from_list(value)
            elif isinstance(value, dict):
                sampled_config[key] = self._sample(value)
            else:
                sampled_config[key] = value
        return sampled_config

    def sample(self):
        return self._sample(self.config)
