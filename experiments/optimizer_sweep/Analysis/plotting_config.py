"""Centralized plotting styles for optimizer comparison figures.

Exports:
- `color_map`: consistent color per optimizer key.
- `correct_name`: readable names for optimizer keys.
- `line_style`: dashed/solid style per optimizer. dashed optimizers are scalar-level and solid optimizers are matrix-level.
"""

color_map = {
    "mars": "#1f77b4",  # blue
    "muon": "#ff7f0e",  # orange
    "lion": "#2ca02c",  # green
    "adamw": "#d62728",  # red
    "nadamw": "#9467bd",  # purple
    "kron": "#8c564b",  # brown
    "scion": "#e377c2",  # pink
    "cautious": "#7f7f7f",  # gray
    "soape": "#bcbd22",  # yellow-green
    "sophia": "#17becf",  # cyan
    "mini": "#aec7e8",  # light blue
}

correct_name = {
    "adamw": "AdamW",
    "lion": "Lion",
    "mini": "Adam-Mini",
    "scion": "Scion",
    "cautious": "Cautious",
    "mars": "Mars",
    "nadamw": "NAdamW",
    "muon": "Muon",
    "soape": "Soap",
    "kron": "Kron",
}


line_style = {
    "adamw": "--",
    "mars": "--",
    "nadamw": "--",
    "muon": "-",
    "soape": "-",
    "kron": "-",
    "lion": "--",
    "mini": "--",
    "scion": "-",
    "cautious": "--",
}