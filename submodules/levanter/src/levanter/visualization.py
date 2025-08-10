# This module has been moved to levanter.analysis.visualization
# This file is kept for backward compatibility

import numpy as np

from levanter.analysis.visualization import compute_and_diff_log_probs  # noqa
from levanter.analysis.visualization import compute_and_visualize_log_probs  # noqa
from levanter.analysis.visualization import visualize_log_prob_diff, visualize_log_probs


# dumb main to test it out
if __name__ == "__main__":
    np.random.seed(1)
    tokens = [["Hello", "world", "!"], ["This", "is", "a", "test", "."]]
    log_probs = np.log(np.random.uniform(size=(2, 5)))
    visualize_log_probs(tokens, log_probs, "test.html")

    # test diff
    log_probs_a = np.log(np.random.uniform(size=(2, 5)))
    log_probs_b = np.log(np.random.uniform(size=(2, 5)))
    visualize_log_prob_diff(tokens, log_probs_a, log_probs_b, "test_diff.html")
