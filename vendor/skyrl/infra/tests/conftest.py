import os
import sys

# Keep legacy top-level infra imports working and expose namespace packages such
# as ``infra.rl_analysis`` exactly as they are invoked from the repository root.
INFRA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOSITORY_ROOT = os.path.dirname(INFRA_ROOT)
SKYRL_TRAIN_ROOT = os.path.join(REPOSITORY_ROOT, "skyrl-train")
sys.path.insert(0, REPOSITORY_ROOT)
sys.path.insert(0, INFRA_ROOT)
sys.path.insert(0, SKYRL_TRAIN_ROOT)
