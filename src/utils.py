import yaml, random, torch
import numpy as np


# ===========================================================
# YAML Loader
# ===========================================================

def load_yaml(fpath: str):
    """Load YAML file and return it as a Python dict."""
    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================
# Seed Fix for Reproducibility
# ===========================================================

def set_seed(seed: int):
    """
    Fix seed for reproducibility.
    Applies to Python, NumPy, PyTorch (CPU/GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
