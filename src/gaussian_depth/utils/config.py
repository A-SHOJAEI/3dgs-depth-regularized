"""Configuration utilities."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(config: dict):
    """Create output directories."""
    for key in ["output_dir", "results_dir", "logs_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
