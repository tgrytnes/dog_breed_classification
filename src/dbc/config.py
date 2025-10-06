from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
