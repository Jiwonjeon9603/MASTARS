import os
from typing import Any, Dict, Optional

import yaml


class Config:
    """Attribute-style access to a flat dict of hyper-parameters."""

    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def update(self, **kwargs: Any) -> "Config":
        self.__dict__.update(kwargs)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        lines = [f"    {k}: {v}" for k, v in self.__dict__.items()]
        return "[ utils/config ] Config:\n" + "\n".join(lines)


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a flat YAML config; `overrides` (non-None values) take precedence."""
    with open(path, "r") as f:
        spec = yaml.safe_load(f) or {}
    if not isinstance(spec, dict):
        raise ValueError(f"Config file {path} must contain a flat mapping")
    if overrides:
        spec.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**spec)
