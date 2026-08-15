"""Load algorithm/config.yaml and resolve its paths against the algorithm/ dir."""

from __future__ import annotations

from pathlib import Path

import yaml

ALGO_DIR = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, data: dict):
        self._d = data

    def __getitem__(self, k):
        return self._d[k]

    def get(self, k, default=None):
        return self._d.get(k, default)

    def path(self, key: str) -> Path:
        """Resolve a path from the `paths:` block against algorithm/."""
        rel = self._d["paths"][key]
        return (ALGO_DIR / rel).resolve()


def load(path: Path | None = None) -> Config:
    p = Path(path) if path else (ALGO_DIR / "config.yaml")
    return Config(yaml.safe_load(p.read_text()))
