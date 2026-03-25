"""Configuration profile manager: save and load JSON profiles."""
from __future__ import annotations

import json
from pathlib import Path


class ConfigManager:
    """Manages named configuration profiles stored as JSON files."""

    def __init__(self, profiles_dir):
        self.dir = Path(profiles_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, profile: dict) -> None:
        """Save *profile* to <profiles_dir>/<profile['name']>.json."""
        path = self.dir / f"{profile['name']}.json"
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)

    def load(self, name: str) -> dict:
        """Load and return the profile named *name*."""
        path = self.dir / f"{name}.json"
        with open(path) as f:
            return json.load(f)
