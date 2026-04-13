"""Configuration profile manager: save and load JSON profiles."""
from __future__ import annotations

import json
from pathlib import Path

from src.config.profile import Profile


class ConfigManager:
    """Manages named configuration profiles stored as JSON files."""

    def __init__(self, profiles_dir):
        self.dir = Path(profiles_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Legacy dict-based API (kept for backward compatibility)
    # ------------------------------------------------------------------

    def save(self, profile: dict) -> None:
        """Save *profile* dict to <profiles_dir>/<profile['name']>.json."""
        path = self.dir / f"{profile['name']}.json"
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)

    def load(self, name: str) -> dict:
        """Load and return the profile named *name* as a raw dict."""
        path = self.dir / f"{name}.json"
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Structured Profile API
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        """Return the lowest unused auto-increment ID."""
        max_id = 0
        for path in self.dir.glob("*.json"):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                max_id = max(max_id, int(d.get("id", 0)))
            except Exception:
                pass
        return max_id + 1

    def save_profile(self, profile: Profile) -> None:
        """Serialise and save a Profile object.  Auto-assigns ID if 0."""
        if profile.id == 0:
            profile.id = self._next_id()
        self.save(profile.to_dict())

    def load_profile(self, name: str) -> Profile:
        """Load and deserialise a Profile object by name."""
        return Profile.from_dict(self.load(name))

    def list_profiles(self) -> list[str]:
        """Return a sorted list of all saved profile names."""
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def list_profiles_full(self) -> list[dict]:
        """Return [{id, name}, ...] sorted by id."""
        result = []
        for path in self.dir.glob("*.json"):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                result.append({"id": int(d.get("id", 0)), "name": d["name"]})
            except Exception:
                pass
        return sorted(result, key=lambda x: x["id"])

    def delete_profile(self, name: str) -> None:
        """Delete the profile file for *name*.

        Raises:
            FileNotFoundError: if no profile with that name exists.
        """
        path = self.dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Profile '{name}' not found in {self.dir}")
        path.unlink()
