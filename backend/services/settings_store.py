"""Persistent runtime settings for the dashboard/backend."""

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_PATH = PROJECT_ROOT / "data" / "runtime" / "dashboard_settings.json"


class SettingsStore:
    """Load and save user-adjustable runtime settings as JSON."""

    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = path

    def load(self) -> dict:
        """Return saved settings, or an empty dict if none exist."""
        try:
            if not self.path.exists():
                return {}
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def save(self, settings: dict) -> None:
        """Atomically persist settings to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(str(tmp), str(self.path))

