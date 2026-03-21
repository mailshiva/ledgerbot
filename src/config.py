"""
Config Loader
Reads ~/.bank_parser/config.yaml and exposes settings to the rest of the app.

Config file location: ~/.bank_parser/config.yaml

Minimal example:
    database:
      path: /Users/you/Documents/bank_data/transactions.db

Full example:
    database:
      path: /Users/you/Documents/bank_data/transactions.db

    watcher:
      folder:   /Users/you/Downloads/statements
      log_file: /Users/you/.bank_parser/watcher.log

    parser:
      default_year: 2026
      confidence_threshold: 0.5

    cli:
      date_format: "%Y-%m-%d"
"""

import os
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

CONFIG_DIR  = Path.home() / ".bank_parser"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

DEFAULTS = {
    "database": {
        "path": str(Path.home() / ".bank_parser" / "transactions_raw.db"),
    },
    "watcher": {
        "folder":   str(Path.home() / "statements"),
        "log_file": str(Path.home() / ".bank_parser" / "watcher.log"),
    },
    "parser": {
        "default_year": 2026,
        "confidence_threshold": 0.5,
    },
    "cli": {
        "date_format": "%Y-%m-%d",
    },
}


class Config:
    """Thin wrapper around the YAML config with dot-access helpers."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def db_path(self) -> str:
        return self._get("database", "path")

    @property
    def watch_folder(self) -> str:
        return self._get("watcher", "folder")

    @property
    def watch_log_file(self) -> str:
        return self._get("watcher", "log_file")

    @property
    def default_year(self) -> int:
        return int(self._get("parser", "default_year"))

    @property
    def confidence_threshold(self) -> float:
        return float(self._get("parser", "confidence_threshold"))

    @property
    def date_format(self) -> str:
        return self._get("cli", "date_format")

    def _get(self, section: str, key: str) -> Any:
        return (
            self._data
            .get(section, {})
            .get(key, DEFAULTS[section][key])
        )

    def __repr__(self) -> str:
        return f"<Config db_path={self.db_path!r} watch_folder={self.watch_folder!r}>"


def load_config() -> Config:
    """
    Load ~/.bank_parser/config.yaml, creating it with defaults if absent.
    If pyyaml is not installed, returns a Config with all defaults and
    prints a one-time warning so the user knows what to install.
    """
    if not _YAML_AVAILABLE:
        print(
            "⚠️  pyyaml is not installed — using default config values.\n"
            "    Run: pip install pyyaml"
        )
        return Config({})

    if not CONFIG_FILE.exists():
        _create_default_config()

    with open(CONFIG_FILE, "r") as f:
        raw = yaml.safe_load(f) or {}

    return Config(raw)


def _create_default_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    default_yaml = f"""\
# Bank Statement Parser - Configuration
# Location: {CONFIG_FILE}

database:
  # Path to the SQLite database file (outside the project folder).
  path: {DEFAULTS['database']['path']}

watcher:
  # Folder the watcher scans for new PDF statements.
  folder: {DEFAULTS['watcher']['folder']}
  # Log file written by the watcher on each run.
  log_file: {DEFAULTS['watcher']['log_file']}

parser:
  # Year assumed when a PDF has no year in its dates (e.g. Robinhood MM/DD format).
  default_year: {DEFAULTS['parser']['default_year']}
  # Minimum confidence score (0-1) for a parsed transaction to be accepted.
  confidence_threshold: {DEFAULTS['parser']['confidence_threshold']}

cli:
  # Date display format in terminal output.
  date_format: "{DEFAULTS['cli']['date_format']}"
"""
    with open(CONFIG_FILE, "w") as f:
        f.write(default_yaml)

    print(f"📝  Created default config at: {CONFIG_FILE}")
    print(f"    DB will be stored at      : {DEFAULTS['database']['path']}")
    print(f"    Watcher will scan         : {DEFAULTS['watcher']['folder']}")
    print(f"    Edit {CONFIG_FILE} to customise.\n")
