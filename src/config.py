"""
src/config.py
─────────────
Loads ~/.bank_parser/config.yaml and exposes a typed Config dataclass.

This is the Week 1 config pattern, intact and extended with an [nlp]
section for Week 2.

~/.bank_parser/config.yaml minimal example
-------------------------------------------
database:
  path: ~/.bank_parser/bank_statements.db

parser:
  output_dir: ~/Documents/bank_exports
  supported_banks:
    - capital_one
    - citi
    - bofa
    - robinhood

nlp:
  spacy_model: en_core_web_sm
  fuzzy_threshold: 80

Usage
-----
    from src.config import Config

    cfg = Config.load()
    print(cfg.db_path)       # PosixPath('/Users/you/.bank_parser/bank_statements.db')
    print(cfg.spacy_model)   # 'en_core_web_sm'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Well-known locations  (same as Week 1)
# ---------------------------------------------------------------------------

CONFIG_DIR  = Path.home() / ".bank_parser"
CONFIG_FILE = CONFIG_DIR  / "config.yaml"
DEFAULT_DB  = CONFIG_DIR  / "bank_data.db"

_DEFAULT_YAML = """\
database:
  path: ~/.bank_parser/bank_data.db

parser:
  output_dir: ~/Documents/bank_exports
  supported_banks:
    - capital_one
    - citi
    - bofa
    - robinhood

nlp:
  spacy_model: en_core_web_sm
  fuzzy_threshold: 80
"""

DEFAULTS = {
    "database": {"path": "default_db.sqlite"},
    "watcher": {"folder": "./statements", "log_file": "watcher.log"},
    "parser": {"default_year": 2023}
}

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        db_section      = config_dict.get("database", {})
        parser_section  = config_dict.get("parser", {})
        watcher_section = config_dict.get("watcher", {})
        nlp_section     = config_dict.get("nlp", {})

        # Store raw values so Config({}) returns exactly what was passed in
        self.db_path         = db_section.get("path", DEFAULTS["database"]["path"])
        self.watch_folder    = watcher_section.get("folder",   DEFAULTS["watcher"]["folder"])
        self.watch_log_file  = watcher_section.get("log_file", DEFAULTS["watcher"]["log_file"])
        self.default_year    = parser_section.get("default_year", DEFAULTS["parser"]["default_year"])
        self.output_dir      = parser_section.get("output_dir", "~/statements")
        self.supported_banks = parser_section.get("supported_banks", ["capital_one", "citi", "bofa", "robinhood"])
        self.spacy_model     = nlp_section.get("spacy_model", "en_core_web_sm")
        self.fuzzy_threshold = int(nlp_section.get("fuzzy_threshold", 80))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load from *config_path* (default: ~/.bank_parser/config.yaml).

        Creates the file with sensible defaults on first run so the user
        doesn't need to create it manually.
        """
        path = Path(config_path) if config_path else CONFIG_FILE

        if not path.exists():
            log.info("Config not found at %s – writing defaults.", path)
            cls._write_default(path)

        raw = yaml.safe_load(path.read_text()) or {}

        cfg = cls(raw)
        # Expand ~ but don't resolve() — resolve() adds /private/ prefix on macOS symlinks
        cfg.db_path = Path(cfg.db_path).expanduser()
        cfg.output_dir = Path(cfg.output_dir).expanduser()
        log.debug("Config loaded  db_path=%s", cfg.db_path)
        return cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_default(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_YAML)
        log.info("Default config written to %s", path)

    def ensure_db_dir(self) -> None:
        """Create the parent directory for the DB file if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path=None) -> "Config":
    """
    Load config from YAML, falling back to ~/sqlLite_DB/bank_data.db if not found.

    This is the function imported by DatabaseManager.
    """
    path = Path(config_path) if config_path else CONFIG_FILE

    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        db_raw = raw.get("database", {}).get("path") or str(Path.home() / "sqlLite_DB" / "bank_data.db")
    else:
        db_raw = str(Path.home() / "sqlLite_DB" / "bank_data.db")

    return Config({"database": {"path": db_raw}})
